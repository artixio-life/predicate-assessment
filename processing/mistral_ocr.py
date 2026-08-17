# processing/mistral_ocr.py
#
# OCR fallback for scanned/low-text-density PDFs that the primary
# pdf-inspector/pypdf text-layer extraction can't read.
#
# Ported near-verbatim from regulatory-explorer's
# processing/mistral_ocr_client.py — self-contained (only needs `requests`
# and MISTRAL_API_KEY), no dependency on that app's infra.
#
# 4-step pattern recommended by Mistral:
#   1. Upload raw bytes to Files API (multipart, no base64 inflation)
#   2. GET signed URL (tiny request, 1h expiry)
#   3. POST /v1/ocr referencing the URL (small JSON body, no payload)
#   4. DELETE uploaded file (best-effort cleanup)
#
# Large PDFs (>_MAX_CHUNK_BYTES) are split page-by-page, each chunk OCR'd in
# parallel, then merged in page order.
from __future__ import annotations

import io
import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

_MISTRAL_API_BASE = "https://api.mistral.ai/v1"
_OCR_MODEL = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
_MAX_CHUNK_BYTES = int(os.getenv("MISTRAL_MAX_CHUNK_BYTES", str(45 * 1024 * 1024)))  # 45 MB
_REQUEST_TIMEOUT = int(os.getenv("MISTRAL_OCR_TIMEOUT_S", "120"))


class MistralOcrError(Exception):
    pass


def _api_key() -> str | None:
    return os.getenv("MISTRAL_API_KEY", "").strip() or None


def is_configured() -> bool:
    return bool(_api_key())


def _auth_headers() -> dict:
    key = _api_key()
    if not key:
        raise MistralOcrError("MISTRAL_API_KEY is not set")
    return {"Authorization": f"Bearer {key}"}


def _build_table_map(pages: list) -> dict:
    table_map: dict[str, str] = {}
    for page in pages:
        for tbl in page.get("tables", []):
            tbl_id = tbl.get("id", "")
            content = tbl.get("content", "")
            if not tbl_id:
                continue
            table_map[tbl_id] = content
            bare = tbl_id[:-3] if tbl_id.endswith(".md") else tbl_id
            table_map[bare] = content
            table_map[bare + ".md"] = content
    return table_map


def _inline_tables(text: str, table_map: dict) -> str:
    def replacer(m):
        return table_map.get(m.group(2), m.group(0))
    return re.sub(r'\[([^\]]*)\]\(([^)]+\.md)\)', replacer, text)


def _split_pdf(file_bytes: bytes, max_bytes: int) -> list[bytes]:
    """Split a PDF into page-groups each under max_bytes."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(io.BytesIO(file_bytes))
    chunks: list[bytes] = []
    group: list[int] = []

    for i in range(len(reader.pages)):
        group.append(i)
        writer = PdfWriter()
        for p in group:
            writer.add_page(reader.pages[p])
        buf = io.BytesIO()
        writer.write(buf)
        if buf.tell() >= max_bytes or i == len(reader.pages) - 1:
            buf.seek(0)
            chunks.append(buf.read())
            group = []

    return chunks or [file_bytes]


def _ocr_chunk(file_bytes: bytes, filename: str) -> str:
    """Run the 4-step Mistral OCR flow on a single file and return markdown."""
    import requests

    hdrs = _auth_headers()
    ctype = "application/pdf" if filename.lower().endswith(".pdf") else "application/octet-stream"

    # Step 1: Upload
    try:
        r = requests.post(
            f"{_MISTRAL_API_BASE}/files",
            headers=hdrs,
            files={"file": (filename, io.BytesIO(file_bytes), ctype)},
            data={"purpose": "ocr"},
            timeout=_REQUEST_TIMEOUT,
        )
    except Exception as e:
        raise MistralOcrError(f"Mistral file upload failed: {e}")
    if not r.ok:
        raise MistralOcrError(f"Mistral file upload {r.status_code}: {r.text[:300]}")
    file_id = r.json().get("id")
    if not file_id:
        raise MistralOcrError("Mistral file upload returned no id")

    try:
        # Step 2: Signed URL
        try:
            r2 = requests.get(
                f"{_MISTRAL_API_BASE}/files/{file_id}/url",
                headers=hdrs,
                timeout=30,
            )
        except Exception as e:
            raise MistralOcrError(f"Mistral signed URL request failed: {e}")
        if not r2.ok:
            raise MistralOcrError(f"Mistral signed URL {r2.status_code}: {r2.text[:300]}")
        signed_url = r2.json().get("url")
        if not signed_url:
            raise MistralOcrError("Mistral signed URL response missing url field")

        # Step 3: OCR
        try:
            r3 = requests.post(
                f"{_MISTRAL_API_BASE}/ocr",
                headers={**hdrs, "Content-Type": "application/json"},
                json={
                    "model": _OCR_MODEL,
                    "document": {"type": "document_url", "document_url": signed_url},
                    "table_format": "markdown",  # tables -> proper markdown, not prose
                    "include_image_base64": False,
                },
                timeout=_REQUEST_TIMEOUT,
            )
        except Exception as e:
            raise MistralOcrError(f"Mistral OCR request failed: {e}")
        if not r3.ok:
            raise MistralOcrError(f"Mistral OCR {r3.status_code}: {r3.text[:300]}")

        pages = r3.json().get("pages", [])
        if not pages:
            return ""

        table_map = _build_table_map(pages)
        parts = []
        for page in pages:
            md = page.get("markdown", "")
            if table_map:
                md = _inline_tables(md, table_map)
            if md.strip():
                parts.append(md)

        return "\n\n".join(parts)

    finally:
        # Step 4: Cleanup (best-effort)
        try:
            requests.delete(
                f"{_MISTRAL_API_BASE}/files/{file_id}",
                headers=hdrs,
                timeout=15,
            )
        except Exception:
            pass


def extract_markdown(file_bytes: bytes, filename: str = "document.pdf") -> str:
    """
    Extract text from file_bytes using Mistral OCR.
    PDFs larger than _MAX_CHUNK_BYTES are split into chunks, OCR'd in parallel,
    and merged in page order.
    Raises MistralOcrError on any failure.
    """
    if not is_configured():
        raise MistralOcrError("MISTRAL_API_KEY is not set")

    is_pdf = filename.lower().endswith(".pdf")

    if is_pdf and len(file_bytes) > _MAX_CHUNK_BYTES:
        size_mb = len(file_bytes) // (1024 * 1024)
        logger.info(f"[Mistral OCR] {filename} is {size_mb}MB — splitting into chunks")
        chunks = _split_pdf(file_bytes, _MAX_CHUNK_BYTES)
        logger.info(f"[Mistral OCR] Split into {len(chunks)} chunk(s)")

        results: list[str | None] = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as pool:
            future_map = {
                pool.submit(_ocr_chunk, chunk, f"part{i}_{filename}"): i
                for i, chunk in enumerate(chunks)
            }
            for fut in as_completed(future_map):
                results[future_map[fut]] = fut.result()

        return "\n\n".join(r for r in results if r and r.strip())

    return _ocr_chunk(file_bytes, filename)

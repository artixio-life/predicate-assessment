"""
Stage B: extract text from drug.products.source_url into document_text,
then chunk it into drug.product_chunks for Stage C.

Primary extraction is pdf-inspector (installed as `pdf_inspector`), which
classifies a PDF as text-based vs scanned before extracting — so we only
pay for Mistral OCR on the pages that actually need it, rather than always
routing through a paid OCR API. A cheap post-hoc sanity check (extracted
chars per page) catches misclassifications pdf-inspector's own confidence
score didn't.
"""
import logging
import os
import threading

import pdf_inspector
from psycopg2.extras import RealDictCursor

import storage
from db import get_db_connection
from processing import mistral_ocr
from processing.chunking import chunk_text
from processing.claim import claim_text_extraction, count_text_extraction_pending
from processing.eu_section_extraction import extract_annex_iii
from processing.progress import Progress
from processing.retry import run_with_retries, RetriesExhausted
from processing.workers import run_worker_pool

logger = logging.getLogger(__name__)

# Countries whose document_text is a large structured regulatory PDF where
# only one section is worth chunking for the LLM fold — see each function's
# module docstring for why. Keyed by drug.regulatory_geography.country_name.
_SECTION_EXTRACTORS = {
    "European Union": extract_annex_iii,
}

# Below this average chars/page, treat pdf-inspector's text-layer result as
# unusable even if it classified the PDF as text_based — mirrors
# regulatory-explorer's INSUFFICIENT_TEXT / LOW_OCR_CONFIDENCE thresholds.
MIN_CHARS_PER_PAGE = 50
MIN_TOTAL_CHARS = 50


def _extract_document_text(file_bytes: bytes, filename: str) -> str:
    classification = pdf_inspector.classify_pdf_bytes(file_bytes)
    needs_ocr = bool(classification.pages_needing_ocr) or classification.pdf_type != "text_based"

    text = ""
    if not needs_ocr:
        text = pdf_inspector.extract_text_bytes(file_bytes) or ""
        page_count = max(1, classification.page_count)
        if len(text) < MIN_TOTAL_CHARS or (len(text) / page_count) < MIN_CHARS_PER_PAGE:
            logger.info(
                f"[text_extraction] {filename}: text-layer extraction too sparse "
                f"({len(text)} chars / {page_count} pages) — falling back to OCR"
            )
            needs_ocr = True

    if needs_ocr:
        text = mistral_ocr.extract_markdown(file_bytes, filename=filename)

    return text


def _select_chunk_source(document_text, country_name, filename):
    """
    Most countries chunk the full document_text. A country in
    _SECTION_EXTRACTORS instead chunks only the section its extractor
    returns — falling back to the full document if that section isn't
    found (e.g. a template shape the extractor doesn't recognise yet), so a
    miss degrades to today's behaviour rather than producing zero chunks.
    """
    extractor = _SECTION_EXTRACTORS.get(country_name)
    if not extractor:
        return document_text
    section = extractor(document_text)
    if section:
        return section
    logger.warning(
        f"[text_extraction] {filename}: {country_name} section extractor found nothing — "
        f"falling back to chunking the full document_text"
    )
    return document_text


# Countries whose records carry several documents per product AND for which a
# SECOND document is worth chunking alongside the primary one. UK/MHRA records
# hold an SPC (Summary of Product Characteristics — the prescriber document the
# pipeline has always read) plus one or more PILs (Patient Information Leaflet)
# and sometimes a PAR, all listed in json_data.documents[]. The PIL restates
# indications/warnings in patient-facing wording and sometimes carries pack
# details the SPC omits, so one PIL is chunked too.
#
# Only ONE extra PIL is taken even when a record lists several: they are
# near-duplicate revisions of the same leaflet (observed: two PILs differing by
# a few KB), so chunking all of them would multiply Stage C's token cost for
# almost no new information.
_SECOND_DOC_TYPE_BY_COUNTRY = {
    "United Kingdom": "PIL",
}


def _select_documents(product):
    """
    Which document(s) to extract and chunk for this product, as a list of
    (s3_path, doc_type) in the order their chunks should be stored.

    Always the primary document (drug.products.source_url — index 0 of the
    crawler's document_url[], which is what this pipeline has always used).
    For a country in _SECOND_DOC_TYPE_BY_COUNTRY, also the FIRST document of
    that country's second type, unless the primary already IS that type — a
    record whose first document is the PIL is left exactly as-is, per the
    same rule.

    json_data.documents[] is positionally aligned with the crawler's
    document_url[] (verified against multi-document UK records: index i of one
    is index i of the other), so documents[i].doc_type describes
    document_url[i] and documents[i].s3_path is the path to fetch.
    """
    source_url = product["source_url"]
    primary_type = None
    docs = []
    json_data = product.get("json_data") or {}
    raw_docs = [d for d in (json_data.get("documents") or []) if isinstance(d, dict)]

    for d in raw_docs:
        if d.get("s3_path") and d["s3_path"] == source_url:
            primary_type = d.get("doc_type")
            break

    docs.append((source_url, primary_type))

    wanted = _SECOND_DOC_TYPE_BY_COUNTRY.get(product.get("country_name"))
    if not wanted or primary_type == wanted:
        return docs

    for d in raw_docs:
        path = d.get("s3_path")
        if d.get("doc_type") == wanted and path and path != source_url:
            docs.append((path, wanted))
            break
    return docs


def _process_one(cursor, product):
    documents = _select_documents(product)

    extracted = []  # (doc_type, path, text)
    for path, doc_type in documents:
        filename = os.path.basename(path)
        try:
            file_bytes = storage.download_file(path)
            text = _extract_document_text(file_bytes, filename)
        except Exception as e:
            # The PRIMARY document failing is a real failure — re-raise so the
            # row is retried/marked FAILED as before. A SECONDARY document
            # failing is not: the product is still fully usable from its
            # primary document, so log and carry on rather than losing the
            # whole row over a supplementary leaflet.
            if path == product["source_url"]:
                raise
            logger.warning(
                f"[text_extraction] product={product['id']}: secondary {doc_type} "
                f"{filename} failed, keeping primary document only: {e}"
            )
            continue
        if text and text.strip():
            extracted.append((doc_type, path, text))
        elif path == product["source_url"]:
            raise ValueError(f"No text could be extracted from {path}")

    if not extracted:
        raise ValueError(f"No text could be extracted from {product['source_url']}")

    # document_text keeps the PRIMARY document's text only. It is a single
    # column consumed elsewhere as "the product's document" (e.g. the US
    # manual extractor's regexes), so concatenating a patient leaflet into it
    # would change what those readers see. The PIL's contribution lives in
    # product_chunks, which is what Stage C actually reads.
    primary_text = extracted[0][2]

    cursor.execute(
        """
        UPDATE drug.products
        SET document_text = %s,
            text_extraction_status = 'DONE',
            text_extraction_attempts = text_extraction_attempts + 1,
            text_extraction_error = NULL,
            processing_status = 'PARSED',
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        """,
        (primary_text, product["id"]),
    )
    cursor.execute("DELETE FROM drug.product_chunks WHERE product_id = %s", (product["id"],))

    # One continuous chunk_index across all documents, primary first, so
    # Stage C (which reads ORDER BY chunk_index) still sees the primary
    # document before any supplementary one.
    index = 0
    for doc_type, path, text in extracted:
        filename = os.path.basename(path)
        chunk_source = _select_chunk_source(text, product.get("country_name"), filename)
        for chunk in chunk_text(chunk_source):
            cursor.execute(
                """
                INSERT INTO drug.product_chunks
                    (product_id, chunk_index, chunk_text, char_count, doc_type, source_path)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (product["id"], index, chunk, len(chunk), doc_type, path),
            )
            index += 1
    if len(extracted) > 1:
        logger.info(
            f"[text_extraction] product={product['id']}: chunked "
            f"{', '.join(d[0] or '?' for d in extracted)} ({index} chunk(s) total)"
        )
    return index


def _skip_no_source_url(cursor, product):
    """No document to extract from. If json_data exists Stage C can still
    run off it; otherwise there's genuinely nothing to work with."""
    has_json_data = bool(product["json_data"])
    processing_status = "PARSED" if has_json_data else "NEEDS_REVIEW"
    cursor.execute(
        """
        UPDATE drug.products
        SET text_extraction_status = 'SKIPPED',
            processing_status = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        """,
        (processing_status, product["id"]),
    )


def _mark_failed(cursor, product, attempts, error):
    has_json_data = bool(product["json_data"])
    processing_status = "PARSED" if has_json_data else "FAILED"
    cursor.execute(
        """
        UPDATE drug.products
        SET text_extraction_status = 'FAILED',
            text_extraction_attempts = %s,
            text_extraction_error = %s,
            processing_status = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        """,
        (attempts, str(error)[:2000], processing_status, product["id"]),
    )


def _worker(country, remaining, remaining_lock, progress):
    """
    One worker's claim loop. claim_text_extraction() atomically flips a row
    to PROCESSING before this worker ever starts the slow download/OCR work,
    so no other worker (thread or separate process) can pick it up too.
    """
    conn = get_db_connection()
    stats = {"done": 0, "skipped": 0, "failed": 0}
    try:
        while True:
            if remaining is not None:
                with remaining_lock:
                    if remaining[0] <= 0:
                        break
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                product = claim_text_extraction(cur, country=country)
                conn.commit()
            if not product:
                break
            if remaining is not None:
                with remaining_lock:
                    remaining[0] -= 1

            if not product["source_url"]:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    _skip_no_source_url(cur, product)
                    conn.commit()
                stats["skipped"] += 1
                progress.advance()
                continue

            try:
                def _attempt():
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        n_chunks = _process_one(cur, product)
                        conn.commit()
                        return n_chunks

                n_chunks = run_with_retries(_attempt, label=f"text_extraction product={product['id']}")
                stats["done"] += 1
                logger.info(f"[text_extraction] product={product['id']} -> {n_chunks} chunk(s)")
            except RetriesExhausted as e:
                conn.rollback()
                stats["failed"] += 1
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    _mark_failed(cur, product, e.attempts, e.last_exception)
                    conn.commit()
                logger.error(f"[text_extraction] product={product['id']} failed after {e.attempts} attempts: {e.last_exception}")
            progress.advance()
    finally:
        conn.close()
    return stats


def extract_pending(limit=None, country=None, workers=1):
    """
    Run text extraction for every drug.products row with
    text_extraction_status = 'PENDING', using `workers` concurrent claim
    loops. `country` (matches drug.regulatory_geography.country_name, e.g.
    "Brazil") scopes the run to one country. `limit` across concurrent
    workers is best-effort (may overshoot by up to `workers - 1` rows).
    Returns a dict of outcome -> count.
    """
    remaining = [limit] if limit else None
    remaining_lock = threading.Lock()

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            pending = count_text_extraction_pending(cur, country=country)
        conn.commit()
    finally:
        conn.close()
    total = min(pending, limit) if limit else pending
    progress = Progress("text_extraction", total=total, workers=workers)

    totals = run_worker_pool(
        # No sharding needed: claim_text_extraction flips a PERSISTED status to
        # PROCESSING, so a claimed row is invisible to other workers regardless
        # of commit timing (unlike Stage A — see promote._claim_one_raw_record).
        lambda _i, _n: _worker(country, remaining, remaining_lock, progress),
        workers, label="text_extraction"
    )
    progress.finish()
    logger.info(f"[text_extraction] done: {totals}")
    return totals

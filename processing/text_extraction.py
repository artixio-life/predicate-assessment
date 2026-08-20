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


def _process_one(cursor, product):
    source_url = product["source_url"]
    filename = os.path.basename(source_url)
    file_bytes = storage.download_file(source_url)
    document_text = _extract_document_text(file_bytes, filename)

    if not document_text or not document_text.strip():
        raise ValueError(f"No text could be extracted from {source_url}")

    chunk_source = _select_chunk_source(document_text, product.get("country_name"), filename)
    chunks = chunk_text(chunk_source)

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
        (document_text, product["id"]),
    )
    cursor.execute("DELETE FROM drug.product_chunks WHERE product_id = %s", (product["id"],))
    for index, chunk in enumerate(chunks):
        cursor.execute(
            """
            INSERT INTO drug.product_chunks (product_id, chunk_index, chunk_text, char_count)
            VALUES (%s, %s, %s, %s)
            """,
            (product["id"], index, chunk, len(chunk)),
        )
    return len(chunks)


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

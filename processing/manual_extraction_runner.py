"""
Runs a deterministic (no-LLM) extractor over one country's queue instead of
Stage C's LLM fold — for countries like Saudi Arabia (SFDA) whose json_data is
already fully structured (see processing/saudi_manual_extraction.py) or the US
(processing/us_manual_extraction.py).

Reuses processing.claim.claim_ai_extraction for the exact same atomic
claim-or-skip semantics Stage C uses (see processing/claim.py and
processing/workers.py) — so running this with multiple `workers` is safe the
same way running ai_extraction.py with multiple workers is: SKIP LOCKED
guarantees no two workers ever claim the same row.

Also reuses ai_extraction._persist_with_regnum_fallback, so a manually-mapped
row is written and marked DONE/ENRICHED through the IDENTICAL code path a
successful Stage C row goes through. That's what actually stops
ai_extraction.extract_pending() from re-claiming a manually-processed row on a
later run: claim_ai_extraction only ever selects ai_extraction_status =
'PENDING' (or a stale PROCESSING). Writing product_data/columns without also
flipping ai_extraction_status to 'DONE' leaves the row looking untouched to
that query, and the next `python main.py` run does the AI extraction anyway —
this module's whole point is to avoid exactly that.
"""
import logging
import threading

from psycopg2.extras import RealDictCursor

from db import get_db_connection
from processing.ai_extraction import PRODUCT_DATA_KEYS, _persist_with_regnum_fallback
from processing.claim import claim_ai_extraction, count_ai_extraction_pending
from processing.progress import Progress
from processing.workers import run_worker_pool

logger = logging.getLogger(__name__)


def _mark_failed(cursor, product_id, error):
    cursor.execute(
        """
        UPDATE drug.products
        SET ai_extraction_status = 'FAILED',
            ai_extraction_attempts = ai_extraction_attempts + 1,
            ai_extraction_error = %s,
            processing_status = 'FAILED',
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        """,
        (str(error)[:2000], product_id),
    )


def _process_one(cursor, product, extract_fn):
    result = extract_fn(product.get("json_data"), product.get("document_text"))
    product_data = result.get("product_data") or {}
    accumulated = {"columns": result.get("columns") or {}, "product_data": product_data}

    has_all_keys = all(key in product_data for key in PRODUCT_DATA_KEYS)
    if has_all_keys:
        ai_status, processing_status = "DONE", "ENRICHED"
    else:
        # Same rule ai_extraction._extract_one uses: an incomplete result is
        # NEEDS_REVIEW, never a false DONE. ai_extraction_status stays a
        # terminal (non-PENDING) value either way, so it is still never
        # re-claimed automatically — a human has to act on NEEDS_REVIEW rows.
        ai_status, processing_status = "NEEDS_REVIEW", "NEEDS_REVIEW"

    _persist_with_regnum_fallback(
        cursor, product["id"], accumulated, ai_status, 0, None, processing_status
    )
    return ai_status


def _worker(country, extract_fn, remaining, remaining_lock, progress):
    conn = get_db_connection()
    stats = {"DONE": 0, "NEEDS_REVIEW": 0, "FAILED": 0}
    try:
        while True:
            if remaining is not None:
                with remaining_lock:
                    if remaining[0] <= 0:
                        break
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                product = claim_ai_extraction(cur, country=country)
                conn.commit()
            if not product:
                break
            if remaining is not None:
                with remaining_lock:
                    remaining[0] -= 1

            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    outcome = _process_one(cur, product, extract_fn)
                    conn.commit()
                logger.info(f"[manual_extraction] product={product['id']} -> {outcome}")
            except Exception as e:
                conn.rollback()
                outcome = "FAILED"
                logger.exception(f"[manual_extraction] product={product['id']} errored — marking FAILED")
                try:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        _mark_failed(cur, product["id"], e)
                        conn.commit()
                except Exception:
                    conn.rollback()
                    logger.exception(f"[manual_extraction] product={product['id']}: could not record failure")
            stats[outcome] = stats.get(outcome, 0) + 1
            progress.advance()
    finally:
        conn.close()
    return stats


def run(country, extract_fn, limit=None, workers=1):
    """
    `extract_fn(json_data, document_text) -> {"columns": ..., "product_data": ...}`
    — the same shape processing/saudi_manual_extraction.extract and
    processing/us_manual_extraction.extract return. `country` must match
    drug.regulatory_geography.country_name exactly (e.g. "Saudi Arabia").

    Prerequisite: rows must already be at processing_status = 'PARSED' (i.e.
    Stage A promote + Stage B text-extraction have already run for this
    country — text extraction is a no-op SKIP for a country with no
    source_url/PDF, but it's still what flips PENDING -> PARSED). Run
    `python main.py` with PIPELINE_STAGES=promote,text and PIPELINE_COUNTRY
    set first if that hasn't happened yet.
    """
    remaining = [limit] if limit else None
    remaining_lock = threading.Lock()

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            pending = count_ai_extraction_pending(cur, country=country)
        conn.commit()
    finally:
        conn.close()
    total = min(pending, limit) if limit else pending
    progress = Progress("manual_extraction", total=total, workers=workers)

    totals = run_worker_pool(
        lambda _i, _n: _worker(country, extract_fn, remaining, remaining_lock, progress),
        workers, label="manual_extraction",
    )
    progress.finish()
    logger.info(f"[manual_extraction] done for {country}: {totals}")
    return totals

"""
Stage A: promote source.drug_predicate_raw_records rows (owned by the
source-predicate crawler) into drug.products.

One raw record -> one product row for now (see README on splitting
multi-strength records later). Idempotent via drug.products.raw_record_id's
unique constraint, so reruns only pick up genuinely new raw records.

Concurrency: source.drug_predicate_raw_records has no promotion-status
column of its own (it's owned by a different repo), so unlike Stages B/C
this doesn't use processing/claim.py's PROCESSING-flip pattern. Instead,
`FOR UPDATE OF r SKIP LOCKED` on the raw record means two workers racing
the same candidate never both start promoting it, and the INSERT's
`ON CONFLICT (raw_record_id) DO NOTHING` is a second, independent
correctness net — even in the rare case both got past the lock, only one
INSERT actually lands.
"""
import logging
import threading

from psycopg2.extras import Json, RealDictCursor

from db import get_db_connection
from processing.geography import resolve_geography_id
from processing.workers import run_worker_pool

logger = logging.getLogger(__name__)

# Best-effort registration-number lookup across the 5 crawlers' differing
# json_data shapes (checked against each crawler's actual field names):
#   Brazil (ANVISA):        numeroRegistro
#   South Africa (SAHPRA):  detail.registration_number, else application_no
#   Australia (TGA):        artg_id
#   China (NMPA):           acceptance_no
#   United Kingdom (MHRA):  not captured yet (dedup is by name only) -> None
_TOP_LEVEL_KEYS = (
    'registration_number', 'numeroRegistro', 'artg_id', 'acceptance_no',
    'application_no', 'reg_number', 'license_number', 'pl_number',
)


def guess_registration_number(json_data):
    if not json_data:
        return None
    detail = json_data.get('detail') if isinstance(json_data.get('detail'), dict) else {}
    if detail.get('registration_number'):
        return str(detail['registration_number'])
    for key in _TOP_LEVEL_KEYS:
        if json_data.get(key):
            return str(json_data[key])
    return None


def _claim_one_raw_record(cursor, country=None):
    query = """
        SELECT r.id, r.name, r.country_id, r.document_url, r.json_data
        FROM source.drug_predicate_raw_records r
        LEFT JOIN drug.products p ON p.raw_record_id = r.id
        LEFT JOIN source.country c ON c.id = r.country_id
        WHERE p.id IS NULL
    """
    params = []
    if country:
        query += " AND c.name = %s"
        params.append(country)
    query += " ORDER BY r.id FOR UPDATE OF r SKIP LOCKED LIMIT 1"
    cursor.execute(query, tuple(params))
    return cursor.fetchone()


def _regulator_for_geography(cursor, geography_id):
    """The resolved drug.regulatory_geography row already carries the
    authoritative agency acronym (FDA, ANVISA, TGA, ...) — pull it here
    rather than asking the AI extraction stage to guess it from prose."""
    if geography_id is None:
        return None
    cursor.execute(
        "SELECT agency_acronym FROM drug.regulatory_geography WHERE id = %s",
        (geography_id,),
    )
    row = cursor.fetchone()
    return row["agency_acronym"] if row else None


def _promote_one(cursor, raw_record):
    geography_id = resolve_geography_id(cursor, raw_record['country_id'])
    regulator = _regulator_for_geography(cursor, geography_id)
    document_urls = raw_record['document_url'] or []
    source_url = document_urls[0] if document_urls else None
    json_data = raw_record['json_data']
    registration_number = guess_registration_number(json_data)
    # No geography match is a review signal, not a blocker — the row still
    # promotes so text/AI extraction can proceed independently of it.
    processing_status = 'PENDING' if geography_id is not None else 'NEEDS_REVIEW'

    cursor.execute(
        """
        INSERT INTO drug.products
            (raw_record_id, product_name, country_id, regulator, source_url, json_data,
             registration_number, processing_status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (raw_record_id) DO NOTHING
        RETURNING id
        """,
        (
            raw_record['id'], raw_record['name'], geography_id, regulator, source_url,
            Json(json_data) if json_data is not None else None,
            registration_number, processing_status,
        ),
    )
    return cursor.fetchone()


def _worker(country, remaining, remaining_lock):
    """
    One worker's claim loop: keeps claiming and promoting raw records until
    none are left (or `remaining` runs out). Each claim+promote is one short
    transaction — promotion is a single fast INSERT, no slow external calls,
    so there's no need for retry-across-an-open-transaction here: on any
    failure this worker just rolls back (releasing the row lock so it's
    immediately claimable again, by this worker or another) and moves on.
    """
    conn = get_db_connection()
    stats = {"promoted": 0, "failed": 0}
    try:
        while True:
            if remaining is not None:
                with remaining_lock:
                    if remaining[0] <= 0:
                        break
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                raw_record = _claim_one_raw_record(cur, country=country)
                if not raw_record:
                    conn.commit()
                    break
                try:
                    result = _promote_one(cur, raw_record)
                    conn.commit()
                    if result:
                        stats["promoted"] += 1
                        logger.info(f"[promote] raw_record={raw_record['id']} -> product={result['id']}")
                    else:
                        logger.info(f"[promote] raw_record={raw_record['id']} already promoted (race), skipping")
                except Exception as e:
                    conn.rollback()
                    stats["failed"] += 1
                    logger.error(f"[promote] raw_record={raw_record['id']} failed: {e}")
            if remaining is not None:
                with remaining_lock:
                    remaining[0] -= 1
    finally:
        conn.close()
    return stats


def promote_pending(limit=None, country=None, workers=1):
    """
    Promote every raw record with no matching drug.products row yet, using
    `workers` concurrent claim loops (see _worker). `country` (matches
    source.country.name, e.g. "Brazil") scopes the run to one country —
    handy for testing a single pipeline stage in isolation. `limit` across
    concurrent workers is best-effort (may overshoot by up to `workers - 1`
    rows) rather than exactly enforced — fine for a dev/test convenience.
    Returns a dict of outcome -> count.
    """
    remaining = [limit] if limit else None
    remaining_lock = threading.Lock()
    totals = run_worker_pool(
        lambda: _worker(country, remaining, remaining_lock), workers, label="promote"
    )
    logger.info(f"[promote] done: {totals}")
    return totals

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

from psycopg2.errors import UniqueViolation
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


def _looks_like_identifier(value):
    """
    Reject values that are prose rather than a registration identifier.

    Some sources put a category or status string in the same field a
    registration number would occupy (observed: 'Old Medicine'). Storing that
    is worse than storing nothing: it is not an identifier, and because many
    products share the same phrase they all collide on the
    (country_id, registration_number) unique constraint, so every one after
    the first fails to promote.

    Every real registration number we handle contains at least one digit —
    '126750089', '1.0573.0562', 'PL 29831/0647', 'A99/999', 'Z20050001' — so
    requiring a digit is enough to separate them from prose, without needing a
    per-country format rule.
    """
    text = str(value).strip()
    if not text:
        return False
    return any(ch.isdigit() for ch in text)


def guess_registration_number(json_data):
    if not json_data:
        return None
    candidates = []
    detail = json_data.get('detail') if isinstance(json_data.get('detail'), dict) else {}
    if detail.get('registration_number'):
        candidates.append(detail['registration_number'])
    for key in _TOP_LEVEL_KEYS:
        if json_data.get(key):
            candidates.append(json_data[key])

    for candidate in candidates:
        if _looks_like_identifier(candidate):
            return str(candidate).strip()

    if candidates:
        logger.warning(
            f"[promote] ignoring non-identifier registration number "
            f"{candidates[0]!r} — storing NULL instead"
        )
    return None


def _claim_one_raw_record(cursor, country=None, exclude_ids=None):
    """
    `exclude_ids` is the run's set of raw records that already failed to
    promote. It is REQUIRED to avoid an infinite loop: the claim holds only a
    transaction-scoped row lock (FOR UPDATE), and source.drug_predicate_raw_records
    has no status column of its own to record an attempt (it belongs to the
    crawler repo). So a failed promotion rolls back, releases the lock, leaves
    the row still unpromoted, and would be claimed again immediately — forever.
    Excluding known failures is what makes the loop terminate.
    """
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
    if exclude_ids:
        query += " AND r.id <> ALL(%s)"
        params.append(list(exclude_ids))
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


def _worker(country, remaining, remaining_lock, failed_ids, failed_lock):
    """
    One worker's claim loop: keeps claiming and promoting raw records until
    none are left (or `remaining` runs out). Each claim+promote is one short
    transaction — promotion is a single fast INSERT, no slow external calls,
    so there is no retry-across-an-open-transaction here.

    A failed row is added to `failed_ids` (shared by every worker in the run)
    and excluded from subsequent claims. Without that the loop never
    terminates — see _claim_one_raw_record's docstring.
    """
    conn = get_db_connection()
    stats = {"promoted": 0, "duplicate": 0, "failed": 0}
    try:
        while True:
            if remaining is not None:
                with remaining_lock:
                    if remaining[0] <= 0:
                        break
            with failed_lock:
                skip_ids = set(failed_ids)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                raw_record = _claim_one_raw_record(cur, country=country, exclude_ids=skip_ids)
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
                except UniqueViolation:
                    # A sibling raw record describing the SAME product was already
                    # promoted — crawlers can emit the same product twice (observed
                    # for SAHPRA, where application_no is duplicated despite being
                    # its dedup key). One drug.products row per real product is the
                    # wanted outcome, so this is a routine skip, not a failure.
                    conn.rollback()
                    stats["duplicate"] += 1
                    with failed_lock:
                        failed_ids.add(raw_record['id'])
                    logger.info(
                        f"[promote] raw_record={raw_record['id']} skipped: another raw record "
                        f"already produced the product for registration_number="
                        f"{guess_registration_number(raw_record['json_data'])!r}"
                    )
                except Exception as e:
                    conn.rollback()
                    stats["failed"] += 1
                    with failed_lock:
                        failed_ids.add(raw_record['id'])
                    logger.error(f"[promote] raw_record={raw_record['id']} failed (will not be retried this run): {e}")
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
    # Shared across workers so one worker's known-bad row is not picked up and
    # re-failed by another. Per-run only — a fixed row is retried next run.
    failed_ids = set()
    failed_lock = threading.Lock()
    totals = run_worker_pool(
        lambda: _worker(country, remaining, remaining_lock, failed_ids, failed_lock),
        workers,
        label="promote",
    )
    # Only genuine failures are worth a warning; `duplicate` is expected whenever
    # a crawler emitted the same product more than once.
    if totals.get("failed"):
        logger.warning(f"[promote] {totals['failed']} raw record(s) failed — see errors above")
    if totals.get("duplicate"):
        logger.info(
            f"[promote] {totals['duplicate']} raw record(s) skipped as duplicates of an "
            f"already-promoted product"
        )
    logger.info(f"[promote] done: {totals}")
    return totals

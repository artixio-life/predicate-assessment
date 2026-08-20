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
import os
import threading

from psycopg2.extras import Json, RealDictCursor, execute_values

from db import get_db_connection
from processing.progress import Progress
from processing.workers import run_worker_pool

logger = logging.getLogger(__name__)

# Rows claimed and inserted per round trip. Promotion is a plain INSERT with no
# external calls, so batching is nearly free and removes the per-row commit
# (fsync) that dominated Stage A's runtime.
BATCH_SIZE = int(os.getenv("PROMOTE_BATCH_SIZE", "200"))

# Best-effort registration-number lookup across the 7 crawlers' differing
# json_data shapes (checked against each crawler's actual field names):
#   Brazil (ANVISA):        numeroRegistro
#   South Africa (SAHPRA):  detail.registration_number, else application_no
#   Australia (TGA):        artg_id
#   China (NMPA):           acceptance_no
#   United States (FDA):    application_number (the ApplNo, e.g. "020892")
#   United Kingdom (MHRA):  not captured yet (dedup is by name only) -> None
#   Saudi Arabia (SFDA):    registration_number (top-level; SFDA's own registerNumber,
#                           already the first key checked below)
_TOP_LEVEL_KEYS = (
    'registration_number', 'numeroRegistro', 'artg_id', 'acceptance_no',
    'application_no', 'application_number', 'reg_number', 'license_number', 'pl_number',
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


def _claim_raw_records(cursor, country=None, exclude_ids=None,
                        shard_index=None, shard_count=None, batch_size=1,
                        after_id=0):
    """
    `exclude_ids` is the run's set of raw records that already failed to
    promote. It is REQUIRED to avoid an infinite loop: the claim holds only a
    transaction-scoped row lock (FOR UPDATE), and source.drug_predicate_raw_records
    has no status column of its own to record an attempt (it belongs to the
    crawler repo). So a failed promotion rolls back, releases the lock, leaves
    the row still unpromoted, and would be claimed again immediately — forever.
    Excluding known failures is what makes the loop terminate.

    `shard_index`/`shard_count` partition the queue by `id % shard_count`, so
    each worker only ever looks at its own disjoint slice. SKIP LOCKED alone is
    not enough here: the row lock is dropped the moment the INSERT commits, but
    the LEFT JOIN only starts excluding the row once that commit is visible, so
    concurrent workers repeatedly landed on the same lowest-id row and burned an
    attempt discovering it was already promoted. Sharding removes the contention
    by construction rather than relying on that timing.

    `after_id` is the worker's high-water mark and is what keeps this linear.
    Without it, `WHERE p.id IS NULL ORDER BY r.id LIMIT n` has to walk past
    every already-promoted row to reach the next unpromoted one, so batch k
    re-scans everything batches 1..k-1 already covered — quadratic, and very
    slow once the table is large. Because rows are always taken in ascending
    id order and a processed row never becomes claimable again within a run
    (it is either promoted, or recorded in exclude_ids), everything at or
    below the mark can be skipped outright.
    """
    query = """
        SELECT r.id, r.name, r.country_id, r.document_url, r.json_data
        FROM source.drug_predicate_raw_records r
        LEFT JOIN drug.products p ON p.raw_record_id = r.id
        LEFT JOIN source.country c ON c.id = r.country_id
        WHERE p.id IS NULL
    """
    params = []
    if after_id:
        query += " AND r.id > %s"
        params.append(after_id)
    if country:
        query += " AND c.name = %s"
        params.append(country)
    if shard_count and shard_count > 1:
        query += " AND (r.id %% %s) = %s"
        params.extend([shard_count, shard_index])
    if exclude_ids:
        query += " AND r.id <> ALL(%s)"
        params.append(list(exclude_ids))
    query += " ORDER BY r.id FOR UPDATE OF r SKIP LOCKED LIMIT %s"
    params.append(batch_size)
    cursor.execute(query, tuple(params))
    return cursor.fetchall()


def count_pending_raw_records(cursor, country=None):
    """
    How many raw records are still unpromoted — the same predicate
    _claim_raw_records() uses, minus the sharding/watermark bookkeeping, run
    once up front so the stage can report progress and an ETA. Counts every
    candidate, including ones that will turn out to be duplicates and get
    skipped, so this is an upper bound on the work.
    """
    query = """
        SELECT COUNT(*) AS n
        FROM source.drug_predicate_raw_records r
        LEFT JOIN drug.products p ON p.raw_record_id = r.id
        LEFT JOIN source.country c ON c.id = r.country_id
        WHERE p.id IS NULL
    """
    params = []
    if country:
        query += " AND c.name = %s"
        params.append(country)
    cursor.execute(query, tuple(params))
    return cursor.fetchone()["n"]


def load_geography_cache(cursor):
    """
    Resolve EVERY source.country -> (drug.regulatory_geography.id, agency_acronym)
    in one query pair, up front.

    This used to be 4 queries per promoted row (3 inside resolve_geography_id,
    1 for the agency acronym) — but there are only a few dozen countries, and
    the mapping cannot change mid-run, so per-row lookups were pure overhead
    and the dominant cost of Stage A. Matching happens here in Python to keep
    the exact ISO-code-then-name precedence resolve_geography_id used.
    """
    cursor.execute("SELECT id, name, code FROM source.country")
    countries = cursor.fetchall()
    cursor.execute(
        "SELECT id, country_name, country_code, agency_acronym FROM drug.regulatory_geography"
    )
    geographies = cursor.fetchall()

    by_code, by_name = {}, {}
    for g in geographies:
        if g["country_code"]:
            by_code.setdefault(g["country_code"].strip().upper(), g)
        if g["country_name"]:
            by_name.setdefault(g["country_name"].strip().upper(), g)

    cache, unresolved = {}, []
    for c in countries:
        match = None
        if c["code"]:
            match = by_code.get(c["code"].strip().upper())
        if match is None and c["name"]:
            match = by_name.get(c["name"].strip().upper())
        if match is None:
            unresolved.append(c["name"])
            cache[c["id"]] = (None, None)
        else:
            cache[c["id"]] = (match["id"], match["agency_acronym"])

    if unresolved:
        logger.warning(
            f"[promote] no drug.regulatory_geography match for: {sorted(unresolved)} "
            f"— products from these countries promote with country_id NULL "
            f"and processing_status NEEDS_REVIEW"
        )
    logger.info(f"[promote] geography cache: {len(cache)} country/countries resolved once")
    return cache


def _row_values(raw_record, geo_cache):
    """Build the INSERT tuple for one raw record. No queries — everything it
    needs is either on the record or in the pre-loaded geography cache."""
    geography_id, regulator = geo_cache.get(raw_record['country_id'], (None, None))
    document_urls = raw_record['document_url'] or []
    source_url = document_urls[0] if document_urls else None
    json_data = raw_record['json_data']
    # No geography match is a review signal, not a blocker — the row still
    # promotes so text/AI extraction can proceed independently of it.
    processing_status = 'PENDING' if geography_id is not None else 'NEEDS_REVIEW'
    return (
        raw_record['id'], raw_record['name'], geography_id, regulator, source_url,
        Json(json_data) if json_data is not None else None,
        guess_registration_number(json_data), processing_status,
    )


def _promote_batch(cursor, raw_records, geo_cache):
    """
    Insert a whole batch in ONE statement, returning the raw_record_ids that
    actually landed.

    `ON CONFLICT DO NOTHING` with no conflict target absorbs BOTH unique
    constraints — uq_products_raw_record (already promoted) and
    uq_products_country_regnum (a sibling raw record described the same
    product). Both are routine skips rather than errors, so letting Postgres
    drop them means the batch never aborts and no per-row exception handling
    or SAVEPOINT is needed.
    """
    values = [_row_values(r, geo_cache) for r in raw_records]
    rows = execute_values(
        cursor,
        """
        INSERT INTO drug.products
            (raw_record_id, product_name, country_id, regulator, source_url, json_data,
             registration_number, processing_status)
        VALUES %s
        ON CONFLICT DO NOTHING
        RETURNING raw_record_id, id
        """,
        values,
        fetch=True,
    )
    return {row["raw_record_id"]: row["id"] for row in rows}


def _worker(shard_index, shard_count, country, remaining, remaining_lock,
            failed_ids, failed_lock, progress):
    """
    One worker's claim loop, working a BATCH at a time: claim up to
    BATCH_SIZE rows from this worker's shard, insert them in one statement,
    commit once. That is 2 round trips and 1 fsync per batch instead of ~6
    queries + 1 fsync per row.

    A batch that fails outright is added to `failed_ids` (shared by every
    worker) and excluded from later claims. Without that the loop never
    terminates — see _claim_raw_records's docstring.
    """
    conn = get_db_connection()
    stats = {"promoted": 0, "duplicate": 0, "failed": 0}
    # High-water mark: this worker never looks at ids <= watermark again. Keeps
    # the claim scan linear instead of quadratic (see _claim_raw_records).
    watermark = 0
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            geo_cache = load_geography_cache(cur)
            conn.commit()

        while True:
            batch_size = BATCH_SIZE
            if remaining is not None:
                with remaining_lock:
                    if remaining[0] <= 0:
                        break
                    batch_size = min(batch_size, remaining[0])
            with failed_lock:
                skip_ids = {i for i in failed_ids if i > watermark}

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                raw_records = _claim_raw_records(
                    cur, country=country, exclude_ids=skip_ids,
                    shard_index=shard_index, shard_count=shard_count,
                    batch_size=batch_size, after_id=watermark,
                )
                if not raw_records:
                    conn.commit()
                    break
                try:
                    promoted = _promote_batch(cur, raw_records, geo_cache)
                    conn.commit()
                    # Anything not returned by the INSERT hit ON CONFLICT DO
                    # NOTHING: already promoted, or a sibling raw record already
                    # covered the same product. Either way it must not be re-claimed.
                    skipped = [r['id'] for r in raw_records if r['id'] not in promoted]
                    stats["promoted"] += len(promoted)
                    stats["duplicate"] += len(skipped)
                    if skipped:
                        with failed_lock:
                            failed_ids.update(skipped)
                    logger.info(
                        f"[promote] shard {shard_index}/{shard_count}: batch of "
                        f"{len(raw_records)} -> {len(promoted)} promoted, "
                        f"{len(skipped)} already covered"
                    )
                except Exception as e:
                    conn.rollback()
                    stats["failed"] += len(raw_records)
                    with failed_lock:
                        failed_ids.update(r['id'] for r in raw_records)
                    logger.error(
                        f"[promote] batch of {len(raw_records)} starting at "
                        f"raw_record={raw_records[0]['id']} failed "
                        f"(not retried this run): {e}"
                    )
            # Advance past everything just handled, whether promoted, skipped
            # as a duplicate, or failed — none of it is claimable again.
            watermark = max(r['id'] for r in raw_records)
            progress.advance(len(raw_records))
            if remaining is not None:
                with remaining_lock:
                    remaining[0] -= len(raw_records)
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

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            pending = count_pending_raw_records(cur, country=country)
        conn.commit()
    finally:
        conn.close()
    total = min(pending, limit) if limit else pending
    progress = Progress("promote", total=total, workers=workers)

    totals = run_worker_pool(
        lambda i, n: _worker(i, n, country, remaining, remaining_lock,
                              failed_ids, failed_lock, progress),
        workers,
        label="promote",
    )
    progress.finish()
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

"""
Atomic row-claiming for concurrent workers.

`UPDATE ... SET status = 'PROCESSING' WHERE id = (SELECT ... FOR UPDATE
SKIP LOCKED LIMIT 1) RETURNING ...` claims exactly one row in a single
round trip: the SELECT's row lock means two workers racing this statement
at the same instant can never land on the same row — one gets it, the
other's subquery skips past the now-locked row and finds a different one
(or nothing). This holds whether the two workers are threads in this
process or two separate `docker compose` runs against the same database.

A row stuck in PROCESSING past STALE_MINUTES (its worker crashed or was
killed mid-flight, so it never reached a terminal status) is treated as
claimable again automatically — no separate reaper process needed.
"""

STALE_MINUTES = 30


def claim_text_extraction(cursor, country=None):
    params = []
    country_clause = ""
    if country:
        country_clause = " AND g.country_name = %s"
        params.append(country)
    cursor.execute(
        f"""
        UPDATE drug.products AS p
        SET text_extraction_status = 'PROCESSING', updated_at = CURRENT_TIMESTAMP
        WHERE p.id = (
            SELECT p2.id
            FROM drug.products p2
            LEFT JOIN drug.regulatory_geography g ON g.id = p2.country_id
            WHERE (p2.text_extraction_status = 'PENDING'
                   OR (p2.text_extraction_status = 'PROCESSING'
                       AND p2.updated_at < CURRENT_TIMESTAMP - INTERVAL '{STALE_MINUTES} minutes')){country_clause}
            ORDER BY p2.created_at
            FOR UPDATE OF p2 SKIP LOCKED
            LIMIT 1
        )
        RETURNING p.id, p.source_url, p.json_data
        """,
        tuple(params),
    )
    return cursor.fetchone()


def claim_ai_extraction(cursor, country=None):
    params = []
    country_clause = ""
    if country:
        country_clause = " AND g.country_name = %s"
        params.append(country)
    cursor.execute(
        f"""
        UPDATE drug.products AS p
        SET ai_extraction_status = 'PROCESSING', updated_at = CURRENT_TIMESTAMP
        WHERE p.id = (
            SELECT p2.id
            FROM drug.products p2
            LEFT JOIN drug.regulatory_geography g ON g.id = p2.country_id
            WHERE p2.processing_status = 'PARSED'
              AND (p2.ai_extraction_status = 'PENDING'
                   OR (p2.ai_extraction_status = 'PROCESSING'
                       AND p2.updated_at < CURRENT_TIMESTAMP - INTERVAL '{STALE_MINUTES} minutes')){country_clause}
            ORDER BY p2.created_at
            FOR UPDATE OF p2 SKIP LOCKED
            LIMIT 1
        )
        RETURNING p.id, p.product_name, p.registration_number, p.json_data
        """,
        tuple(params),
    )
    return cursor.fetchone()

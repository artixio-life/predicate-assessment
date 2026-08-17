"""
Resolve a source-predicate `source.country` row to the matching
`drug.regulatory_geography` row (owned by the regulatory-explorer app).

These are two independently-seeded tables that happen to describe the same
real-world countries. We never write into drug.regulatory_geography from
here — it's a curated reference table maintained by the other app, and
guessing agency/region data into it on a promotion miss would corrupt a
shared table. An unresolved country is a NEEDS_REVIEW signal, not a reason
to insert a best-effort row.
"""
import logging

logger = logging.getLogger(__name__)


def resolve_geography_id(cursor, source_country_id):
    """
    Look up source.country(source_country_id), then find the matching
    drug.regulatory_geography row: first by ISO country_code (case-
    insensitive), falling back to country_name (case-insensitive, trimmed).
    Returns the drug.regulatory_geography.id, or None if source_country_id
    is unset or no match exists in either table.
    """
    if source_country_id is None:
        return None

    cursor.execute(
        "SELECT name, code FROM source.country WHERE id = %s",
        (source_country_id,),
    )
    row = cursor.fetchone()
    if not row:
        logger.warning(f"[geography] source.country id={source_country_id} not found")
        return None
    country_name, country_code = row["name"], row["code"]

    if country_code:
        cursor.execute(
            "SELECT id FROM drug.regulatory_geography WHERE UPPER(country_code) = UPPER(%s)",
            (country_code,),
        )
        match = cursor.fetchone()
        if match:
            return match["id"]

    if country_name:
        cursor.execute(
            "SELECT id FROM drug.regulatory_geography WHERE UPPER(TRIM(country_name)) = UPPER(TRIM(%s))",
            (country_name,),
        )
        match = cursor.fetchone()
        if match:
            return match["id"]

    logger.warning(
        f"[geography] No drug.regulatory_geography match for source.country "
        f"'{country_name}' ({country_code}) — leaving country_id unresolved, "
        f"row should be flagged NEEDS_REVIEW."
    )
    return None

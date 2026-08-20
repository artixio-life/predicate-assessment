"""
Run deterministic (no-LLM) extraction for one country, instead of Stage C's
LLM fold — see processing/manual_extraction_runner.py and
processing/saudi_manual_extraction.py.

Assumes Stage A (promote) and Stage B (text extraction) have already run for
this country, so rows are sitting at processing_status='PARSED',
ai_extraction_status='PENDING'. If they haven't, run those stages first:

    PIPELINE_COUNTRY="Saudi Arabia" PIPELINE_STAGES=promote,text python main.py

Usage:
    python tools/run_manual_extraction.py --country "Saudi Arabia" --workers 4
    python tools/run_manual_extraction.py --country "Saudi Arabia" --limit 20 --dry-run
"""
import argparse
import json
import logging

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Country name (as it appears in drug.regulatory_geography.country_name) ->
# the deterministic extractor module for it. Add an entry here (and a new
# processing/<country>_manual_extraction.py) for the next fully-structured,
# no-document source before reaching for the LLM path.
_EXTRACTORS = {}


def _load_extractors():
    if not _EXTRACTORS:
        from processing import australia_manual_extraction, saudi_manual_extraction, us_manual_extraction
        _EXTRACTORS["Saudi Arabia"] = saudi_manual_extraction.extract
        _EXTRACTORS["United States"] = us_manual_extraction.extract
        _EXTRACTORS["Australia"] = australia_manual_extraction.extract
    return _EXTRACTORS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--country", required=True, help="e.g. 'Saudi Arabia' — must match drug.regulatory_geography.country_name")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                         help="Print what N sample rows would map to, without touching the database")
    parser.add_argument("--dry-run-sample", type=int, default=5)
    args = parser.parse_args()

    load_dotenv()
    extractors = _load_extractors()
    if args.country not in extractors:
        raise SystemExit(
            f"No manual extractor registered for {args.country!r}. "
            f"Known: {sorted(extractors)}. Add one to _EXTRACTORS in this file."
        )
    extract_fn = extractors[args.country]

    if args.dry_run:
        from db import get_db_connection
        from psycopg2.extras import RealDictCursor
        conn = get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT p.id, p.json_data, p.document_text
                    FROM drug.products p
                    LEFT JOIN drug.regulatory_geography g ON g.id = p.country_id
                    WHERE g.country_name = %s AND p.ai_extraction_status = 'PENDING'
                    ORDER BY p.id LIMIT %s
                    """,
                    (args.country, args.dry_run_sample),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        if not rows:
            print(f"No PENDING rows found for {args.country!r} — nothing to preview.")
            return
        for row in rows:
            result = extract_fn(row["json_data"], row.get("document_text"))
            print(f"--- product {row['id']} ---")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    from processing import manual_extraction_runner
    totals = manual_extraction_runner.run(
        args.country, extract_fn, limit=args.limit, workers=args.workers
    )
    print(f"Done: {totals}")


if __name__ == "__main__":
    main()

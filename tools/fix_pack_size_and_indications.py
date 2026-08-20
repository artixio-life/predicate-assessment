"""
CLI wrapper for processing/llm_field_repair.py — see that module for the exact
selection rule and design rationale. main.py runs the same repair when
RUN_LLM_FIX=true is set; this script is for manual and dry-run use.

Only rows whose stored presentation pack_size is null AND whose raw json_data
component actually has pack_size text are touched. A row whose raw pack_size
is null/empty is never sent to the model.

Usage:
    # preview what would change, nothing written
    python tools/fix_pack_size_and_indications.py --dry-run --limit 5

    # apply for real
    python tools/fix_pack_size_and_indications.py
"""
import argparse
import logging
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--country", default="Australia")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4,
                         help="Concurrent repair loops (default 4). Bounded by the LLM "
                              "provider's rate limit — a burst of 429s means it's too high.")
    args = parser.parse_args()

    load_dotenv()
    from processing import llm_field_repair
    stats = llm_field_repair.run(
        country=args.country, limit=args.limit, dry_run=args.dry_run,
        workers=args.workers,
    )
    print(f"Done: {stats}")


if __name__ == "__main__":
    main()

import logging
import os

from dotenv import load_dotenv

from processing import runner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    Runs the pipeline only. The schema is NOT applied here — apply
    schema/schema.sql by hand (every statement in it is idempotent, so it is
    safe to re-run). db.init_db() is still available if you want to apply it
    from Python instead.
    """
    load_dotenv()

    limit_env = os.getenv("PIPELINE_LIMIT")
    limit = int(limit_env) if limit_env else None
    country = os.getenv("PIPELINE_COUNTRY") or None
    workers = int(os.getenv("PIPELINE_WORKERS", "4"))
    stages = runner.parse_stages(os.getenv("PIPELINE_STAGES"))
    runner.run(limit=limit, country=country, workers=workers, stages=stages)

    # Opt-in, off by default: LLM-only repair of Australia/TGA
    # presentations.pack_size/per_value and indications on rows the
    # deterministic mapper left with a null pack_size despite the source
    # having pack text to read — see processing/llm_field_repair.py for the
    # exact selection rule. Runs AFTER the normal pipeline, against whatever
    # is already in drug.products, and writes only those two fields.
    if os.getenv("RUN_LLM_FIX", "false").lower() == "true":
        from processing import llm_field_repair
        dry_run = os.getenv("RUN_LLM_FIX_DRY_RUN", "false").lower() == "true"
        logger.info(
            f"=== LLM field repair (RUN_LLM_FIX=true) === {country or 'Australia'}"
            f"{' [DRY RUN]' if dry_run else ''}"
        )
        llm_field_repair.run(country=country or "Australia", limit=limit, dry_run=dry_run)


if __name__ == "__main__":
    main()

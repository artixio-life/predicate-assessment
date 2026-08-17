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


if __name__ == "__main__":
    main()

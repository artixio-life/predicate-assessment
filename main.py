import logging
import os

from dotenv import load_dotenv

from db import init_db
from processing import runner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    load_dotenv()
    logger.info("Applying schema...")
    init_db()

    limit_env = os.getenv("PIPELINE_LIMIT")
    limit = int(limit_env) if limit_env else None
    country = os.getenv("PIPELINE_COUNTRY") or None
    workers = int(os.getenv("PIPELINE_WORKERS", "4"))
    runner.run(limit=limit, country=country, workers=workers)


if __name__ == "__main__":
    main()

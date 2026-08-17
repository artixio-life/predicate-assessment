"""
Sequences the pipeline: promote -> text extraction -> AI extraction.

Each stage only touches rows left in the state the previous stage leaves
them in, so re-running this after a partial/failed run picks up exactly
where it left off (see each stage module's status semantics).
"""
import logging

from processing import promote, text_extraction, ai_extraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(limit=None, country=None, workers=1):
    """
    `country` (e.g. "Brazil") scopes all three stages to one country's
    products — handy for testing a change against a small, known slice
    before running it over everything. `workers` runs that many concurrent
    claim loops per stage (see processing/claim.py + processing/workers.py
    for how two workers are kept from ever picking the same row).
    """
    logger.info(f"=== Stage A: promote raw records -> drug.products ==={' (' + country + ')' if country else ''} [{workers} worker(s)]")
    promote.promote_pending(limit=limit, country=country, workers=workers)

    logger.info(f"=== Stage B: text extraction === [{workers} worker(s)]")
    text_extraction.extract_pending(limit=limit, country=country, workers=workers)

    logger.info(f"=== Stage C: AI extraction === [{workers} worker(s)]")
    ai_extraction.extract_pending(limit=limit, country=country, workers=workers)

"""
Sequences the pipeline: promote -> text extraction -> AI extraction.

Each stage only touches rows left in the state the previous stage leaves
them in, so re-running this after a partial/failed run picks up exactly
where it left off (see each stage module's status semantics).

Stages can be selected individually (see `stages` / PIPELINE_STAGES), which
is how you stop promoting new raw records and only re-process rows that are
already in drug.products.
"""
import logging

from processing import promote, text_extraction, ai_extraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALL_STAGES = ("promote", "text", "ai")

# Accepted spellings for each stage, so PIPELINE_STAGES is forgiving.
_ALIASES = {
    "promote": "promote", "a": "promote", "migrate": "promote", "migration": "promote",
    "text": "text", "b": "text", "text_extraction": "text", "ocr": "text",
    "ai": "ai", "c": "ai", "ai_extraction": "ai", "extract": "ai",
}


def parse_stages(value):
    """
    Turn a PIPELINE_STAGES string into an ordered tuple of stage keys.
    Empty/unset means all three. Unknown names raise rather than silently
    running the wrong thing.
    """
    if value is None or not str(value).strip():
        return ALL_STAGES
    requested = [part.strip().lower() for part in str(value).split(",") if part.strip()]
    resolved = []
    for part in requested:
        if part not in _ALIASES:
            raise ValueError(
                f"Unknown pipeline stage {part!r}. Valid values: "
                f"promote, text, ai (aliases: a/b/c, migration, text_extraction, ai_extraction)."
            )
        key = _ALIASES[part]
        if key not in resolved:
            resolved.append(key)
    # Always run in pipeline order regardless of how they were listed.
    return tuple(s for s in ALL_STAGES if s in resolved)


def run(limit=None, country=None, workers=1, stages=ALL_STAGES):
    """
    `country` (e.g. "Brazil") scopes every stage to one country's products —
    handy for testing a change against a small, known slice before running it
    over everything. `workers` runs that many concurrent claim loops per stage
    (see processing/claim.py + processing/workers.py for how two workers are
    kept from ever picking the same row). `stages` selects which stages run,
    e.g. ("text", "ai") to re-process existing drug.products rows without
    promoting any new raw records.
    """
    scope = f" ({country})" if country else ""
    skipped = [s for s in ALL_STAGES if s not in stages]
    if skipped:
        logger.info(f"Running stages {list(stages)}; skipping {skipped}")

    if "promote" in stages:
        logger.info(f"=== Stage A: promote raw records -> drug.products ==={scope} [{workers} worker(s)]")
        promote.promote_pending(limit=limit, country=country, workers=workers)

    if "text" in stages:
        logger.info(f"=== Stage B: text extraction ==={scope} [{workers} worker(s)]")
        text_extraction.extract_pending(limit=limit, country=country, workers=workers)

    if "ai" in stages:
        logger.info(f"=== Stage C: AI extraction ==={scope} [{workers} worker(s)]")
        ai_extraction.extract_pending(limit=limit, country=country, workers=workers)

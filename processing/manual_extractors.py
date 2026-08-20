"""
Countries whose crawler json_data is structured enough that Stage C (the LLM
fold, processing/ai_extraction.py) should never run on them at all — not "try
manual mapping first, fall back to the LLM," but a hard exclusion.

processing/runner.py checks this registry and routes Stage C to
processing/manual_extraction_runner.py instead of ai_extraction.py whenever
`country` matches an entry here. ai_extraction.py itself also excludes every
country listed here from its own claim query (see claim.claim_ai_extraction's
`exclude_countries`), so even an UNSCOPED run (no PIPELINE_COUNTRY set, every
country in one pass) can never send one of these countries' rows to the paid
OpenRouter/self-hosted LLM by accident.

Only add a country here once its json_data is genuinely complete enough that
nothing worth extracting is lost by skipping the LLM read entirely — Saudi
Arabia (SFDA) qualifies because its records carry no document/prose at all
(see processing/saudi_manual_extraction.py's module docstring). A country like
the US, where processing/us_manual_extraction.py exists but the LLM path still
captures label prose (indications, key_risks, pivotal_evidence, excipients)
the rule-based mapper explicitly can't, does NOT belong here — running it
manually is an opt-in choice (see tools/run_manual_extraction.py), not an
automatic pipeline substitution.
"""
from processing import saudi_manual_extraction

MANUAL_EXTRACTORS = {
    "Saudi Arabia": saudi_manual_extraction.extract,
}

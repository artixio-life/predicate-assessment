"""
Guard: tools/harness_prompt_standalone.py must match Stage C's live prompt.

The standalone copy exists for machines without this repo on disk, so nothing
imports it in normal operation — which means it can go stale silently the next
time ai_extraction.SYSTEM_PROMPT changes, and a cloud run would then benchmark a
prompt the pipeline is no longer using.

Fix a failure by regenerating, never by hand-editing the generated file:

    python tools/generate_standalone_prompt.py
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))


def test_standalone_prompt_matches_production():
    import processing.ai_extraction as ai
    import harness_prompt_standalone

    assert harness_prompt_standalone.SYSTEM_PROMPT == ai.SYSTEM_PROMPT, (
        "tools/harness_prompt_standalone.py is stale — regenerate it with "
        "`python tools/generate_standalone_prompt.py`"
    )


def test_harness_sends_the_production_prompt():
    """
    The harness must benchmark the prompt the pipeline actually uses. It used to
    patch extra rules on at import time; those rules now live in
    ai_extraction.SYSTEM_PROMPT, so any divergence here means a benchmark that
    measures something production never sends.
    """
    import tools.local_llm_harness as harness
    import processing.ai_extraction as ai

    assert harness.SYSTEM_PROMPT == ai.SYSTEM_PROMPT


def test_rules_appear_exactly_once():
    """
    Moving the rules into production while the harness still patched them on
    would duplicate every one of them. Pin the counts.
    """
    import processing.ai_extraction as ai

    for needle in ("TWO VIEWS OF THE SAME FACTS",
                   "Under-counting a list this way is a missed extraction",
                   "NOT per treatment arm and NOT per dose",
                   "must ALWAYS be present in your output"):
        assert ai.SYSTEM_PROMPT.count(needle) == 1, f"{needle!r} appears more than once"


@pytest.mark.parametrize("needle", [
    # A sample of the per-field semantics a hand-written reconstruction loses.
    "Taohe Chengqi Decoction Granules",          # product_name_en translation
    "The mere existence of a currently-published label does NOT prove Active",
    "'ARTG Start Date' or 'Start Date' (Australia TGA)",
    "'Produced at' date is NEVER this field",
    "CONTAINER MATERIAL IS NOT part of the identity",
    "is a symptom, not a therapeutic area",
    "use the SAME canonical term every time",
    # The extraction rules.
    "Under-counting a list this way is a missed extraction",
    "Do NOT skip a fact just because other strengths",
    "applying EXCLUSIVELY to a different strength",
    "NOT per treatment arm and NOT per dose",
    "TWO VIEWS OF THE SAME FACTS",
])
def test_standalone_prompt_retains(needle):
    import harness_prompt_standalone
    assert needle in harness_prompt_standalone.SYSTEM_PROMPT

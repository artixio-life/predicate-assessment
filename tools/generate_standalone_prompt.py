"""
Emit tools/harness_prompt_standalone.py — Stage C's system prompt as one
self-contained literal, for machines that don't have this repo on disk.

A cloud notebook or scratch VM running only the harness script can import the
generated module instead of processing/ai_extraction.py, or paste the literal in
directly, and still send byte-identical bytes.

Run after ANY change to ai_extraction.SYSTEM_PROMPT, otherwise the standalone
copy silently goes stale:

    python tools/generate_standalone_prompt.py

Equality against the live harness prompt is asserted before the file is
written, so a drifted generator fails here rather than in a benchmark.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import processing.ai_extraction as ai
from processing.chunking import count_tokens

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "harness_prompt_standalone.py")

DOCSTRING = '''"""
Self-contained copy of Stage C's system prompt (processing/ai_extraction.py).

GENERATED FILE — do not hand-edit. Regenerate with:

    python tools/generate_standalone_prompt.py

Stage C's prompt lives in processing/ai_extraction.py, which needs this repo on
disk. On a machine that has only the harness script, import this module instead:

    from harness_prompt_standalone import SYSTEM_PROMPT

or paste the literal below straight into the script. It is asserted
byte-identical to processing/ai_extraction.py's SYSTEM_PROMPT at generation
time, so a run either way sends what the pipeline sends.

Everything a hand-written reconstruction of this prompt tends to lose, and this
keeps in full:

  - product_name_en translation / transliteration rules
  - brand_name_local "original script only" definition
  - generic_name = INN base, not the salt form
  - mah_address / manufacturer_address English-rendering rules (translate
    structural words, keep proper nouns, append the country)
  - product_type = the regulator's OWN class label, translated
  - status: REGISTRATION status only, with the explicit warning that a
    dispensing/legal classification (prescription-only, OTC) is NOT it, and
    that a published label does not prove Active
  - the registration_date / approval_date regulator-label mapping table (TGA
    "ARTG Start Date", SAHPRA "Date Registered", MHRA/EU SmPC section 9, FDA
    ORIG-1 action date) plus the never-infer-from-a-PDF-footer caution
  - is_generic's "anything other than the innovator itself" definition
  - therapeutic_areas: the two-source derivation order, the "never restate the
    indication text" prohibition, canonical-specialty-naming requirement
  - symptoms vs adverse_reactions (what it treats vs what it causes)
  - strengths: include the volume for per-volume presentations
  - the per_value / pack_size distinction (concentration denominator vs pack
    count) and that pack sizes exist in product_data only
  - the full presentations dedup contract: identity is strength set + form +
    route + pack_size; a combination is ONE entry; container material is NOT
    identity; a null pack_size means "not yet known", not "no pack"
  - key_risks re-ranking semantics and how they differ from the comprehensive
    columns.adverse_reactions list
  - excipients value/unit conventions and "most documents give no quantity"
  - the atc_code / reference_product "leave null rather than infer" rules
  - the LANGUAGE block: everything in English EXCEPT brand_name_local,
    mah_name, manufacturer, registration_number, atc_code, source_language
  - the TERMINOLOGY block: use the same canonical clinical term every time
  - the note that columns.indications and product_data.indications are
    different fields serving different purposes, and both must be filled

Plus the extraction rules: exhaustive enumeration, carry-forward of
established identity fields, multi-strength sentence handling, the
pivotal_evidence one-entry-per-study contract, always emit all 7 product_data
keys, and the columns/product_data mirroring requirement.
"""
'''


def as_literal(text, name="SYSTEM_PROMPT"):
    """
    Render `text` as a readable triple-quoted literal.

    The prompt has no backslashes and no triple-double-quote, both asserted
    below, so it needs no escaping at all and stays diffable line by line —
    which a repr() one-liner would not be. A leading \\ keeps the first prompt
    line off the assignment line without adding a newline to the value.
    """
    assert '"""' not in text, 'prompt contains a triple-double-quote; escaping needed'
    assert "\\" not in text, "prompt contains a backslash; escaping needed"
    return f'{name} = """\\\n{text}"""\n'


def main():
    merged = ai.SYSTEM_PROMPT
    assert merged.endswith("\n"), "prompt should end with a newline"
    assert "TWO VIEWS OF THE SAME FACTS" in merged, (
        "ai_extraction.SYSTEM_PROMPT is missing the extraction rules — this "
        "generator exports production's prompt, not a patched variant"
    )

    body = (
        DOCSTRING
        + "\n"
        + as_literal(merged)
        + "\n"
        + f"# Generated verbatim from processing/ai_extraction.py's SYSTEM_PROMPT.\n"
        + f"# {count_tokens(merged)} tokens, {len(merged)} chars.\n"
    )

    with open(OUT_PATH, "w") as fh:
        fh.write(body)

    # Import what was actually written and prove it round-trips, so a quoting
    # bug cannot ship a prompt that merely looks right.
    sys.path.insert(0, os.path.dirname(OUT_PATH))
    import importlib
    import harness_prompt_standalone
    importlib.reload(harness_prompt_standalone)
    if harness_prompt_standalone.SYSTEM_PROMPT != merged:
        raise SystemExit("FAILED: generated literal does not match the harness prompt")

    print(f"wrote {OUT_PATH}")
    print(f"  {count_tokens(merged):,} tokens, {len(merged):,} chars")
    print("  byte-identical to processing/ai_extraction.py SYSTEM_PROMPT: yes")


if __name__ == "__main__":
    main()

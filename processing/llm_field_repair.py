"""
LLM-only repair for two fields on existing Australia/TGA product rows:
product_data.presentations[].pack_size/pack_unit (+ per_value/per_unit), and
columns.indications / product_data.indications.

This exists to re-fix rows that processing/australia_manual_extraction.py's
deterministic mapping got wrong. It deliberately contains NO parsing, regex,
or chunk-counting of its own — the whole point is that the rule-based approach
could not read these two fields reliably, so here the model is the only thing
that interprets the raw source text. The only Python-side checks are on the
model's OUTPUT (see _validate_presentations): that it didn't alter fields it
was told to copy through, and that it didn't emit duplicate pack entries.
Those are guards against a bad response, not a second opinion on the source.

WHICH ROWS RUN (exactly this, nothing else):
  - The row's stored product_data has a presentation with pack_size NULL, AND
  - the raw json_data component actually has pack_size text to read.
A component whose raw pack_size is null/empty is SKIPPED — there is nothing
for the model to read, so calling it could only invent a value. Rows whose
stored pack_size is already populated are also skipped: they are not the
corrupted ones this repair targets.

Indications are repaired on the same pass for a selected row when the
component has indications text, since the same call covers both fields at no
extra cost.
"""
import json
import logging

from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are fixing exactly TWO parts of an already-extracted pharmaceutical product \
record from the Australian TGA (Therapeutic Goods Administration) product register: the \
`presentations` array's pack_size/pack_unit/per_value/per_unit fields, and `indications`. \
active_ingredients, form, and route in `presentations` are already correct — copy them through \
unchanged.

INPUT:
{
  "presentations": [<existing presentation object(s) — form, route, and active_ingredients are \
ALREADY CORRECT. per_value/per_unit are usually null; they may already be filled in some cases>],
  "pack_size_raw": "<raw pack description string from the source>",
  "indications_raw": "<raw indications text from the source>"|null
}

OUTPUT — exactly these two top-level keys, nothing else:
{
  "presentations": [<corrected presentation objects: SAME active_ingredients/form/route as a matching \
input presentation, copied verbatim — field-for-field, value-for-value, no changes, no rewording — \
with pack_size/pack_unit filled in and per_value/per_unit filled in where pack_size_raw states a \
per-unit fill volume/weight. One entry per DISTINCT pack configuration named in pack_size_raw, with \
exact duplicates (same pack_size AND pack_unit) removed.>],
  "indications": ["<verbatim indication statement, exact substring of indications_raw except for \
whitespace normalisation>", ...]|null
}

RULES for presentations/pack_size/per_value/per_unit:
- NEVER alter active_ingredients, form, or route — copy them exactly from the input presentation they \
belong with. In particular, copy strength_value/strength_unit exactly as given; never recompute them \
against a fill volume.
- Read EVERY configuration listed in pack_size_raw. It is usually a comma-separated list of several \
alternative packs sold under the same registration, and you must return one entry per distinct \
configuration — not just the first or the easiest one to read.
- pack_size is the COUNT of individual dosage units (ampoules, tablets, sachets, vials, syringes...) \
in that pack — NEVER a volume or weight. A liquid's fill volume per unit ('2mL', '4g') is not a pack_size.
- The source writes 'volume x count' in some records and 'count x volume' in others for the exact same \
kind of product — read each string in context and determine which number is genuinely the COUNT of \
packaged units versus which is a per-unit fill volume/weight. Do not assume a fixed left-to-right order.
- If pack_size_raw ALSO states a fill volume/weight per unit (e.g. '1x Vial of 2.4 mL', '2mL x 5 \
ampoules'), and the input presentation's per_value/per_unit are null, fill them in from that stated \
volume (per_value=2.4, per_unit='mL' for the first example). If the input's per_value/per_unit are \
already non-null, or pack_size_raw states no such volume, leave per_value/per_unit exactly as given — \
never invent one and never overwrite an existing value with a different one.
- When pack_size_raw lists several configurations of the SAME product and states the per-unit fill \
volume on only some of them, apply that one volume to EVERY entry — the product has a single fill \
volume per unit and the other chunks simply don't restate it. Example: for '2mL x 2, 10 ampoule pack, \
2mL x 5' all three entries get per_value=2, per_unit='mL' (including the '10 ampoule pack' one), \
because they are the same 2 mL ampoule sold in different quantities. Only use a different per_value \
for an entry if pack_size_raw explicitly states a different volume for that specific configuration.
- CRITICAL: pack_size_raw very often states ONLY a fill volume or weight and NO unit count at all — \
e.g. "4 mL", "15 mL", "500 mL", "50 g", "30mL Glass Type III Bottle", "75 mL". In that case pack_size \
and pack_unit MUST stay null, and the volume/weight goes to per_value/per_unit instead (per_value=4, \
per_unit="mL" for "4 mL"). NEVER put a volume or weight number in pack_size, and NEVER use a volume or \
weight unit ('ml', 'mL', 'g', 'mg', 'L') as pack_unit — pack_unit is only ever a countable dosage-unit \
noun ('tablet', 'ampoule', 'vial', 'sachet', 'capsule', 'syringe', 'bottle', 'patch', 'pen'). If you \
cannot name such a countable noun from the text, pack_size and pack_unit are both null.
- If a configuration bundles unrelated components (a kit, an applicator, a device with several separate \
parts) or is prose you cannot confidently reduce to one count+unit, OMIT that configuration entirely — \
never guess a number for it. If NO configuration in pack_size_raw can be read that way, return the \
input presentations unchanged with pack_size/pack_unit still null.
- Do NOT output two presentations with the same (pack_size, pack_unit) pair — that is a duplicate; keep \
only one.
- pack_unit is lowercase and SINGULAR ('tablet', not 'tablets'; 'ampoule', not 'ampoules').

RULES for indications:
- Return the FULL indications_raw text split into separate array entries, one per distinct indication/\
condition/use. Every word of the original text must appear in some entry — you are splitting, not \
summarising or shortening anything.
- Do NOT reword, paraphrase, or rephrase anything. Each entry must be an exact, verbatim substring of \
indications_raw (only whitespace may be normalised).
- If indications_raw already uses line breaks or a '; ' list, split on those. If it has none (one \
continuous block covering many conditions — e.g. an oncology drug's long list of cancer types run \
together with commas), split at each sentence boundary (after a '.' or '!' where a new indication \
clearly begins) — never split in the middle of a sentence, and never split on a comma that sits inside \
a sentence (e.g. 'Stage IIB, IIC, or III' is NOT a split point).
- A short disease/condition-name heading immediately preceding its own full sentence (e.g. \
'Melanoma,X is indicated for melanoma...') stays attached to that sentence as ONE entry.
- No two entries may be identical. If indications_raw genuinely repeats a statement, return it once.
- Do NOT fix typos, misspellings, capitalisation, or grammar in indications_raw, even when a word is \
obviously misspelled. If the source says 'Ondersertron' where it clearly means 'Ondansetron', your \
entry must still say 'Ondersertron' — this field is a verbatim record of what the regulator published, \
and silently correcting it makes the stored text differ from the source document. The ONLY permitted \
change is collapsing/trimming whitespace.

EXAMPLES:

Example 1 — presentations: [{"form": "injection, solution", "route": "intravenous, intramuscular", \
"per_value": null, "per_unit": null, "pack_size": null, "pack_unit": null, "active_ingredients": \
[{"substance": "Ondansetron", "strength_value": 5, "strength_unit": "mg", "salt": {"substance": \
"Ondansetron hydrochloride dihydrate", "value": null, "unit": null}}]}], \
pack_size_raw: "2mL x 2 (plastic), 10 ampoule pack, 2mL x 5 (glass and plastic), 2 mL x 4, 2mL x 1 (plastic)"
-> presentations: FIVE entries, identical except for pack_size, each with per_value=2, per_unit="mL", \
pack_unit="ampoule", and pack_size = 2, 10, 5, 4, 1 respectively.
(2mL is the fill volume PER AMPOULE — the same across all five configurations, so it is not a count; \
'10 ampoule pack' states the count directly; the '(plastic)'/'(glass and plastic)' material notes are \
dropped, there is no field for them. All five configurations must appear — returning only the one \
easiest chunk is wrong. active_ingredients including its nested salt object is copied through \
byte-for-byte on every entry.)

Example 2 — pack_size_raw: "5 x 20 mL ampoules, 10 x 20 mL ampoules"
-> two entries: pack_size 5 and 10, pack_unit "ampoule", per_value=20, per_unit="mL"
(the COUNT comes FIRST here — the opposite order from Example 1's source record, read correctly from \
context rather than assumed)

Example 3 — presentations: [{"form": "injection, solution", "route": "subcutaneous", "per_value": null, \
"per_unit": null, "pack_size": null, "pack_unit": null, "active_ingredients": [{"substance": \
"Pembrolizumab", "strength_value": 395, "strength_unit": "mg/mL", "salt": null}]}], \
pack_size_raw: "1x Vial of 2.4 mL"
-> one entry: pack_size=1, pack_unit="vial", per_value=2.4, per_unit="mL", active_ingredients unchanged
(strength_value/strength_unit stay exactly '395'/'mg/mL' even though '395 mg/mL' next to a 2.4 mL fill \
volume looks like it might multiply out to a total content — that arithmetic is not your job, only copy \
what was already there)

Example 4 — pack_size_raw: "1 kit containing 300 mg powder for injection vial + 1 x 2 mL diluent vial + 1 x 3 mL syringe"
-> presentations: unchanged from input, pack_size/pack_unit/per_value/per_unit all left null (a \
multi-component kit, not a single discrete pack count or a single fill volume — omit rather than guess)

Example 5 — presentations: [{"form": "solution", "route": "topical", "per_value": null, "per_unit": null, \
"pack_size": null, "pack_unit": null, "active_ingredients": [{"substance": "Povidone-iodine", \
"strength_value": 10, "strength_unit": "% w/v", "salt": null}]}], pack_size_raw: "15 mL"
-> one entry: pack_size=null, pack_unit=null, per_value=15, per_unit="mL", active_ingredients unchanged
(the source states ONLY a fill volume and no count of units, so pack_size/pack_unit stay null. Returning \
pack_size=15 with pack_unit="mL" would be WRONG — 15 mL is a volume, not a count of dosage units.)

Example 6 — indications_raw: "\\n\\nMelanoma,KEYTRUDA SC® (pembrolizumab) is indicated as monotherapy for \
the treatment of unresectable or metastatic melanoma in adults.,KEYTRUDA SC® (pembrolizumab) is indicated \
for the adjuvant treatment of adult and adolescent (12 years and older) patients with Stage IIB, IIC, or \
III melanoma who have undergone complete resection.,Non-small cell lung cancer (NSCLC),KEYTRUDA SC® ..."
-> indications: ["Melanoma,KEYTRUDA SC® (pembrolizumab) is indicated as monotherapy for the treatment of \
unresectable or metastatic melanoma in adults.", "KEYTRUDA SC® (pembrolizumab) is indicated for the \
adjuvant treatment of adult and adolescent (12 years and older) patients with Stage IIB, IIC, or III \
melanoma who have undergone complete resection.", "Non-small cell lung cancer (NSCLC),KEYTRUDA SC® ..."]
(the heading 'Melanoma' stays attached to its own sentence; the internal comma in 'Stage IIB, IIC, or \
III' is NOT a split point — only the sentence boundary is)

Respond with ONLY the JSON object — no markdown fences, no commentary.
"""


def _presentation_identity(p):
    """The part of a presentation the model must copy through untouched —
    used to verify its output rather than trust its claim. per_value/per_unit
    are deliberately NOT here: the model may FILL those in from null (checked
    separately below), which is a different rule from "must match exactly"."""
    return (
        p.get("form"), p.get("route"),
        tuple(sorted(
            (i.get("substance"), i.get("strength_value"), i.get("strength_unit"),
             json.dumps(i.get("salt"), sort_keys=True))
            for i in (p.get("active_ingredients") or [])
        )),
    )


# pack_unit must name something COUNTABLE. A volume/weight unit here means the
# model put a fill volume in pack_size (e.g. '15 mL' -> pack_size=15,
# pack_unit='ml'), which the prompt forbids but which it did do on real rows.
# Caught here so a wrong value can't reach the database even when the prompt
# is ignored — the volume is moved to per_value/per_unit where it belongs.
_VOLUME_WEIGHT_UNITS = frozenset((
    "ml", "l", "g", "mg", "mcg", "kg", "gm", "litre", "liter", "millilitre",
    "milliliter", "gram", "milligram", "microgram", "iu", "%",
))


def _validate_presentations(original, returned):
    """
    Guard against a bad model response, NOT a re-derivation of the answer.
    Returns the validated list, or None (caller keeps `original` untouched) if
    the model altered active_ingredients/form/route, or overwrote a per_value/
    per_unit that was already non-null. Duplicate (pack_size, pack_unit)
    entries are dropped here regardless of what the model did — a real
    duplicate reached the database once when the prompt was the only safeguard.
    A volume/weight pack_unit is corrected in place (see _VOLUME_WEIGHT_UNITS)
    rather than rejecting the whole row, since the value the model found is
    right — it just belongs in per_value/per_unit instead.
    """
    if not isinstance(returned, list) or not returned or not original:
        return None
    allowed_identities = {_presentation_identity(p) for p in original}
    orig_per_value, orig_per_unit = original[0].get("per_value"), original[0].get("per_unit")

    seen_pack, out = set(), []
    for p in returned:
        if not isinstance(p, dict):
            return None
        if _presentation_identity(p) not in allowed_identities:
            logger.warning(
                "[llm_field_repair] model altered active_ingredients/form/route — "
                "rejecting its presentations output entirely"
            )
            return None
        if orig_per_value is not None and p.get("per_value") != orig_per_value:
            logger.warning("[llm_field_repair] model overwrote a stated per_value — rejecting output")
            return None
        if orig_per_unit is not None and p.get("per_unit") != orig_per_unit:
            logger.warning("[llm_field_repair] model overwrote a stated per_unit — rejecting output")
            return None

        p = dict(p)
        pack_unit = (p.get("pack_unit") or "").strip().lower()
        if pack_unit and pack_unit in _VOLUME_WEIGHT_UNITS:
            volume = p.get("pack_size")
            logger.info(
                f"[llm_field_repair] pack_unit={p.get('pack_unit')!r} is a volume/weight unit, not a "
                f"countable one — moving {volume!r} to per_value/per_unit and leaving pack_size null"
            )
            p["pack_size"], p["pack_unit"] = None, None
            if orig_per_value is None and p.get("per_value") is None and volume is not None:
                p["per_value"], p["per_unit"] = volume, pack_unit if pack_unit != "ml" else "mL"

        key = (p.get("pack_size"), p.get("pack_unit"))
        if key in seen_pack:
            continue
        seen_pack.add(key)
        out.append(p)
    return out


def _dedupe(strings):
    out, seen = [], set()
    for s in strings or []:
        key = s.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(s.strip())
    return out


def _validate_indications(raw, returned):
    """
    Every returned entry must be a verbatim substring of the raw source text
    (whitespace-normalised), or the whole list is rejected and the caller
    keeps what was already stored.

    This is not paranoia: the model silently "fixed" the source's own typo
    ('Ondersertron' -> 'Ondansetron') on a real record, which would have made
    the stored indication differ from the published label. The prompt now
    forbids that, but a prompt rule alone has already proven insufficient
    twice in this module, so the claim is checked rather than trusted.
    """
    if not returned:
        return None
    normalise = lambda s: " ".join(s.split()).lower()
    raw_norm = normalise(raw or "")
    if not raw_norm:
        return None
    cleaned = _dedupe(returned)
    for entry in cleaned:
        if normalise(entry) not in raw_norm:
            logger.warning(
                f"[llm_field_repair] indication entry is not a verbatim substring of the source "
                f"(model reworded or corrected the text) — rejecting indications output: {entry[:120]!r}"
            )
            return None
    return cleaned


def repair_product(json_data, stored_product_data):
    """
    `stored_product_data` is the row's EXISTING product_data (what the
    deterministic extractor previously wrote and what this repair is
    correcting) — not a fresh re-extraction.

    Returns (repaired_product_data, indications_list, called). `called` is
    False when no component qualified, in which case nothing is changed and
    no LLM call was made (zero cost).

    Selection rule. The raw component must have pack_size text at all — a
    null/empty raw pack_size is always skipped, since there is nothing to read
    and a call could only invent a value. Given that, a row qualifies if
    EITHER:
      (a) a stored presentation has pack_size null — the mapper got nothing; or
      (b) the raw text lists multiple configurations (contains a comma) but
          only ONE presentation was stored — the mapper got *some* of them and
          silently dropped the rest.
    (b) exists because (a) alone missed real corruption: ZOFRAN's raw text
    lists five packs ('2mL x 2 (plastic), 10 ampoule pack, 2mL x 5 ...'), the
    mapper resolved only the one plainly-worded chunk to pack_size=10, and a
    non-null stored pack_size made the row look healthy while four real
    configurations were missing. This is a count comparison only — nothing
    here interprets what the raw text says.
    """
    from processing import llm_service_client

    components = [c for c in (json_data.get("components") or []) if isinstance(c, dict)]
    stored = stored_product_data or {}
    presentations = list(stored.get("presentations") or [])
    stored_indications = [i.get("condition") for i in (stored.get("indications") or [])
                          if isinstance(i, dict) and i.get("condition")]

    if not components or not presentations:
        return stored, stored_indications, False

    # Every presentation this repair might replace belongs to some component;
    # with one component (every TGA record observed) they are all its own.
    # Guard rather than assume, so a future multi-component record is skipped
    # loudly instead of having its presentations mismapped.
    if len(components) > 1:
        logger.warning(
            f"[llm_field_repair] {len(components)} components on one product — "
            f"skipping (presentation-to-component mapping is not established for this shape)"
        )
        return stored, stored_indications, False

    component = components[0]
    raw_pack = component.get("pack_size")
    raw_pack = str(raw_pack).strip() if raw_pack is not None else ""
    if not raw_pack:
        return stored, stored_indications, False

    # See the selection rule in this function's docstring.
    stored_pack_is_null = any(p.get("pack_size") is None for p in presentations)
    partially_parsed = "," in raw_pack and len(presentations) == 1
    if not (stored_pack_is_null or partially_parsed):
        return stored, stored_indications, False

    raw_ind = component.get("indications")
    user = json.dumps({
        "presentations": presentations,
        "pack_size_raw": raw_pack,
        "indications_raw": raw_ind if raw_ind else None,
    }, ensure_ascii=False)

    # Pinned to OpenRouter deliberately, not left to the usual
    # self-hosted-then-fallback selection: this repair is a one-off correction
    # pass whose output is judged against a hosted model's behaviour (the
    # prompt's rules and examples were tuned against it), so it must not
    # silently run on a self-hosted model just because LLM_SERVICE_BASE_URL
    # happens to be set in the environment.
    obj, provider, meta = llm_service_client.call_json(
        SYSTEM_PROMPT, user, max_tokens=8192, temperature=0, json_mode=True,
        parse_retries=1, provider=llm_service_client.OPENROUTER,
    )
    usage = meta.get("usage") if obj else None
    if usage:
        logger.info(
            f"[llm_field_repair] provider={provider} "
            f"prompt={getattr(usage, 'prompt_tokens', '?')} completion={getattr(usage, 'completion_tokens', '?')}"
        )
    if obj is None:
        logger.error("[llm_field_repair] LLM call failed or returned unparseable JSON — row left unchanged")
        return stored, stored_indications, False

    fixed = _validate_presentations(presentations, obj.get("presentations"))
    new_presentations = fixed if fixed is not None else presentations

    validated_ind = _validate_indications(raw_ind, obj.get("indications"))
    new_indications = validated_ind if validated_ind is not None else stored_indications

    repaired = dict(stored)
    repaired["presentations"] = new_presentations
    repaired["indications"] = [
        {"condition": c, "population": None, "line_of_therapy": None,
         "biomarker": None, "approval_date": None} for c in new_indications
    ]
    return repaired, new_indications, True


def _persist(cursor, product_id, product_data, indications):
    """Writes ONLY the two repaired fields — product_data and the flat
    indications column. Every other column on the row is left exactly as it
    is, so this can never disturb a field it wasn't asked to fix."""
    from psycopg2.extras import Json
    cursor.execute(
        """
        UPDATE drug.products
        SET product_data = %s,
            indications = %s,
            ai_extraction_status = 'DONE',
            ai_extraction_error = NULL,
            processing_status = 'ENRICHED',
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        """,
        (Json(product_data), indications, product_id),
    )


def run(country="Australia", limit=None, dry_run=False):
    """
    Repairs pack_size/indications on `country`'s TGA rows that meet the
    selection rule in repair_product's docstring: stored pack_size null AND
    raw json_data pack_size present. Sequential — an LLM call per row is the
    slow part and the qualifying set is small.
    """
    from db import get_db_connection
    from processing import llm_service_client

    conn = get_db_connection()
    stats = {}
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Mirrors repair_product's selection rule in SQL so rows that
            # cannot qualify are never fetched: raw pack_size must be present
            # and non-empty, AND either a stored presentation has pack_size
            # null, or the raw text lists several configs (has a comma) while
            # only one presentation was stored (partially parsed).
            query = """
                SELECT p.id, p.json_data, p.product_data
                FROM drug.products p
                LEFT JOIN drug.regulatory_geography g ON g.id = p.country_id
                WHERE g.country_name = %(country)s
                  AND p.json_data ? 'artg_id'
                  AND EXISTS (
                    SELECT 1 FROM jsonb_array_elements(p.json_data->'components') c
                    WHERE COALESCE(btrim(c->>'pack_size'), '') <> ''
                  )
                  AND (
                    EXISTS (
                      SELECT 1 FROM jsonb_array_elements(p.product_data->'presentations') pr
                      WHERE pr->'pack_size' = 'null'::jsonb OR pr->>'pack_size' IS NULL
                    )
                    OR (
                      jsonb_array_length(COALESCE(p.product_data->'presentations', '[]'::jsonb)) = 1
                      AND EXISTS (
                        SELECT 1 FROM jsonb_array_elements(p.json_data->'components') c
                        WHERE c->>'pack_size' LIKE '%%,%%'
                      )
                    )
                  )
                ORDER BY p.id
            """
            if limit:
                query += " LIMIT %(limit)s"
            cur.execute(query, {"country": country, "limit": limit})
            products = cur.fetchall()
        conn.commit()

        logger.info(f"[llm_field_repair] {len(products)} qualifying row(s) for {country}")
        for product in products:
            try:
                repaired, indications, called = repair_product(
                    product["json_data"], product["product_data"])
                if not called:
                    outcome = "skipped"
                elif dry_run:
                    print(f"=== product {product['id']} — DRY RUN, not persisted ===")
                    print(json.dumps({
                        "presentations": repaired["presentations"],
                        "indications": indications,
                    }, indent=2, ensure_ascii=False))
                    outcome = "dry_run"
                else:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        _persist(cur, product["id"], repaired, indications)
                        conn.commit()
                    outcome = "repaired"
            except Exception:
                conn.rollback()
                logger.exception(f"[llm_field_repair] product={product['id']} errored")
                outcome = "error"
            stats[outcome] = stats.get(outcome, 0) + 1
            logger.info(f"[llm_field_repair] product={product['id']} -> {outcome}")
    finally:
        conn.close()

    logger.info(f"[llm_field_repair] done for {country}: {stats}")
    logger.info(f"[llm_field_repair] LLM usage: {llm_service_client.provider_stats()}")
    return stats

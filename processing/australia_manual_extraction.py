"""
Deterministic (no-LLM) extraction for Australia / TGA (Therapeutic Goods
Administration) ARTG records — NOT the PBS (Pharmaceutical Benefits Scheme)
shape that also appears under "Australia" in this crawl.

Two genuinely different regulators' data are mixed into this country's raw
records:
  - TGA (artg_id / artg_category / approval_area present): the actual
    regulatory approval registry (ARTG — Australian Register of Therapeutic
    Goods). This is what production handles, and what this module maps.
  - PBS (pbs_code / item_restrictions / pricing present, no artg_id): a
    reimbursement/subsidy scheme sitting downstream of TGA approval, not a
    regulatory record itself. Out of scope here — see NotTgaRecordError.

TGA's ARTG covers three categories seen in this data (artg_category):
  - "Registered": TGA has assessed quality, safety, AND efficacy. Includes
    prescription and higher-risk OTC medicines. Indications here read as
    genuine clinical claims ("Relief from the burning pain of cystitis").
  - "Listed" / "Listed (Export Only)": lower-risk complementary/herbal/
    vitamin medicines, self-certified by the sponsor for safety only — TGA
    does NOT assess efficacy for these, and by law they cannot claim to
    treat a disease. Indications here read as marketing-style health claims
    ("Helps enhance/promote bone health"), not clinical conditions. This
    isn't a data-quality problem to fix; it's what "Listed" legally means.
All three share one JSON shape (`components[]` holding the actual
formulation), so one mapper covers all three — the distinction that matters
downstream is captured verbatim in product_type/registration_class, not
smoothed over.

What TGA's ARTG summary genuinely does NOT carry, left null/empty rather than
guessed: manufacturer name/address (only `sponsor_name` — the MAH-equivalent
— is given, no separate manufacturer), atc_code and therapeutic_areas (no
ATC data in this shape — that's PBS-only), approval.pathway/is_generic (no
originator/generic signal at all for these entries), symptoms/adverse_
reactions/contraindications/pivotal_evidence (no clinical trial or ADR data
in an ARTG summary).
"""
import re

from processing.us_manual_extraction import _clean, _title, base_substance

# ---------------------------------------------------------------- vocabulary

# Same idea as the other two manual extractors' infer_modality — a suffix or
# keyword is chemistry, not TGA-specific vocabulary.
_BIOLOGIC_SUFFIXES = ("mab", "cept", "kinra", "ase", "gene", "cel")
_VACCINE_KEYWORDS = ("vaccine", "toxoid")

# "Listed"/"Listed (Export Only)" medicines are legally restricted to
# lower-risk ingredients — a biological-pathway active ingredient can only be
# marketed via "Registered", never "Listed". This is a real regulatory rule,
# not an assumption about any specific ingredient, so defaulting these two
# categories to small_molecule is a rule, not a guess. "Registered" covers
# everything from OTC creams to biologics, so it gets no such default —
# only the keyword/suffix check below, same as the other two categories.
_LISTED_CATEGORIES = ("Listed", "Listed (Export Only)")


class NotTgaRecordError(ValueError):
    """Raised when json_data is PBS-shaped (or otherwise not a TGA ARTG
    record) — see the module docstring. The caller (processing/
    manual_extraction_runner.py) marks the row FAILED with this message
    rather than force-mapping fields that mean something different in PBS's
    schema, which would silently produce wrong data."""


def _require_tga_shape(json_data):
    if not json_data.get("artg_id"):
        if json_data.get("pbs_code") is not None:
            raise NotTgaRecordError(
                "PBS-shaped record (has pbs_code, no artg_id) — this pipeline "
                "only handles TGA ARTG data for Australia, see module docstring"
            )
        raise NotTgaRecordError("no artg_id on record — not a recognised TGA ARTG shape")


def _split_top_level(text, sep=";"):
    """Split on `sep` only OUTSIDE parentheses. Needed because an ingredient's
    '(Equivalent: X, Qty Y; Equivalent: Z, Qty W)' clause uses the same
    separator internally — a plain .split(sep) would cut it in half."""
    parts, depth, current = [], 0, []
    for ch in text or "":
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


# unit is everything after the number to end of string, not just the next
# token — 'Quantity: 10 billion CFU' has a two-word unit ('billion CFU'); a
# single-token capture silently drops the CFU qualifier, which for a
# probiotic count is the part that actually makes the number meaningful.
_QUANTITY_RE = re.compile(r"^(?P<name>[^,]+?),\s*Quantity:\s*(?P<value>[\d.]+)\s*(?P<unit>.+?)\s*$", re.I)


def _parse_ingredient(chunk):
    """
    'Silybum marianum, Quantity: 140 mg (Equivalent: Silybum marianum, Qty
    4.9 g)' -> ('Silybum marianum', 140, 'mg'). The '(Equivalent: ...)'
    clause is a unit-conversion footnote (e.g. raw herb equivalent for an
    extract), not a salt or a distinct fact this schema has a place for, so
    it's dropped rather than mapped somewhere it doesn't belong.
    """
    chunk = chunk.split("(")[0].strip()
    m = _QUANTITY_RE.match(chunk)
    if m:
        try:
            value = float(m.group("value"))
        except ValueError:
            value = None
        value = int(value) if value is not None and value == int(value) else value
        return _clean(m.group("name")), value, _clean(m.group("unit"))
    return _clean(chunk) or None, None, None


def infer_modality(bases, artg_category):
    joined = " ".join(b.lower() for b in bases if b)
    if any(k in joined for k in _VACCINE_KEYWORDS):
        return "vaccine"
    if any(joined.endswith(s) for s in _BIOLOGIC_SUFFIXES):
        return "biologic"
    if artg_category in _LISTED_CATEGORIES:
        return "small_molecule"
    return None


def _parse_pack_size(raw):
    """
    Usually a single integer ('56'). Sometimes a comma-separated LIST of
    alternative pack configurations sold under the one ARTG entry (e.g.
    '28 x 4g sachets, 8 sachets, 16 sachets, ...') — enumerating that into
    distinct presentations would mean parsing free-form pack descriptions
    ("28 x 4g" vs "8 sachets" aren't even the same kind of quantity), which
    risks inventing a pack count that isn't cleanly stated. Left null in that
    case rather than guessed; the raw text isn't a value this schema's
    numeric pack_size field can honestly represent anyway.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _to_iso_date(value):
    text = _clean(value)
    if not text:
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def build_presentation(component):
    names_raw = _split_top_level(component.get("active_ingredients") or "")
    ingredients = []
    for chunk in names_raw:
        name, value, unit = _parse_ingredient(chunk)
        base, salt_form = base_substance(name) if name else (None, None)
        ingredients.append({
            "substance": base,
            "strength_value": value,
            "strength_unit": unit,
            "salt": ({"substance": salt_form, "value": None, "unit": None}
                     if salt_form else None),
        })

    pack_size = _parse_pack_size(component.get("pack_size"))
    form = _clean(component.get("dosage_form"))
    route = _clean(component.get("route_of_administration"))

    return {
        # TGA's ARTG summary states total ingredient content per dosage unit
        # (a capsule, a sachet), not a per-volume concentration — there is no
        # per-mL/per-dose denominator field in this shape the way an
        # injectable's would carry one.
        "per_value": None,
        "per_unit": None,
        "pack_size": pack_size,
        # pack_size's own text often names the unit ('sachets') redundantly;
        # when pack_size couldn't be parsed as a plain number, pack_unit has
        # nothing reliable to pair it with either.
        "pack_unit": None,
        "form": form.lower() if form else None,
        "route": route.lower() if route else None,
        "active_ingredients": ingredients,
    }


def _indications_from(component):
    """
    One entry per LINE of the component's own 'indications' text, stored
    verbatim — not split further, not shortened. TGA's Listed-medicine format
    puts one claim per line (each line's trailing ' ; ' is the source's own
    list separator, not sentence punctuation to react to); Registered
    medicines instead give plain prose with no per-line structure, which
    correctly falls out of this as a single verbatim entry rather than being
    sentence-split into pieces.
    """
    text = component.get("indications")
    if not text:
        return []
    out, seen = [], set()
    for line in text.split("\n"):
        line = _clean(line.strip(" ;"))
        if not line or line.lower() in seen:
            continue
        seen.add(line.lower())
        out.append(line)
    return out


def substance_from(presentations, artg_category):
    bases, salts = [], []
    for p in presentations:
        for ing in p["active_ingredients"]:
            if ing["substance"] and ing["substance"] not in bases:
                bases.append(ing["substance"])
            if ing.get("salt") and ing["salt"]["substance"] not in salts:
                salts.append(ing["salt"]["substance"])
    return {
        "inn": " + ".join(bases) if bases else None,
        "salt_form": salts[0] if len(salts) == 1 and len(bases) == 1 else None,
        "modality": infer_modality(bases, artg_category),
        "target": None,
        "moa": None,
    }


def columns_from(json_data, presentations, substance, indications):
    artg_category = json_data.get("artg_category")
    product_name = _clean(json_data.get("product_name"))
    registration_date = _to_iso_date(json_data.get("start_date"))
    # TGA's own 'effective date' for this ARTG entry — real regulatory
    # metadata, not a PDF-generation timestamp, but it can also just reflect
    # an administrative re-publication rather than an actual label content
    # change. Mapped here as the closest available analogue since this
    # schema has no separate label-version field, with that caveat on record
    # (see provenance in extract()).
    label_revision_date = _to_iso_date(json_data.get("effective_date"))
    warnings = _clean(json_data.get("warnings"))

    return {
        "product_name_en": None,
        "brand_name": product_name,
        "brand_name_local": None,
        "generic_name": substance["inn"],
        "mah_name": _clean(json_data.get("sponsor_name")),
        "mah_address": None,
        # No manufacturer name/address field exists in this ARTG shape — only
        # the sponsor (MAH-equivalent) is given.
        "manufacturer": None,
        "manufacturer_address": None,
        "registration_number": _clean(json_data.get("artg_id")),
        "product_type": artg_category,
        # No registration-validity field (e.g. Cancelled/Suspended) exists in
        # this shape to map to REGISTRATION status — see module docstring.
        "status": None,
        "registration_date": registration_date,
        "approval_date": registration_date,
        "market_authorization_date": registration_date,
        "expiry_date": None,
        "withdrawal_date": None,
        "label_revision_date": label_revision_date,
        # No ATC classification in this shape — that's PBS-only data.
        "atc_code": None,
        "is_generic": None,
        "reference_product": None,
        "source_language": "en",
        "therapeutic_areas": [],
        "indications": indications,
        "symptoms": [],
        "adverse_reactions": [],
        "contraindications": [],
        "active_ingredients": sorted({i["substance"] for p in presentations
                                      for i in p["active_ingredients"] if i["substance"]}),
        "strengths": sorted({f"{i['strength_value']}{(' ' + i['strength_unit']) if i['strength_unit'] else ''}"
                             for p in presentations for i in p["active_ingredients"]
                             if i["strength_value"] is not None}),
        "dosage_forms": sorted({p["form"] for p in presentations if p["form"]}),
        "routes": sorted({p["route"] for p in presentations if p["route"]}),
        # A regulator/sponsor-stated safety warning (e.g. an allergen
        # declaration) is real safety information, not a guess — carried
        # verbatim rather than dropped just because it's outside the usual
        # ADR/contraindication shape.
        "_warnings_for_key_risks": [warnings] if warnings else [],
    }


def excipients_from(component):
    raw = component.get("excipient_ingredients")
    if not raw:
        return []
    out, seen = [], set()
    for chunk in _split_top_level(raw):
        name = _title(_clean(chunk))
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        # TGA's ARTG summary states excipient NAMES only, never a
        # per-dosage-unit quantity — same situation as the US module's
        # excipients_from, so value/unit stay null rather than invented.
        out.append({"name": name, "name_local": None, "value": None, "unit": None})
    return out


def extract(json_data, document_text=None):
    """
    Map one Australia/TGA ARTG raw record to the same {"columns",
    "product_data"} shape processing/ai_extraction.py produces. No network
    calls, no model, and no document_text is read — the ARTG summary's own
    JSON (components[]) already carries everything this module maps,
    including indications as a real structured field rather than prose.
    `document_text` is accepted only for interface parity; raises
    NotTgaRecordError for a PBS-shaped (or otherwise unrecognised) record —
    see the module docstring.
    """
    json_data = json_data or {}
    _require_tga_shape(json_data)

    components = [c for c in (json_data.get("components") or []) if isinstance(c, dict)]
    presentations = [build_presentation(c) for c in components]
    indications = []
    for c in components:
        for ind in _indications_from(c):
            if ind not in indications:
                indications.append(ind)

    substance = substance_from(presentations, json_data.get("artg_category"))
    columns = columns_from(json_data, presentations, substance, indications)
    key_risks = columns.pop("_warnings_for_key_risks")

    excipients = []
    for c in components:
        for exc in excipients_from(c):
            if exc not in excipients:
                excipients.append(exc)

    product_data = {
        "_schema_version": "1",
        "substance": substance,
        "presentations": presentations,
        "indications": [{"condition": c, "population": None, "line_of_therapy": None,
                         "biomarker": None, "approval_date": None} for c in indications],
        "approval": {
            "pathway": None,
            "registration_class": columns["product_type"],
            "conditional": None,
            "priority_review": None,
            "reference_product": None,
        },
        "pivotal_evidence": [],
        "key_risks": key_risks,
        "excipients": excipients,
    }
    provenance = {
        "substance.inn": "field map (components[].active_ingredients, comma+Quantity parse + salt strip)",
        "substance.modality": ("rule (artg_category='Listed'* -> small_molecule; else name suffix/keyword)"
                               if substance["modality"] else "unavailable (Registered category, no keyword match)"),
        "substance.target": "unavailable (no data in ARTG summary)",
        "substance.moa": "unavailable (no data in ARTG summary)",
        "presentations": "field map (components[] dosage_form/route/active_ingredients)",
        "columns.indications": "verbatim, one entry per line of components[].indications — not sentence-split",
        "columns.label_revision_date": ("field map (effective_date) — TGA registry metadata, but may reflect an "
                                        "administrative update rather than an actual label content change"
                                        if columns["label_revision_date"] else "unavailable"),
        "columns.atc_code/therapeutic_areas": "unavailable (ATC data is PBS-only, not present in the TGA ARTG shape)",
        "columns.is_generic/approval.pathway": "unavailable (no originator/generic signal in the TGA ARTG shape)",
        "key_risks": ("field map (top-level 'warnings')" if key_risks else "unavailable (no warnings field on record)"),
        "excipients": "field map (components[].excipient_ingredients) — names only, source gives no quantities",
    }
    return {"columns": columns, "product_data": product_data, "provenance": provenance}

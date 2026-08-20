"""
Deterministic (no-LLM) extraction for Saudi Arabia / SFDA records.

Unlike every other crawler in this pipeline, SFDA's json_data is a single
flat, fully-structured registry row per product — no PDF, no document_url, no
prose at all (confirmed against the whole Saudi Arabia raw-record set: every
one of the ~1,120 rows carries the exact same key set, and fields like
drug_type/marketing_status/authorization_status only ever take a handful of
fixed values). There is nothing here an LLM would be reading; it would just be
doing the same field-by-field mapping this module does, at API cost and
latency. So Stage C (processing/ai_extraction.py) should never run on this
country — see processing/manual_extraction_runner.py for how a row processed
by this module is marked DONE the same way a successful Stage C row is, which
is what stops it being re-claimed on a later pipeline run.

Returns the same {"columns", "product_data", "provenance"} shape as
processing/ai_extraction.py and processing/us_manual_extraction.py, so results
are directly comparable across countries.

What SFDA's registry genuinely does NOT carry, and which are therefore always
left null/empty rather than guessed: indications, symptoms, adverse_reactions,
contraindications, key_risks, pivotal_evidence, excipients, atc-derived
target/moa, and — despite `register_year` being known — the flat
registration_date/approval_date/market_authorization_date columns (they are
native DATE columns with no year-only mode; see `_registration_year`). The
year itself is not thrown away: it's preserved at its real precision in
product_data.approval.registration_year, which has no such type constraint.
"""
import re

from processing.us_manual_extraction import _clean, _title, base_substance

# ---------------------------------------------------------------- vocabulary

# SFDA's own product classification. "NCE" (New Chemical Entity) is SFDA's
# term for a first-in-market/originator product — the closest equivalent to
# an "Innovator" NDA in the US module's _CLASS_LABEL.
_PRODUCT_TYPE_LABEL = {
    "Generic": "Generic",
    "Biological": "Biological",
    "NCE": "New Chemical Entity",
}
_PATHWAY_BY_DRUG_TYPE = {
    "Generic": "generic",
    "NCE": "innovator",
    # A "Biological" entry could itself be an originator biologic or a
    # biosimilar of one — SFDA's own field doesn't distinguish them, and
    # nothing else in this record does either, so this is left unmapped
    # rather than assumed either way.
    "Biological": None,
}

# authorization_status as observed across the full Saudi corpus is only ever
# "Valid" today, but the CHECK constraint on drug.products.status only
# accepts a closed set of REGISTRATION-state words, so an unrecognised future
# value (e.g. "Cancelled", "Suspended") maps to None rather than guessing at
# its English wording.
_STATUS_MAP = {"valid": "Active"}

# WHO ATC classification is a published, country-independent ontology, so
# mapping its codes to a broad medical specialty is a rule, not a guess — the
# same category of move as the US module's openfda.pharm_class_epc mapping.
# Checked most-specific prefix first (3-char) then the ATC level-1 letter.
_ATC_PREFIX_AREA = {
    "A10": "Endocrinology",
    "B01": "Hematology",
    "M05": "Rheumatology",
    "N01": "Anaesthesiology",
    "N02": "Pain Management",
    "N05": "Psychiatry",
    "N06": "Psychiatry",
    "S01": "Ophthalmology",
    "S02": "Otolaryngology",
}
_ATC_LEVEL1_AREA = {
    "A": "Gastroenterology",
    "B": "Hematology",
    "C": "Cardiology",
    "D": "Dermatology",
    "G": "Urology",
    "H": "Endocrinology",
    "J": "Infectious Diseases",
    "L": "Oncology",
    "M": "Rheumatology",
    "N": "Neurology",
    "P": "Infectious Diseases",
    "R": "Pulmonology",
    "S": "Ophthalmology",
    # V (various) has no single specialty and is deliberately left unmapped.
}

# Reused for substance.modality, same idea as us_manual_extraction's
# infer_modality — an ingredient-name suffix/keyword is not FDA-specific
# vocabulary, it's chemistry, so the same heuristic applies here.
_BIOLOGIC_SUFFIXES = ("mab", "cept", "kinra", "ase", "gene", "cel")
_PEPTIDE_NAMES = (
    "insulin", "calcitonin", "octreotide", "leuprolide", "goserelin",
    "desmopressin", "vasopressin", "oxytocin", "glucagon", "teriparatide",
    "liraglutide", "semaglutide", "exenatide", "corticotropin", "cosyntropin",
)


def _registration_year(json_data):
    """
    SFDA gives `register_year` (an integer) but no month/day. drug.products'
    registration_date/approval_date/market_authorization_date are native
    Postgres DATE columns (schema/schema.sql) — a DATE has no year-only mode,
    so storing a bare year there would mean inventing a day and month that
    aren't in the source (e.g. stamping January 1st, which states a precision
    the registry doesn't have). Rather than fabricate one, those three flat
    columns stay null for Saudi Arabia; the year itself is preserved in
    product_data.approval.registration_year instead (see extract() below),
    since product_data is JSONB and has no such type constraint.
    """
    year = json_data.get("register_year")
    if not year:
        return None
    try:
        return int(year)
    except (TypeError, ValueError):
        return None


def _status(json_data):
    raw = (json_data.get("authorization_status") or "").strip().lower()
    return _STATUS_MAP.get(raw)


def _product_type(drug_type):
    return _PRODUCT_TYPE_LABEL.get(drug_type, drug_type)


def _atc_area(atc_code):
    code = (atc_code or "").strip().upper()
    if not code:
        return None
    return _ATC_PREFIX_AREA.get(code[:3]) or _ATC_LEVEL1_AREA.get(code[:1])


def _route(administration_route):
    """'Parenteral use' -> 'parenteral'. SFDA always suffixes the route with
    ' use'; stripping it mechanically covers every value in the corpus
    without needing an exhaustive per-route lookup table."""
    text = _clean(administration_route)
    if not text:
        return None
    text = re.sub(r"\s+use\s*$", "", text, flags=re.I)
    return text.lower() or None


# SFDA spells multiplier-prefixed inorganic salts as two words ('TRI SODIUM
# CITRATE'), where the shared base_substance() (processing/us_manual_extraction,
# written against FDA's always-fused 'TRISODIUM CITRATE' spelling) only
# recognises the fused form. Left un-normalised, 'TRI SODIUM CITRATE' strips as
# base='Tri' + salt='Tri sodium citrate' — nonsense, since 'Tri' is not a
# substance. Fusing the prefix first makes it hit the same is-it-just-a-
# counter-ion guard that already protects 'SODIUM CHLORIDE' from being split.
_MULTIPLIER_PREFIX_RE = re.compile(
    r"\b(mono|di|tri)\s+(sodium|potassium)\b", re.I)


def _ingredient_names(scientific_name):
    raw = _clean(scientific_name) or ""
    names = [n.strip() for n in raw.split(",") if n.strip()]
    return [_MULTIPLIER_PREFIX_RE.sub(lambda m: m.group(1) + m.group(2), n)
            for n in names]


def _strength_values(strength):
    raw = _clean(strength) or ""
    if not raw:
        return []

    def num(token):
        token = token.strip()
        if not token:
            return None
        try:
            val = float(token)
        except ValueError:
            return None
        return int(val) if val == int(val) else val

    return [num(t) for t in raw.split(",")]


def infer_modality(bases, drug_type):
    if drug_type == "Biological":
        return "biologic"
    joined = " ".join(b.lower() for b in bases if b)
    if not joined:
        return None
    if "vaccine" in joined or "toxoid" in joined:
        return "vaccine"
    if any(p in joined for p in _PEPTIDE_NAMES):
        return "peptide"
    if any(joined.endswith(s) for s in _BIOLOGIC_SUFFIXES):
        return "biologic"
    return "small_molecule"


def build_presentation(json_data):
    """One SFDA record IS one presentation — unlike the US module, there is no
    products[] table to fan out over; each raw record already describes a
    single strength/form/route/pack combination."""
    names = _ingredient_names(json_data.get("scientific_name"))
    values = _strength_values(json_data.get("strength"))
    unit = _clean(json_data.get("strength_unit"))

    ingredients = []
    for idx, name in enumerate(names or [None]):
        base, salt_form = base_substance(name) if name else (None, None)
        value = values[idx] if idx < len(values) else None
        ingredients.append({
            "substance": base,
            "strength_value": value,
            "strength_unit": unit,
            "salt": ({"substance": salt_form, "value": None, "unit": None}
                     if salt_form else None),
        })

    per_value = json_data.get("size")
    per_unit = _clean(json_data.get("size_unit"))
    pack_size = json_data.get("package_size")
    pack_unit = _clean(json_data.get("package_type"))

    return {
        "per_value": per_value,
        "per_unit": per_unit,
        "pack_size": pack_size,
        "pack_unit": pack_unit.lower() if pack_unit else None,
        "form": _clean(json_data.get("pharmaceutical_form")).lower()
                if json_data.get("pharmaceutical_form") else None,
        "route": _route(json_data.get("administration_route")),
        "active_ingredients": ingredients,
    }


def substance_from(presentation, drug_type):
    bases = [i["substance"] for i in presentation["active_ingredients"] if i["substance"]]
    salts = [i["salt"]["substance"] for i in presentation["active_ingredients"] if i.get("salt")]
    return {
        "inn": " + ".join(bases) if bases else None,
        "salt_form": salts[0] if len(salts) == 1 and len(bases) == 1 else None,
        "modality": infer_modality(bases, drug_type),
        # No label/monograph text exists in this source, so neither of these
        # can be filled without fabricating a claim the registry never makes.
        "target": None,
        "moa": None,
    }


def _manufacturer_fields(manufacturers):
    """
    [{'name': 'ANIKA THERAPEUTICS', 'country': 'United States'},
     {'name': 'Laboratories Biove', 'country': 'France'}] ->
    ("ANIKA THERAPEUTICS; Laboratories Biove", "United States; France")

    Country is all this source gives beyond the name — no street/city — so it
    is what fills manufacturer_address, kept as its own field (rather than
    folded into the name) and aligned positionally with `manufacturer` so the
    Nth name and the Nth address entry are the same manufacturer.
    Deduplicated by name, order-preserving.
    """
    names, addresses, seen = [], [], set()
    for m in manufacturers or []:
        if not isinstance(m, dict) or not m.get("name"):
            continue
        name = _clean(m["name"])
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        names.append(name)
        addresses.append(_clean(m.get("country")) or "")
    return "; ".join(names) or None, "; ".join(addresses) or None


def columns_from(json_data, presentation, substance):
    drug_type = json_data.get("drug_type")
    pathway = _PATHWAY_BY_DRUG_TYPE.get(drug_type)
    trade_name = _clean(json_data.get("trade_name"))
    trade_name_ar = _clean(json_data.get("trade_name_ar"))
    atc = _clean(json_data.get("atc_code_1")) or _clean(json_data.get("atc_code_2"))
    manufacturer, manufacturer_address = _manufacturer_fields(json_data.get("manufacturers"))

    return {
        # trade_name is already Latin-script/English in every sample checked
        # (it is SFDA's own English registry field), so this stays null per
        # the same rule the US module applies to its already-English names.
        "product_name_en": None,
        "brand_name": trade_name,
        "brand_name_local": trade_name_ar,
        "generic_name": substance["inn"],
        "mah_name": _clean(json_data.get("marketing_company")),
        # Only the COUNTRY is captured for MAH/manufacturer in this source, not
        # a street address — filled with just that rather than left null, on
        # the same basis as manufacturer_address below.
        "mah_address": _clean(json_data.get("marketing_company_country")),
        "manufacturer": manufacturer,
        "manufacturer_address": manufacturer_address,
        "registration_number": _clean(json_data.get("registration_number")),
        "product_type": _product_type(drug_type),
        "status": _status(json_data),
        # Not fabricated to a day/month — see _registration_year and extract()'s
        # product_data.approval.registration_year, which carries the real value.
        "registration_date": None,
        "approval_date": None,
        "market_authorization_date": None,
        "expiry_date": None,
        "withdrawal_date": None,
        "label_revision_date": None,
        "atc_code": atc,
        "is_generic": (None if pathway is None else pathway != "innovator"),
        "reference_product": None,
        "source_language": "en",
        "therapeutic_areas": [a for a in [_atc_area(atc)] if a],
        "indications": [],
        "symptoms": [],
        "adverse_reactions": [],
        "contraindications": [],
        "active_ingredients": sorted({i["substance"] for i in presentation["active_ingredients"]
                                      if i["substance"]}),
        "strengths": sorted({f"{i['strength_value']}{(' ' + i['strength_unit']) if i['strength_unit'] else ''}"
                             for i in presentation["active_ingredients"]
                             if i["strength_value"] is not None}),
        "dosage_forms": [presentation["form"]] if presentation["form"] else [],
        "routes": [presentation["route"]] if presentation["route"] else [],
    }


def extract(json_data, document_text=None):
    """
    Map one Saudi Arabia (SFDA) raw record to the same {"columns",
    "product_data"} shape processing/ai_extraction.py produces. No network
    calls, no model — every field below is either copied straight from
    json_data or derived by a fixed rule (salt stripping, ATC->specialty,
    route-suffix stripping). `document_text` is accepted only for interface
    parity with the other extractors; SFDA records never have one.
    """
    json_data = json_data or {}
    presentation = build_presentation(json_data)
    substance = substance_from(presentation, json_data.get("drug_type"))
    columns = columns_from(json_data, presentation, substance)

    registration_year = _registration_year(json_data)
    product_data = {
        "_schema_version": "1",
        "substance": substance,
        "presentations": [presentation],
        "indications": [],
        "approval": {
            "pathway": _PATHWAY_BY_DRUG_TYPE.get(json_data.get("drug_type")),
            "registration_class": columns["product_type"],
            "conditional": None,
            "priority_review": None,
            "reference_product": None,
            # Extra key beyond the standard 6-key product_data contract
            # (schema/product_data_spec.md) — added here, not to the flat
            # DATE columns, because it's the only place a bare year (no
            # fabricated day/month) can be stored. See _registration_year.
            "registration_year": registration_year,
        },
        "pivotal_evidence": [],
        "key_risks": [],
        "excipients": [],
    }
    provenance = {
        "substance.inn": "field map (scientific_name, comma-split + salt strip)",
        "substance.modality": "rule (drug_type='Biological' + name suffix/keyword)",
        "substance.target": "unavailable (no label text in this source)",
        "substance.moa": "unavailable (no label text in this source)",
        "presentations": "field map (strength/strength_unit/size/package_* — one record is one presentation)",
        "approval.pathway": "rule (drug_type: Generic->generic, NCE->innovator, Biological->unmapped)",
        "approval.registration_year": ("field map (register_year)" if registration_year
                                       else "unavailable (no register_year on record)"),
        "columns.therapeutic_areas": ("rule (WHO ATC code -> specialty)"
                                      if columns["therapeutic_areas"] else "unavailable (no atc_code on record)"),
        "columns.registration_date/approval_date/market_authorization_date": (
            "unavailable as a DATE (only register_year known, no day/month — "
            "not fabricated; see product_data.approval.registration_year instead)"
        ),
        "columns.status": ("field map (authorization_status)" if columns["status"] else "unavailable (unrecognised authorization_status value)"),
        "indications/key_risks/pivotal_evidence/excipients": "unavailable (registry record, no label/monograph text)",
    }
    return {"columns": columns, "product_data": product_data, "provenance": provenance}

"""
Deterministic (no-LLM) extraction for United States / Drugs@FDA records.

The FDA crawler's json_data is already structured — `products[]` carries a
one-line strength/form/route/ingredient table per marketed product,
`approval_history` carries the dated approval actions, `application_type_code`
carries the pathway, `openfda` carries brand/generic/manufacturer/route and
FDA's own pharmacologic class, and `spl_labels[]` carries the label already
split into named SPL sections (indications_and_usage, mechanism_of_action,
boxed_warning, clinical_studies, ...).

So for the US, most of the product_data contract can be filled by mapping
fields rather than by asking a model to read prose. This module does that
mapping. It returns exactly the same two-key shape as
processing/ai_extraction.py ({"columns": ..., "product_data": ...}) so the two
can be compared, or this one substituted, without touching persistence.

Where a field genuinely only exists as prose (indications wording, pivotal
trial designs), this module extracts what a rule can defend and leaves the
rest null rather than guessing — see `provenance` in the returned dict for
which keys were filled by which mechanism.
"""
import re

# ---------------------------------------------------------------- vocabulary

# Salt/ester/hydrate qualifiers FDA appends to an ingredient name. Stripping
# these yields the INN base — 'ABACAVIR SULFATE' -> 'Abacavir'. Order matters:
# multiword forms must be tried before their single-word prefixes.
_SALT_TOKENS = (
    "hydrochloride dihydrate", "hydrochloride monohydrate", "hydrogen tartrate",
    "sodium succinate", "sodium phosphate", "sodium sulfate", "sodium chloride",
    "calcium chloride", "hydrobromide", "hydrochloride", "dihydrochloride",
    "methylbromide", "monohydrate", "dihydrate", "trihydrate", "anhydrous",
    "besylate", "bitartrate", "camsylate", "citrate", "decanoate", "dipropionate",
    "esylate", "fumarate", "gluconate", "hyclate", "lactate", "maleate",
    "mesylate", "napsylate", "nitrate", "oxalate", "palmitate", "pamoate",
    "phosphate", "propionate", "salicylate", "stearate", "succinate", "sulfate",
    "trisodium", "disodium", "monosodium", "dipotassium", "monopotassium",
    "sulphate", "tannate", "tartrate", "tosylate", "valerate", "acetate",
    "benzoate", "bromide", "chloride", "iodide", "sodium", "potassium",
    "calcium", "magnesium", "meglumine", "hydrate", "base",
)

# Suffixes that identify a biologic without needing the application type.
_BIOLOGIC_SUFFIXES = ("mab", "cept", "kinra", "ase", "gene", "cel")
_PEPTIDE_NAMES = (
    "insulin", "calcitonin", "octreotide", "leuprolide", "goserelin",
    "desmopressin", "vasopressin", "oxytocin", "glucagon", "teriparatide",
    "liraglutide", "semaglutide", "exenatide", "corticotropin", "cosyntropin",
)
# Large protein hormones — biologics rather than peptides, regardless of
# whether FDA approved them under an NDA (many pre-date the BLA transition).
_PROTEIN_NAMES = ("gonadotropin", "somatropin", "menotropins", "urofollitropin",
                  "follitropin", "lutropin", "thyrotropin", "streptokinase")

# strength_unit normalisation: FDA writes them uppercase and unspaced.
_UNIT_MAP = {
    "MG": "mg", "G": "g", "GM": "g", "GRAM": "g", "MCG": "mcg", "UG": "mcg",
    "KG": "kg", "ML": "mL", "L": "L", "IU": "IU", "UNIT": "units",
    "UNITS": "units", "MEQ": "mEq", "MMOL": "mmol", "%": "%", "INH": "inhalation",
    "VIAL": "vial", "BOT": "bottle", "TBSP": "tablespoon", "SPRAY": "spray",
    "ACT": "actuation", "MG/ML": "mg/mL",
}

# FDA dosage_form_route strings are uppercase catalogue values; map to the
# lowercase clinical wording the rest of the schema uses.
_ROUTE_MAP = {
    "ORAL": "oral", "INJECTION": "injection", "INTRAVENOUS": "intravenous",
    "INTRAMUSCULAR": "intramuscular", "SUBCUTANEOUS": "subcutaneous",
    "TOPICAL": "topical", "OPHTHALMIC": "ophthalmic", "OTIC": "otic",
    "NASAL": "nasal", "INHALATION": "inhalation", "RECTAL": "rectal",
    "VAGINAL": "vaginal", "SUBLINGUAL": "sublingual", "BUCCAL": "buccal",
    "TRANSDERMAL": "transdermal", "INTRA-ARTERIAL": "intra-arterial",
    "INTRATHECAL": "intrathecal", "INTRAVESICAL": "intravesical",
    "EPIDURAL": "epidural", "IMPLANTATION": "implantation",
    "INTRAOCULAR": "intraocular", "IRRIGATION": "irrigation",
    "PERCUTANEOUS": "percutaneous", "INTRACAVERNOUS": "intracavernous",
    "INTRAPERITONEAL": "intraperitoneal", "DENTAL": "dental",
    "FOR RX COMPOUNDING": "for prescription compounding",
    "SINGLE-USE": None, "N/A": None,
}


def _clean(text):
    if text is None:
        return None
    text = str(text)
    # Orange Book footnote markers are appended straight onto the value:
    #   '100MG **Federal Register determination that ...**'
    text = re.sub(r"\*\*.*?\*\*", " ", text, flags=re.S)
    text = re.sub(r"\s+", " ", text).strip(" ;,")
    return text or None


def _title(name):
    """
    Normalise an FDA all-caps substance name to sentence case, the casing the
    rest of the schema uses ('Ondansetron hydrochloride dihydrate' — only the
    first word capitalised, since the salt qualifier is not a proper noun).

    FDA also inverts some names for alphabetical listing the way an index does:
    'GONADOTROPIN, CHORIONIC' is the entry for chorionic gonadotropin. A single
    comma with no digits on either side is that inversion, so flip it back.
    """
    cleaned = _clean(name)
    if not cleaned:
        return None
    if cleaned.count(",") == 1 and not any(c.isdigit() for c in cleaned):
        head, _, tail = cleaned.partition(",")
        if head.strip() and tail.strip():
            cleaned = f"{tail.strip()} {head.strip()}"
    words = cleaned.lower().split()
    out = []
    for i, w in enumerate(words):
        # Keep alphanumeric designators as-is: 'b12', 'a1c', 'HCl' style tokens.
        if any(c.isdigit() for c in w):
            out.append(w)
        else:
            out.append(w.capitalize() if i == 0 else w)
    return " ".join(out)


def base_substance(name):
    """'ABACAVIR SULFATE' -> 'Abacavir'. Returns (base, salt_form_or_None)."""
    cleaned = _clean(name)
    if not cleaned:
        return None, None
    low = cleaned.lower()
    stripped = low
    matched = False
    changed = True
    # Strip repeatedly: 'HYDROCORTISONE SODIUM SUCCINATE' needs one pass,
    # 'ONDANSETRON HYDROCHLORIDE DIHYDRATE' also one (multiword token first),
    # but 'X SODIUM ANHYDROUS' needs two.
    while changed:
        changed = False
        for token in _SALT_TOKENS:
            if not stripped.endswith(" " + token):
                continue
            remainder = stripped[: -(len(token) + 1)].strip()
            # An inorganic salt is BOTH halves — 'SODIUM CHLORIDE' is the
            # substance, not sodium formulated as a chloride. Refuse the strip
            # whenever what is left behind is itself only a counter-ion.
            if remainder in _SALT_TOKENS:
                continue
            stripped = remainder
            matched = changed = True
            break
    if not stripped:
        return _title(cleaned), None
    return _title(stripped), (_title(cleaned) if matched else None)


def _norm_unit(unit):
    if not unit:
        return None
    key = unit.strip().upper()
    return _UNIT_MAP.get(key, unit.strip().lower())


_STRENGTH_RE = re.compile(
    r"""^\s*
    (?:EQ\s+)?                                  # 'EQ 600MG BASE' = base equivalent
    (?P<value>[\d.,]+)\s*
    (?P<unit>%|[A-Za-z/]+)                      # MG, GM, UNITS, %, MG/ML
    (?:\s+(?P<basis>BASE|IRON|[A-Z]+))?         # 'BASE' / 'IRON' qualifier
    (?:\s*/\s*(?P<per_value>[\d.,]*)\s*(?P<per_unit>[A-Za-z%]+))?   # /2ML, /VIAL, /ML
    \s*$""",
    re.X | re.I,
)


def parse_strength(text):
    """
    Parse one FDA strength token into (value, unit, per_value, per_unit).

    Handles: '500MG', '0.5MG', '10,000 UNITS/ML', 'EQ 600MG BASE',
    'EQ 1GM BASE/VIAL', '5MG/ML', '4MG/2ML', '10%', '0.12MG/INH',
    '100GM/BOT'. A trailing parenthetical is a restatement of the same
    concentration ('125MG/125ML (1MG/ML)') and is dropped.
    """
    text = _clean(text)
    if not text:
        return (None, None, None, None)
    text = re.sub(r"\(.*?\)", " ", text).strip()
    m = _STRENGTH_RE.match(text)
    if not m:
        return (None, None, None, None)

    def num(raw):
        if raw in (None, ""):
            return None
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            return None
        return int(val) if val == int(val) else val

    value = num(m.group("value"))
    unit = m.group("unit")
    per_unit = m.group("per_unit")
    per_value = num(m.group("per_value"))
    # 'MG/ML' arrives as one unit token; split it into strength + basis.
    # 'MG/ML' and 'UNITS/VIAL' arrive as one greedy unit token; split them.
    if unit and "/" in unit and not per_unit:
        unit, _, per_unit = unit.partition("/")
        per_value = None
    # '/ML' with no number means per 1 unit of volume. A '/VIAL' or '/BOT'
    # denominator is a container, not a measured volume, so it gets no
    # per_value at all (see product_data_spec.md example 3).
    if per_unit and per_value is None and _norm_unit(per_unit) in ("mL", "L"):
        per_value = 1
    return (value, _norm_unit(unit), per_value, _norm_unit(per_unit))


def parse_form_route(product):
    """
    'TABLET, DELAYED RELEASE;ORAL' -> ('tablet, delayed release', 'oral').
    Falls back to the separate dosage_form/route keys some records carry
    instead of the combined one.
    """
    combined = _clean(product.get("dosage_form_route"))
    form = route = None
    if combined:
        parts = [p.strip() for p in combined.split(";")]
        form = parts[0] or None
        if len(parts) > 1 and parts[1]:
            routes = [_ROUTE_MAP.get(r.strip().upper(), r.strip().lower())
                      for r in parts[1].split(",")]
            routes = [r for r in routes if r]
            route = ", ".join(routes) or None
    form = _clean(product.get("dosage_form")) or form
    if product.get("route"):
        mapped = [_ROUTE_MAP.get(r.strip().upper(), r.strip().lower())
                  for r in _clean(product["route"]).split(",")]
        route = ", ".join([r for r in mapped if r]) or route
    return (form.lower() if form else None, route)


def _split_ingredients(product):
    raw = _clean(product.get("active_ingredients")) or ""
    return [i.strip() for i in raw.split(";") if i.strip()]


def build_presentation(product):
    """One `products[]` row -> one product_data.presentations entry."""
    form, route = parse_form_route(product)
    names = _split_ingredients(product)
    strengths = [s.strip() for s in (_clean(product.get("strength")) or "").split(";")]

    ingredients = []
    per_value = per_unit = None
    for idx, name in enumerate(names or [None]):
        # Strengths align positionally with ingredients. A single strength for a
        # multi-ingredient product means only the total is published — attach it
        # to nothing rather than to an arbitrary ingredient.
        token = strengths[idx] if idx < len(strengths) else (
            strengths[0] if len(strengths) == 1 and len(names) <= 1 else None
        )
        value, unit, pv, pu = parse_strength(token)
        per_value = per_value or pv
        per_unit = per_unit or pu
        base, salt_form = base_substance(name)
        ingredients.append({
            "substance": base,
            "strength_value": value,
            "strength_unit": unit,
            # 'EQ 600MG BASE' states the strength as base, so the salt is the
            # formulated species with an unstated quantity.
            "salt": ({"substance": salt_form, "value": None, "unit": None}
                     if salt_form else None),
        })
    return {
        "per_value": per_value,
        "per_unit": per_unit,
        # FDA's product table has no pack-size column; NDC package data would be
        # needed for it, and this crawler doesn't capture it.
        "pack_size": None,
        "pack_unit": None,
        "form": form,
        "route": route,
        "active_ingredients": ingredients,
    }


def _dedupe_presentations(entries):
    """Identity = strength set + form + route (per the extraction contract)."""
    out, seen = [], set()
    for e in entries:
        key = (
            e["form"], e["route"], e["per_value"], e["per_unit"],
            tuple(sorted((i["substance"], i["strength_value"], i["strength_unit"])
                         for i in e["active_ingredients"])),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


# ----------------------------------------------------------------- substance

def infer_modality(bases, app_type, openfda):
    joined = " ".join(b.lower() for b in bases if b)
    if not joined:
        return None
    if "vaccine" in joined or "toxoid" in joined:
        return "vaccine"
    if any(p in joined for p in _PROTEIN_NAMES):
        return "biologic"
    if any(p in joined for p in _PEPTIDE_NAMES):
        return "peptide"
    if joined.endswith("mab") or " vedotin" in joined or "deruxtecan" in joined:
        return "adc" if ("vedotin" in joined or "deruxtecan" in joined) else "biologic"
    if any(joined.endswith(s) for s in _BIOLOGIC_SUFFIXES):
        return "biologic"
    if app_type == "BLA":
        return "biologic"
    return "small_molecule"


def _first_sentences(text, limit=2, max_chars=400):
    text = _clean(text)
    if not text:
        return None
    # SPL sections open with their own numbered heading, and the heading text
    # is repeated verbatim from a fixed list, so match it by name rather than
    # by shape (a shape-based rule clipped 'Mechanism of Action' to 'Action').
    text = re.sub(
        r"^(?:\d+(?:\.\d+)*\s*)?(?:Mechanism\s+of\s+Action|Indications\s+and\s+Usage"
        r"|Clinical\s+Studies|Clinical\s+Pharmacology|Description|Pharmacodynamics)\s*:?\s*",
        "", text, flags=re.I)
    text = re.sub(r"^\d+(?:\.\d+)*\s+", "", text)
    parts = re.split(r"(?<=[.!])\s+(?=[A-Z(])", text)
    out = " ".join(parts[:limit]).strip()
    return out[:max_chars] or None


def _spl_sections(json_data):
    """Merge every SPL label version into one section -> text mapping,
    newest effective_time first so the current label wins."""
    labels = [l for l in (json_data.get("spl_labels") or []) if isinstance(l, dict)]
    labels.sort(key=lambda l: str(l.get("effective_time") or ""), reverse=True)
    merged = {}
    for label in labels:
        for key, value in label.items():
            if key in merged or not isinstance(value, list) or not value:
                continue
            if all(isinstance(v, str) for v in value):
                merged[key] = "\n".join(value)
    return merged


def substance_from(json_data, presentations, sections):
    bases, salts = [], []
    for p in presentations:
        for ing in p["active_ingredients"]:
            if ing["substance"] and ing["substance"] not in bases:
                bases.append(ing["substance"])
            if ing.get("salt") and ing["salt"]["substance"] not in salts:
                salts.append(ing["salt"]["substance"])
    openfda = json_data.get("openfda") or {}
    if not bases:
        bases = [_title(n) for n in (openfda.get("substance_name") or [])]
        bases = [b for b in bases if b]

    moa = _first_sentences(sections.get("mechanism_of_action"), limit=2)
    # FDA's own established-pharmacologic-class strings are a defensible
    # target/moa when the label section is absent: 'Serotonin 3 Receptor
    # Antagonist [EPC]'.
    epc = [re.sub(r"\s*\[(EPC|MoA|CS|PE)\]\s*$", "", c)
           for c in (openfda.get("pharm_class_moa") or []) + (openfda.get("pharm_class_epc") or [])]
    return {
        "inn": " + ".join(bases) if bases else None,
        "salt_form": salts[0] if len(salts) == 1 and len(bases) == 1 else None,
        "modality": infer_modality(bases, json_data.get("application_type_code"), openfda),
        "target": epc[0] if epc else None,
        "moa": moa or (epc[0] if epc else None),
    }


# ------------------------------------------------------------------ approval

_PATHWAY_BY_APP_TYPE = {"ANDA": "generic", "NDA": None, "BLA": None}
_CLASS_LABEL = {"generic": "Generic", "innovator": "Innovator",
                "biosimilar": "Biosimilar", "hybrid": "Hybrid"}

# The RLD is named in the approval letter in a handful of fixed phrasings.
# Tried in order; the first that matches wins. The '... of <Company>' anchor is
# the most precise, so it comes first, with unanchored variants as fallbacks
# for letters where a line break splits the sentence.
_RLD_RES = (
    re.compile(r"reference\s+listed\s+drug\s*\(RLD\)\s*,?\s*"
               r"(?P<name>[A-Z][^,]{2,60}(?:,\s*[^,]{1,40})?)\s*,?\s+of\s+[A-Z]", re.I),
    # '(RLD) upon which you have based your ANDA, Epzicom Tablets, 600 mg
    # (base)/300 mg of VIIV Healthcare Company, is subject to ...' — the ')'
    # after RLD is optional because some letters spell the phrase out in full.
    re.compile(r"\bRLD\)?\s+upon\s+which\s+you\s+have\s+based\s+your\s+(?:ANDA|NDA)\s*,\s*"
               r"(?:[A-Z][\w&.\- ]{2,40}[’']s\s+)?"
               r"(?P<name>[A-Z][^,]{2,60}(?:,\s*[^,]{1,60})?)\s*,", re.I),
    # Loosest fallback: '(RLD), Zytiga Tablets' with the sentence cut short by a
    # line break. The lookahead rejects the OTHER phrasing this would otherwise
    # match, '(RLD) upon which you have based your ANDA, ...', whose brand name
    # comes later — that one is handled by the pattern above.
    re.compile(r"reference\s+listed\s+drug\s*\(RLD\)\s*,\s*"
               r"(?!(?:upon|which|for|is|was|and|the|to|that)\b)"
               r"(?P<name>[A-Z][A-Za-z\-]{2,30}(?:\s+(?:Tablets?|Capsules?|Injection|"
               r"Solution|Suspension|Cream|Ointment|Gel))?)", re.I),
)
_BIOSIM_RE = re.compile(r"biosimilar(?:ity)?\s+to\s+(?P<name>[A-Z][\w\- ]+)", re.I)
_EXPEDITED_RE = re.compile(
    r"expedited\s+review|President[’']?s?\s+Emergency\s+Plan\s+for\s+AIDS\s+Relief"
    r"|\bPEPFAR\b|priority\s+review\s+(?:was\s+)?(?:granted|designat)"
    r"|fast\s+track\s+designat|breakthrough\s+therapy\s+designat", re.I)


# approval_history rows come in TWO shapes, because the crawler falls back
# between two FDA sources. Both appear in the same corpus, so both must be read
# — handling only the first silently lost the approval date on half the records:
#
#   Drugs@FDA web page:  submission='ORIG-1', action_date='03/28/2017',
#                        action_type='Approval',
#                        review_priority_orphan_status='PRIORITY',
#                        submission_classification='Type 2 - ...'
#   openFDA API:         submission_type='ORIG', submission_number='1',
#                        submission_status='AP', submission_status_date='1956-03-09',
#                        review_priority='PRIORITY',
#                        submission_class_description='Type 2 - ...'
def _normalise_history_row(row):
    """Return (iso_date, submission_label, review_priority, classification)."""
    if not isinstance(row, dict):
        return None
    date = _to_iso(row.get("action_date")) or _to_iso(row.get("submission_status_date"))
    submission = _clean(row.get("submission"))
    if not submission and row.get("submission_type"):
        submission = f"{row['submission_type']}-{row.get('submission_number') or ''}".rstrip("-")
    submission = (submission or "").upper()
    # 'AP' is openFDA's approved status code; the web shape spells it out.
    action = (row.get("action_type") or "").strip().lower()
    status = (row.get("submission_status") or "").strip().upper()
    approved = action == "approval" or status == "AP" or submission.startswith("ORIG")
    priority = str(row.get("review_priority_orphan_status")
                   or row.get("review_priority") or "")
    classification = str(row.get("submission_classification")
                         or row.get("submission_class_description")
                         or row.get("submission_class_code") or "")
    return (date, submission, priority, classification) if approved else None


def _dates(json_data):
    """Every approval action, earliest first, in either history shape."""
    history = json_data.get("approval_history") or {}
    rows = list(history.get("original_approvals") or []) + list(history.get("supplements") or [])
    approvals = []
    for row in rows:
        norm = _normalise_history_row(row)
        if norm and norm[0]:
            approvals.append(norm)
    approvals.sort()
    return approvals


def _to_iso(value):
    value = _clean(value)
    if not value:
        return None
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", value)
    if m:
        month, day, year = m.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value
    if re.match(r"^\d{8}$", value):  # SPL effective_time
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return None


def approval_from(json_data, document_text, sections):
    app_type = json_data.get("application_type_code")
    products = [p for p in (json_data.get("products") or []) if isinstance(p, dict)]
    # 'reference_drug'/'rld' == Yes means THIS application is the reference
    # listed drug, i.e. the innovator.
    is_rld = any(str(p.get("rld") or p.get("reference_drug") or "").strip().lower() == "yes"
                 for p in products)
    pathway = _PATHWAY_BY_APP_TYPE.get(app_type)
    if pathway is None:
        if app_type == "BLA":
            pathway = "biosimilar" if (document_text and _BIOSIM_RE.search(document_text)) else "innovator"
        elif app_type == "NDA":
            # A 505(b)(2) NDA relies on another product's data — the US
            # equivalent of a hybrid. Everything else is an innovator NDA.
            if document_text and re.search(r"505\(b\)\(2\)", document_text):
                pathway = "hybrid"
            else:
                pathway = "innovator" if is_rld or not products else "innovator"

    approvals = _dates(json_data)
    statuses = " ".join(a[2] for a in approvals).upper()
    classifications = " ".join(a[3] for a in approvals).upper()
    # openFDA writes 'UNKNOWN' where the web shape writes an empty string;
    # neither is a statement that review was standard.
    if statuses.replace("UNKNOWN", "").strip() == "":
        statuses = ""

    reference = None
    if document_text:
        for pattern in _RLD_RES:
            m = pattern.search(document_text)
            if m:
                # The sentence continues '... of <Marketing Company>, is subject
                # to ...'. The company is not part of the product name, and the
                # regex has to run past it to find the sentence boundary, so
                # trim it back off here.
                reference = _clean(re.sub(r"\s+of\s+[A-Z].*$", "", m.group("name")))
                break
        if reference is None and pathway == "biosimilar":
            b = _BIOSIM_RE.search(document_text)
            reference = _clean(b.group("name")) if b else None

    return {
        "pathway": pathway,
        "registration_class": _CLASS_LABEL.get(pathway),
        # Accelerated approval is the closest US analogue of a conditional
        # marketing authorisation; nothing else in Drugs@FDA implies one.
        "conditional": (
            True if "ACCELERATED" in statuses + classifications
            else True if (document_text and re.search(
                r"accelerated\s+approval|subpart\s+H\b", document_text, re.I))
            # No accelerated-approval signal anywhere in a record that DOES
            # carry review metadata means an ordinary, unconditional approval.
            else False if statuses or classifications else None),
        # Drugs@FDA only carries a review-priority field for newer submissions.
        # When it is absent, the letter itself often names an expedited review
        # programme (PEPFAR, fast track, breakthrough, accelerated), which is
        # the same signal.
        "priority_review": (
            True if "PRIORITY" in statuses
            else True if (document_text and _EXPEDITED_RE.search(document_text))
            else False if statuses else None),
        "reference_product": reference,
    }


# --------------------------------------------------------------- label prose

# Indications live in prose. SPL 'indications_and_usage' is formulaic enough
# ('X is indicated for the treatment of Z') that a regex CAN carve out just
# the condition phrase — an earlier version of this function did exactly
# that. But the carve-out is itself a source of error on the one field a
# reviewer leans on most: the wrong split point, a swallowed or truncated
# clause, a phrase that reads wrong out of context. Since this module's whole
# premise is "map deterministically, don't introduce a claim the source
# doesn't literally support," indications_from now keeps the WHOLE sentence
# verbatim instead of trying to extract just the disease name from it. This
# still needs a trigger to decide which sentences ARE indication statements —
# "indicated" is what the label itself uses for this — but past that trigger,
# nothing is reworded, trimmed, or split.
_POPULATION_RE = re.compile(
    r"\b(adults?(?:\s+and\s+(?:paediatric|pediatric|children)[^,.;]{0,60})?"
    r"|p(?:a)?ediatric\s+patients?[^,.;]{0,60}|children[^,.;]{0,40}|neonates?|adolescents?)\b",
    re.I,
)


def _name_anchors(json_data):
    """
    Known brand/ingredient names for THIS product, used only to find where a
    real sentence starts inside indications_from's output — never to alter or
    supply the indication content itself. SPL section/subsection headings
    ('1 INDICATIONS AND USAGE', '1.1 Mantle Cell Lymphoma') have no
    sentence-ending punctuation, so splitting text on '. ' glues the heading
    onto the front of the next real sentence. A generic shape-based rule
    can't reliably tell a heading from a genuine sentence opener, but this
    record's OWN stated names can: real indication sentences start with the
    product's brand or ingredient name ('JAYPIRCA is indicated for...'), so
    trimming to the earliest of those known strings removes the glued-on
    heading without touching anything the label actually says.
    """
    anchors = set()
    openfda = json_data.get("openfda") or {}
    for name in (openfda.get("brand_name") or []):
        if name and len(str(name).strip()) >= 3:
            anchors.add(str(name).strip())
    for p in (json_data.get("products") or []):
        if not isinstance(p, dict):
            continue
        if p.get("drug_name") and len(str(p["drug_name"]).strip()) >= 3:
            anchors.add(str(p["drug_name"]).strip())
        for ing in _split_ingredients(p):
            if len(ing) >= 3:
                anchors.add(ing)
    return anchors


def _trim_heading_prefix(sentence, name_anchors):
    """See _name_anchors. Leaves the sentence untouched if no known name is
    found in it — no known anchor means no defensible trim point, not a
    license to guess one."""
    lower = sentence.lower()
    positions = [p for p in (lower.find(a.lower()) for a in name_anchors) if p >= 0]
    return sentence[min(positions):] if positions else sentence


def indications_from(json_data, sections):
    """
    One entry per SENTENCE containing 'indicated', stored as the full
    sentence — not a regex-extracted condition phrase — after trimming any
    glued-on section-heading prefix (see _trim_heading_prefix). `population`
    is still pulled separately (a same-window regex match, not a rewording of
    the sentence itself), since it's an addition alongside the verbatim text
    rather than a substitute for any part of it.
    """
    text = sections.get("indications_and_usage")
    if not text:
        return []
    anchors = _name_anchors(json_data)
    out, seen = [], set()
    for sentence in re.split(r"(?<=[.!])\s+(?=[A-Z(])", text):
        sentence = _clean(sentence)
        if not sentence or "indicated" not in sentence.lower():
            continue
        sentence = _trim_heading_prefix(sentence, anchors)
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        start = text.find(sentence)
        window = (text[max(0, start - 120): start + len(sentence) + 120]
                  if start >= 0 else sentence)
        pop = _POPULATION_RE.search(window)
        out.append({
            "condition": sentence,
            "population": _clean(pop.group(0)).lower() if pop else None,
            "line_of_therapy": None,
            "biomarker": None,
            "approval_date": None,
        })
    return out[:12]


# Boxed-warning risk headings: SPL renders them as short capitalised phrases.
_RISK_HEAD_RE = re.compile(r"\b([A-Z][A-Z /,\-]{6,60}[A-Z])\b")
_TRAILING_CONNECTOR_RE = re.compile(r"(?:\s+(?:and|or|with|of|in|to|the))+$", re.I)
_RISK_STOP = {
    "WARNING", "WARNINGS", "WARNINGS AND PRECAUTIONS", "SEE FULL PRESCRIBING INFORMATION",
    "FULL PRESCRIBING INFORMATION", "BOXED WARNING", "IMPORTANT", "AND", "OR",
}


# Words that plausibly START the body sentence after an SPL subsection heading,
# and never end a Title Case heading. Used to peel over-captured headings.
_SENTENCE_OPENERS = frozenset((
    "as", "in", "the", "use", "this", "if", "when", "a", "an", "for", "to",
    "of", "and", "or", "with", "patients", "avoid", "do", "monitor", "dual",
    "consider", "either", "correct", "there", "it", "these", "because",
))


def key_risks_from(sections):
    boxed = sections.get("boxed_warning")
    risks = []
    if boxed:
        for m in _RISK_HEAD_RE.finditer(boxed):
            phrase = _clean(m.group(1))
            if not phrase:
                continue
            pretty = phrase.title() if phrase.isupper() else phrase
            if phrase.upper() in _RISK_STOP or pretty in risks:
                continue
            risks.append(pretty)
    if len(risks) < 8:
        # Fall back to the label's own Warnings & Precautions subsection
        # headings, which SPL numbers as '5.1 Lactic Acidosis and Severe
        # Hepatomegaly'. The heading is a run of Capitalised words plus lowercase
        # connectors; the body sentence that follows it is also capitalised, so
        # take the whole run and then trim the trailing connector — this
        # over-captures by a word or two on some labels, which is the main
        # reason key_risks is the weakest field this module produces.
        for section in ("warnings_and_cautions", "warnings"):
            for m in re.finditer(
                    r"\b\d+\.\d+\s+((?:[A-Z][A-Za-z\-]+|and|or|with|of|in|to|the)"
                    r"(?:\s+(?:[A-Z][A-Za-z\-]+|and|or|with|of|in|to|the)){0,7})",
                    sections.get(section) or ""):
                phrase, text = m.group(1), sections.get(section) or ""
                # The body sentence that follows the heading also starts with a
                # capital, so the run above swallows its first word
                # ('... Steatosis Lactic acidosis has been reported'). If what
                # comes next is lowercase, that last word opened the sentence
                # rather than closing the heading — drop it.
                if re.match(r"\s+[a-z]", text[m.end():m.end() + 3] or ""):
                    phrase = phrase.rsplit(" ", 1)[0]
                # ONE strip is not always enough: the run's lowercase-connector
                # alternation lets it swallow a sentence opener AND its article,
                # e.g. '5.4 Impaired Hepatic Function As the majority of ...'
                # captures '... Function As the', which the single strip above
                # leaves as '... Function As'. Keep peeling while the tail is a
                # word that opens a sentence rather than ending a Title Case
                # heading.
                words = phrase.split()
                while len(words) > 1 and words[-1].lower() in _SENTENCE_OPENERS:
                    words.pop()
                phrase = _clean(_TRAILING_CONNECTOR_RE.sub("", " ".join(words)))
                if not phrase or len(phrase) <= 4:
                    continue
                # A boxed-warning heading and its W&P subsection heading are
                # often the same risk stated twice.
                if any(phrase.lower() == r.lower() or phrase.lower() in r.lower()
                       or r.lower() in phrase.lower() for r in risks):
                    continue
                risks.append(phrase)
    return risks[:8]


# Pivotal-trial facts in prose. Only the unambiguous bits are taken: an
# explicit enrolment count and an explicit design phrase.
_N_RE = re.compile(r"\b(?:N\s*=\s*|enrolled\s+|randomized\s+|a\s+total\s+of\s+)(\d{2,5})\b", re.I)
_DESIGN_RE = re.compile(
    r"\b((?:double-blind|open-label|randomized|randomised|placebo-controlled|"
    r"single-arm|multicenter|multicentre|active-controlled|crossover)"
    r"(?:[,\s]+(?:double-blind|open-label|randomized|randomised|placebo-controlled|"
    r"single-arm|multicenter|multicentre|active-controlled|crossover))*"
    r"(?:\s+(?:phase\s+[1-4I]{1,3}))?\s+(?:study|trial|studies|trials))", re.I)
_STUDY_ID_RE = re.compile(r"\b(?:Study|Trial|Protocol)\s+([A-Z0-9][A-Z0-9\-]{2,15})\b")


# A US label's reliable study_id is a named trial acronym (ONTARGET, TRANSCEND).
# The stop list keeps section boilerplate and units out.
_TRIAL_NAME_RE = re.compile(r"\b([A-Z][A-Z0-9\-]{4,20})\b")
_TRIAL_NAME_STOP = frozenset((
    "CLINICAL", "STUDIES", "ADVERSE", "REACTIONS", "WARNINGS", "TABLE", "NOTE",
    "DOSAGE", "USAGE", "MMHG", "USP", "ACE", "ARBS", "INDICATIONS", "PRECAUTIONS",
    "OVERDOSAGE", "DESCRIPTION", "CONTRAINDICATIONS", "PATIENTS", "PLACEBO",
))


def pivotal_evidence_from(sections):
    """
    One entry per SENTENCE describing a trial, rather than per design phrase
    found anywhere in the section.

    Sentence scoping is what prevents cross-contamination: a section describing
    ONTARGET and then TRANSCEND previously paired the first design phrase with
    whichever enrolment count appeared next in a 500-character window, yielding
    an entry ('double-blind study', n=2954) whose design came from one trial and
    whose N came from the other — a fact that is in neither.
    """
    text = sections.get("clinical_studies")
    if not text:
        return []
    out, seen = [], set()
    for sentence in re.split(r"(?<=[.])\s+(?=[A-Z(])", text):
        m = _DESIGN_RE.search(sentence)
        if not m:
            continue
        n = _N_RE.search(sentence)
        sid = _STUDY_ID_RE.search(sentence)
        study_id = sid.group(1) if sid else None
        if not study_id:
            names = [t for t in _TRIAL_NAME_RE.findall(sentence)
                     if t not in _TRIAL_NAME_STOP]
            study_id = names[0] if names else None
        entry = {
            "study_id": study_id,
            "design": _clean(m.group(1)).lower(),
            "n": int(n.group(1)) if n else None,
            # Endpoint/value/outcome are stated too variably across labels to
            # pull by rule without fabricating; left null deliberately.
            "endpoint": None, "value": None, "unit": None,
            "comparator": None, "outcome": None,
        }
        key = (entry["study_id"], entry["design"], entry["n"])
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out[:6]


# ------------------------------------------------------------------- columns

# 'The tablets contain the following inactive ingredients: crospovidone, lactose
# monohydrate, magnesium stearate, meglumine, povidone and sodium hydroxide
# pellets.' — near-fixed phrasing across US labels, so the list IS
# rule-extractable from SPL prose even though Drugs@FDA publishes no structured
# excipient field.
_EXCIPIENT_RE = re.compile(
    r"inactive\s+ingredient(?:s)?\s*(?:are|include|:)?\s*:?\s*(?P<list>[^.]{5,400})",
    re.I)
# Trailing dosage-form nouns the sentence ends on, not part of the excipient name.
_EXCIPIENT_TAIL_RE = re.compile(r"\s+(pellets|powder|granules|q\.s\.)$", re.I)


def excipients_from(sections):
    """
    Excipient NAMES only. US labels state the formulation qualitatively and
    essentially never give a per-dosage-unit quantity, so value/unit stay null
    rather than being invented — product_data_spec.md's excipient note says a
    missing quantity is normal and must not be guessed.
    """
    for key in ("description", "inactive_ingredient", "spl_patient_package_insert"):
        m = _EXCIPIENT_RE.search(sections.get(key) or "")
        if not m:
            continue
        out, seen = [], set()
        for part in re.split(r",|\band\b", m.group("list")):
            name = _clean(_EXCIPIENT_TAIL_RE.sub("", _clean(part) or ""))
            if not name or len(name) < 3 or name.lower() in seen:
                continue
            seen.add(name.lower())
            out.append({"name": _title(name), "name_local": None,
                        "value": None, "unit": None})
        if out:
            return out[:30]
    return []


def columns_from(json_data, document_text, sections, presentations, substance, approval):
    openfda = json_data.get("openfda") or {}
    products = [p for p in (json_data.get("products") or []) if isinstance(p, dict)]
    approvals = _dates(json_data)
    orig = [a for a in approvals if a[1].startswith("ORIG")]
    approval_date = (orig or approvals or [(None,)])[0][0]

    def first(key):
        vals = openfda.get(key) or []
        return _clean(vals[0]) if vals else None

    brand = first("brand_name")
    drug_names = [_clean(p.get("drug_name")) for p in products if p.get("drug_name")]
    marketing = {(_clean(p.get("marketing_status")) or "").lower() for p in products}
    label_dates = [_to_iso(l.get("effective_time"))
                   for l in (json_data.get("spl_labels") or []) if isinstance(l, dict)]
    label_dates = sorted(d for d in label_dates if d)

    return {
        "product_name_en": None,  # US records are already English
        "brand_name": brand or (drug_names[0] if drug_names else None),
        "brand_name_local": None,
        "generic_name": substance["inn"],
        "mah_name": _clean(json_data.get("company")),
        "mah_address": None,
        "manufacturer": first("manufacturer_name") or _clean(json_data.get("company")),
        "manufacturer_address": None,
        "registration_number": _clean(json_data.get("application_number")),
        "product_type": approval["registration_class"],
        "status": ("Withdrawn" if marketing and marketing <= {"discontinued", "none (tentative approval)"}
                   else ("Active" if any("prescription" in s or "over-the-counter" in s
                                          for s in marketing) else None)),
        "registration_date": approval_date,
        "approval_date": approval_date,
        "market_authorization_date": approval_date,
        "expiry_date": None,
        "withdrawal_date": None,
        "label_revision_date": label_dates[-1] if label_dates else None,
        "atc_code": None,  # FDA does not publish ATC
        "is_generic": (None if approval["pathway"] is None
                       else approval["pathway"] != "innovator"),
        "reference_product": approval["reference_product"],
        "source_language": "en",
        "therapeutic_areas": [],  # requires clinical judgement — left to review
        "indications": [i["condition"] for i in indications_from(json_data, sections)],
        "symptoms": [],
        "adverse_reactions": [],
        "contraindications": [],
        "active_ingredients": sorted({i["substance"] for p in presentations
                                      for i in p["active_ingredients"] if i["substance"]}),
        "strengths": sorted({s for s in (_clean(p.get("strength")) for p in products) if s}),
        "dosage_forms": sorted({p["form"] for p in presentations if p["form"]}),
        "routes": sorted({p["route"] for p in presentations if p["route"]}),
    }


# ---------------------------------------------------------------------- main

def extract(json_data, document_text=None):
    """
    Map one US raw record to the same {"columns", "product_data"} shape
    processing/ai_extraction.py produces. No network calls, no model.

    Also returns "provenance": key -> the mechanism that filled it, so a
    reviewer can tell a mapped field from a regexed one.
    """
    json_data = json_data or {}
    sections = _spl_sections(json_data)
    products = [p for p in (json_data.get("products") or []) if isinstance(p, dict)]

    presentations = _dedupe_presentations([build_presentation(p) for p in products])
    substance = substance_from(json_data, presentations, sections)
    approval = approval_from(json_data, document_text, sections)
    indications = indications_from(json_data, sections)
    columns = columns_from(json_data, document_text, sections,
                           presentations, substance, approval)

    product_data = {
        "_schema_version": "1",
        "substance": substance,
        "presentations": presentations,
        "indications": indications,
        "approval": approval,
        "pivotal_evidence": pivotal_evidence_from(sections),
        "key_risks": key_risks_from(sections),
        "excipients": excipients_from(sections),
    }
    provenance = {
        "substance.inn": "field map (products[].active_ingredients + salt strip)",
        "substance.modality": "rule (application_type_code + name suffix)",
        "substance.target": "field map (openfda.pharm_class_*)" if substance["target"] else "unavailable",
        "substance.moa": ("regex (SPL mechanism_of_action)" if sections.get("mechanism_of_action")
                          else ("field map (openfda.pharm_class_*)" if substance["moa"] else "unavailable")),
        "presentations": "field map (products[] strength/dosage_form_route)",
        "approval.pathway": "rule (application_type_code + rld flag + letter text)",
        "approval.reference_product": ("regex (approval letter RLD sentence)"
                                       if approval["reference_product"] else "unavailable"),
        "approval.priority_review": ("field map (approval_history review_priority)"
                                      if approval["priority_review"] is not None else "unavailable"),
        "indications": ("verbatim sentence (SPL indications_and_usage, 'indicated' trigger — "
                        "not a regex-carved condition phrase)" if indications else "unavailable"),
        "key_risks": ("regex (SPL boxed_warning / warnings)"
                      if product_data["key_risks"] else "unavailable"),
        "pivotal_evidence": ("regex (SPL clinical_studies)"
                             if product_data["pivotal_evidence"] else "unavailable"),
        "excipients": ("regex (SPL inactive-ingredient list)"
                       if product_data["excipients"] else "unavailable"),
    }
    return {"columns": columns, "product_data": product_data, "provenance": provenance}

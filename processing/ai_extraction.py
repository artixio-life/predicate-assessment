"""
Stage C: the agentic chunk-fold AI extraction.

Feeds a product's chunks (source.product_chunks, in order) to the LLM one
at a time. Each call is shown the JSON accumulated from every prior chunk
and asked to return the complete, updated object — not a diff — so the
result after the last chunk is the merged extraction across the whole
document. The crawler's raw json_data (application details, product
tables, approval history, therapeutic equivalents, ...) is always folded
in too, chunked the same way as document text — not only as a
registry-only fallback when there's no document at all, but as extra
context alongside document_text whenever both exist (see
schema/schema.sql's json_data comment). Document prose often doesn't
restate structured facts like application numbers or approval dates, so
skipping json_data whenever a document was present used to silently drop
real signal.

Two things get written per product:
- the flat `columns` on drug.products itself (brand_name, mah_name,
  manufacturer, atc_code, therapeutic_areas, indications, adverse_reactions,
  etc.) — these already exist on the table (schema/schema.sql /
  products.sql) and are what issue #1's worked examples show as the actual
  row shape.
- `product_data`, the predicate-assessment contract based on
  schema/product_data_spec.md's 6 keys (substance, presentations,
  indications, approval, pivotal_evidence, key_risks) plus `excipients` —
  the one field that spec explicitly flags as "the one to think about" and
  a cheap, non-breaking addition once pharmaceutical-equivalence work needs
  it, which issue #1's worked examples already show populated.
These are deliberately separate JSON sections in the LLM's response (see
SYSTEM_PROMPT) so persistence can route each to the right place without
re-parsing.

Fields Stage A (promote.py) already sets — product_name, country_id,
regulator, source_url, json_data, raw_record_id — are never touched here.
registration_number is Stage A's best-effort guess from crawler json_data;
Stage C is only allowed to fill it in if Stage A left it null (see the
COALESCE in the UPDATE below), never override a value Stage A already found.
"""
import json
import logging
import os
import threading

import psycopg2.errors
from openai import OpenAI
from psycopg2.extras import Json, RealDictCursor

from db import get_db_connection
from processing.chunking import DEFAULT_TARGET_TOKENS, count_tokens
from processing.claim import claim_ai_extraction, count_ai_extraction_pending
from processing import llm_service_client
from processing.progress import Progress
from processing.retry import run_with_retries, RetriesExhausted
from processing.workers import run_worker_pool

logger = logging.getLogger(__name__)

# The OpenRouter model. Since this stage now runs on the self-hosted LLM first
# (processing/llm_service_client.py), this is the FALLBACK model — it is what
# llm_service_client's LLM_FALLBACK_MODEL defaults to, and is only reached when
# the self-hosted endpoint is unreachable or keeps erroring.
EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "google/gemini-2.5-flash")

# Cap on the reply. The fold re-emits the WHOLE accumulated object every call, so
# this must cover the largest accumulated object, not the largest excerpt.
#
# Sized from measurement, not headroom: observed completions run ~1,000-1,600
# tokens and the largest stored accumulator across every DONE row is ~3,560.
# Over-declaring is NOT free on vLLM — the scheduler reserves KV cache for the
# declared max_tokens, and defers requests it cannot guarantee that space for
# (visible as `Deferred: N reqs` alongside `Running: 1`), so an 8192 default
# throttled concurrency to a fraction of what the GPU could actually run.
# Truncation is recoverable anyway: json_utils closes a cut-off object and
# _call_llm_merge logs a warning naming this variable.
MAX_COMPLETION_TOKENS = int(os.getenv("AI_EXTRACTION_MAX_TOKENS", "3072"))

# Headroom between our token estimate and the server's real count.
#
# Two sources of error, and only one of them is a constant:
#   - chat-template overhead (role wrappers, BOS/EOS): a fixed handful of tokens
#   - TOKENISER MISMATCH: count_tokens uses tiktoken cl100k_base, but the served
#     model has its own tokeniser. Gemma's SentencePiece vocab measured 1.68%
#     HIGHER than cl100k on real fold payloads (17,872 actual vs 17,571
#     estimated). That error is PROPORTIONAL to prompt size, so a flat margin
#     silently stops covering it as prompts grow — at a 17.5k prompt a 300-token
#     margin is exactly consumed, and the request lands 1 token over the limit.
# Hence a ratio with a flat floor, rather than a constant.
CONTEXT_SAFETY_MARGIN = int(os.getenv("AI_EXTRACTION_SAFETY_MARGIN", "300"))
CONTEXT_SAFETY_RATIO = float(os.getenv("AI_EXTRACTION_SAFETY_RATIO", "0.06"))


def _safety_margin(prompt_estimate):
    """Headroom to reserve for a prompt we estimated at `prompt_estimate`."""
    return max(CONTEXT_SAFETY_MARGIN, int(prompt_estimate * CONTEXT_SAFETY_RATIO))
# Below this much headroom a call is pointless — the object the fold must
# re-emit will not fit in what is left, so the batch is split instead.
MIN_COMPLETION_TOKENS = int(os.getenv("AI_EXTRACTION_MIN_COMPLETION", "512"))

# Chunks are ~1800 tokens each (processing/chunking.py). 7 was right while this
# stage ran on OpenRouter's 1M-token context: batching cut a 14-chunk document
# from 14 calls to 2 at no accuracy cost, since the fold still carries the
# accumulated JSON forward between batches.
#
# Now that the self-hosted LLM is the primary provider, the binding constraint
# is its context window (~22k), which must also hold a ~4.9k system prompt plus
# an accumulator that grows for the whole fold. At 7 excerpts that overflows on
# real products and triggers the batch splitting in _fold_slice, and a split
# re-sends the system prompt and accumulator from scratch — so it costs MORE
# than the extra round-trip it was meant to save. 5 is the setting that avoids
# most splits; raise it when pointing this stage back at a large-context model.
CHUNKS_PER_CALL = int(os.getenv("AI_EXTRACTION_CHUNKS_PER_CALL", "5"))

# Hard cap on how many excerpts a single product may send, across all its calls.
# 0 (the default) means no cap.
#
# This is a COST/TIME control, not a quality one: the tail of a long product is
# genuinely dropped, so anything only stated in excerpt 21 of 51 is lost. A
# capped row therefore lands NEEDS_REVIEW rather than DONE even when every call
# succeeded — it has not actually been fully read, and marking it ENRICHED would
# assert something untrue about the data.
MAX_CHUNKS_PER_PRODUCT = int(os.getenv("AI_EXTRACTION_MAX_CHUNKS", "0"))


def _batch(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]

PRODUCT_DATA_KEYS = (
    "substance", "presentations", "indications",
    "approval", "pivotal_evidence", "key_risks", "excipients",
)

# Flat drug.products columns this stage is responsible for. Scalars fill
# once and only get corrected on a direct contradiction; arrays accumulate
# distinct values across chunks. registration_number is handled specially
# (COALESCE, see module docstring) — it's here so the LLM still reports it
# when found, but the UPDATE below decides whether it's actually applied.
COLUMN_SCALAR_FIELDS = (
    "product_name_en", "brand_name", "brand_name_local", "generic_name",
    "mah_name", "mah_address", "manufacturer", "manufacturer_address",
    "registration_number", "product_type", "status",
    "registration_date", "approval_date", "market_authorization_date",
    "expiry_date", "withdrawal_date", "label_revision_date",
    "atc_code", "is_generic", "reference_product", "source_language",
)
COLUMN_ARRAY_FIELDS = (
    "therapeutic_areas", "indications", "symptoms", "adverse_reactions",
    "contraindications", "active_ingredients", "strengths",
    "dosage_forms", "routes",
)

SYSTEM_PROMPT = """You are extracting structured predicate-assessment data from a drug \
regulatory document (SmPC, bula, product information leaflet, or similar), for a \
database that matches products across countries and evaluates regulatory precedent.

You will be shown ONE excerpt at a time from a longer document, plus the JSON object \
built so far from earlier excerpts. Your job each time is to return the COMPLETE, \
UPDATED JSON object — not a diff, not just the new fields.

Output JSON must have exactly two top-level keys, "columns" and "product_data":

{
  "columns": {
    "product_name_en": "the product's FULL name (including strength and dosage form, as \
printed) rendered in English — translate it, or transliterate where no translation \
exists: '桃核承气汤颗粒' -> 'Taohe Chengqi Decoction Granules', 'ANSENTRON Solução \
Injetável 4 mg/2 mL' -> 'ANSENTRON Solution for Injection 4 mg/2 mL'. Set to null ONLY \
if the source name is already entirely English — never echo an unchanged non-English \
name here"|null,
    "brand_name": str|null,
    "brand_name_local": "the trade name in its ORIGINAL script — null when the source is \
already Latin script and would just duplicate brand_name"|null,
    "generic_name": "the INN/base substance name — same base form as active_ingredients \
below, e.g. 'Ondansetron', NOT the salt form 'Ondansetron hydrochloride dihydrate' (the \
salt belongs in product_data.substance.salt_form instead)"|null,
    "mah_name": str|null,
    "mah_address": "the address rendered in ENGLISH: translate the generic/structural \
words (Av./Avenida -> Avenue, Rua -> Street, andar -> Floor, Bairro -> District, 区 -> \
District, 市 -> City) and transliterate any non-Latin script, but keep PROPER NOUNS as \
they are — street, district, city and state names are not translated ('Av. Brigadeiro \
Faria Lima, 201 - 1º ao 4º andar, São Paulo - SP' -> 'Brigadeiro Faria Lima Avenue 201, \
1st to 4th Floor, São Paulo - SP, Brazil'). Append the country in English if the \
document makes it clear. Keep numbers, unit/floor designators and postal codes exactly \
as given"|null,
    "manufacturer": str|null,
    "manufacturer_address": "same English rendering rules as mah_address"|null,
    "registration_number": str|null,
    "product_type": "the regulator's OWN registration/product classification, TRANSLATED \
to English (e.g. Brazil's 'Medicamento Similar' -> 'Similar Medicine', not left in \
Portuguese), or 'Generic'/'Innovator'/'Biosimilar' — NOT the dosage form (that is \
dosage_forms below)"|null,
    "status": "REGISTRATION status only — Active, Withdrawn, or Suspended — and ONLY if \
the document explicitly states the registration itself is in that state. Do NOT put \
legal/dispensing status here (prescription-only, restricted-to-hospitals, OTC, etc. — \
there is no field for that in this schema, so that information is simply not captured). \
The mere existence of a currently-published label does NOT prove Active — leave null \
unless the document says so directly."|null,
    "registration_date": "YYYY-MM-DD — when the product/entry was first registered or \
listed with the regulator. Regulators label this differently; map ANY of these to this \
field: 'ARTG Start Date' or 'Start Date' (Australia TGA), 'Date Registered' (South \
Africa SAHPRA), 'DATE OF FIRST AUTHORISATION/RENEWAL OF THE AUTHORISATION' (UK MHRA/EU \
SmPC section 9 — use the FIRST authorisation date, not a later renewal date, when both \
appear on that line), 'Original Approval Date' or 'Effective Date of Issue' (used by \
some regulators as the product's founding/registration event rather than a distinct \
approval step), or an equivalent 'registered on'/'listed since' date under any other \
label. Fill it whenever ONE of these is explicitly stated — do not infer from a label \
revision, publication, or document-generation date/timestamp (e.g. a PDF footer's \
'Produced at' date is NEVER this field, and is not the same event as registration)"|null,
    "approval_date": "YYYY-MM-DD — when the regulator approved the product, e.g. an \
'Action Date' whose 'Action Type' is Approval (US FDA's ORIG-1/original approval row), \
or any other explicit approval-action date. Same caution as registration_date: never \
infer from a label revision, publication, or document-generation date"|null,
    "market_authorization_date": "YYYY-MM-DD"|null, "expiry_date": "YYYY-MM-DD"|null,
    "withdrawal_date": "YYYY-MM-DD"|null, "label_revision_date": "YYYY-MM-DD"|null,
    "atc_code": str|null,
    "is_generic": "true if this product is anything OTHER than the original/innovator/\
reference product itself (a generic, a country-specific 'similar'/branded-generic \
category, a biosimilar, etc.) — false only if THIS document IS the innovator/reference \
product. A country-specific sub-classification (e.g. Brazil's genérico vs similar vs \
referência) still means true here; that nuance lives in product_type, not this flag. \
null only if the document gives no signal either way."|null,
    "reference_product": str|null,
    "source_language": "ISO 639-1 code, e.g. en/pt/zh"|null,
    "therapeutic_areas": "[str] — the broad MEDICAL SPECIALTY / DISEASE-AREA classification(s) \
this product is used within (e.g. Oncology, Anaesthesiology, Cardiology, Neurology, \
Infectious Diseases, Endocrinology, Psychiatry, Gastroenterology, Dermatology, \
Rheumatology, Nephrology, Pain Management, Supportive Care). Two sources, in order: \
(1) an explicitly stated therapeutic class/classification field if the document or \
registry record has one — e.g. a registry field giving 'ANALGESICOS NAO NARCOTICOS' \
(non-narcotic analgesics) maps to Pain Management, 'ANTIBIOTICOS' to Infectious \
Diseases; translate the class to its standard English specialty name rather than \
copying the class label verbatim. (2) failing that, derive from what the indications \
are FOR — chemotherapy/radiotherapy-induced nausea implies Oncology + Supportive Care; \
postoperative nausea implies Anaesthesiology + Supportive Care. Do NOT leave this empty \
when the record states a therapeutic class, even if it states no indications at all. \
This is NEVER a restatement of the indication or symptom text itself — 'Nausea and \
vomiting' is a symptom, not a therapeutic area, and must never appear here. Always use \
the standard, recognized name for the specialty, the same way every time — do not \
paraphrase or invent alternate wording for it.",
    "indications": [str — condition names only, English],
    "symptoms": [str — DISEASE symptoms the indication treats, e.g. Nausea, Vomiting \
— NOT what the drug causes, that is adverse_reactions],
    "adverse_reactions": [str — every ADR term found, normalised English, this IS meant \
to be a comprehensive list unlike columns.product_data.key_risks below],
    "contraindications": [str], "active_ingredients": [str — base substance, not salt],
    "strengths": [str — when a presentation is defined per volume (e.g. an injectable \
solution), ALWAYS include the volume: "4 mg/2 mL", not just "4 mg". Otherwise just the \
strength: "225 mg"],
    "dosage_forms": [str], "routes": [str]
  },
  "product_data": {
    "_schema_version": "1",
    "substance": {"inn": "for a COMBINATION product with multiple active ingredients, join them with ' + ', e.g. 'Amoxicillin + Clavulanic acid' — the full per-ingredient strength/salt breakdown goes in presentations[].active_ingredients below, not here"|null, "salt_form": "null for combination products (salt is per-ingredient, see presentations[].active_ingredients[].salt)"|null, "modality": "small_molecule|biologic|adc|vaccine|peptide"|null, "target": str|null, "moa": str|null},
    "presentations": [{"per_value": "CONCENTRATION DENOMINATOR ONLY, for liquids/injectables where strength is stated per volume — '4 mg/2 mL' -> per_value=2, per_unit='mL'. NEVER a pack size or tablet count: a box of 20 tablets is NOT per_value=20, it is per_value=null (pack sizes are not stored in this schema at all)"|number|null, "per_unit": "e.g. 'mL' — null for solid oral forms"|null, "pack_size": "how many units the pack contains, e.g. 10 for a box of 10 tablets, 1 for a single ampoule. One presentation entry per distinct pack size"|number|null, "pack_unit": "what pack_size counts, e.g. 'tablets', 'capsules', 'ampoules', 'vials'"|str|null, "form": str|null, "route": str|null, "active_ingredients": [{"substance": "English, base form not salt", "strength_value": number|null, "strength_unit": str|null, "salt": {"substance": str, "value": number, "unit": str}|null}]}],
    "indications": [{"condition": str, "population": str|null, "line_of_therapy": str|null, "biomarker": str|null, "approval_date": "YYYY-MM-DD"|null}],
    "approval": {"pathway": "innovator|generic|biosimilar|hybrid|similar"|null, "registration_class": "same value as columns.product_type, in English"|null, "conditional": bool|null, "priority_review": bool|null, "reference_product": str|null},
    "pivotal_evidence": [{"study_id": str|null, "design": str, "n": number|null, "endpoint": str, "value": number|null, "unit": str|null, "comparator": str|null, "outcome": "met|not_met"|null}],
    "key_risks": ["string — AT MOST 8 of the most severe / most label-driving risks, not a full ADR list"],
    "excipients": [{"name": "English", "name_local": str|null, "value": "quantity per dosage unit if stated, e.g. 'Lactose monohydrate 24.75 mg per capsule' -> 24.75 — most documents give NO quantity at all, which is normal; leave null rather than guessing"|number|null, "unit": "JUST the measurement unit, e.g. 'mg' — never 'mg per capsule' or similar, the per-dosage-unit basis is implied the same way it is for presentations.strength_unit"|null}]
  }
}

Note columns.indications (flat condition-name strings, for search) and \
product_data.indications (structured objects with population/line_of_therapy/biomarker) \
are DIFFERENT fields serving different purposes — fill both.

LANGUAGE: this database is cross-country, so every value must be in English for \
uniformity — indications, adverse_reactions, key_risks, contraindications, warnings, \
product_type/registration_class, symptoms, therapeutic_areas, dosage_forms, routes, \
substance/moa/target, mah_address, manufacturer_address, everything. Translate rather \
than copy the source language verbatim (e.g. Portuguese "Medicamento Similar" -> \
"Similar Medicine", not left as-is). Default to English whenever there is any doubt; \
only the short list below is exempt.
Exceptions — these stay exactly as printed in the source document, NEVER translated:
  - brand_name_local (the whole point of this field is the original script/spelling)
  - mah_name, manufacturer (registered legal entity names — 'Aché Laboratórios \
Farmacêuticos S.A.' stays as-is, it is the company's actual legal name, not prose. For \
a non-Latin-script company give the officially-used English/romanised name when the \
document itself provides one, otherwise leave the original)
  - registration_number, atc_code (identifiers/codes, not language content)
  - source_language (this field's VALUE is an ISO code like "pt", describing the \
document's original language — it is metadata, not something to translate)

TERMINOLOGY: use standard, established clinical/medical terminology throughout — \
therapeutic areas, adverse reaction terms, indication names, mechanism-of-action \
language — and use the SAME canonical term every time you mean the same thing. Do not \
casually paraphrase, abbreviate loosely, or invent alternate wording for a well-defined \
clinical concept from one excerpt to the next; that produces inconsistent output and is \
treated as an error, not stylistic variation. When in doubt about the exact standard \
term, use the closest recognized medical terminology rather than a plain-language \
description of the effect.

Rules:
- Fill a field only when THIS excerpt (or an earlier one already reflected in the JSON \
so far) gives direct evidence for it. Never fabricate or guess. atc_code and \
reference_product in particular: many documents never state these directly (a bula/SmPC \
usually doesn't name its reference product even when it says "equivalent to the \
reference medicine") — leave null rather than inferring from outside knowledge.
- Registration/administrative facts (status, registration_date, approval_date) are the \
most common place to accidentally over-infer. A document being currently in print, or \
carrying a dispensing/legal classification (prescription-only, restricted-use, OTC), \
proves nothing about registration status — leave these null unless stated outright.
- Never blank out, delete, or downgrade a field that is already correctly populated, \
unless this excerpt directly contradicts it (e.g. a corrected dose).
- All array fields (columns.therapeutic_areas/indications/symptoms/adverse_reactions/\
contraindications/active_ingredients/strengths/dosage_forms/routes, and \
product_data.presentations/indications/pivotal_evidence/excipients) accumulate: keep \
every entry already present and append genuinely new, distinct entries found in this \
excerpt. For excipients specifically, only the active substance(s) belong in \
active_ingredients/substance — everything else in the formulation (fillers, coatings, \
preservatives, solvents) goes in excipients instead.
- A product_data.presentations ENTRY is identified by (strength set + form + route + \
pack_size) and NOTHING ELSE. Deduplicate ruthlessly against that key:
  * More than one active ingredient is still ONE entry. A combination tablet with \
amoxicillin 500 mg + clavulanic acid 125 mg is ONE presentation entry with \
active_ingredients = [{"substance":"Amoxicillin","strength_value":500,...}, \
{"substance":"Clavulanic acid","strength_value":125,...}], NOT two entries.
  * PACK SIZE IS part of the identity: one entry per distinct pack size. Montelukast \
4 mg chewable tablet in 10s, 30s and 100s plus 5 mg in 10s, 30s and 100s is SIX entries \
(pack_size 10/30/100 for each of the two strengths). Put the quantity in pack_size — \
never in per_value, which means something entirely different.
  * CONTAINER MATERIAL IS NOT part of the identity. Blister material, carton type, \
hospital-vs-retail labelling and per-presentation registry codes are not stored anywhere \
in this schema, so they never create a new entry. A registry listing "750 mg tablet, \
clear blister, x20", "750 mg tablet, opaque blister, x20" and "750 mg tablet, alu/alu \
blister, x20" is ONE entry (750 mg, film-coated tablet, oral, pack_size 20) — the three \
rows differ only in a field this schema does not keep. Emitting the same \
strength+form+route+pack_size twice is always an error.
  * Genuinely different strengths ARE separate entries (4 mg/2 mL and 8 mg/4 mL of the \
same injectable are two).
  * NULL pack_size is not a distinct value from a populated one — it means "pack size not \
yet known," not "no pack." If an entry already in the JSON has the same strength set + \
form + route with pack_size null, and this excerpt (or an earlier one) gives a pack_size/\
pack_unit for that same presentation, fill it into the existing entry instead of adding a \
new one. Only keep them as two separate entries if there is a real distinguishing \
difference beyond the null vs. populated pack_size (e.g. the populated one is actually a \
different quantity than another already-known non-null pack_size for that same strength \
set + form + route).
- product_data.key_risks does NOT accumulate the same way — it is a curated shortlist, \
not an adverse-reaction dump (columns.adverse_reactions is where the full list goes). \
Every time, re-rank across this excerpt PLUS everything already listed, and keep only \
the max 8 risks that most drive labelling, contraindications, or active patient \
monitoring (e.g. boxed-warning-tier events, fatal/life-threatening reactions, major \
contraindications) — prefer these over routine/common ADRs like headache or nausea. It \
is correct and expected to drop a previously-listed lower-severity risk once 8 more \
important ones are known, even though that means shortening the list.
- If this excerpt has nothing relevant, return the JSON unchanged.
- Some excerpts are a JSON blob of raw registry metadata (application details, product \
tables, approval history, therapeutic equivalents, ...) rather than document prose — \
these may appear before, after, or interleaved with document excerpts for the same \
product. Extract from whatever fields are present the same way regardless of which \
form an excerpt takes.
- When a single excerpt lists multiple distinct values for the same field in \
one sentence (e.g. "available in pack sizes of 14, 21, 56, 60, 84, 90, 100 or 112 \
capsules", or a paragraph naming several separate clinical trials), you MUST create \
one separate array entry per distinct value — never one entry that merges, represents, \
or summarises them. Enumerate exhaustively: if the text names N distinct pack sizes, \
trials, or list items, your output must contain N entries, not fewer. Under-counting a \
list this way is a missed extraction, not an acceptable summarisation.
- When exhaustively enumerating per the rule above, each new entry still carries the \
OTHER identity fields already established for this same fact elsewhere in the excerpts \
— do not drop them just because this particular sentence only mentions the one field \
you are enumerating. Example: if the strength/form/route for this product were already \
established as 200 mg / hard capsule / oral, and a later sentence says "available in \
pack sizes of 14, 21, 56...", each resulting presentation entry must still carry \
strength_value=200, form="hard capsule", route="oral" — not leave them null just \
because this sentence itself only names the pack size.
- A label commonly lists MULTIPLE STRENGTHS of the same product family in one \
sentence (e.g. "Pregabalin Accord 25/50/75/100/150/200/225/300mg hard capsules are \
available in pack sizes of 14, 21, 56..."). Read carefully which strengths the sentence \
actually covers:
  * If the sentence's strength list INCLUDES the exact strength named in "Product:"/\
"Registration number:" in the user message, the fact DOES apply to this product — \
extract it. Do NOT skip a fact just because other strengths are also named alongside \
yours in the same sentence.
  * Only exclude a fact when it is stated as applying EXCLUSIVELY to a different \
strength than this product's (e.g. "Additionally, Pregabalin Accord 75mg hard capsules \
are ALSO available in pack sizes of 70" names only 75mg, not this product — exclude \
that one fact, but still use the earlier sentence that included your strength).
- product_data.pivotal_evidence holds EFFICACY trials that support the labelled \
indications — NOT safety/epidemiology/interaction studies. Skip observational studies \
about combining this drug with something else, adverse-event odds-ratio studies, and \
post-marketing surveillance; those inform key_risks instead, never pivotal_evidence.
  * One entry per DISTINCT STUDY — a distinct patient population plus a distinct \
primary endpoint — NOT per treatment arm and NOT per dose within that same \
study/endpoint.
  * Put the study's own treated-arm result in "value"/"unit", and fold any other \
arm(s) (placebo, comparator dose, lower dose) into "comparator" as short text, e.g. \
"Placebo: 18%" or "7 mg/kg/day: not significant". Do NOT create two separate entries \
for the treatment arm and the placebo arm of the same comparison — that is ONE entry.
  * A new entry is only warranted when the population or the specific clinical outcome \
being measured genuinely changes (e.g. adult neuropathic pain vs paediatric seizures vs \
generalised anxiety disorder are 3 separate studies, so 3 entries) — not for every \
percentage figure mentioned in the paragraph.
- The 7 product_data keys (substance, presentations, indications, approval, \
pivotal_evidence, key_risks, excipients) must ALWAYS be present in your output, even \
as an empty object/array, for every batch, from the first call onward.
- "columns" and "product_data" are TWO VIEWS OF THE SAME FACTS, not two independent \
extraction targets — every fact you put in product_data MUST also be mirrored into the \
matching flat columns.* field, every single call. Concretely:
  * Every product_data.presentations[].form -> add to columns.dosage_forms
  * Every product_data.presentations[].route -> add to columns.routes
  * Every product_data.presentations[].active_ingredients[].substance and \
.strength_value+.strength_unit -> add to columns.active_ingredients and columns.strengths \
respectively
  * Every product_data.indications[].condition -> add to columns.indications
  * product_data.substance.inn -> also belongs in columns.active_ingredients if not \
already present
If you find yourself about to return a non-empty product_data field alongside an empty \
columns.* array that should contain the same facts, that is a mistake — go back and \
populate the columns.* array before returning your answer. This applies from the very \
first call, not only once product_data has multiple entries.
- Respond with ONLY the JSON object. No markdown fences, no commentary.
"""


def _client():
    """
    The OpenRouter client. No longer called on the main path — Stage C goes
    through llm_service_client, which reaches this only as its fallback (see
    llm_service_client.get_fallback_client).
    """
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )


# Running usage/cost totals for the current extract_pending() run, shared
# across worker threads. Counts BOTH providers, so the totals reflect the whole
# run; the per-provider split (and which of them actually cost money) comes from
# llm_service_client.provider_stats() in extract_pending's summary. `cost` is
# only ever populated for OpenRouter — a self-hosted call has no API cost, so
# the field stays absent and contributes nothing.
_usage_lock = threading.Lock()
_usage_totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}


def _record_usage(usage, product_id, batch_label):
    if usage is None:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", None) or 0
    completion_tokens = getattr(usage, "completion_tokens", None) or 0
    cost = getattr(usage, "cost", None)
    cost_str = f"${cost:.6f}" if cost is not None else "n/a"
    logger.info(
        f"[ai_extraction] usage product={product_id} {batch_label}: "
        f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} cost={cost_str}"
    )
    with _usage_lock:
        _usage_totals["calls"] += 1
        _usage_totals["prompt_tokens"] += prompt_tokens
        _usage_totals["completion_tokens"] += completion_tokens
        if cost is not None:
            _usage_totals["cost"] += cost


def _reset_usage_totals():
    with _usage_lock:
        _usage_totals.update(calls=0, prompt_tokens=0, completion_tokens=0, cost=0.0)


def _usage_summary():
    with _usage_lock:
        totals = dict(_usage_totals)
    return (
        f"{totals['calls']} call(s), "
        f"{totals['prompt_tokens']} prompt tokens, "
        f"{totals['completion_tokens']} completion tokens, "
        f"${totals['cost']:.4f} total cost"
    )


def _build_user_message(product, accumulated, batch_texts, batch_start, batch_end, total):
    identity = (
        f"Product: {product.get('product_name') or 'unknown'}\n"
        f"Registration number: {product.get('registration_number') or 'unknown'}\n"
    )
    excerpts = "\n\n".join(
        f"--- Excerpt {batch_start + i} of {total} ---\n{text}"
        for i, text in enumerate(batch_texts)
    )
    return (
        f"{identity}\n"
        f"Excerpts {batch_start}-{batch_end} of {total} (read them in order; they are "
        f"consecutive pieces of the same document).\n\n"
        f"JSON accumulated so far:\n{json.dumps(accumulated, ensure_ascii=False)}\n\n"
        f"New excerpts:\n{excerpts}"
    )


class ContextOverflow(RuntimeError):
    """
    The batch does not fit the model's context window.

    Distinct from every other failure because the remedy is the opposite: send
    LESS, do not retry. Retrying an oversized payload fails identically each
    time, and failing it over to the other provider does not make it fit either
    — so this propagates past both the retry loop (see no_retry_on below) and
    llm_service_client's fallback, up to _fold_slice, which splits the batch.
    """


def _fit_completion_tokens(user_message, context_window):
    """
    How many reply tokens are left after the system prompt, the accumulated
    JSON and this batch's excerpts. Returns (max_tokens, prompt_estimate), with
    max_tokens None when there is not enough room for a usable reply.

    Done as token arithmetic before the call rather than by waiting for the
    server to reject it: the pre-check is free, deterministic, and the same
    answer.
    """
    prompt_estimate = count_tokens(SYSTEM_PROMPT) + count_tokens(user_message) + 16
    available = context_window - prompt_estimate - _safety_margin(prompt_estimate)
    if available < MIN_COMPLETION_TOKENS:
        return None, prompt_estimate
    return max(MIN_COMPLETION_TOKENS, min(MAX_COMPLETION_TOKENS, available)), prompt_estimate


def _call_llm_merge(product, accumulated, batch_texts, batch_start, batch_end, total,
                    context_window):
    """
    One fold call, on the self-hosted LLM with an automatic OpenRouter fallback
    (processing/llm_service_client.py). Raises ContextOverflow when the payload
    cannot fit, ValueError for anything else the caller should retry.
    """
    label = f"excerpts={batch_start}-{batch_end}/{total}"
    user_message = _build_user_message(
        product, accumulated, batch_texts, batch_start, batch_end, total)

    max_tokens, prompt_estimate = _fit_completion_tokens(user_message, context_window)
    if max_tokens is None:
        raise ContextOverflow(
            f"prompt ~{prompt_estimate} tokens leaves under {MIN_COMPLETION_TOKENS} "
            f"for a reply in a {context_window}-token window"
        )

    try:
        parsed, provider, meta = llm_service_client.call_json(
            SYSTEM_PROMPT, user_message, max_tokens=max_tokens, temperature=0,
            json_mode=True, parse_retries=1,
        )
    except llm_service_client.LLMServiceError as e:
        if llm_service_client.is_context_error(e):
            raise ContextOverflow(str(e)) from e
        raise ValueError(f"LLM call failed: {e}") from e

    _record_usage(meta.get("usage"), product["id"], f"{label} provider={provider}")
    if parsed is None:
        raise ValueError("LLM returned an empty or unparseable response")
    if not isinstance(parsed, dict) or "columns" not in parsed or "product_data" not in parsed:
        raise ValueError(
            f"LLM JSON missing columns/product_data sections: "
            f"{list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}"
        )
    if meta.get("finish_reason") == "length":
        # Not fatal — json_utils closes the object — but the tail of the reply is
        # genuinely lost, so say so rather than letting it pass silently.
        logger.warning(
            f"[ai_extraction] product={product['id']} {label}: reply hit max_tokens "
            f"({max_tokens}) and was truncated; raise AI_EXTRACTION_MAX_TOKENS"
        )
    return parsed


def _flatten_json_fields(data, prefix=""):
    """
    Walk a nested dict, yielding (dotted.path, value) for every leaf that is
    either a scalar or a non-empty list. Nested dicts (e.g. a crawler's
    approval_history: {original_approvals: [...], supplements: [...]}) are
    descended into rather than treated as one opaque leaf, so each of their
    list-valued fields packs independently below.
    """
    for key, value in (data or {}).items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flatten_json_fields(value, path)
        elif value not in (None, [], {}, ""):
            yield path, value


def _pack_list_field(path, items, target_tokens=DEFAULT_TARGET_TOKENS):
    """
    Greedily pack a list field's items into ~target_tokens-sized groups —
    the same greedy-packing idea as chunk_text's paragraph packing, just
    keyed on list items instead of paragraphs. Needed because chunk_text
    itself doesn't work on pretty-printed JSON: it has no blank-line
    paragraph breaks for chunk_text's normal split, and no sentence-ending
    punctuation for its oversized-paragraph fallback to find, so a large
    array-shaped json_data (a crawler's products/documents/approval_history
    entries) would otherwise come back as a single unsplit multi-thousand-
    token blob — confirmed live against a 400-entry synthetic payload before
    this function existed.
    """
    groups, current, current_tokens, start = [], [], 0, 1
    for i, item in enumerate(items, start=1):
        text = json.dumps(item, ensure_ascii=False, default=str)
        tokens = count_tokens(text)
        if current and current_tokens + tokens > target_tokens:
            groups.append((start, i - 1, current))
            current, current_tokens, start = [], 0, i
        current.append(text)
        current_tokens += tokens
    if current:
        groups.append((start, len(items), current))

    return [
        f"Raw structured registry metadata — {path} (items {s}-{e} of {len(items)}):\n"
        f"[{', '.join(chunk_items)}]"
        for s, e, chunk_items in groups
    ]


def _json_data_units(product):
    """
    Split product['json_data'] into one or more chunk-sized text units: one
    "summary" unit of every scalar field (application_number, company,
    source_url, ...), plus one or more units per list-valued field
    (products, documents, approval_history.original_approvals, ...), each
    itself packed to ~DEFAULT_TARGET_TOKENS via _pack_list_field. This is
    what makes a large json_data blob actually get chunked, unlike naively
    reusing chunk_text on the whole serialized JSON (see _pack_list_field).
    """
    json_data = product.get("json_data")
    if not json_data:
        return []

    scalars = {}
    units = []
    for path, value in _flatten_json_fields(json_data):
        if isinstance(value, list):
            units.extend(_pack_list_field(path, value))
        else:
            scalars[path] = value

    if scalars:
        units.insert(
            0,
            "Raw structured registry metadata (application/product summary):\n"
            + json.dumps(scalars, ensure_ascii=False, default=str, indent=2),
        )
    if not units:
        # json_data had only empty/falsy values under _flatten_json_fields'
        # filter — fall back to dumping it whole rather than silently
        # dropping it (still tiny by construction, since anything
        # substantial would have been caught above).
        units.append(
            "Raw structured registry metadata:\n"
            + json.dumps(json_data, ensure_ascii=False, default=str, indent=2)
        )
    return units


def _input_units(cursor, product):
    cursor.execute(
        "SELECT chunk_text FROM drug.product_chunks WHERE product_id = %s ORDER BY chunk_index",
        (product["id"],),
    )
    document_chunks = [row["chunk_text"] for row in cursor.fetchall()]

    # json_data goes first: it's the structured "skeleton" (application
    # number, product table, approval dates, TE cross-references) that
    # document prose often doesn't restate, so the fold sees it before the
    # narrative detail in document_chunks. Included whenever present, not
    # only when document_chunks is empty — see module docstring.
    return _json_data_units(product) + document_chunks


def _persist(cursor, product_id, accumulated, ai_status, attempts, error_summary,
             processing_status, write_registration_number=True):
    """
    `write_registration_number=False` persists everything except the
    AI-supplied registration number — the fallback when writing it would
    violate uq_products_country_regnum (see _persist_with_regnum_fallback).
    """
    columns = accumulated.get("columns") or {}
    product_data = accumulated.get("product_data") or {}

    set_clauses = ["product_data = %(product_data)s"]
    params = {
        "product_data": Json(product_data),
        "ai_status": ai_status,
        "attempts": attempts,
        "error_summary": error_summary,
        "processing_status": processing_status,
        "product_id": product_id,
    }

    for field in COLUMN_SCALAR_FIELDS:
        if field == "registration_number":
            # Never override Stage A's value (from crawler json_data) — only
            # fill it in if Stage A left it null, and only if no other product
            # in the same country already claims that number. Sources where
            # Stage A captures no registration number (MHRA) otherwise let the
            # LLM's value collide on uq_products_country_regnum.
            if write_registration_number:
                set_clauses.append("""
                    registration_number = COALESCE(
                        products.registration_number,
                        CASE WHEN %(registration_number)s IS NOT NULL AND NOT EXISTS (
                            SELECT 1 FROM drug.products other
                            WHERE other.country_id IS NOT DISTINCT FROM products.country_id
                              AND other.registration_number = %(registration_number)s
                              AND other.id <> products.id
                        ) THEN %(registration_number)s END
                    )
                """)
            else:
                set_clauses.append("registration_number = products.registration_number")
        else:
            set_clauses.append(f"{field} = %({field})s")
        params[field] = columns.get(field)

    for field in COLUMN_ARRAY_FIELDS:
        set_clauses.append(f"{field} = %({field})s")
        value = columns.get(field)
        params[field] = value if isinstance(value, list) else None

    cursor.execute(
        f"""
        UPDATE drug.products AS products
        SET {', '.join(set_clauses)},
            ai_extraction_status = %(ai_status)s,
            ai_extraction_attempts = products.ai_extraction_attempts + %(attempts)s,
            ai_extraction_error = %(error_summary)s,
            processing_status = %(processing_status)s,
            updated_at = CURRENT_TIMESTAMP
        WHERE products.id = %(product_id)s
        """,
        params,
    )


def _persist_with_regnum_fallback(cursor, product_id, accumulated, ai_status, attempts,
                                  error_summary, processing_status):
    """
    _persist's NOT EXISTS guard closes the common case, but two workers can
    still persist the same brand-new registration number at the same instant
    (each one's guard passes before the other commits). A SAVEPOINT lets us
    absorb that unique violation and re-persist without the number, instead of
    letting it abort the transaction and kill the worker thread — the LLM's
    other extracted fields are still worth keeping.
    """
    cursor.execute("SAVEPOINT persist_product")
    try:
        _persist(cursor, product_id, accumulated, ai_status, attempts,
                 error_summary, processing_status)
    except psycopg2.errors.UniqueViolation:
        cursor.execute("ROLLBACK TO SAVEPOINT persist_product")
        logger.warning(
            f"[ai_extraction] product={product_id}: AI registration_number collides with an "
            f"existing product in the same country — persisting everything else without it"
        )
        _persist(cursor, product_id, accumulated, ai_status, attempts,
                 error_summary, processing_status, write_registration_number=False)
    cursor.execute("RELEASE SAVEPOINT persist_product")


_context_window_cache = None
_context_window_lock = threading.Lock()


def _context_window():
    """
    The primary provider's context window, discovered once per process.

    Asked of the server (GET /models -> max_model_len) rather than assumed, so
    a model swap on the GPU does not silently start overflowing; falls back to
    LLM_SERVICE_CONTEXT_TOKENS when the server does not report one. Cached
    because every fold call needs it and it cannot change mid-run.
    """
    global _context_window_cache
    if _context_window_cache is None:
        with _context_window_lock:
            if _context_window_cache is None:
                _context_window_cache = llm_service_client.discover_context_window()
                logger.info(f"[ai_extraction] context window: {_context_window_cache} tokens")
    return _context_window_cache


def _fold_slice(product, accumulated, batch_texts, offset, total, context_window,
                failed_chunks, depth=0):
    """
    Fold one slice of the excerpt list into the accumulator, returning the
    updated accumulator.

    On a ContextOverflow the slice is halved and each half recursed, so every
    excerpt is still folded in — at the cost of more, smaller calls — rather
    than the whole batch being abandoned. This matters much more on the
    self-hosted model than it did on OpenRouter: with a ~22k window, a long
    product's accumulator eventually crowds out the excerpts, and without
    splitting the tail of that product's document would simply never be read.

    An excerpt that overflows even alone cannot be folded at all; it is recorded
    in `failed_chunks` so the row lands NEEDS_REVIEW instead of a false DONE.
    """
    if not batch_texts:
        return accumulated

    batch_start, batch_end = offset + 1, offset + len(batch_texts)
    span = f"{batch_start}-{batch_end}"
    try:
        return run_with_retries(
            _call_llm_merge, product, accumulated, batch_texts,
            batch_start, batch_end, total, context_window,
            label=f"ai_extraction product={product['id']} excerpts={span}/{total}",
            # Splitting is the remedy for an overflow, not retrying — see
            # ContextOverflow. Without this the row waits out the full backoff
            # three times before the split it needed all along.
            no_retry_on=(ContextOverflow,),
        )
    except ContextOverflow as e:
        if len(batch_texts) == 1:
            logger.error(
                f"[ai_extraction] product={product['id']} excerpt {span}/{total} does not "
                f"fit the {context_window}-token window even alone ({e}) — skipped. "
                f"Re-chunk this document with a smaller "
                f"chunking.DEFAULT_TARGET_TOKENS, or use a larger-context model."
            )
            failed_chunks.append((span, f"excerpt exceeds context window: {e}"))
            return accumulated
        mid = len(batch_texts) // 2
        logger.warning(
            f"[ai_extraction] product={product['id']} excerpts {span}/{total} overflow "
            f"({e}) — splitting into {batch_start}-{offset + mid} and "
            f"{offset + mid + 1}-{batch_end}"
        )
        accumulated = _fold_slice(product, accumulated, batch_texts[:mid], offset,
                                  total, context_window, failed_chunks, depth + 1)
        return _fold_slice(product, accumulated, batch_texts[mid:], offset + mid,
                           total, context_window, failed_chunks, depth + 1)
    except RetriesExhausted as e:
        logger.error(
            f"[ai_extraction] product={product['id']} excerpts {span}/{total} "
            f"permanently failed after {e.attempts} attempts: {e.last_exception}"
        )
        failed_chunks.append((span, str(e.last_exception)))
        return accumulated


def _extract_one(cursor, product):
    units = _input_units(cursor, product)
    if not units:
        cursor.execute(
            """
            UPDATE drug.products
            SET ai_extraction_status = 'SKIPPED', processing_status = 'NEEDS_REVIEW',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            """,
            (product["id"],),
        )
        return "skipped"

    capped_from = None
    if MAX_CHUNKS_PER_PRODUCT and len(units) > MAX_CHUNKS_PER_PRODUCT:
        capped_from = len(units)
        logger.warning(
            f"[ai_extraction] product={product['id']}: capping {capped_from} excerpts "
            f"to {MAX_CHUNKS_PER_PRODUCT} (AI_EXTRACTION_MAX_CHUNKS) — excerpts "
            f"{MAX_CHUNKS_PER_PRODUCT + 1}-{capped_from} will NOT be read, so this "
            f"row lands NEEDS_REVIEW rather than DONE"
        )
        units = units[:MAX_CHUNKS_PER_PRODUCT]

    accumulated = {"columns": {}, "product_data": {}}
    failed_chunks = []
    total = len(units)
    context_window = _context_window()

    offset = 0
    for batch_texts in _batch(units, CHUNKS_PER_CALL):
        accumulated = _fold_slice(
            product, accumulated, batch_texts, offset, total,
            context_window, failed_chunks,
        )
        offset += len(batch_texts)

    product_data = accumulated.get("product_data") or {}
    has_all_keys = all(key in product_data for key in PRODUCT_DATA_KEYS)
    attempts = len(failed_chunks)
    if capped_from:
        # Recorded like a failed batch so the reason survives in
        # ai_extraction_error and the status logic below cannot return DONE.
        failed_chunks.append((
            f"{MAX_CHUNKS_PER_PRODUCT + 1}-{capped_from}",
            f"not sent: AI_EXTRACTION_MAX_CHUNKS={MAX_CHUNKS_PER_PRODUCT}",
        ))

    if not product_data and not accumulated.get("columns"):
        ai_status, processing_status = "FAILED", "FAILED"
    elif failed_chunks or not has_all_keys:
        ai_status, processing_status = "NEEDS_REVIEW", "NEEDS_REVIEW"
    else:
        ai_status, processing_status = "DONE", "ENRICHED"

    error_summary = "; ".join(f"excerpts {p}: {err}" for p, err in failed_chunks)[:2000] or None

    _persist_with_regnum_fallback(
        cursor, product["id"], accumulated, ai_status, attempts, error_summary, processing_status
    )
    return ai_status


def _mark_failed(cursor, product_id, error):
    """Terminal status for a row whose extraction raised something unexpected,
    so the next run doesn't re-claim it and hit the same wall."""
    cursor.execute(
        """
        UPDATE drug.products
        SET ai_extraction_status = 'FAILED',
            ai_extraction_attempts = ai_extraction_attempts + 1,
            ai_extraction_error = %s,
            processing_status = 'FAILED',
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        """,
        (str(error)[:2000], product_id),
    )


def _worker(country, remaining, remaining_lock, progress):
    """
    One worker's claim loop. claim_ai_extraction() atomically flips a row to
    PROCESSING before this worker starts its (potentially multi-minute,
    multi-chunk) LLM fold, so no other worker can pick it up too.
    """
    conn = get_db_connection()
    stats = {"DONE": 0, "NEEDS_REVIEW": 0, "FAILED": 0, "skipped": 0}
    try:
        while True:
            if remaining is not None:
                with remaining_lock:
                    if remaining[0] <= 0:
                        break
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                product = claim_ai_extraction(cur, country=country)
                conn.commit()
            if not product:
                break
            if remaining is not None:
                with remaining_lock:
                    remaining[0] -= 1

            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    outcome = _extract_one(cur, product)
                    conn.commit()
                logger.info(f"[ai_extraction] product={product['id']} -> {outcome}")
            except Exception as e:
                # One unexpected row must not take the whole worker down: before
                # this, any escaping exception ended the thread and the pool
                # silently shrank until nothing was processing. Mark the row
                # FAILED so it is not re-claimed forever, and carry on.
                conn.rollback()
                outcome = "FAILED"
                logger.exception(f"[ai_extraction] product={product['id']} errored — marking FAILED")
                try:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        _mark_failed(cur, product["id"], e)
                        conn.commit()
                except Exception:
                    conn.rollback()
                    logger.exception(f"[ai_extraction] product={product['id']}: could not record failure")
            stats[outcome] = stats.get(outcome, 0) + 1
            progress.advance()
    finally:
        conn.close()
    return stats


def extract_pending(limit=None, country=None, workers=1):
    """
    Run AI extraction for every drug.products row that has text (or
    json_data) ready and hasn't been AI-extracted yet, using `workers`
    concurrent claim loops. `country` (matches
    drug.regulatory_geography.country_name, e.g. "Brazil") scopes the run to
    one country. `limit` across concurrent workers is best-effort (may
    overshoot by up to `workers - 1` rows). Returns a dict of outcome -> count.
    """
    remaining = [limit] if limit else None
    remaining_lock = threading.Lock()
    _reset_usage_totals()
    llm_service_client.reset_stats()

    # Confirm a provider answers before claiming any rows. Without this, a dead
    # self-hosted endpoint with no fallback configured would flip every claimed
    # row to PROCESSING, fail it, and burn the whole queue's retry budget before
    # anyone noticed the GPU was down.
    llm_service_client.health_check()
    if llm_service_client.self_hosted_available():
        logger.info(
            f"[ai_extraction] primary: {llm_service_client.LLM_SERVICE_MODEL} @ "
            f"{llm_service_client.LLM_SERVICE_BASE_URL} | fallback: "
            f"{llm_service_client.LLM_FALLBACK_MODEL} "
            f"({'enabled' if llm_service_client.fallback_available() else 'unavailable'})"
        )
    else:
        # Say WHICH of the two reasons, so an accidentally-missing env var is not
        # mistaken for a deliberate hosted-only run.
        reason = ("LLM_SERVICE_ENABLED=false"
                  if not llm_service_client.LLM_SERVICE_ENABLED
                  else "LLM_SERVICE_BASE_URL is unset")
        logger.warning(
            f"[ai_extraction] self-hosted path OFF ({reason}) — every call goes "
            f"to {llm_service_client.LLM_FALLBACK_MODEL} via OpenRouter (PAID)"
        )

    logger.info(
        f"[ai_extraction] batching: {CHUNKS_PER_CALL} excerpts/call, "
        f"max_tokens {MAX_COMPLETION_TOKENS}, chunk size "
        f"{DEFAULT_TARGET_TOKENS} tokens, per-product excerpt cap "
        f"{MAX_CHUNKS_PER_PRODUCT or 'none'}"
    )

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            pending = count_ai_extraction_pending(cur, country=country)
        conn.commit()
    finally:
        conn.close()
    total = min(pending, limit) if limit else pending
    progress = Progress("ai_extraction", total=total, workers=workers)

    totals = run_worker_pool(
        # No sharding needed — see the note in text_extraction.extract_pending.
        lambda _i, _n: _worker(country, remaining, remaining_lock, progress),
        workers, label="ai_extraction"
    )
    progress.finish()
    logger.info(f"[ai_extraction] done: {totals}")

    provider_stats = llm_service_client.provider_stats()
    calls = provider_stats["calls"]
    logger.info(
        f"[ai_extraction] provider split: self-hosted={calls.get('self_hosted', 0)} "
        f"call(s), OpenRouter={calls.get('openrouter', 0)} call(s)"
    )
    if calls.get("openrouter"):
        # OpenRouter is a paid API either way, but distinguish "we chose this"
        # from "we wanted the GPU and could not have it" — the row statuses look
        # identical, so only this line tells them apart.
        openrouter = provider_stats["tokens"]["openrouter"]
        cost = openrouter.get("cost") or 0.0
        role = ("configured provider" if not llm_service_client.self_hosted_available()
                else "fallback")
        logger.warning(
            f"[ai_extraction] {calls['openrouter']} call(s) went to the PAID OpenRouter "
            f"{role} ({llm_service_client.LLM_FALLBACK_MODEL}): "
            f"{openrouter['prompt']} prompt + {openrouter['completion']} completion tokens"
            + (f", ${cost:.4f}" if cost else "")
            + (" — the self-hosted endpoint was unreachable for part of this run"
               if provider_stats["degraded"] else "")
        )
    self_hosted = provider_stats["tokens"]["self_hosted"]
    if calls.get("self_hosted"):
        logger.info(
            f"[ai_extraction] self-hosted usage: {self_hosted['prompt']} prompt + "
            f"{self_hosted['completion']} completion tokens (no API cost)"
        )
    logger.info(f"[ai_extraction] combined usage: {_usage_summary()}")
    return totals

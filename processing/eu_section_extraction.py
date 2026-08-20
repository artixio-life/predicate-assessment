"""
For European Union products, drug.products.document_text is the full EMA
"Product Information" PDF — Annex I (Summary of Product Characteristics),
Annex II (manufacturer / marketing-authorisation conditions — regulatory
paperwork, not patient- or prescriber-facing), then Annex III (Labelling +
Package Leaflet), always in that order per EMA's QRD template. Annex I alone
routinely runs to dozens or hundreds of pages of dense clinical-trial prose;
Annex III is the much shorter, already-summarised version of the same core
facts (indication, form, strength, storage, pack contents, MAH) that the AI
extraction fold (processing/ai_extraction.py) actually needs. So only Annex
III gets chunked into drug.product_chunks for this country — Annex I/II stay
in document_text (kept for reference / any future use) but are never sent to
the LLM. Verified against three structurally distinct real EMA documents
(infusion solution, biosimilar injection pen, oral tablet) — see
tests/test_eu_section_extraction.py.

Headings are matched case-sensitively on the literal word "ANNEX" (all
caps): EMA's own template only writes it uppercase in a heading; every
inline cross-reference in the body ("see Annex I: Summary of Product
Characteristics, section 4.2") uses title case, so case-sensitivity alone
tells a real section boundary apart from a cross-reference without needing
markdown/heading-position heuristics. This also makes the same regex work
unchanged whether document_text came from pdf-inspector's plain-text layer
("ANNEX III", "A. LABELLING") or Mistral OCR's markdown ("# **ANNEX III**",
"# A. LABELLING") — see processing/text_extraction.py's two extraction paths.

Not handled: the older EMA template some pre-~2012 authorisations still use,
which splits Annex III into two independent annexes ("ANNEX IIIA" /
"ANNEX IIIB") instead of one "ANNEX III" with "A. LABELLING" / "B. PACKAGE
LEAFLET" subsections — `\bIII\b` doesn't match "IIIA"/"IIIB", so those
products fall through to `None` (caller chunks the whole document instead)
rather than risk mis-slicing a shape this hasn't been checked against.
"""
import re

_LABELLING_START = re.compile(r"A\.\s*LABELLING")
_ANNEX_III_START = re.compile(r"ANNEX\s+III\b")
_NEXT_ANNEX = re.compile(r"ANNEX\s+(?:I|II|IV|V)\b")


def extract_annex_iii(document_text):
    """
    Return the Annex III slice (Labelling + Package Leaflet) of `document_text`,
    or None if no recognisable Annex III heading was found. None is a signal,
    not an error: the caller (processing/text_extraction.py) falls back to
    chunking the full document rather than silently producing zero chunks.
    """
    if not document_text:
        return None

    start_match = _LABELLING_START.search(document_text) or _ANNEX_III_START.search(document_text)
    if not start_match:
        return None

    next_match = _NEXT_ANNEX.search(document_text, start_match.end())
    end = next_match.start() if next_match else len(document_text)

    section = document_text[start_match.start():end].strip()
    return section or None

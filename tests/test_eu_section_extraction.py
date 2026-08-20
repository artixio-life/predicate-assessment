"""
Tests for processing/eu_section_extraction.py.

Every fixture in tests/fixtures/eu_*.{txt,md} is a real excerpt taken from
actual EMA "Product Information" PDFs already crawled into
source.drug_predicate_raw_records for country_id 18 (European Union) —
LysaKare (radiopharmaceutical infusion, kept in full at 43KB) and Tyenne
(biosimilar injection pen) and Rubraca (oral tablet, via Mistral OCR
markdown) trimmed around the Annex I/II/III boundary and the package-leaflet
tail to keep fixture size reasonable. The trimmed gaps are real omissions,
clearly marked, and no assertion relies on their contents — but everything
present in a fixture is copied verbatim from the extracted document, per
this repo's convention (see tests/test_us_manual_extraction.py) of pinning
behaviour against real strings rather than synthetic ones.

Run: python -m unittest discover -s tests
"""
import os
import unittest

from processing.eu_section_extraction import extract_annex_iii

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _read(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return f.read()


class TestExtractAnnexIII(unittest.TestCase):
    def test_lysakare_plain_text_starts_at_labelling(self):
        # pdf-inspector's plain-text layer: "A. LABELLING" as a bare line.
        section = extract_annex_iii(_read("eu_lysakare_full.txt"))
        self.assertTrue(section.startswith("A. LABELLING"))

    def test_lysakare_excludes_annex_i_and_ii(self):
        section = extract_annex_iii(_read("eu_lysakare_full.txt"))
        self.assertNotIn("SUMMARY OF PRODUCT CHARACTERISTICS", section)
        self.assertNotIn("MANUFACTURER RESPONSIBLE FOR BATCH RELEASE", section)

    def test_lysakare_includes_package_leaflet_tail(self):
        section = extract_annex_iii(_read("eu_lysakare_full.txt"))
        self.assertIn("Package leaflet: Information", section)
        self.assertIn("Detailed information on this medicine", section)

    def test_tyenne_plain_text_boundary(self):
        # Real excerpt: Annex I opening + Annex II/III/A.LABELLING boundary
        # + package-leaflet tail, with the (marked) internal gaps omitted.
        section = extract_annex_iii(_read("eu_tyenne_excerpt.txt"))
        self.assertTrue(section.startswith("A. LABELLING"))
        self.assertNotIn("Rheumatoid arthritis (RA)", section)  # Annex I, before the boundary
        self.assertIn("MINIMUM PARTICULARS TO APPEAR ON SMALL IMMEDIATE PACKAGING", section)

    def test_rubraca_ocr_markdown_headings(self):
        # Mistral OCR markdown: headings carry "#"/"##" and "**bold**" noise
        # around the same words ("# **ANNEX III**", "# A. LABELLING").
        section = extract_annex_iii(_read("eu_rubraca_ocr_excerpt.md"))
        self.assertTrue(section.startswith("A. LABELLING"))
        self.assertNotIn("DATE OF FIRST AUTHORISATION", section)  # Annex I, before the boundary
        self.assertIn("METHOD AND ROUTE(S) OF ADMINISTRATION", section)

    def test_inline_annex_cross_reference_is_not_mistaken_for_a_heading(self):
        # "Annex I" in title case (an inline cross-reference) must not be
        # picked up as the next-annex cutoff after a real "A. LABELLING" match.
        text = (
            "A. LABELLING\n\nRestricted medical prescription "
            "(see Annex I: Summary of Product Characteristics, section 4.2).\n\n"
            "Keep this leaflet, tail content."
        )
        section = extract_annex_iii(text)
        self.assertIn("Keep this leaflet, tail content.", section)

    def test_no_annex_heading_returns_none(self):
        self.assertIsNone(extract_annex_iii("Just some unrelated document text."))

    def test_empty_input_returns_none(self):
        self.assertIsNone(extract_annex_iii(""))
        self.assertIsNone(extract_annex_iii(None))


if __name__ == "__main__":
    unittest.main()

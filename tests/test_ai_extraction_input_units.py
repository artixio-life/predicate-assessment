"""
Tests for processing/ai_extraction.py::_input_units.

Covers the _SKIP_JSON_DATA_WHEN_DOCUMENT exception: European Union products
drop json_data once the document that was ACTUALLY chunked is confirmed to be
the expected "product_information" PDF (their json_data is a thin metadata
skeleton that just restates that specific document) — matched by
document["s3_path"] == product["source_url"], via _primary_document_type.
json_data is still included when no chunks exist yet (same as every other
country), or when the chunked document is something else (an assessment
report, a variation notice, or simply not listed in json_data["documents"]
at all) — that "it's just a redundant skeleton" reasoning doesn't hold for a
document it was never checked against.

Run: python -m unittest discover -s tests
"""
import unittest

from processing.ai_extraction import _input_units, _primary_document_type


class _FakeCursor:
    """Just enough of a DB cursor for _input_units: records the query it
    was given, and fetchall() returns the chunk rows it was constructed with."""

    def __init__(self, chunk_rows):
        self._chunk_rows = chunk_rows

    def execute(self, query, params):
        pass

    def fetchall(self):
        return [{"chunk_text": text} for text in self._chunk_rows]


class TestInputUnits(unittest.TestCase):
    def test_eu_with_confirmed_product_information_drops_json_data(self):
        # documents[] has an entry whose s3_path matches source_url, AND that
        # entry's document_type is "product_information" — the one case the
        # skip is meant for.
        product = {
            "id": 1,
            "country_name": "European Union",
            "source_url": "european_union/18/tyenne.pdf",
            "json_data": {
                "application_number": "EU/1/19/1381/001", "company": "Novartis",
                "documents": [{"s3_path": "european_union/18/tyenne.pdf",
                               "document_type": "product_information"}],
            },
        }
        cursor = _FakeCursor(["labelling text", "package leaflet text"])
        units = _input_units(cursor, product)
        self.assertEqual(units, ["labelling text", "package leaflet text"])

    def test_eu_with_different_document_type_still_includes_json_data(self):
        # Same shape, but the chunked document is an assessment report, not
        # the product_information PDF — json_data is NOT confirmed redundant,
        # so it must still be included.
        product = {
            "id": "1b",
            "country_name": "European Union",
            "source_url": "european_union/18/assessment_report.pdf",
            "json_data": {
                "application_number": "EU/1/19/1381/001",
                "documents": [{"s3_path": "european_union/18/assessment_report.pdf",
                               "document_type": "assessment_report"}],
            },
        }
        cursor = _FakeCursor(["assessment report text"])
        units = _input_units(cursor, product)
        # json_data included (more than just the one document chunk), the
        # document chunk still comes last, and the scalar summary is present
        # somewhere ahead of it — exact unit count isn't asserted here since
        # a list-valued json_data field (documents[]) legitimately gets its
        # own unit alongside the scalar-fields summary.
        self.assertGreater(len(units), 1)
        self.assertEqual(units[-1], "assessment report text")
        self.assertTrue(any("EU/1/19/1381/001" in u for u in units[:-1]))

    def test_eu_with_no_matching_documents_entry_still_includes_json_data(self):
        # documents[] doesn't even mention the chunked document (or json_data
        # has no documents[] at all) — document_type is unknown, which must
        # NOT be treated as a confirmed match.
        product = {
            "id": "1c",
            "country_name": "European Union",
            "source_url": "european_union/18/tyenne.pdf",
            "json_data": {"application_number": "EU/1/19/1381/001"},
        }
        cursor = _FakeCursor(["labelling text"])
        units = _input_units(cursor, product)
        self.assertEqual(len(units), 2)

    def test_primary_document_type_matches_by_s3_path(self):
        product = {
            "source_url": "european_union/18/tyenne.pdf",
            "json_data": {"documents": [
                {"s3_path": "european_union/18/other.pdf", "document_type": "annex"},
                {"s3_path": "european_union/18/tyenne.pdf", "document_type": "product_information"},
            ]},
        }
        self.assertEqual(_primary_document_type(product), "product_information")

    def test_eu_with_no_chunks_still_falls_back_to_json_data(self):
        product = {
            "id": 2,
            "country_name": "European Union",
            "json_data": {"application_number": "EU/1/19/1381/001"},
        }
        cursor = _FakeCursor([])
        units = _input_units(cursor, product)
        self.assertTrue(units)
        self.assertIn("EU/1/19/1381/001", units[0])

    def test_non_eu_country_still_folds_json_data_alongside_chunks(self):
        product = {
            "id": 3,
            "country_name": "Brazil",
            "json_data": {"application_number": "12345"},
        }
        cursor = _FakeCursor(["document chunk one"])
        units = _input_units(cursor, product)
        self.assertEqual(len(units), 2)
        self.assertIn("12345", units[0])
        self.assertEqual(units[1], "document chunk one")

    def test_missing_country_name_key_does_not_raise(self):
        # Rows claimed before this field existed, or a country with no
        # regulatory_geography match (NULL country_id) — country_name is
        # simply absent/None, and must not be treated as a skip match.
        product = {"id": 4, "json_data": {"foo": "bar"}}
        cursor = _FakeCursor(["chunk"])
        units = _input_units(cursor, product)
        self.assertEqual(len(units), 2)


if __name__ == "__main__":
    unittest.main()

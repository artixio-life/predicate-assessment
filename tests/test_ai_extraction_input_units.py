"""
Tests for processing/ai_extraction.py::_input_units.

Covers the _SKIP_JSON_DATA_WHEN_DOCUMENT exception: European Union products
drop json_data once a document was actually chunked (their json_data is a
thin metadata skeleton that just restates the document), but still fall
back to json_data when no chunks exist yet, same as every other country.

Run: python -m unittest discover -s tests
"""
import unittest

from processing.ai_extraction import _input_units


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
    def test_eu_with_document_chunks_drops_json_data(self):
        product = {
            "id": 1,
            "country_name": "European Union",
            "json_data": {"application_number": "EU/1/19/1381/001", "company": "Novartis"},
        }
        cursor = _FakeCursor(["labelling text", "package leaflet text"])
        units = _input_units(cursor, product)
        self.assertEqual(units, ["labelling text", "package leaflet text"])

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

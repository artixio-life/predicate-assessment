"""
Tests for processing/us_manual_extraction.py.

Every case here is a real string taken from source.drug_predicate_raw_records
for country_id 3 (United States) — the point of a rule-based extractor over an
LLM one is that its behaviour is pinned, so the rules get a regression suite
rather than a spot-check.

Run: python -m unittest discover -s tests
"""
import unittest

from processing.us_manual_extraction import (
    base_substance, parse_strength, parse_form_route, build_presentation,
    infer_modality, extract, _to_iso,
)


class TestBaseSubstance(unittest.TestCase):
    def test_strips_single_salt_token(self):
        self.assertEqual(base_substance("ABACAVIR SULFATE"), ("Abacavir", "Abacavir sulfate"))

    def test_strips_multiword_salt(self):
        self.assertEqual(base_substance("HYDROCORTISONE SODIUM SUCCINATE"),
                         ("Hydrocortisone", "Hydrocortisone sodium succinate"))

    def test_strips_prefixed_sodium_salt(self):
        self.assertEqual(base_substance("GADOFOSVESET TRISODIUM"),
                         ("Gadofosveset", "Gadofosveset trisodium"))

    def test_no_salt_leaves_salt_form_null(self):
        self.assertEqual(base_substance("LAMIVUDINE"), ("Lamivudine", None))

    def test_deinverts_index_style_name(self):
        # FDA lists this one alphabetically by the second word.
        self.assertEqual(base_substance("GONADOTROPIN, CHORIONIC")[0], "Chorionic gonadotropin")

    def test_salt_only_name_is_not_stripped_to_nothing(self):
        self.assertEqual(base_substance("SODIUM CHLORIDE")[0], "Sodium chloride")


class TestParseStrength(unittest.TestCase):
    def test_plain_mg(self):
        self.assertEqual(parse_strength("500MG"), (500, "mg", None, None))

    def test_decimal(self):
        self.assertEqual(parse_strength("0.5MG"), (0.5, "mg", None, None))

    def test_thousands_separator_and_container_denominator(self):
        # A vial is a container, not a measured volume, so it gets no per_value.
        self.assertEqual(parse_strength("10,000 UNITS/VIAL"), (10000, "units", None, "vial"))

    def test_base_equivalent(self):
        self.assertEqual(parse_strength("EQ 600MG BASE"), (600, "mg", None, None))

    def test_base_equivalent_per_vial(self):
        self.assertEqual(parse_strength("EQ 1GM BASE/VIAL"), (1, "g", None, "vial"))

    def test_concentration_per_ml_implies_per_value_one(self):
        self.assertEqual(parse_strength("5MG/ML"), (5, "mg", 1, "mL"))

    def test_concentration_per_stated_volume(self):
        self.assertEqual(parse_strength("2440MG/10ML (244MG/ML)"), (2440, "mg", 10, "mL"))

    def test_percentage(self):
        self.assertEqual(parse_strength("10%"), (10, "%", None, None))

    def test_orange_book_footnote_is_ignored(self):
        self.assertEqual(
            parse_strength("100MG **Federal Register determination that product was "
                           "not discontinued or withdrawn for safety or effectiveness reasons**"),
            (100, "mg", None, None))

    def test_unparseable_returns_all_null(self):
        self.assertEqual(parse_strength("N/A"), (None, None, None, None))


class TestFormRoute(unittest.TestCase):
    def test_combined_field(self):
        self.assertEqual(parse_form_route({"dosage_form_route": "TABLET;ORAL"}),
                         ("tablet", "oral"))

    def test_modified_release_form_keeps_its_qualifier(self):
        self.assertEqual(parse_form_route({"dosage_form_route": "TABLET, DELAYED RELEASE;ORAL"}),
                         ("tablet, delayed release", "oral"))

    def test_multiple_routes(self):
        self.assertEqual(parse_form_route({"dosage_form_route": "TABLET;BUCCAL, SUBLINGUAL"}),
                         ("tablet", "buccal, sublingual"))

    def test_form_with_no_route(self):
        self.assertEqual(parse_form_route({"dosage_form_route": "TABLET"}), ("tablet", None))

    def test_non_route_qualifier_is_dropped_not_echoed(self):
        # 'SINGLE-USE' describes the container, not a route of administration.
        self.assertEqual(parse_form_route({"dosage_form_route": "VIAL;SINGLE-USE"}),
                         ("vial", None))


class TestPresentation(unittest.TestCase):
    def test_combination_product_is_one_entry_with_aligned_strengths(self):
        pres = build_presentation({
            "strength": "EQ 600MG BASE;300MG",
            "dosage_form_route": "TABLET;ORAL",
            "active_ingredients": "ABACAVIR SULFATE; LAMIVUDINE",
        })
        self.assertEqual(pres["form"], "tablet")
        self.assertEqual(pres["route"], "oral")
        self.assertEqual(len(pres["active_ingredients"]), 2)
        abacavir, lamivudine = pres["active_ingredients"]
        self.assertEqual((abacavir["substance"], abacavir["strength_value"]), ("Abacavir", 600))
        self.assertEqual(abacavir["salt"]["substance"], "Abacavir sulfate")
        # 'EQ 600MG BASE' states the BASE quantity, so the salt quantity is unknown.
        self.assertIsNone(abacavir["salt"]["value"])
        self.assertEqual((lamivudine["substance"], lamivudine["strength_value"]),
                         ("Lamivudine", 300))
        self.assertIsNone(lamivudine["salt"])

    def test_pack_size_is_never_guessed(self):
        pres = build_presentation({"strength": "500MG", "dosage_form_route": "TABLET;ORAL",
                                    "active_ingredients": "METFORMIN HYDROCHLORIDE"})
        self.assertIsNone(pres["pack_size"])
        self.assertIsNone(pres["pack_unit"])


class TestModality(unittest.TestCase):
    def test_small_molecule_default(self):
        self.assertEqual(infer_modality(["Abacavir"], "ANDA", {}), "small_molecule")

    def test_bla_is_biologic(self):
        self.assertEqual(infer_modality(["Something novel"], "BLA", {}), "biologic")

    def test_monoclonal_suffix(self):
        self.assertEqual(infer_modality(["Trastuzumab"], "BLA", {}), "biologic")

    def test_adc_beats_biologic(self):
        self.assertEqual(infer_modality(["Disitamab vedotin"], "BLA", {}), "adc")

    def test_peptide(self):
        self.assertEqual(infer_modality(["Insulin glargine"], "NDA", {}), "peptide")

    def test_protein_hormone_under_an_nda_is_still_biologic(self):
        self.assertEqual(infer_modality(["Chorionic gonadotropin"], "NDA", {}), "biologic")


class TestDates(unittest.TestCase):
    def test_us_slash_format(self):
        self.assertEqual(_to_iso("03/28/2017"), "2017-03-28")

    def test_iso_passthrough(self):
        self.assertEqual(_to_iso("1956-03-09"), "1956-03-09")

    def test_spl_effective_time(self):
        self.assertEqual(_to_iso("20180201"), "2018-02-01")

    def test_junk(self):
        self.assertIsNone(_to_iso(""))


class TestApprovalHistoryShapes(unittest.TestCase):
    """Both crawler shapes must yield the same approval date and priority flag."""

    WEB_SHAPE = {
        "application_number": "091144",
        "application_type_code": "ANDA",
        "products": [{"strength": "500MG", "dosage_form_route": "TABLET;ORAL",
                      "active_ingredients": "IBUPROFEN", "marketing_status": "Prescription"}],
        "approval_history": {"original_approvals": [{
            "submission": "ORIG-1", "action_date": "03/28/2017", "action_type": "Approval",
            "review_priority_orphan_status": "PRIORITY",
            "submission_classification": "Type 2 - New Active Ingredient"}]},
    }
    API_SHAPE = {
        "application_number": "091144",
        "application_type_code": "ANDA",
        "products": [{"strength": "500MG", "dosage_form_route": "TABLET;ORAL",
                      "active_ingredients": "IBUPROFEN", "marketing_status": "Prescription"}],
        "approval_history": {"original_approvals": [{
            "submission_type": "ORIG", "submission_number": "1", "submission_status": "AP",
            "submission_status_date": "2017-03-28", "review_priority": "PRIORITY",
            "submission_class_description": "Type 2 - New Active Ingredient"}]},
    }

    def test_web_shape(self):
        result = extract(self.WEB_SHAPE)
        self.assertEqual(result["columns"]["approval_date"], "2017-03-28")
        self.assertTrue(result["product_data"]["approval"]["priority_review"])

    def test_api_shape_gives_the_same_answer(self):
        self.assertEqual(extract(self.API_SHAPE)["columns"],
                         extract(self.WEB_SHAPE)["columns"])

    def test_unknown_review_priority_is_not_a_standard_review(self):
        payload = dict(self.API_SHAPE)
        payload["approval_history"] = {"original_approvals": [dict(
            self.API_SHAPE["approval_history"]["original_approvals"][0],
            review_priority="UNKNOWN", submission_class_description=None,
            submission_class_code="UNKNOWN")]}
        self.assertIsNone(extract(payload)["product_data"]["approval"]["priority_review"])


class TestApprovalPathway(unittest.TestCase):
    def _payload(self, app_type, **extra):
        return {"application_type_code": app_type,
                "products": [dict({"strength": "500MG", "dosage_form_route": "TABLET;ORAL",
                                   "active_ingredients": "IBUPROFEN"}, **extra)]}

    def test_anda_is_generic(self):
        self.assertEqual(extract(self._payload("ANDA"))["product_data"]["approval"]["pathway"],
                         "generic")

    def test_nda_is_innovator(self):
        self.assertEqual(extract(self._payload("NDA"))["product_data"]["approval"]["pathway"],
                         "innovator")

    def test_505b2_nda_is_hybrid(self):
        result = extract(self._payload("NDA"),
                         document_text="submitted pursuant to section 505(b)(2) of the Act")
        self.assertEqual(result["product_data"]["approval"]["pathway"], "hybrid")

    def test_reference_listed_drug_is_pulled_from_the_letter(self):
        letter = ("The reference listed drug (RLD) upon which you have based your ANDA, "
                  "Epzicom Tablets, 600 mg (base)/300 mg of VIIV Healthcare Company, is "
                  "subject to periods of patent protection.")
        result = extract(self._payload("ANDA"), document_text=letter)
        self.assertEqual(result["product_data"]["approval"]["reference_product"],
                         "Epzicom Tablets, 600 mg (base)/300 mg")

    def test_rld_phrase_without_a_named_drug_yields_null_not_a_stopword(self):
        letter = "The reference listed drug (RLD) upon which you have based your ANDA."
        self.assertIsNone(
            extract(self._payload("ANDA"), document_text=letter)["product_data"]["approval"]["reference_product"])


class TestExtractContract(unittest.TestCase):
    def test_empty_input_still_returns_the_full_shape(self):
        result = extract({})
        self.assertEqual(set(result), {"columns", "product_data", "provenance"})
        self.assertEqual(result["product_data"]["_schema_version"], "1")
        for key in ("substance", "presentations", "indications", "approval",
                    "pivotal_evidence", "key_risks", "excipients"):
            self.assertIn(key, result["product_data"])

    def test_identical_input_gives_byte_identical_output(self):
        payload = TestApprovalHistoryShapes.WEB_SHAPE
        self.assertEqual(extract(payload), extract(payload))


if __name__ == "__main__":
    unittest.main()

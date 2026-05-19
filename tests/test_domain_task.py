import unittest
from types import SimpleNamespace

from dspy_domain_train.domain_task import build_error_report, parse_prediction
from dspy_domain_train.training import ErrorCase


class ParsePredictionTests(unittest.TestCase):
    def test_filters_invalid_domains_and_removes_none_when_specific_exists(
        self,
    ) -> None:
        pred = SimpleNamespace(domains=["Taxi", "none", "invalid", "taxi"])

        self.assertEqual(parse_prediction(pred), ["taxi"])

    def test_falls_back_to_none_for_empty_or_invalid_prediction(self) -> None:
        pred = SimpleNamespace(domains=[])

        self.assertEqual(parse_prediction(pred), ["none"])

    def test_parses_legacy_json_string_prediction(self) -> None:
        pred = SimpleNamespace(domains_json='prefix {"domains": ["Hotel", "none"]}')

        self.assertEqual(parse_prediction(pred), ["hotel"])


class ErrorReportTests(unittest.TestCase):
    def test_builds_general_guidance_for_possessive_and_location_errors(self) -> None:
        errors: list[ErrorCase] = [
            {
                "history": "",
                "turn": "Find a museum near my hotel.",
                "gt": ["attraction"],
                "pred": ["attraction", "hotel"],
            }
        ]

        report = build_error_report(errors)

        self.assertIn("possessive pronouns", report)
        self.assertIn("location anchors", report)
        self.assertIn("Extra domains predicted", report)


if __name__ == "__main__":
    unittest.main()

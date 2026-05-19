import unittest
from types import SimpleNamespace

import dspy

from dspy_domain_train.domain_task import (
    domain_metric,
    normalize_domains,
    parse_prediction,
)


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


class NormalizeDomainsTests(unittest.TestCase):
    def test_deduplicates_and_lowercases(self) -> None:
        result = normalize_domains(["Hotel", "HOTEL", "taxi"])
        self.assertEqual(result, ["hotel", "taxi"])

    def test_none_exclusive(self) -> None:
        self.assertEqual(normalize_domains(["none", "hotel"]), ["hotel"])

    def test_empty_returns_none(self) -> None:
        self.assertEqual(normalize_domains([]), ["none"])

    def test_filters_invalid_labels(self) -> None:
        self.assertEqual(normalize_domains(["invalid", "bus"]), ["bus"])


class DomainMetricTests(unittest.TestCase):
    def test_correct_prediction_returns_1(self) -> None:
        example = dspy.Example(
            dialogue_context="", turn="Book a taxi.", domains=["taxi"]
        ).with_inputs("dialogue_context", "turn")
        pred = dspy.Prediction(domains=["taxi"])

        score = domain_metric(example, pred)
        self.assertEqual(score, 1.0)

    def test_wrong_prediction_returns_0(self) -> None:
        example = dspy.Example(
            dialogue_context="", turn="Book a taxi.", domains=["taxi"]
        ).with_inputs("dialogue_context", "turn")
        pred = dspy.Prediction(domains=["hotel"])

        score = domain_metric(example, pred)
        self.assertEqual(score, 0.0)

    def test_feedback_returned_when_pred_name_set(self) -> None:
        example = dspy.Example(
            dialogue_context="", turn="Book a taxi.", domains=["taxi"]
        ).with_inputs("dialogue_context", "turn")
        pred = dspy.Prediction(domains=["hotel"])

        result = domain_metric(example, pred, pred_name="predict")
        assert isinstance(result, dspy.Prediction)
        self.assertEqual(result.score, 0.0)
        self.assertIn("Missing", result.feedback)
        self.assertIn("Extra", result.feedback)

    def test_correct_feedback(self) -> None:
        example = dspy.Example(
            dialogue_context="", turn="Hi", domains=["none"]
        ).with_inputs("dialogue_context", "turn")
        pred = dspy.Prediction(domains=["none"])

        result = domain_metric(example, pred, pred_name="predict")
        assert isinstance(result, dspy.Prediction)
        self.assertEqual(result.score, 1.0)
        self.assertIn("Correct", result.feedback)


if __name__ == "__main__":
    unittest.main()

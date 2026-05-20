import unittest

import dspy

from dspy_domain_train.domain_task import (
    canonical_gold,
    canonical_prediction,
    domain_metric,
)


class CanonicalGoldTests(unittest.TestCase):
    def test_lowercases_and_deduplicates(self) -> None:
        result = canonical_gold(["Hotel", "HOTEL", "taxi"])
        self.assertEqual(result, frozenset({"hotel", "taxi"}))

    def test_empty_returns_none(self) -> None:
        self.assertEqual(canonical_gold([]), frozenset({"none"}))

    def test_raises_on_invalid_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid gold domain labels"):
            canonical_gold(["invalid", "bus"])

    def test_raises_on_mixed_none(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot mix 'none'"):
            canonical_gold(["none", "hotel"])


class CanonicalPredictionTests(unittest.TestCase):
    def test_valid_labels(self) -> None:
        self.assertEqual(
            canonical_prediction(["Hotel", "taxi"]), frozenset({"hotel", "taxi"})
        )

    def test_invalid_label_returns_none(self) -> None:
        self.assertIsNone(canonical_prediction(["invalid", "hotel"]))

    def test_empty_returns_none_label(self) -> None:
        self.assertEqual(canonical_prediction([]), frozenset({"none"}))

    def test_deduplicates(self) -> None:
        self.assertEqual(canonical_prediction(["taxi", "Taxi"]), frozenset({"taxi"}))

    def test_mixed_none_returns_none(self) -> None:
        self.assertIsNone(canonical_prediction(["none", "hotel"]))


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

    def test_invalid_prediction_returns_0(self) -> None:
        example = dspy.Example(
            dialogue_context="", turn="Book a taxi.", domains=["taxi"]
        ).with_inputs("dialogue_context", "turn")
        pred = dspy.Prediction(domains=["invalid_label"])

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
        self.assertIn("Gold", result.feedback)
        self.assertIn("Predicted", result.feedback)

    def test_correct_feedback(self) -> None:
        example = dspy.Example(
            dialogue_context="", turn="Hi", domains=["none"]
        ).with_inputs("dialogue_context", "turn")
        pred = dspy.Prediction(domains=["none"])

        result = domain_metric(example, pred, pred_name="predict")
        assert isinstance(result, dspy.Prediction)
        self.assertEqual(result.score, 1.0)
        self.assertIn("Missing: []", result.feedback)
        self.assertIn("Extra: []", result.feedback)


if __name__ == "__main__":
    unittest.main()

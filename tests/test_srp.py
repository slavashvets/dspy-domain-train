import unittest
from types import SimpleNamespace
from unittest.mock import patch

import dspy

from dspy_domain_train.srp import SRP


def metric(example: dspy.Example, pred: dspy.Prediction, **_: object) -> float:
    return float(set(example.domains) == set(getattr(pred, "domains", [])))


def row(
    turn: str,
    gold: list[str],
    predicted: list[str],
    score: float,
) -> tuple[dspy.Example, dspy.Prediction, float]:
    example = dspy.Example(
        dialogue_context="",
        turn=turn,
        domains=gold,
    ).with_inputs("dialogue_context", "turn")
    return example, dspy.Prediction(domains=predicted), score


class SplitSRP(SRP):
    def __init__(self, results_by_id: dict[int, SimpleNamespace]) -> None:
        super().__init__(metric=metric)
        self.results_by_id = results_by_id
        self.calls: list[int] = []

    def _evaluate(
        self,
        program: dspy.Module,
        evalset: list[dspy.Example],
        eval_kwargs: dict,
    ) -> SimpleNamespace:
        del program, eval_kwargs
        self.calls.append(id(evalset))
        return self.results_by_id[id(evalset)]


class SRPTests(unittest.TestCase):
    def test_score_and_results_scores_val_but_returns_workset_results(self) -> None:
        workset = [dspy.Example(turn="work")]
        valset = [dspy.Example(turn="val")]
        work_rows = [row("work error", ["restaurant"], ["hotel"], 0.0)]
        val_rows = [row("val ok", ["taxi"], ["taxi"], 1.0)]
        optimizer = SplitSRP(
            {
                id(workset): SimpleNamespace(score=12.0, results=work_rows),
                id(valset): SimpleNamespace(score=98.0, results=val_rows),
            }
        )

        score, results = optimizer._score_and_results(
            dspy.Module(),
            workset=workset,
            score_set=valset,
            eval_kwargs={},
        )

        self.assertEqual(score, 98.0)
        self.assertIs(results, work_rows)
        self.assertEqual(optimizer.calls, [id(workset), id(valset)])

    def test_structural_error_category_does_not_parse_turn_words(self) -> None:
        category = SRP._error_category(
            inputs={"turn": "taxi from whatever arbitrary endpoint"},
            gold={"taxi"},
            predicted={"hotel"},
            missing=["taxi"],
            extra=["hotel"],
        )

        self.assertEqual(category, "label_substitution")

    def test_result_metrics_include_secondary_metrics(self) -> None:
        results = [
            row("ok", ["taxi"], ["taxi"], 1.0),
            row("partial", ["taxi", "hotel"], ["taxi"], 0.0),
            row("none", ["none"], ["hotel"], 0.0),
        ]

        metrics = SRP._result_metrics(results)

        self.assertEqual(metrics["total"], 3)
        self.assertAlmostEqual(metrics["exact"], 1 / 3)
        self.assertGreater(metrics["mean_jaccard"], metrics["exact"])
        self.assertIn("taxi", metrics["per_domain_f1"])
        self.assertEqual(metrics["none_exact"], 0.0)

    def test_error_fingerprint_tracks_examples_not_wrong_predictions(self) -> None:
        example = dspy.Example(
            dialogue_context="",
            turn="need a ride",
            domains=["taxi"],
            example_id="dev:7",
        ).with_inputs("dialogue_context", "turn")

        first = [(example, dspy.Prediction(domains=["hotel"]), 0.0)]
        second = [(example, dspy.Prediction(domains=["restaurant"]), 0.0)]

        self.assertEqual(SRP._error_fingerprint(first), SRP._error_fingerprint(second))
        self.assertEqual(SRP._error_ids(first), ["dev:7"])

    def test_evaluate_aborts_infrastructure_errors_instead_of_scoring_zero(
        self,
    ) -> None:
        optimizer = SRP(metric=metric)
        evalset = [dspy.Example(turn="x")]
        program = dspy.Module()

        with patch("dspy_domain_train.srp.dspy.Evaluate") as evaluate:
            evaluate.return_value.return_value = SimpleNamespace(
                score=100.0, results=[]
            )

            optimizer._evaluate(program, evalset, {"num_threads": 4})

        evaluate.assert_called_once()
        self.assertEqual(evaluate.call_args.kwargs["max_errors"], 1)
        self.assertEqual(evaluate.call_args.kwargs["num_threads"], 4)


if __name__ == "__main__":
    unittest.main()

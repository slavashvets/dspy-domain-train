from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from random import Random
from typing import Any

import dspy
from dspy.teleprompt.teleprompt import Teleprompter
from pydantic import BaseModel, Field


class SRPErrorCase(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)
    gold: dict[str, Any] = Field(default_factory=dict)
    prediction: dict[str, Any] = Field(default_factory=dict)
    score: float
    feedback: str | None = None


class SRPRefinerSig(dspy.Signature):
    """Generate concise rules to append to a DSPy predictor instruction.

    Produce general rules that address the observed errors. Do not rewrite the
    full instruction. Do not include worked examples. Return 1 to 3 rules.
    """

    current_instruction: str = dspy.InputField(
        desc="Current instruction of the predictor being optimized."
    )
    error_cases: list[SRPErrorCase] = dspy.InputField(
        desc="Representative errors from the current program."
    )
    rules: list[str] = dspy.OutputField(
        desc="1 to 3 concise, general, non-duplicative rules to append."
    )


class SRP(Teleprompter):
    def __init__(
        self,
        *,
        metric: Callable[..., Any],
        prompt_model: dspy.LM | None = None,
        max_iters: int = 6,
        patience: int = 2,
        max_examples: int | None = None,
        max_error_cases: int = 12,
        num_threads: int | None = None,
        seed: int = 0,
        target_predictor: str | None = None,
        display_progress: bool = False,
    ) -> None:
        if max_iters < 1:
            raise ValueError("max_iters must be >= 1")
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if max_examples is not None and max_examples < 1:
            raise ValueError("max_examples must be >= 1")
        if max_error_cases < 1:
            raise ValueError("max_error_cases must be >= 1")

        self.metric = metric
        self.prompt_model = prompt_model
        self.max_iters = max_iters
        self.patience = patience
        self.max_examples = max_examples
        self.max_error_cases = max_error_cases
        self.num_threads = num_threads
        self.seed = seed
        self.target_predictor = target_predictor
        self.display_progress = display_progress
        self.refiner = dspy.Predict(SRPRefinerSig)

    def compile(
        self,
        student: dspy.Module,
        *,
        trainset: list[dspy.Example],
        teacher: dspy.Module | None = None,
        valset: list[dspy.Example] | None = None,
        **kwargs: Any,
    ) -> dspy.Module:
        del teacher

        if not trainset:
            raise ValueError("trainset must not be empty")

        eval_kwargs: dict[str, Any] = {
            "num_threads": self.num_threads,
            "display_progress": self.display_progress,
            "display_table": False,
        }
        eval_kwargs.update(kwargs.pop("eval_kwargs", {}) or {})
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected compile kwargs: {unknown}")

        workset = self._select_examples(trainset)
        score_set = valset if valset is not None else workset

        current = student.deepcopy()
        predictor_name, _ = self._target(current)

        best_score, current_results = self._score_and_results(
            current,
            workset=workset,
            score_set=score_set,
            eval_kwargs=eval_kwargs,
        )
        best = current.deepcopy()

        candidate_programs: list[dict[str, Any]] = [
            self._candidate_record(
                iteration=0,
                score=best_score,
                program=current,
                rules=[],
                accepted=True,
            )
        ]

        if best_score >= 100.0:
            best.candidate_programs = candidate_programs
            best.trial_logs = {
                "best_score": best_score,
                "stopped_reason": "perfect_score",
                "max_iters": self.max_iters,
                "patience": self.patience,
            }
            return best

        no_improvement = 0
        stopped_reason = "max_iters"

        for iteration in range(1, self.max_iters + 1):
            errors = self._collect_errors(current_results, predictor_name)
            if not errors:
                stopped_reason = "no_errors"
                break

            rules = self._propose_rules(self._instruction(current), errors)
            if not rules:
                stopped_reason = "no_rules"
                break

            candidate = self._with_appended_rules(current, rules)
            candidate_score, candidate_results = self._score_and_results(
                candidate,
                workset=workset,
                score_set=score_set,
                eval_kwargs=eval_kwargs,
            )

            accepted = candidate_score > best_score
            candidate_programs.append(
                self._candidate_record(
                    iteration=iteration,
                    score=candidate_score,
                    program=candidate,
                    rules=rules,
                    accepted=accepted,
                )
            )

            if accepted:
                current = candidate
                current_results = candidate_results
                best = candidate.deepcopy()
                best_score = candidate_score
                no_improvement = 0

                if best_score >= 100.0:
                    stopped_reason = "perfect_score"
                    break
            else:
                no_improvement += 1
                if no_improvement >= self.patience:
                    stopped_reason = "patience"
                    break

        best.candidate_programs = candidate_programs
        best.trial_logs = {
            "best_score": best_score,
            "stopped_reason": stopped_reason,
            "max_iters": self.max_iters,
            "patience": self.patience,
        }
        return best

    def _select_examples(self, trainset: Sequence[dspy.Example]) -> list[dspy.Example]:
        examples = list(trainset)
        if self.max_examples is None or len(examples) <= self.max_examples:
            return examples
        return Random(self.seed).sample(examples, self.max_examples)

    def _evaluate(
        self,
        program: dspy.Module,
        evalset: list[dspy.Example],
        eval_kwargs: dict[str, Any],
    ) -> Any:
        evaluator = dspy.Evaluate(
            devset=evalset,
            metric=self.metric,
            **eval_kwargs,
        )
        return evaluator(program)

    def _score_and_results(
        self,
        program: dspy.Module,
        *,
        workset: list[dspy.Example],
        score_set: list[dspy.Example],
        eval_kwargs: dict[str, Any],
    ) -> tuple[float, list[tuple[dspy.Example, dspy.Prediction, Any]]]:
        work_result = self._evaluate(program, workset, eval_kwargs)
        if score_set is workset:
            return float(work_result.score), work_result.results

        score_result = self._evaluate(program, score_set, eval_kwargs)
        return float(score_result.score), work_result.results

    def _collect_errors(
        self,
        results: list[tuple[dspy.Example, dspy.Prediction, Any]],
        predictor_name: str,
    ) -> list[SRPErrorCase]:
        errors: list[SRPErrorCase] = []

        for example, prediction, score in results:
            score_value = self._score_value(score)
            if score_value >= 1.0:
                continue

            errors.append(
                SRPErrorCase(
                    inputs=dict(example.inputs()),
                    gold=dict(example.labels()),
                    prediction=dict(prediction),
                    score=score_value,
                    feedback=self._metric_feedback(example, prediction, predictor_name),
                )
            )

            if len(errors) >= self.max_error_cases:
                break

        return errors

    def _propose_rules(
        self,
        current_instruction: str,
        errors: list[SRPErrorCase],
    ) -> list[str]:
        context = (
            dspy.context(lm=self.prompt_model)
            if self.prompt_model is not None
            else nullcontext()
        )
        with context:
            prediction = self.refiner(
                current_instruction=current_instruction,
                error_cases=errors,
                config={"temperature": 1.0},
            )

        return self._clean_rules(getattr(prediction, "rules", []), current_instruction)

    def _with_appended_rules(
        self,
        program: dspy.Module,
        rules: Sequence[str],
    ) -> dspy.Module:
        updated = program.deepcopy()
        _, predictor = self._target(updated)
        predictor.signature = predictor.signature.with_instructions(
            self._append_rule_block(predictor.signature.instructions or "", rules)
        )
        return updated

    def _target(self, program: dspy.Module) -> tuple[str, dspy.Predict]:
        predictors = list(program.named_predictors())
        if not predictors:
            raise ValueError("SRP requires at least one dspy.Predict in the program")

        if self.target_predictor is None:
            if len(predictors) != 1:
                names = ", ".join(name for name, _ in predictors)
                raise ValueError(
                    "SRP needs target_predictor when the program has multiple "
                    f"predictors: {names}"
                )
            return predictors[0]

        for name, predictor in predictors:
            if name == self.target_predictor:
                return name, predictor

        names = ", ".join(name for name, _ in predictors)
        raise ValueError(
            f"Unknown target_predictor={self.target_predictor!r}. Available: {names}"
        )

    def _instruction(self, program: dspy.Module) -> str:
        _, predictor = self._target(program)
        return predictor.signature.instructions or ""

    def _candidate_record(
        self,
        *,
        iteration: int,
        score: float,
        program: dspy.Module,
        rules: Sequence[str],
        accepted: bool,
    ) -> dict[str, Any]:
        return {
            "iteration": iteration,
            "score": score,
            "program": program.deepcopy(),
            "rules": list(rules),
            "accepted": accepted,
            "instruction": self._instruction(program),
        }

    def _metric_feedback(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        predictor_name: str,
    ) -> str | None:
        result = self.metric(
            example,
            prediction,
            pred_name=predictor_name,
            pred_trace=None,
        )
        value = getattr(result, "feedback", None)
        return str(value) if value else None

    @staticmethod
    def _score_value(score: Any) -> float:
        if isinstance(score, bool):
            return float(score)
        if isinstance(score, (int, float)):
            return float(score)
        if hasattr(score, "score"):
            return float(score.score)
        raise TypeError(
            f"Metric score must be numeric, bool, or have .score: {score!r}"
        )

    @staticmethod
    def _clean_rules(raw_rules: Any, current_instruction: str) -> list[str]:
        if isinstance(raw_rules, str):
            raw_rules = [raw_rules]

        rules: list[str] = []
        seen: set[str] = set()

        for raw_rule in raw_rules or []:
            rule = " ".join(str(raw_rule).strip().split())
            if not rule:
                continue
            key = rule.lower()
            if key in seen or rule in current_instruction:
                continue
            seen.add(key)
            rules.append(rule)
            if len(rules) == 3:
                break

        return rules

    @staticmethod
    def _append_rule_block(instruction: str, rules: Sequence[str]) -> str:
        cleaned = [" ".join(str(rule).strip().split()) for rule in rules]
        cleaned = [rule for rule in cleaned if rule]

        if not cleaned:
            return instruction.strip()

        block = "\n".join(f"- {rule}" for rule in cleaned)
        parts = [instruction.strip(), f"Additional rules:\n{block}"]
        return "\n\n".join(part for part in parts if part)

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from random import Random
from typing import Any

import dspy
from dspy.teleprompt.teleprompt import Teleprompter
from pydantic import BaseModel, Field

type EvalRows = list[tuple[dspy.Example, dspy.Prediction, Any]]
type CandidateEval = tuple[float, dspy.Module, EvalRows, dict[str, Any]]

SERVICE_DOMAINS = (
    "attraction",
    "bus",
    "hospital",
    "hotel",
    "police",
    "restaurant",
    "taxi",
    "train",
    "none",
)
SERVICE_DOMAIN_SET = frozenset(SERVICE_DOMAINS)


class SRPErrorCase(BaseModel):
    index: int = 0
    inputs: dict[str, Any] = Field(default_factory=dict)
    gold: dict[str, Any] = Field(default_factory=dict)
    prediction: dict[str, Any] = Field(default_factory=dict)
    score: float
    missing: list[str] = Field(default_factory=list)
    extra: list[str] = Field(default_factory=list)
    category: str = "other"
    feedback: str | None = None


class SRPRefinerSig(dspy.Signature):
    """Revise DSPy predictor instructions using metric feedback.

    Generate mode-diverse complete instruction candidates. Preserve the task
    output contract. Do not include dataset-specific examples or quote the
    feedback examples.
    """

    current_instruction: str = dspy.InputField(
        desc="Current instruction of the predictor being optimized."
    )
    feedback_report: str = dspy.InputField(
        desc="Aggregate metric feedback and representative errors."
    )
    candidate_instructions: list[str] = dspy.OutputField(
        desc=(
            "Several complete revised instruction candidates, best first. "
            "Use different strategies such as minimal patch, pruning, "
            "context-first, label-specific, multi-label recall, and "
            "high-precision variants."
        )
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
        num_candidates: int = 5,
        candidate_retries: int = 1,
        proposal_temperature: float = 1.0,
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
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")
        if candidate_retries < 1:
            raise ValueError("candidate_retries must be >= 1")
        if proposal_temperature < 0.0:
            raise ValueError("proposal_temperature must be >= 0")

        self.metric = metric
        self.prompt_model = prompt_model
        self.max_iters = max_iters
        self.patience = patience
        self.max_examples = max_examples
        self.max_error_cases = max_error_cases
        self.num_candidates = num_candidates
        self.candidate_retries = candidate_retries
        self.proposal_temperature = proposal_temperature
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
                accepted=True,
                results=current_results,
            )
        ]

        if best_score >= 100.0:
            return self._finish_perfect_baseline(
                best,
                best_score,
                candidate_programs,
                score_source="valset" if valset is not None else "workset",
            )

        no_improvement = 0
        stopped_reason = "max_iters"

        for iteration in range(1, self.max_iters + 1):
            errors = self._collect_errors(current_results, predictor_name)
            if not errors:
                stopped_reason = "no_errors"
                break

            current_instruction = self._instruction(current)
            instructions = self._propose_instructions(
                current_instruction,
                errors,
                iteration,
            )
            if not instructions:
                stopped_reason = "no_rules"
                break

            candidates = self._score_candidates(
                current,
                instructions=instructions,
                workset=workset,
                score_set=score_set,
                eval_kwargs=eval_kwargs,
            )
            candidate_score, candidate, candidate_results, _ = candidates[0]
            accepted = candidate_score > best_score

            for rank, (score, program, results, metrics) in enumerate(candidates, 1):
                candidate_programs.append(
                    self._candidate_record(
                        iteration=iteration,
                        score=score,
                        program=program,
                        accepted=accepted and rank == 1,
                        rank=rank,
                        results=results,
                        metrics=metrics,
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

        frontier_programs = self._frontier_records(candidate_programs, best_score)
        best.candidate_programs = candidate_programs
        best.frontier_programs = frontier_programs
        best.trial_logs = {
            "best_score": best_score,
            "stopped_reason": stopped_reason,
            "max_iters": self.max_iters,
            "patience": self.patience,
            "num_candidates": self.num_candidates,
            "candidate_retries": self.candidate_retries,
            "feedback_source": "workset",
            "score_source": "valset" if valset is not None else "workset",
            "frontier_size": len(frontier_programs),
        }
        return best

    def _score_candidates(
        self,
        current: dspy.Module,
        *,
        instructions: Sequence[str],
        workset: list[dspy.Example],
        score_set: list[dspy.Example],
        eval_kwargs: dict[str, Any],
    ) -> list[CandidateEval]:
        candidates: list[CandidateEval] = []

        for instruction in instructions:
            candidate = self._with_instruction(current, instruction)
            score, results = self._score_and_results(
                candidate,
                workset=workset,
                score_set=score_set,
                eval_kwargs=eval_kwargs,
            )
            metrics = self._result_metrics(results)
            candidates.append((score, candidate, results, metrics))

        candidates.sort(
            key=lambda item: (
                item[0],
                item[3]["macro_f1"],
                item[3]["mean_jaccard"],
                -item[3]["invalid_output_rate"],
                -len(self._instruction(item[1]).split()),
            ),
            reverse=True,
        )
        return candidates

    def _finish_perfect_baseline(
        self,
        best: dspy.Module,
        best_score: float,
        candidate_programs: list[dict[str, Any]],
        score_source: str,
    ) -> dspy.Module:
        frontier_programs = self._frontier_records(candidate_programs, best_score)
        best.candidate_programs = candidate_programs
        best.frontier_programs = frontier_programs
        best.trial_logs = {
            "best_score": best_score,
            "stopped_reason": "perfect_score",
            "max_iters": self.max_iters,
            "patience": self.patience,
            "num_candidates": self.num_candidates,
            "candidate_retries": self.candidate_retries,
            "feedback_source": "workset",
            "score_source": score_source,
            "frontier_size": len(frontier_programs),
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
        safe_eval_kwargs = dict(eval_kwargs)
        safe_eval_kwargs.setdefault("max_errors", 1)
        evaluator = dspy.Evaluate(
            devset=evalset,
            metric=self.metric,
            **safe_eval_kwargs,
        )
        return evaluator(program)

    def _score_and_results(
        self,
        program: dspy.Module,
        *,
        workset: list[dspy.Example],
        score_set: list[dspy.Example],
        eval_kwargs: dict[str, Any],
    ) -> tuple[float, EvalRows]:
        work_result = self._evaluate(program, workset, eval_kwargs)
        if score_set is workset:
            return float(work_result.score), work_result.results

        score_result = self._evaluate(program, score_set, eval_kwargs)
        return float(score_result.score), work_result.results

    def _collect_errors(
        self,
        results: EvalRows,
        predictor_name: str,
    ) -> list[SRPErrorCase]:
        errors: list[SRPErrorCase] = []

        for index, (example, prediction, score) in enumerate(results):
            score_value = self._score_value(score)
            if score_value >= 1.0:
                continue

            gold = self._gold_dict(example)
            pred = dict(prediction)
            gold_domains = self._domains(gold.get("domains"), empty_as_none=True)
            predicted_domains = self._domains(
                pred.get("domains"),
                empty_as_none=True,
            )
            missing = sorted(gold_domains - predicted_domains)
            extra = sorted(predicted_domains - gold_domains)

            errors.append(
                SRPErrorCase(
                    index=index,
                    inputs=dict(example.inputs()),
                    gold=gold,
                    prediction=pred,
                    score=score_value,
                    missing=missing,
                    extra=extra,
                    category=self._error_category(
                        inputs=dict(example.inputs()),
                        gold=gold_domains,
                        predicted=predicted_domains,
                        missing=missing,
                        extra=extra,
                    ),
                    feedback=self._metric_feedback(example, prediction, predictor_name),
                )
            )

        return self._select_error_cases(errors)

    def _propose_instructions(
        self,
        current_instruction: str,
        errors: list[SRPErrorCase],
        iteration: int = 0,
    ) -> list[str]:
        feedback_report = self._feedback_report(errors)
        context = (
            dspy.context(lm=self.prompt_model)
            if self.prompt_model is not None
            else nullcontext()
        )
        candidates: list[str] = []
        with context:
            for retry in range(self.candidate_retries):
                prediction = self.refiner(
                    current_instruction=current_instruction,
                    feedback_report=feedback_report,
                    config={
                        "temperature": self.proposal_temperature,
                        "rollout_id": iteration * 1000 + retry,
                    },
                )
                candidates.extend(getattr(prediction, "candidate_instructions", []))

        return self._clean_instructions(candidates, current_instruction)

    def _with_instruction(
        self,
        program: dspy.Module,
        instruction: str,
    ) -> dspy.Module:
        updated = program.deepcopy()
        _, predictor = self._target(updated)
        predictor.signature = predictor.signature.with_instructions(instruction)
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
        accepted: bool,
        rank: int = 0,
        results: EvalRows | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        instruction = self._instruction(program)
        result_metrics = metrics
        if result_metrics is None and results is not None:
            result_metrics = self._result_metrics(results)

        return {
            "iteration": iteration,
            "rank": rank,
            "score": score,
            "program": program.deepcopy(),
            "accepted": accepted,
            "instruction": instruction,
            "instruction_words": len(instruction.split()),
            "metrics": result_metrics or {},
            "error_fingerprint": self._error_fingerprint(results or []),
            "error_ids": self._error_ids(results or []),
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
    def _domains(value: Any, *, empty_as_none: bool = False) -> set[str]:
        if value is None:
            return {"none"} if empty_as_none else set()
        if isinstance(value, str):
            values = {value.lower().strip()} if value.strip() else set()
            return values or ({"none"} if empty_as_none else set())
        if isinstance(value, Sequence):
            values = {
                str(item).lower().strip() for item in value if str(item).lower().strip()
            }
            return values or ({"none"} if empty_as_none else set())
        text = str(value).lower().strip()
        if not text:
            return {"none"} if empty_as_none else set()
        return {text}

    def _feedback_report(self, errors: Sequence[SRPErrorCase]) -> str:
        missing: Counter[str] = Counter()
        extra: Counter[str] = Counter()
        categories: Counter[str] = Counter(error.category for error in errors)

        for error in errors:
            gold = self._domains(error.gold.get("domains"), empty_as_none=True)
            predicted = self._domains(
                error.prediction.get("domains"),
                empty_as_none=True,
            )
            missing.update(gold - predicted)
            extra.update(predicted - gold)

        lines: list[str] = ["Score summary:"]
        lines.append(f"- Selected feedback errors: {len(errors)}")
        lines.append(f"- Max error cases: {self.max_error_cases}")
        if missing:
            lines.append(f"- Missing labels: {dict(sorted(missing.items()))}")
        if extra:
            lines.append(f"- Extra labels: {dict(sorted(extra.items()))}")
        if categories:
            lines.append(f"- Error categories: {dict(sorted(categories.items()))}")

        lines.append("")
        lines.append("Candidate generation request:")
        lines.append(f"- Return up to {self.num_candidates} complete instructions.")
        lines.append("- Use mode-diverse search, not small wording variants.")
        lines.append(
            "- Cover minimal patch, pruning, context-first, label-specific, "
            "multi-label recall, and high-precision variants when relevant."
        )
        lines.append(
            "- Prefer 80-140 words; include one longer candidate only if needed."
        )

        lines.append("")
        lines.append("Error clusters:")
        for category, grouped in self._group_errors(errors).items():
            lines.append(f"- {category}: {len(grouped)}")
            for error in grouped[:3]:
                lines.append(f"  Example #{error.index}:")
                lines.append(f"    Inputs: {error.inputs}")
                lines.append(f"    Gold: {error.gold}")
                lines.append(f"    Predicted: {error.prediction}")
                lines.append(f"    Missing: {error.missing} | Extra: {error.extra}")
                if error.feedback:
                    lines.append(f"    Metric feedback: {error.feedback}")

        return "\n".join(lines)

    @staticmethod
    def _group_errors(
        errors: Sequence[SRPErrorCase],
    ) -> dict[str, list[SRPErrorCase]]:
        grouped: dict[str, list[SRPErrorCase]] = defaultdict(list)
        for error in errors:
            grouped[error.category].append(error)
        return dict(sorted(grouped.items()))

    def _select_error_cases(
        self,
        errors: Sequence[SRPErrorCase],
    ) -> list[SRPErrorCase]:
        if len(errors) <= self.max_error_cases:
            return list(errors)

        selected: list[SRPErrorCase] = []
        selected_indexes: set[int] = set()

        def add(error: SRPErrorCase) -> None:
            if len(selected) >= self.max_error_cases:
                return
            if error.index in selected_indexes:
                return
            selected.append(error)
            selected_indexes.add(error.index)

        category_priority = (
            "invalid_output",
            "none_mismatch",
            "multi_domain",
            "label_substitution",
            "missing_only",
            "extra_only",
            "partial_score",
        )
        by_category = self._group_errors(errors)
        for category in category_priority:
            for error in by_category.get(category, [])[:1]:
                add(error)

        labels = sorted(
            {label for error in errors for label in error.missing + error.extra}
        )
        for label in labels:
            for error in errors:
                if label in error.missing or label in error.extra:
                    add(error)
                    break

        for error in errors:
            add(error)

        return selected

    @classmethod
    def _error_category(
        cls,
        *,
        inputs: dict[str, Any],
        gold: set[str],
        predicted: set[str],
        missing: list[str],
        extra: list[str],
    ) -> str:
        del inputs

        if cls._prediction_invalid(predicted) or "<invalid-output>" in extra:
            return "invalid_output"
        if gold == {"none"} or predicted == {"none"}:
            return "none_mismatch"
        if len(gold - {"none"}) > 1 or len(predicted - {"none"}) > 1:
            return "multi_domain"
        if missing and extra:
            return "label_substitution"
        if missing:
            return "missing_only"
        if extra:
            return "extra_only"
        return "partial_score"

    @classmethod
    def _prediction_invalid(cls, predicted: set[str]) -> bool:
        if not predicted:
            return False
        if predicted - SERVICE_DOMAIN_SET:
            return True
        return "none" in predicted and len(predicted) > 1

    @classmethod
    def _result_metrics(cls, results: EvalRows) -> dict[str, Any]:
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "exact": 0.0,
                "mean_jaccard": 0.0,
                "macro_f1": 0.0,
                "invalid_output_rate": 0.0,
                "multi_domain_exact": None,
                "none_exact": None,
                "per_domain_f1": {},
            }

        exact_count = 0
        invalid_count = 0
        jaccards: list[float] = []
        counts = {domain: {"tp": 0, "fp": 0, "fn": 0} for domain in SERVICE_DOMAINS}
        multi_total = 0
        multi_exact = 0
        none_total = 0
        none_exact = 0

        for example, prediction, score in results:
            score_value = cls._score_value(score)
            if score_value >= 1.0:
                exact_count += 1

            gold = cls._domains(
                cls._gold_dict(example).get("domains"),
                empty_as_none=True,
            )
            predicted = cls._domains(
                dict(prediction).get("domains"),
                empty_as_none=True,
            )
            if cls._prediction_invalid(predicted):
                invalid_count += 1

            union = gold | predicted
            jaccards.append((len(gold & predicted) / len(union)) if union else 1.0)

            if len(gold - {"none"}) > 1:
                multi_total += 1
                if score_value >= 1.0:
                    multi_exact += 1
            if gold == {"none"}:
                none_total += 1
                if score_value >= 1.0:
                    none_exact += 1

            for domain in SERVICE_DOMAINS:
                in_gold = domain in gold
                in_pred = domain in predicted
                if in_gold and in_pred:
                    counts[domain]["tp"] += 1
                elif in_pred:
                    counts[domain]["fp"] += 1
                elif in_gold:
                    counts[domain]["fn"] += 1

        per_domain_f1: dict[str, float] = {}
        for domain, values in counts.items():
            tp = values["tp"]
            fp = values["fp"]
            fn = values["fn"]
            denom = (2 * tp) + fp + fn
            if denom:
                per_domain_f1[domain] = (2 * tp) / denom

        macro_values = list(per_domain_f1.values())
        return {
            "total": total,
            "exact": exact_count / total,
            "mean_jaccard": sum(jaccards) / total,
            "macro_f1": (sum(macro_values) / len(macro_values))
            if macro_values
            else 0.0,
            "invalid_output_rate": invalid_count / total,
            "multi_domain_exact": (multi_exact / multi_total) if multi_total else None,
            "none_exact": (none_exact / none_total) if none_total else None,
            "per_domain_f1": dict(sorted(per_domain_f1.items())),
        }

    @classmethod
    def _error_fingerprint(cls, results: EvalRows) -> list[str]:
        return [
            hashlib.sha1(identity.encode()).hexdigest()[:16]
            for identity in cls._error_ids(results)
        ]

    @classmethod
    def _error_ids(cls, results: EvalRows) -> list[str]:
        ids: list[str] = []
        for example, prediction, score in results:
            del prediction
            if cls._score_value(score) >= 1.0:
                continue
            example_id = getattr(example, "example_id", None)
            if example_id:
                ids.append(str(example_id))
                continue
            payload = json.dumps(
                {
                    "inputs": dict(example.inputs()),
                    "labels": cls._gold_dict(example),
                },
                sort_keys=True,
                default=str,
            )
            ids.append(hashlib.sha1(payload.encode()).hexdigest()[:16])
        return ids

    @staticmethod
    def _gold_dict(example: dspy.Example) -> dict[str, Any]:
        labels = dict(example.labels())
        return {"domains": labels.get("domains", getattr(example, "domains", []))}

    @staticmethod
    def _frontier_records(
        records: Sequence[dict[str, Any]],
        best_score: float,
    ) -> list[dict[str, Any]]:
        frontier: list[dict[str, Any]] = []
        seen_fingerprints: set[tuple[str, ...]] = set()
        for record in records:
            if abs(float(record["score"]) - best_score) > 1e-9:
                continue
            fingerprint = tuple(record.get("error_fingerprint", []))
            if fingerprint in seen_fingerprints:
                continue
            frontier.append(record)
            seen_fingerprints.add(fingerprint)
        return frontier

    def _clean_instructions(
        self,
        raw_instructions: Any,
        current_instruction: str,
    ) -> list[str]:
        if isinstance(raw_instructions, str):
            raw_instructions = [raw_instructions]

        short_instructions: list[str] = []
        long_instructions: list[str] = []
        seen: set[str] = set()
        current_key = " ".join(current_instruction.lower().split())

        for raw_instruction in raw_instructions or []:
            instruction = str(raw_instruction).strip()
            if not instruction:
                continue
            key = " ".join(instruction.lower().split())
            if key in seen or key == current_key:
                continue
            seen.add(key)
            if len(instruction.split()) <= 180:
                short_instructions.append(instruction)
            else:
                long_instructions.append(instruction)
            if len(short_instructions) >= self.num_candidates:
                break

        instructions = short_instructions[: self.num_candidates]
        if len(instructions) < self.num_candidates and long_instructions:
            instructions.append(long_instructions[0])
        return instructions[: self.num_candidates]

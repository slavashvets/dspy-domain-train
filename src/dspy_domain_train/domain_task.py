import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, get_args

import dspy

Domain = Literal[
    "attraction",
    "bus",
    "hospital",
    "hotel",
    "police",
    "restaurant",
    "taxi",
    "train",
    "none",
]

DOMAINS: tuple[str, ...] = get_args(Domain)


class DomainClassificationSig(dspy.Signature):
    """Classify the active service domains in the current user turn."""

    dialogue_context: str = dspy.InputField(desc="Previous dialogue context, if any.")
    turn: str = dspy.InputField(desc="Current user turn.")
    domains: list[Domain] = dspy.OutputField(
        desc=(
            "Active domains requested in the current turn."
            " Use 'none' only when no domain is active."
        )
    )


class DomainClassifier(dspy.Module):
    def __init__(self, instructions: str | None = None) -> None:
        super().__init__()
        sig = (
            DomainClassificationSig.with_instructions(instructions)
            if instructions
            else DomainClassificationSig
        )
        self.predict = dspy.Predict(sig)

    def forward(self, dialogue_context: str, turn: str) -> dspy.Prediction:
        return self.predict(dialogue_context=dialogue_context, turn=turn)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def dedup(seq: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def normalize_domains(raw: list[str] | tuple[str, ...]) -> list[str]:
    allowed = set(DOMAINS)
    filtered = dedup([d.lower().strip() for d in raw if d.lower().strip() in allowed])
    if "none" in filtered and len(filtered) > 1:
        filtered = [d for d in filtered if d != "none"]
    return filtered or ["none"]


def parse_prediction(pred: dspy.Prediction) -> list[str]:
    raw_domains = getattr(pred, "domains", None)
    if raw_domains:
        if not isinstance(raw_domains, (list, tuple)):
            raw_domains = [raw_domains]
        return normalize_domains(raw_domains)

    # Legacy path: JSON string fallback for older checkpoints
    raw_str = getattr(pred, "domains_json", "") or ""
    return _parse_domains_from_str(raw_str)


def _parse_domains_from_str(raw: str) -> list[str]:
    """Parse model output into a clean list of allowed domains."""

    def try_load(s: str) -> list[str]:
        try:
            obj = json.loads(s.strip())
            val = obj.get("domains", [])
            if not isinstance(val, list):
                return []
            return [str(v).lower().strip() for v in val]
        except Exception:
            return []

    domains = try_load(raw)
    if not domains:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            domains = try_load(m.group(0))

    return normalize_domains(domains) if domains else ["none"]


def domain_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None,  # noqa: ANN001
    pred_name: str | None = None,
    pred_trace=None,  # noqa: ANN001
) -> float | dspy.Prediction:
    y_true = set(normalize_domains(example.domains))
    y_pred = set(parse_prediction(pred))

    missing = sorted(y_true - y_pred)
    extra = sorted(y_pred - y_true)
    score = float(not missing and not extra)

    if pred_name is None:
        return score

    feedback_parts = []
    if missing:
        feedback_parts.append(f"Missing: {missing}.")
    if extra:
        feedback_parts.append(f"Extra: {extra}.")
    if not feedback_parts:
        feedback_parts.append("Correct.")

    ctx = f"Turn: {example.turn}"
    feedback = f"{ctx} | {' '.join(feedback_parts)}"
    return dspy.Prediction(score=score, feedback=feedback)

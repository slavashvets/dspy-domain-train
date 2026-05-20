from collections.abc import Sequence
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
DOMAIN_SET = frozenset(DOMAINS)


class DomainClassificationSig(dspy.Signature):
    """Classify the active service domains in the current user turn.

    Return one or more from: restaurant, attraction, hotel, taxi, train, bus,
    hospital, police. Use "none" only when no domain is active.

    Do not add a domain just because a named place belongs to that category.
    In taxi requests, pickup/dropoff locations are taxi slot values, not
    additional active domains.
    """

    dialogue_context: str = dspy.InputField(desc="Previous dialogue context, if any.")
    turn: str = dspy.InputField(desc="Current user turn.")
    domains: list[Domain] = dspy.OutputField(
        desc="Active domains requested in the current turn."
    )


class DomainClassifier(dspy.Module):
    def __init__(self, instructions: str | None = None) -> None:
        super().__init__()
        signature = (
            DomainClassificationSig.with_instructions(instructions)
            if instructions
            else DomainClassificationSig
        )
        self.predict = dspy.Predict(signature)

    def forward(self, dialogue_context: str, turn: str) -> dspy.Prediction:
        return self.predict(dialogue_context=dialogue_context, turn=turn)


def normalize_gold(labels: Sequence[str]) -> list[str]:
    cleaned = [
        label
        for label in dict.fromkeys(str(item).lower().strip() for item in labels)
        if label in DOMAIN_SET
    ]
    if "none" in cleaned and len(cleaned) > 1:
        cleaned = [label for label in cleaned if label != "none"]
    return cleaned or ["none"]


def normalize_prediction(labels: Sequence[str]) -> list[str] | None:
    cleaned = [str(item).lower().strip() for item in labels]
    if any(label not in DOMAIN_SET for label in cleaned):
        return None

    deduped = list(dict.fromkeys(cleaned))
    if "none" in deduped and len(deduped) > 1:
        deduped = [label for label in deduped if label != "none"]
    return deduped or ["none"]


def domain_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None,  # noqa: ANN001
    pred_name: str | None = None,
    pred_trace=None,  # noqa: ANN001
) -> float | dspy.Prediction:
    y_true = set(normalize_gold(example.domains))
    y_pred_list = normalize_prediction(getattr(pred, "domains", []))

    if y_pred_list is None:
        score = 0.0
        missing = sorted(y_true)
        extra = ["<invalid-label>"]
    else:
        y_pred = set(y_pred_list)
        missing = sorted(y_true - y_pred)
        extra = sorted(y_pred - y_true)
        score = float(not missing and not extra)

    if pred_name is None:
        return score

    feedback = (
        f"Context: {example.dialogue_context!r} | "
        f"Turn: {example.turn!r} | "
        f"Gold: {sorted(y_true)} | "
        f"Predicted: {y_pred_list if y_pred_list is not None else '<invalid>'} | "
        f"Missing: {missing} | Extra: {extra}"
    )
    return dspy.Prediction(score=score, feedback=feedback)

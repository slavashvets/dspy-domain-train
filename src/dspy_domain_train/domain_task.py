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
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(DomainClassificationSig)

    def forward(self, dialogue_context: str, turn: str) -> dspy.Prediction:
        return self.predict(dialogue_context=dialogue_context, turn=turn)


def canonical_gold(labels: Sequence[str]) -> frozenset[str]:
    values = frozenset(str(label).lower().strip() for label in labels)

    if not values:
        return frozenset({"none"})

    unknown = values - DOMAIN_SET
    if unknown:
        raise ValueError(f"Invalid gold domain labels: {sorted(unknown)}")

    if "none" in values and len(values) > 1:
        raise ValueError(
            f"Gold labels cannot mix 'none' with domains: {sorted(values)}"
        )

    return values


def canonical_prediction(labels: Sequence[str]) -> frozenset[str] | None:
    values = frozenset(str(label).lower().strip() for label in labels)

    if not values:
        return frozenset({"none"})

    if values - DOMAIN_SET:
        return None

    if "none" in values and len(values) > 1:
        return None

    return values


def domain_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None,  # noqa: ANN001
    pred_name: str | None = None,
    pred_trace=None,  # noqa: ANN001
) -> float | dspy.Prediction:
    y_true = canonical_gold(example.domains)
    y_pred = canonical_prediction(getattr(pred, "domains", []))

    if y_pred is None:
        score = 0.0
        missing = sorted(y_true)
        extra = ["<invalid-output>"]
        predicted: list[str] | str = "<invalid>"
    else:
        missing = sorted(y_true - y_pred)
        extra = sorted(y_pred - y_true)
        score = float(not missing and not extra)
        predicted = sorted(y_pred)

    if pred_name is None:
        return score

    feedback = (
        f"Context: {example.dialogue_context!r} | "
        f"Turn: {example.turn!r} | "
        f"Gold: {sorted(y_true)} | "
        f"Predicted: {predicted} | "
        f"Missing: {missing} | Extra: {extra}"
    )
    return dspy.Prediction(score=score, feedback=feedback)

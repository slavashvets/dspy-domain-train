import json
import re
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import dspy

from training import ErrorCase, Example, evaluate

DOMAINS: tuple[str, ...] = (
    "restaurant",
    "attraction",
    "hotel",
    "taxi",
    "train",
    "bus",
    "hospital",
    "police",
    "none",
)


class DomainClassificationSig(dspy.Signature):
    """Return strictly JSON: {"domains": ["..."]} (lowercase, no explanations)."""

    rules: str = dspy.InputField(desc="Classification rules.")
    history: str = dspy.InputField()
    turn: str = dspy.InputField()
    domains_json: str = dspy.OutputField()


class DomainClassifier(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(DomainClassificationSig)

    def forward(self, instructions: str, history: str, turn: str) -> dspy.Prediction:
        return self.predict(rules=instructions, history=history, turn=turn)


class RefinerSig(dspy.Signature):
    """Refine domain classification instructions based on error patterns.

    CRITICAL RULES:
    - No dataset-specific examples or phrases.
    - Create GENERAL rules only.
    - Avoid overprediction: possessive references (my/our/their...) and location anchors (near/at/to/from...)
      do NOT add a domain unless the user explicitly requests something about that entity.
    - Keep the instructions as short and concise as possible: prefer minimal bullet points over long prose.
    """

    current_instructions: str = dspy.InputField(
        desc="Current classification instructions"
    )
    error_report: str = dspy.InputField(
        desc="Analysis of classification errors with patterns"
    )
    revised_instructions: str = dspy.OutputField(
        desc="Revised instructions with ONLY general rules, NO specific examples from dataset"
    )


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_devset(path: Path) -> list[Example]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Example(history=e["history"], turn=e["turn"], gt=e["gt"]) for e in data]


def dedup(seq: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


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

    allowed = set(DOMAINS)
    filtered = dedup([d for d in domains if d in allowed])
    if "none" in filtered and len(filtered) > 1:
        filtered = [d for d in filtered if d != "none"]
    return filtered or ["none"]


def parse_prediction(pred: dspy.Prediction) -> list[str]:
    return _parse_domains_from_str(pred.domains_json)


def build_error_report(errors: Sequence[ErrorCase], max_examples: int = 12) -> str:
    if not errors:
        return "No errors. Keep the instructions minimal, precise, unchanged."

    possessive_errors: list[ErrorCase] = []
    location_errors: list[ErrorCase] = []
    missing_domains: set[str] = set()
    extra_domains: set[str] = set()

    for e in errors:
        turn_lower = e["turn"].lower()
        gt_set = set(e["gt"])
        pred_set = set(e["pred"])

        if any(
            pron in turn_lower
            for pron in [" our ", " my ", " their ", " his ", " her "]
        ):
            possessive_errors.append(e)
        if any(
            prep in turn_lower
            for prep in [" near ", " close to ", " at ", " from ", " to "]
        ):
            location_errors.append(e)

        missing_domains.update(gt_set - pred_set)
        extra_domains.update(pred_set - gt_set)

    lines: list[str] = []
    lines.append("=== PATTERN ANALYSIS ===")

    if possessive_errors:
        lines.append(
            f"\n- {len(possessive_errors)}/{len(errors)} errors involve possessive pronouns (our/my/their)."
        )
        lines.append(
            "  Guideline: DO NOT add a domain solely because an entity is possessed."
        )
        lines.append(
            "  Add that entity's domain ONLY when the user requests info/action about it (book/cancel/change/check/features/availability/issue)."
        )

    if location_errors:
        lines.append(
            f"\n- {len(location_errors)}/{len(errors)} errors involve location anchors (near/close to/at/to/from)."
        )
        lines.append(
            "  Guideline: Treat anchors as context only. Include the domain of the actual request target, not of the anchor."
        )

    if missing_domains:
        lines.append(f"\n- Missing domains seen: {sorted(missing_domains)}")
    if extra_domains:
        lines.append(f"\n- Extra domains predicted: {sorted(extra_domains)}")

    if len(possessive_errors) == len(errors):
        lines.append(
            "\nPriority fix: Add a GENERAL anti-overprediction rule for possessives:"
        )
        lines.append(
            "- Possessive references DO NOT imply engagement; require explicit intent to include that domain."
        )

    lines.append("\n\n=== ERROR EXAMPLES ===")
    for i, e in enumerate(errors[:max_examples]):
        h = e["history"] or "<empty>"
        ann: list[str] = []
        turn_lower = e["turn"].lower()
        if any(
            pron in turn_lower
            for pron in [" our ", " my ", " their ", " his ", " her "]
        ):
            ann.append("HAS_POSSESSIVE")
        if any(
            prep in turn_lower
            for prep in [" near ", " close to ", " at ", " from ", " to "]
        ):
            ann.append("HAS_LOCATION")
        label = f"[{', '.join(ann)}]" if ann else ""
        lines.append(f"\nEx{i + 1} {label}")
        lines.append(f"H: {h}")
        lines.append(f"Ut: {e['turn']}")
        lines.append(f"GT: {e['gt']}")
        lines.append(f"Pred: {e['pred']}")
        miss = set(e["gt"]) - set(e["pred"])
        extra = set(e["pred"]) - set(e["gt"])
        if miss:
            lines.append(f"Missing: {sorted(miss)}")
        if extra:
            lines.append(f"Extra: {sorted(extra)}")

    lines.append("\n\n=== RECOMMENDATION ===")
    lines.append(
        "Refine rules to eliminate extra domains from possessive/location mentions while keeping JSON format and the candidate set unchanged."
    )
    lines.append(
        "\nIMPORTANT: Create GENERAL rules based on patterns, NOT dataset-specific examples."
    )
    lines.append(
        "Priority: Add anti-overprediction rules for possessives and location anchors."
    )
    return "\n".join(lines)


def make_auto_refiner(
    devset: Sequence[Example],
    eval_lm: dspy.LM,
    max_len: int = 50,
    k: int = 5,
    retries: int = 1,
) -> dspy.Module:
    clf_for_eval = DomainClassifier()
    dataset_phrases: set[str] = set()
    for ex in devset:
        dataset_phrases.add(ex["turn"].lower())
        words = ex["turn"].lower().split()
        for i in range(len(words) - 3):
            phrase = " ".join(words[i : i + 4])
            dataset_phrases.add(phrase)

    def reward_fn(instr: str) -> float:
        banned = [
            "think",
            "analyze",
            "consider",
            "prioritize",
            "focus",
            "determine",
            "identify",
            "understand",
            "assess",
            "evaluate",
            "examine",
            "note that",
            "ensure",
            "recognize",
            "indicate",
            "suggest",
            "imply",
            "pay attention",
            "be aware",
            "indicates that",
        ]
        text = instr.lower()

        for phrase in dataset_phrases:
            if len(phrase) > 10 and phrase in text:
                print(f"  [Rejected] Instructions contain dataset example: '{phrase}'")
                return 0.0

        example_indicators = [
            "kid-friendly museum",
            "near our hotel",
            "hilton hotel",
        ]
        for indicator in example_indicators:
            if indicator in text:
                print(
                    f"  [Rejected] Instructions contain specific example: '{indicator}'"
                )
                return 0.0

        banned_hits = sum(tok in text for tok in banned)
        penalty = 0.95**banned_hits

        length = len(instr)
        if length > max_len:
            # Penalize verbose instructions even when accuracy is high.
            length_penalty = (max_len / length) ** 2
            penalty *= length_penalty

        # Always score candidates on the classifier LM to avoid reward mismatch.
        # Suppress progress logging here to avoid noisy reward traces.
        with dspy.context(lm=eval_lm, track_usage=False):
            acc, _ = evaluate(
                clf_for_eval,
                instr,
                devset,
                parse_prediction,
                verbose=False,
            )
        score = acc * penalty
        return max(0.0, float(score))

    class AutoRefiner(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.edit = dspy.Predict(RefinerSig)

        def forward(
            self, current_instructions: str, error_report: str
        ) -> dspy.Prediction:
            candidates: list[str] = []
            for _ in range(retries):
                for _ in range(k):
                    cand = self.edit(
                        current_instructions=current_instructions,
                        error_report=error_report + f"\n\n<!-- nonce:{uuid4()} -->",
                    ).revised_instructions.strip()
                    if cand:
                        candidates.append(cand)

            scored = []
            for c in candidates:
                s = reward_fn(c)
                if c.strip() == current_instructions.strip():
                    s = 0.0
                scored.append((c, s))
            scored = [s for s in scored if s[1] > 0]
            scored.sort(key=lambda x: x[1], reverse=True)
            best = scored[0][0] if scored else current_instructions
            return dspy.Prediction(revised_instructions=best)

    return AutoRefiner()

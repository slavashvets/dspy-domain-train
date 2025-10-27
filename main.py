"""Offline SRP demo for domain classification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypedDict
from uuid import uuid4

import dspy
from environs import Env

from instrumentation import add_usage, format_usage

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


class Example(TypedDict):
    history: str
    turn: str
    gt: list[str]


class ErrorCase(TypedDict):
    history: str
    turn: str
    gt: list[str]
    pred: list[str]


@dataclass
class BestResult:
    instr: str
    acc: float
    errors: list[ErrorCase]


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


class Refiner(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edit = dspy.Predict(RefinerSig)

    def forward(self, current_instructions: str, error_report: str) -> dspy.Prediction:
        return self.edit(
            current_instructions=current_instructions, error_report=error_report
        )


ROOT = Path(__file__).parent
PROMPT_PATH = ROOT / "prompts" / "p0.txt"
DEVSET_PATH = ROOT / "data" / "dev.json"


def configure_lm_from_env() -> dict:
    """Configure DSPy with Azure OpenAI settings. Returns dict with both LMs."""
    env = Env()
    env.read_env()
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    import logging

    logging.getLogger("dspy").setLevel(logging.INFO)
    dspy.settings.configure(track_usage=True)

    # 1) Global LM for classification and evaluation - gpt-4.1 (chat)
    lm_eval = dspy.LM(
        model=f"azure/{env.str('AZURE_OPENAI_DEPLOYMENT_MAIN', 'gpt-4.1')}",
        model_type="chat",
        api_base=env.str("AZURE_OPENAI_ENDPOINT"),
        api_version=env.str("AZURE_OPENAI_API_VERSION"),
        api_key=env.str("AZURE_OPENAI_API_KEY"),
        temperature=env.float("AZURE_OPENAI_TEMPERATURE", 0),
    )
    dspy.configure(lm=lm_eval)  # make it the default LM

    # 2) Dedicated LM for refinement - gpt-5 (responses + reasoning high)
    lm_refine = dspy.LM(
        model=f"azure/{env.str('AZURE_OPENAI_DEPLOYMENT_REFINE', 'gpt-5')}",
        model_type="responses",  # Responses API for reasoning-oriented models
        api_base=env.str("AZURE_OPENAI_ENDPOINT_REFINE"),
        api_version=env.str("AZURE_OPENAI_API_VERSION_REFINE"),
        api_key=env.str("AZURE_OPENAI_API_KEY_REFINE"),
        temperature=1,
        max_tokens=16000,
        # reasoning={"effort": env.str("AZURE_OPENAI_REASONING_EFFORT")},
    )

    return {"eval": lm_eval, "refine": lm_refine}


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


def parse_domains(raw: str) -> list[str]:
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
    # Allow "none" only as a singleton; otherwise drop it.
    if "none" in filtered and len(filtered) > 1:
        filtered = [d for d in filtered if d != "none"]
    # Fall back to "none" if nothing valid was detected.
    return filtered or ["none"]


def make_auto_refiner(
    devset: Sequence[Example],
    max_len: int = 50,
    k: int = 5,  # number of candidates per iteration
    retries: int = 1,  # number of generation rounds
) -> dspy.Module:
    clf_for_eval = DomainClassifier()
    # Collect all phrases from the dataset to guard against memorized snippets.
    dataset_phrases = set()
    for ex in devset:
        dataset_phrases.add(ex["turn"].lower())
        words = ex["turn"].lower().split()
        for i in range(len(words) - 3):  # track phrases of length >= 4 words
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
        penalty = (
            0.95**banned_hits
        )  # soft penalty for filler verbs like ensure/evaluate

        if len(instr) > max_len:
            # Do not zero out long instructions; dampen them for verbosity instead.
            penalty *= max(0.05, max_len / len(instr))

        acc, _ = evaluate(clf_for_eval, instr, devset)
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

            # Score and select the best candidate.
            scored = []
            for c in candidates:
                s = reward_fn(c)
                if c.strip() == current_instructions.strip():
                    s = 0.0  # reject exact copies of the input instructions
                scored.append((c, s))
            scored = [s for s in scored if s[1] > 0]
            scored.sort(key=lambda x: x[1], reverse=True)
            best = scored[0][0] if scored else current_instructions
            return dspy.Prediction(revised_instructions=best)

    return AutoRefiner()


def evaluate(
    program: DomainClassifier, instructions: str, dataset: Sequence[Example]
) -> tuple[float, list[ErrorCase]]:
    # Refiners/optimizers inside DSPy may disable track_usage for speed.
    # Ensure usage collection stays enabled during evaluation.
    dspy.settings.configure(track_usage=True)

    correct = 0
    errors: list[ErrorCase] = []
    usage_total: dict = {}

    n = len(dataset)
    print_every = max(1, n // 4)

    for i, ex in enumerate(dataset, 1):
        out = program(instructions, ex["history"], ex["turn"])
        pred = parse_domains(out.domains_json)
        gt = [d.lower() for d in ex["gt"]]

        try:
            add_usage(usage_total, out.get_lm_usage())
        except Exception:
            pass

        if set(pred) == set(gt):
            correct += 1
        else:
            errors.append(
                ErrorCase(history=ex["history"], turn=ex["turn"], gt=gt, pred=pred)
            )

        if i % print_every == 0 or i == n:
            print(
                f"  progress {i}/{n} | usage {format_usage(usage_total)}",
                flush=True,
            )

    acc = correct / n if n else 0.0
    return acc, errors


def build_error_report(errors: Sequence[ErrorCase], max_examples: int = 12) -> str:
    if not errors:
        return "No errors. Keep the instructions minimal, precise, unchanged."

    possessive_errors = []
    location_errors = []
    missing_domains = set()
    extra_domains = set()

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

    lines = []
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
        ann = []
        if e in possessive_errors:
            ann.append("HAS_POSSESSIVE")
        if e in location_errors:
            ann.append("HAS_LOCATION")
        lines.append(f"\nEx{i + 1} {f'[{", ".join(ann)}]' if ann else ''}")
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


def offline_srp(
    devset: Sequence[Example],
    instructions: str,
    max_iters: int = 6,
    tol: float = 5e-3,
    refiner_lm: dspy.LM | None = None,
) -> BestResult:
    clf = DomainClassifier()
    ref = make_auto_refiner(devset)  # keep the best candidate per our metric
    instr = instructions
    init_errors: list[ErrorCase] = []
    best = BestResult(instr=instr, acc=0.0, errors=init_errors)

    for it in range(1, max_iters + 1):
        # 1) Pre-evaluate the current rules.
        pre_acc, pre_errs = evaluate(clf, instr, devset)
        print(f"[Iter {it} pre]  accuracy={pre_acc:.3f}  errors={len(pre_errs)}")

        prev_best = best.acc
        if pre_acc >= prev_best:
            best.instr = instr
            best.acc = pre_acc
            best.errors = list(pre_errs)

        # 2) Ask the critic to propose an improvement.
        report = build_error_report(pre_errs)
        if refiner_lm is not None:
            with dspy.context(lm=refiner_lm):
                revised = ref(
                    current_instructions=instr,
                    error_report=report,
                ).revised_instructions.strip()
        else:
            revised = ref(
                current_instructions=instr,
                error_report=report,
            ).revised_instructions.strip()

        if not revised or revised == instr:
            # No effective update; check the stop condition based on the pre score.
            if pre_acc == 1.0 or (pre_acc - prev_best < tol and not pre_errs):
                break
            continue

        # 3) Post-evaluate the revision within the same iteration to show progress.
        post_acc, post_errs = evaluate(clf, revised, devset)
        print(f"[Iter {it} post] accuracy={post_acc:.3f}  errors={len(post_errs)}")

        # 4) Apply the improvement only if it is strictly better.
        if post_acc >= pre_acc:
            instr = revised
            if post_acc >= best.acc:
                best.instr = instr
                best.acc = post_acc
                best.errors = list(post_errs)

        # 5) Early stopping.
        improved = best.acc - prev_best
        if best.acc == 1.0 or (improved < tol and not best.errors):
            break

    # Final evaluation of the best iteration.
    final_acc, final_errs = evaluate(DomainClassifier(), best.instr, devset)
    print("\nFinal instructions (P*):\n" + best.instr)
    if final_errs:
        print("\nRemaining errors:")
        for e in final_errs:
            print(f"- Ut: {e['turn']} | GT: {e['gt']} | Pred: {e['pred']}")
    return best


if __name__ == "__main__":
    lms = configure_lm_from_env()
    devset = load_devset(DEVSET_PATH)
    p0 = load_text(PROMPT_PATH)
    offline_srp(devset, p0, refiner_lm=None)

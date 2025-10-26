"""Offline SRP demo for domain classification."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence, TypedDict

import dspy
from environs import Env

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
    """Improve instruction text. Output only revised text."""

    current_instructions: str = dspy.InputField()
    error_report: str = dspy.InputField()
    revised_instructions: str = dspy.OutputField()


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


def configure_lm_from_env() -> None:
    """Configure DSPy with Azure OpenAI settings."""
    env = Env()
    env.read_env()
    lm = dspy.LM(
        model=f"azure/{env.str('AZURE_OPENAI_DEPLOYMENT')}",
        model_type="chat",
        api_base=env.str("AZURE_OPENAI_ENDPOINT"),
        api_version=env.str("AZURE_OPENAI_API_VERSION"),
        api_key=env.str("AZURE_OPENAI_API_KEY"),
        temperature=env.float("AZURE_OPENAI_TEMPERATURE", 1),
        max_tokens=env.int("AZURE_OPENAI_MAX_TOKENS", 16000),
    )
    dspy.configure(lm=lm)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_devset(path: Path) -> list[Example]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Example(history=e["history"], turn=e["turn"], gt=e["gt"]) for e in data
    ]


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
    return dedup([d for d in domains if d in allowed])


def evaluate(
    program: DomainClassifier, instructions: str, dataset: Sequence[Example]
) -> tuple[float, list[ErrorCase]]:
    correct = 0
    errors: list[ErrorCase] = []
    for ex in dataset:
        out = program(instructions, ex["history"], ex["turn"])
        pred = parse_domains(out.domains_json)
        gt = [d.lower() for d in ex["gt"]]
        if set(pred) == set(gt):
            correct += 1
        else:
            errors.append(
                ErrorCase(history=ex["history"], turn=ex["turn"], gt=gt, pred=pred)
            )
    acc = correct / len(dataset) if dataset else 0.0
    return acc, errors


def build_error_report(errors: Sequence[ErrorCase], max_examples: int = 12) -> str:
    if not errors:
        return "No errors. Keep the instructions minimal, precise, unchanged."
    lines: list[str] = []
    for i, e in enumerate(errors[:max_examples]):
        h = e["history"] or "<empty>"
        lines.append(
            f"Ex{i+1}\nH: {h}\nUt: {e['turn']}\nGT: {e['gt']}\nPred: {e['pred']}"
        )
    return (
        "Observed misclassifications on the dev set:\n\n"
        + "\n\n".join(lines)
        + "\n\nRefine the rules to fix these errors while keeping the JSON format "
        + "and candidate list unchanged."
    )


def offline_srp(
    devset: Sequence[Example], instructions: str, max_iters: int = 3, tol: float = 5e-3
) -> dict[str, object]:
    clf = DomainClassifier()
    ref = Refiner()
    instr = instructions
    best: dict[str, object] = {"instr": instr, "acc": 0.0, "errors": []}

    for it in range(1, max_iters + 1):
        acc, errs = evaluate(clf, instr, devset)
        print(f"[Iter {it}] accuracy={acc:.3f}  errors={len(errs)}")
        if acc >= best["acc"]:
            best.update({"instr": instr, "acc": acc, "errors": errs})

        improved = acc - float(best["acc"])
        if acc == 1.0 or (improved < tol and not errs):
            break

        report = build_error_report(errs)
        revised = ref(instr, report).revised_instructions.strip()
        if not revised or revised == instr:
            print("No effective revision produced; stopping.")
            break
        instr = revised

    final_acc, final_errs = evaluate(DomainClassifier(), best["instr"], devset)
    print(f"\nFinal dev accuracy: {final_acc:.3f} on {len(devset)} examples")
    print("\nFinal instructions (P★):\n" + str(best["instr"]))
    if final_errs:
        print("\nRemaining errors:")
        for e in final_errs:
            print(f"- Ut: {e['turn']} | GT: {e['gt']} | Pred: {e['pred']}")
    return best


if __name__ == "__main__":
    configure_lm_from_env()
    devset = load_devset(DEVSET_PATH)
    p0 = load_text(PROMPT_PATH)
    offline_srp(devset, p0)

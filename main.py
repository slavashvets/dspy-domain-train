import json

import dspy
from environs import Env

env = Env()
env.read_env()

lm = dspy.LM(
    model=f"azure/{env.str('AZURE_OPENAI_DEPLOYMENT')}",
    model_type="chat",
    api_base=env.str("AZURE_OPENAI_ENDPOINT"),
    api_version=env.str("AZURE_OPENAI_API_VERSION"),
    api_key=env.str("AZURE_OPENAI_API_KEY"),
    temperature=1,
    max_tokens=16000,
)
dspy.configure(lm=lm)

# --------------- SIGNATURES & MODULES ---------------
class DomainClassificationSig(dspy.Signature):
    """Domain classification with rules in 'rules'.
    Return strictly a JSON object: {"domains": ["..."]} with lowercase names, no explanations."""
    rules: str = dspy.InputField(desc="Refined rules for classification (the prompt).")
    history: str = dspy.InputField()
    turn: str = dspy.InputField()
    domains_json: str = dspy.OutputField(desc='Strict JSON like {"domains": ["taxi","hotel"]}')

class DomainClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(DomainClassificationSig)
    def forward(self, instructions, history, turn):
        # Pass the user's 'instructions' into the signature field 'rules'
        return self.predict(rules=instructions, history=history, turn=turn)

class RefinerSig(dspy.Signature):
    """Improve the domain-classification instructions (P) to fix observed errors.
    Return only the revised instructions string, no preamble, no code fences."""
    current_instructions: str = dspy.InputField()
    error_report: str = dspy.InputField()
    revised_instructions: str = dspy.OutputField()

class Refiner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.edit = dspy.Predict(RefinerSig)
    def forward(self, current_instructions, error_report):
        return self.edit(current_instructions=current_instructions, error_report=error_report)

# --------------- DEV DATA (edge cases) ---------------
DEV = [
    {"history": "", "turn": "I need a taxi to the Hilton Hotel.", "gt": ["taxi"]},
    {"history": "We discussed a restaurant booking earlier.", "turn": "Thanks, bye.", "gt": ["none"]},
    {"history": "", "turn": "Cancel my taxi and book a table for 7pm.", "gt": ["taxi","restaurant"]},
    {"history": "Previous turn was about a taxi.", "turn": "Also, does the hotel have a pool?", "gt": ["hotel"]},
    {"history": "", "turn": "Find a museum near the station.", "gt": ["attraction"]},
    {"history": "", "turn": "Book a room near the central station.", "gt": ["hotel"]},
]

# --------------- BASELINE INSTRUCTIONS (P0) ---------------
P0 = """\
Classify the domain(s) of the USER's current turn Ut given the dialogue history H_{t-1}.
Choose from: ["restaurant","attraction","hotel","taxi","train","bus","hospital","police","none"].
Return strictly JSON: {"domains": ["..."]} with lowercase domain names and no explanations.
"""

# --------------- EVAL & SRP LOOP ---------------
def parse_domains(domains_json_str: str):
    try:
        obj = json.loads(domains_json_str.strip())
        doms = obj.get("domains", [])
        return [d.strip().lower() for d in doms]
    except Exception:
        return []

def evaluate(program: DomainClassifier, instructions: str, dataset):
    correct, errors = 0, []
    for ex in dataset:
        out  = program(instructions, ex["history"], ex["turn"])
        pred = parse_domains(out.domains_json)
        gt   = [d.lower() for d in ex["gt"]]
        if set(pred) == set(gt):
            correct += 1
        else:
            errors.append({
                "history": ex["history"],
                "turn": ex["turn"],
                "gt": gt,
                "pred": pred
            })
    acc = correct / len(dataset)
    return acc, errors

def build_error_report(errors, max_examples=12):
    if not errors:
        return "No errors. Keep the instructions minimal, precise, and unchanged."
    lines = []
    for i, e in enumerate(errors[:max_examples]):
        h = e['history'] if e['history'] else "<empty>"
        lines.append(
            f"Ex{i+1}\nH: {h}\nUt: {e['turn']}\nGT: {e['gt']}\nPred: {e['pred']}"
        )
    return (
        "Observed misclassifications on the dev set:\n\n" +
        "\n\n".join(lines) +
        "\n\nRefine the rules to fix these errors while keeping the JSON format and candidate list unchanged."
    )

def offline_srp(devset=DEV, max_iters=3, tol=5e-3):
    clf    = DomainClassifier()
    ref    = Refiner()
    instr  = P0
    best   = {"instr": instr, "acc": 0.0, "errors": []}

    for it in range(1, max_iters + 1):
        acc, errors = evaluate(clf, instr, devset)
        print(f"[Iter {it}] accuracy={acc:.3f}  errors={len(errors)}")
        improved = acc - best["acc"]
        if acc >= best["acc"]:
            best.update({"instr": instr, "acc": acc, "errors": errors})
        # Early stop if perfect or no tangible improvement + no errors
        if acc == 1.0 or (improved < tol and not errors):
            break
        # Ask LLM-critic to refine P -> P'
        report   = build_error_report(errors)
        revision = ref(instr, report).revised_instructions.strip()
        if not revision:
            print("No revision produced; stopping.")
            break
        instr = revision

    # Final check with best instructions
    final_acc, final_errors = evaluate(DomainClassifier(), best["instr"], devset)
    print(f"\nFinal dev accuracy: {final_acc:.3f} on {len(devset)} examples")
    print("\nFinal instructions (P★):\n" + best["instr"])
    if final_errors:
        print("\nRemaining errors:")
        for e in final_errors:
            print(f"- Ut: {e['turn']} | GT: {e['gt']} | Pred: {e['pred']}")
    return best

if __name__ == "__main__":
    offline_srp()

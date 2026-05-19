# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "datasets>=4.0",
# ]
# ///
"""Convert tuetschek/multi_woz_v22 into train/dev/test JSON splits.

Output format:
  {"turn": "...", "dialogue_context": "...", "domains": ["hotel", ...]}

Per-turn domain labels derived from MultiWOZ 2.2 frame annotations:
active_intent and requested_slots (added in 2.2, ground-truth per-turn).
"""

import json
import re
from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

VALID_DOMAINS = frozenset(
    ("restaurant", "attraction", "hotel", "taxi", "train", "bus", "hospital", "police")
)

_CLOSING_RE = re.compile(
    r"^\s*(?:thanks?|thank\s+you|cheers|bye|goodbye|no[,.]?\s*that'?s?"
    r"\s*(?:all|it|everything)|that'?s?\s*(?:all|it)|great[,.]?\s*thank"
    r"|ok(?:ay)?[,.]?\s*thank|no[,.]?\s*(?:that|i)\s)",
    re.IGNORECASE,
)


def _domain_from_intent(intent: str) -> str | None:
    """Extract domain prefix from intent like 'find_restaurant' or 'book_train'."""
    if not intent or intent.upper() == "NONE":
        return None
    parts = intent.lower().replace("-", "_").split("_")
    for part in parts:
        if part in VALID_DOMAINS:
            return part
    for part in parts[1:]:
        if part in VALID_DOMAINS:
            return part
    return None


def _domain_from_slot(slot_name: str) -> str | None:
    """Extract domain from slot like 'restaurant-pricerange'."""
    prefix = slot_name.split("-")[0].lower()
    return prefix if prefix in VALID_DOMAINS else None


def extract_turn_domains(frames: dict) -> list[str]:
    """Extract active domains from user turn frames."""
    domains: set[str] = set()
    states = frames.get("state", [])

    for i, state in enumerate(states):
        intent = state.get("active_intent", "")
        domain = _domain_from_intent(intent)
        if domain:
            domains.add(domain)

        for slot in state.get("requested_slots", []):
            d = _domain_from_slot(slot)
            if d:
                domains.add(d)

    return sorted(domains)


def convert_dialogue(dialogue: dict, max_context: int = 4) -> list[dict]:
    """Convert one dialogue into per-turn examples."""
    turns = dialogue["turns"]
    utterances = turns["utterance"]
    speakers = turns["speaker"]
    all_frames = turns["frames"]

    examples: list[dict] = []
    history: list[str] = []

    for i in range(len(utterances)):
        utt = utterances[i].strip()
        is_user = speakers[i] == 0

        if is_user:
            frames = all_frames[i]
            domains = extract_turn_domains(frames)

            if not domains:
                if _CLOSING_RE.match(utt):
                    domains = ["none"]
                else:
                    # Ambiguous follow-up, skip
                    history.append(utt)
                    continue

            context = " | ".join(history[-max_context:]) if history else ""
            examples.append(
                {
                    "turn": utt,
                    "dialogue_context": context,
                    "domains": domains,
                }
            )

        history.append(utt)

    return examples


def main() -> None:
    print("Loading tuetschek/multi_woz_v22...")
    ds = load_dataset("tuetschek/multi_woz_v22")

    split_map = {
        "train": "train.json",
        "validation": "dev.json",
        "test": "test.json",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, filename in split_map.items():
        examples: list[dict] = []
        for dialogue in ds[split_name]:
            examples.extend(convert_dialogue(dialogue))

        out_path = OUTPUT_DIR / filename
        out_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False) + "\n")
        print(f"  {split_name} -> {out_path} ({len(examples)} examples)")

    print("Done.")


if __name__ == "__main__":
    main()

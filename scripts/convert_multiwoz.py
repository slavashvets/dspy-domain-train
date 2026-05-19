# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "datasets>=4.0",
# ]
# ///
"""Convert Brendan/multiwoz_turns_v22 into train/dev/test JSON splits.

Output format:
  {"turn": "...", "dialogue_context": "...", "domains": ["hotel", ...]}

Uses the per-turn `turn_slot_values` to determine which domain is active
in the current turn. Falls back to dialogue-level `domains` when
turn_slot_values is empty (follow-up questions, closings).
"""

import json
from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def extract_domains(turn_slot_values: dict[str, dict]) -> list[str]:
    active = sorted(d for d, slots in turn_slot_values.items() if any(slots.values()))
    return active if active else ["none"]


def convert_example(ex: dict) -> dict:
    system_utts = ex["system_utterances"]
    user_utts = ex["user_utterances"]

    history_parts: list[str] = []
    for i in range(len(user_utts) - 1):
        if system_utts[i]:
            history_parts.append(system_utts[i])
        history_parts.append(user_utts[i])
    if system_utts[-1]:
        history_parts.append(system_utts[-1])

    dialogue_context = " | ".join(history_parts[-4:]) if history_parts else ""

    return {
        "turn": user_utts[-1].strip(),
        "dialogue_context": dialogue_context,
        "domains": extract_domains(ex["turn_slot_values"]),
    }


def main() -> None:
    print("Loading Brendan/multiwoz_turns_v22...")
    ds = load_dataset("Brendan/multiwoz_turns_v22")

    split_map = {
        "train": "train.json",
        "validation": "dev.json",
        "test": "test.json",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, filename in split_map.items():
        examples = [convert_example(ex) for ex in ds[split_name]]
        out_path = OUTPUT_DIR / filename
        out_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False) + "\n")
        print(f"  {split_name} -> {out_path} ({len(examples)} examples)")

    print("Done.")


if __name__ == "__main__":
    main()

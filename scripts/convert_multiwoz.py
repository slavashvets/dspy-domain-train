# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "datasets>=4.0",
# ]
# ///
"""Convert tuetschek/multi_woz_v22 into train/dev/test JSON splits.

Output format:
  {"turn": "...", "dialogue_context": "...", "domains": ["hotel", ...]}

Per-turn domain labels derived from MultiWOZ 2.2 frame annotations
using current turn slots, requested slots, and belief-state deltas plus a
skip/closing heuristic.
"""

import json
import re
from pathlib import Path

try:
    from datasets import load_dataset  # type: ignore[import-untyped]
except ModuleNotFoundError:
    load_dataset = None

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

VALID_DOMAINS = frozenset(
    ("restaurant", "attraction", "hotel", "taxi", "train", "bus", "hospital", "police")
)

_CLOSING_RE = re.compile(
    r"^\s*(?:no[,.]?\s*)?(?:"
    r"thanks?|thank\s+you|cheers|bye|goodbye|"
    r"(?:that'?s?|that\s+is)\s*(?:all|it|everything)(?:\s*i\s*need)?|"
    r"i\s*(?:think|believe)?\s*(?:that'?s?|that\s+is)\s*"
    r"(?:all|it|everything)|"
    r"i\s*(?:do\s*not|don'?t)\s*need\s*(?:anything\s*)?(?:else|more)|"
    r"nothing\s*(?:else|more)|no\s*more|all\s*set|"
    r"great[,.]?\s*thanks?|ok(?:ay)?[,.]?\s*thanks?"
    r")(?:\s*(?:(?:[,;.!?]|\band\b)\s*)?(?:"
    r"thanks?|thank\s+you|cheers|bye|goodbye|"
    r"(?:that'?s?|that\s+is)\s*(?:all|it|everything)(?:\s*i\s*need)?|"
    r"i\s*(?:do\s*not|don'?t)\s*need\s*(?:anything\s*)?(?:else|more)|"
    r"nothing\s*(?:else|more)|no\s*more|all\s*set"
    r"))*\s*[.!?]*\s*$",
    re.IGNORECASE,
)

type BeliefState = dict[str, tuple[str, ...]]


def _domain_from_intent(intent: str) -> str | None:
    """Extract domain prefix from intent like 'find_restaurant' or 'book_train'."""
    if not intent or intent.upper() == "NONE":
        return None
    parts = intent.lower().replace("-", "_").split("_")
    for part in parts:
        if part in VALID_DOMAINS:
            return part
    return None


def _domain_from_slot(slot_name: str) -> str | None:
    """Extract domain from slot like 'restaurant-pricerange'."""
    prefix = slot_name.split("-")[0].lower()
    return prefix if prefix in VALID_DOMAINS else None


def _domain_from_act_type(act_type: str) -> str | None:
    """Extract domain from dialogue act like 'Restaurant-Inform'."""
    prefix = act_type.split("-")[0].lower()
    return prefix if prefix in VALID_DOMAINS else None


def _dialogue_act_domains(dialogue_acts: object) -> set[str]:
    domains: set[str] = set()
    if not isinstance(dialogue_acts, dict):
        return domains

    dialog_act = dialogue_acts.get("dialog_act", {})
    if isinstance(dialog_act, dict):
        act_types = dialog_act.get("act_type", [])
        if isinstance(act_types, str):
            act_types = [act_types]
        for act_type in act_types:
            if domain := _domain_from_act_type(str(act_type)):
                domains.add(domain)

    span_info = dialogue_acts.get("span_info", {})
    if isinstance(span_info, dict):
        act_types = span_info.get("act_type", [])
        if isinstance(act_types, str):
            act_types = [act_types]
        for act_type in act_types:
            if domain := _domain_from_act_type(str(act_type)):
                domains.add(domain)

    return domains


def _detailed_dialogue_act_domains(dialogue_acts: object) -> set[str]:
    domains: set[str] = set()
    if not isinstance(dialogue_acts, dict):
        return domains

    dialog_act = dialogue_acts.get("dialog_act", {})
    if isinstance(dialog_act, dict):
        act_types = dialog_act.get("act_type", [])
        act_slots = dialog_act.get("act_slots", [])
        if isinstance(act_types, str):
            act_types = [act_types]
        for index, act_type in enumerate(act_types):
            domain = _domain_from_act_type(str(act_type))
            if not domain:
                continue
            slots = _sequence_item(act_slots, index, {})
            slot_names = slots.get("slot_name", []) if isinstance(slots, dict) else []
            if isinstance(slot_names, str):
                slot_names = [slot_names]
            if any(str(slot).lower() != "none" for slot in slot_names):
                domains.add(domain)

    span_info = dialogue_acts.get("span_info", {})
    if isinstance(span_info, dict):
        act_types = span_info.get("act_type", [])
        if isinstance(act_types, str):
            act_types = [act_types]
        for act_type in act_types:
            if domain := _domain_from_act_type(str(act_type)):
                domains.add(domain)

    return domains


def _slot_values(state: dict) -> BeliefState:
    slots = state.get("slots_values", {})
    names = slots.get("slots_values_name", [])
    values_list = slots.get("slots_values_list", [])

    values_by_slot: BeliefState = {}
    for index, name in enumerate(names):
        raw_values = values_list[index] if index < len(values_list) else []
        if isinstance(raw_values, str):
            values: tuple[str, ...] = (raw_values,)
        else:
            values = tuple(str(value) for value in raw_values)
        values_by_slot[str(name)] = tuple(sorted(values))
    return values_by_slot


def _state_delta_domains(state: dict, previous_state: BeliefState) -> set[str]:
    domains: set[str] = set()
    for slot_name, values in _slot_values(state).items():
        if previous_state.get(slot_name) == values:
            continue
        domain = _domain_from_slot(slot_name)
        if domain:
            domains.add(domain)
    return domains


def _merge_state(frames: dict) -> BeliefState:
    state: BeliefState = {}
    for _, frame_state, _ in _iter_frames(frames):
        state.update(_slot_values(frame_state))
    return state


def _sequence_item(values: object, index: int, default: object) -> object:
    if isinstance(values, list):
        return values[index] if index < len(values) else default
    return default


def _iter_frames(frames: dict | list) -> list[tuple[str, dict, object]]:
    if isinstance(frames, list):
        return [
            (
                str(frame.get("service", "")),
                frame.get("state", {}) or {},
                frame.get("slots", []),
            )
            for frame in frames
        ]

    services = frames.get("service", [])
    states = frames.get("state", [])
    slots = frames.get("slots", [])
    frame_count = max(len(services), len(states), len(slots))

    records: list[tuple[str, dict, object]] = []
    for index in range(frame_count):
        service = str(_sequence_item(services, index, ""))
        state = _sequence_item(states, index, {})
        frame_slots = _sequence_item(slots, index, [])
        records.append((service, state if isinstance(state, dict) else {}, frame_slots))
    return records


def _iter_current_slots(slots: object) -> list[str]:
    if isinstance(slots, dict):
        names = slots.get("slot", [])
        if isinstance(names, str):
            return [names]
        return [str(name) for name in names]

    slot_names: list[str] = []
    if isinstance(slots, list):
        for slot in slots:
            if isinstance(slot, str):
                slot_names.append(slot)
            elif isinstance(slot, dict):
                slot_names.append(
                    str(
                        slot.get("slot")
                        or slot.get("slot_name")
                        or slot.get("name")
                        or ""
                    )
                )
    return slot_names


def _current_slot_domains(slots: object) -> set[str]:
    domains: set[str] = set()
    for slot_name in _iter_current_slots(slots):
        domain = _domain_from_slot(slot_name)
        if domain:
            domains.add(domain)
    return domains


def _taxi_copy_endpoint_domains(frames: dict | list) -> set[str]:
    domains: set[str] = set()
    for service, _, slots in _iter_frames(frames):
        if service != "taxi" or not isinstance(slots, dict):
            continue
        copy_from = slots.get("copy_from", [])
        if isinstance(copy_from, str):
            copy_from = [copy_from]
        for source_slot in copy_from:
            domain = _domain_from_slot(str(source_slot))
            if domain and domain != "taxi":
                domains.add(domain)
    return domains


def extract_turn_domains(
    frames: dict,
    previous_state: BeliefState | None = None,
    dialogue_acts: object | None = None,
) -> list[str]:
    """Extract active domains from user turn frames."""
    previous_state = previous_state or {}
    act_domains = _dialogue_act_domains(dialogue_acts)
    if act_domains:
        if "taxi" in act_domains:
            endpoint_domains = _taxi_copy_endpoint_domains(frames)
            detailed_domains = _detailed_dialogue_act_domains(dialogue_acts)
            act_domains = act_domains - (endpoint_domains - detailed_domains)
        return sorted(act_domains)

    current_slot_domains: set[str] = set()
    requested_domains: set[str] = set()
    changed_domains: set[str] = set()
    active_domains: set[str] = set()

    for service, state, slots in _iter_frames(frames):
        if service in VALID_DOMAINS:
            active_domains.add(service)

        intent = state.get("active_intent", "")
        if domain := _domain_from_intent(intent):
            active_domains.add(domain)

        current_slot_domains.update(_current_slot_domains(slots))
        requested_domains.update(
            d
            for slot in state.get("requested_slots", [])
            if (d := _domain_from_slot(slot))
        )
        changed_domains.update(_state_delta_domains(state, previous_state))

    current_domains = current_slot_domains | requested_domains
    if current_domains:
        return sorted(current_domains)
    if changed_domains:
        return sorted(changed_domains)
    return sorted(active_domains)


def convert_dialogue(dialogue: dict, max_context: int = 4) -> list[dict]:
    """Convert one dialogue into per-turn examples."""
    turns = dialogue["turns"]
    utterances = turns["utterance"]
    speakers = turns["speaker"]
    all_frames = turns["frames"]
    all_dialogue_acts = turns.get("dialogue_acts", [])

    examples: list[dict] = []
    history: list[str] = []
    previous_state: BeliefState = {}

    for i in range(len(utterances)):
        utt = utterances[i].strip()
        is_user = speakers[i] == 0
        prefix = "USER" if is_user else "SYSTEM"

        if is_user:
            if _CLOSING_RE.match(utt):
                domains = ["none"]
                context = " | ".join(history[-max_context:]) if history else ""
                examples.append(
                    {
                        "turn": utt,
                        "dialogue_context": context,
                        "domains": domains,
                    }
                )
                history.append(f"{prefix}: {utt}")
                continue

            frames = all_frames[i]
            dialogue_acts = all_dialogue_acts[i] if i < len(all_dialogue_acts) else None
            context = " | ".join(history[-max_context:]) if history else ""
            domains = extract_turn_domains(frames, previous_state, dialogue_acts)
            previous_state.update(_merge_state(frames))

            if not domains:
                history.append(f"{prefix}: {utt}")
                continue

            examples.append(
                {
                    "turn": utt,
                    "dialogue_context": context,
                    "domains": domains,
                }
            )

        history.append(f"{prefix}: {utt}")

    return examples


def main() -> None:
    if load_dataset is None:
        raise RuntimeError(
            "The datasets package is required. Run with "
            "`uv run --script scripts/convert_multiwoz.py`."
        )

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

        label_counts: dict[str, int] = {}
        for ex in examples:
            for domain in ex["domains"]:
                label_counts[domain] = label_counts.get(domain, 0) + 1
        dist = ", ".join(f"{k}={v}" for k, v in sorted(label_counts.items()))
        print(f"  {split_name} -> {out_path} ({len(examples)} examples)")
        print(f"    labels: {dist}")

    print("Done.")


if __name__ == "__main__":
    main()

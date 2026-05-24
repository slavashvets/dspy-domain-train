"""Build balanced hard slices from converted MultiWOZ domain data.

The slice is for evaluation/optimization diagnostics only. It does not alter
label policy. Examples are selected from already-converted gold labels and
balanced across structural difficulty categories.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

CATEGORIES = (
    "multi_domain",
    "taxi_implicit",
    "none_context",
    "short_answer",
    "booking_followup",
    "implicit_context",
)

# Used only to select a hard diagnostic slice, not to derive labels.
DOMAIN_TERMS = {
    "attraction": (
        "attraction",
        "attractions",
        "museum",
        "park",
        "college",
        "theatre",
        "cinema",
    ),
    "bus": ("bus",),
    "hospital": ("hospital",),
    "hotel": (
        "hotel",
        "guesthouse",
        "guest house",
        "place to stay",
        "accommodation",
        "room",
    ),
    "police": ("police",),
    "restaurant": (
        "restaurant",
        "restaurants",
        "food",
        "eat",
        "dine",
        "dining",
        "meal",
        "table",
    ),
    "taxi": ("taxi", "cab"),
    "train": ("train", "trains", "rail"),
}

SHORT_REPLY_RE = re.compile(
    r"\s*(?:no|yes|sure|ok|okay|that works|sounds good)\b",
    re.IGNORECASE,
)
BOOKING_RE = re.compile(
    r"\b(?:book|booking|reservation|reference|people|person|stay|night|"
    r"nights|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"\d{1,2}:\d{2})\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-category", type=int, default=10)
    parser.add_argument("--dev-per-category", type=int, default=15)
    parser.add_argument("--test-per-category", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Write an imbalanced split instead of failing when a bucket is short.",
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def has_domain_terms(text: str, domains: list[str]) -> bool:
    haystack = text.lower()
    return any(
        term in haystack
        for domain in domains
        for term in DOMAIN_TERMS.get(domain, (domain,))
    )


def primary_category(example: dict[str, Any]) -> str | None:
    domains = [domain for domain in example["domains"] if domain != "none"]
    turn = str(example["turn"])
    context = str(example["dialogue_context"])

    if len(domains) > 1:
        return "multi_domain"
    if "taxi" in domains and context and not has_domain_terms(turn, ["taxi"]):
        return "taxi_implicit"
    if example["domains"] == ["none"] and context:
        return "none_context"
    if (
        domains
        and context
        and SHORT_REPLY_RE.match(turn)
        and not has_domain_terms(turn, domains)
    ):
        return "short_answer"
    if (
        domains
        and context
        and BOOKING_RE.search(turn)
        and not has_domain_terms(turn, domains)
    ):
        return "booking_followup"
    if domains and context and not has_domain_terms(turn, domains):
        return "implicit_context"
    return None


def balance_key(example: dict[str, Any]) -> tuple[str, ...]:
    return tuple(example["domains"])


def balanced_take(
    examples: list[dict[str, Any]],
    *,
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    by_label: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        by_label[balance_key(example)].append(example)

    for items in by_label.values():
        rng.shuffle(items)

    keys = list(by_label)
    rng.shuffle(keys)
    selected: list[dict[str, Any]] = []
    while keys and len(selected) < count:
        next_keys: list[tuple[str, ...]] = []
        for key in keys:
            if by_label[key] and len(selected) < count:
                selected.append(by_label[key].pop())
            if by_label[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def build_split(
    examples: list[dict[str, Any]],
    *,
    split: str,
    per_category: int,
    seed: int,
    allow_short: bool = False,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_index, example in enumerate(examples):
        category = primary_category(example)
        if category is not None:
            buckets[category].append(
                {
                    **example,
                    "example_id": f"{split}:{source_index}",
                    "source_split": split,
                    "source_index": source_index,
                }
            )

    selected: list[dict[str, Any]] = []
    for offset, category in enumerate(CATEGORIES):
        rng = random.Random(seed + offset)
        chosen = balanced_take(
            buckets.get(category, []),
            count=per_category,
            rng=rng,
        )
        if len(chosen) < per_category and not allow_short:
            raise ValueError(
                f"{split} bucket {category!r} has only {len(chosen)} examples; "
                f"requested {per_category}. Use --allow-short to write it anyway."
            )
        for example in chosen:
            selected.append({**example, "hard_category": category})

    random.Random(seed).shuffle(selected)
    return selected


def split_summary(examples: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(example["hard_category"] for example in examples)
    labels = Counter(domain for example in examples for domain in example["domains"])
    return {
        "total": len(examples),
        "categories": dict(sorted(categories.items())),
        "labels": dict(sorted(labels.items())),
    }


def summary(name: str, examples: list[dict[str, Any]]) -> None:
    data = split_summary(examples)
    print(f"{name}: {len(examples)} examples")
    print(f"  categories: {data['categories']}")
    print(f"  labels: {data['labels']}")


def main() -> None:
    args = parse_args()
    quotas = {
        "train": args.train_per_category,
        "dev": args.dev_per_category,
        "test": args.test_per_category,
    }
    manifest: dict[str, Any] = {
        "seed": args.seed,
        "quotas": quotas,
        "categories": list(CATEGORIES),
        "allow_short": args.allow_short,
        "splits": {},
    }

    for split, per_category in quotas.items():
        examples = load_json(DATA_DIR / f"{split}.json")
        hard = build_split(
            examples,
            split=split,
            per_category=per_category,
            seed=args.seed,
            allow_short=args.allow_short,
        )
        out_path = DATA_DIR / f"hard_{split}.json"
        out_path.write_text(
            json.dumps(hard, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary(out_path.name, hard)
        manifest["splits"][split] = split_summary(hard)

    (DATA_DIR / "hard_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

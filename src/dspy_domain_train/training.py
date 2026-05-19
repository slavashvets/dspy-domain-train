import json
import logging
import random
from collections.abc import Callable
from pathlib import Path

import dspy

from .settings import AzureOpenAIModelSettings, Settings


def _make_lm(cfg: AzureOpenAIModelSettings) -> dspy.LM:
    return dspy.LM(
        model=f"azure/{cfg.deployment}",
        model_type=cfg.model_type,
        api_base=str(cfg.endpoint),
        api_version=cfg.api_version,
        api_key=cfg.api_key.get_secret_value(),
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


def configure_lm(settings: Settings) -> tuple[dspy.LM, dspy.LM]:
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    logging.getLogger("dspy").setLevel(logging.INFO)

    lm_eval = _make_lm(settings.eval)
    json_adapter = dspy.JSONAdapter()

    dspy.configure(
        lm=lm_eval,
        adapter=json_adapter,
        track_usage=True,
    )

    lm_refine = _make_lm(settings.refine)
    return lm_eval, lm_refine


def load_examples(path: Path) -> list[dspy.Example]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        dspy.Example(
            dialogue_context=item["dialogue_context"],
            turn=item["turn"],
            domains=item["domains"],
        ).with_inputs("dialogue_context", "turn")
        for item in raw
    ]


def split_dataset(
    examples: list[dspy.Example],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def optimize(
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    metric: Callable,
    initial_instructions: str,
    reflection_lm: dspy.LM,
    gepa_auto: str = "light",
    num_threads: int = 4,
    log_dir: str | None = None,
) -> dspy.Module:
    from .domain_task import DomainClassifier

    classifier = DomainClassifier(instructions=initial_instructions)

    optimizer = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        auto=gepa_auto,
        num_threads=num_threads,
        track_stats=True,
        log_dir=log_dir,
    )

    return optimizer.compile(
        classifier,
        trainset=trainset,
        valset=valset,
    )

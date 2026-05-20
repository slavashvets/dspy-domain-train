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


def configure_lm(settings: Settings) -> dspy.LM:
    """Configure DSPy global LM and return the refine/prompt model."""
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    logging.getLogger("dspy").setLevel(logging.INFO)

    lm_eval = _make_lm(settings.eval)
    dspy.configure(lm=lm_eval, adapter=dspy.JSONAdapter())

    return _make_lm(settings.refine)


def load_examples(
    path: Path, max_samples: int | None = None, seed: int = 42
) -> list[dspy.Example]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if max_samples is not None and max_samples < len(raw):
        raw = random.Random(seed).sample(raw, max_samples)
    return [
        dspy.Example(
            dialogue_context=item["dialogue_context"],
            turn=item["turn"],
            domains=item["domains"],
        ).with_inputs("dialogue_context", "turn")
        for item in raw
    ]


def optimize_copro(
    trainset: list[dspy.Example],
    metric: Callable,
    prompt_model: dspy.LM,
    initial_instructions: str | None = None,
    breadth: int = 10,
    depth: int = 3,
    init_temperature: float = 1.4,
    num_threads: int = 4,
) -> dspy.Module:
    from .domain_task import DomainClassifier

    student = DomainClassifier(instructions=initial_instructions)

    optimizer = dspy.COPRO(
        prompt_model=prompt_model,
        metric=metric,
        breadth=breadth,
        depth=depth,
        init_temperature=init_temperature,
        track_stats=True,
    )

    return optimizer.compile(
        student,
        trainset=trainset,
        eval_kwargs={
            "num_threads": num_threads,
            "display_progress": True,
        },
    )


def optimize_simba(
    trainset: list[dspy.Example],
    metric: Callable,
    prompt_model: dspy.LM,
    initial_instructions: str | None = None,
    max_steps: int = 6,
    bsize: int = 32,
    num_candidates: int = 6,
    num_threads: int = 4,
    seed: int = 0,
) -> dspy.Module:
    from .domain_task import DomainClassifier

    student = DomainClassifier(instructions=initial_instructions)

    optimizer = dspy.SIMBA(
        metric=metric,
        prompt_model=prompt_model,
        max_demos=0,
        max_steps=max_steps,
        bsize=bsize,
        num_candidates=num_candidates,
        num_threads=num_threads,
    )

    return optimizer.compile(student, trainset=trainset, seed=seed)


def optimize_gepa(
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    metric: Callable,
    reflection_lm: dspy.LM,
    initial_instructions: str | None = None,
    gepa_auto: str = "light",
    num_threads: int = 4,
    log_dir: str | None = None,
) -> dspy.Module:
    from .domain_task import DomainClassifier

    student = DomainClassifier(instructions=initial_instructions)

    optimizer = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        auto=gepa_auto,
        num_threads=num_threads,
        track_stats=True,
        log_dir=log_dir,
    )

    return optimizer.compile(student, trainset=trainset, valset=valset)


def optimize_srp(
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    metric: Callable,
    prompt_model: dspy.LM,
    initial_instructions: str | None = None,
    max_iters: int = 6,
    patience: int = 2,
    max_examples: int | None = None,
    max_error_cases: int = 12,
    num_threads: int = 4,
    seed: int = 0,
) -> dspy.Module:
    from .domain_task import DomainClassifier
    from .srp import SRP

    student = DomainClassifier(instructions=initial_instructions)

    optimizer = SRP(
        metric=metric,
        prompt_model=prompt_model,
        max_iters=max_iters,
        patience=patience,
        max_examples=max_examples,
        max_error_cases=max_error_cases,
        num_threads=num_threads,
        seed=seed,
        display_progress=True,
    )

    return optimizer.compile(student, trainset=trainset, valset=valset)


def get_instructions(program: dspy.Module) -> str:
    return program.predict.signature.instructions or ""

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


_COPRO_CONSTRAINT = (
    "\n\nCONSTRAINTS: Keep the instruction under 120 words."
    " Use concise decision rules, not verbose explanations."
    " Do NOT include worked examples in the instruction."
    " The task is MULTI-LABEL: the output may contain one or more domains."
)


def optimize_copro(
    trainset: list[dspy.Example],
    metric: Callable,
    initial_instructions: str,
    prompt_model: dspy.LM,
    breadth: int = 10,
    depth: int = 3,
    init_temperature: float = 1.4,
    num_threads: int = 4,
) -> dspy.Module:
    import dspy.teleprompt.copro_optimizer as copro_mod

    from .domain_task import DomainClassifier

    _patch_copro_signatures(copro_mod)

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


def _patch_copro_signatures(copro_mod: object) -> None:
    """Inject conciseness constraints into COPRO's meta-prompts."""
    if getattr(copro_mod, "_patched", False):
        return

    for cls_name in ("BasicGenerateInstruction", "GenerateInstructionGivenAttempts"):
        orig = getattr(copro_mod, cls_name)
        new_doc = (orig.__doc__ or "") + _COPRO_CONSTRAINT
        patched = type(cls_name, (orig,), {"__doc__": new_doc})
        setattr(copro_mod, cls_name, patched)

    copro_mod._patched = True  # type: ignore[attr-defined]


def optimize_gepa(
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


def optimize_simba(
    trainset: list[dspy.Example],
    metric: Callable,
    initial_instructions: str,
    prompt_model: dspy.LM,
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


def get_instructions(program: dspy.Module) -> str:
    predictor = getattr(program, "predict", None)
    if predictor is None:
        return ""
    signature = getattr(predictor, "signature", None)
    if signature is None:
        return ""
    return getattr(signature, "instructions", "") or ""


def select_best_candidate(
    optimized: dspy.Module,
    evalset: list[dspy.Example],
    metric: Callable,
    max_instruction_words: int,
    num_threads: int = 4,
) -> dspy.Module:
    candidates = getattr(optimized, "candidate_programs", None)
    if not candidates:
        detailed = getattr(optimized, "detailed_results", None)
        if detailed:
            raw = getattr(detailed, "candidates", None)
            if raw:
                candidates = [{"program": p} for p in raw]

    if not candidates:
        return optimized

    evaluator = dspy.Evaluate(
        devset=evalset,
        metric=metric,
        num_threads=num_threads,
        display_progress=False,
    )

    scored: list[tuple[float, int, dspy.Module]] = []
    for item in candidates:
        program = item["program"] if isinstance(item, dict) else item
        if program is None:
            continue

        instructions = get_instructions(program)
        word_count = len(instructions.split())
        if word_count > max_instruction_words:
            continue

        result = evaluator(program)
        scored.append((float(result.score), -word_count, program))

    if not scored:
        return optimized

    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return scored[0][2]

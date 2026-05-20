import json
from datetime import UTC, datetime
from pathlib import Path

import dspy

from .domain_task import domain_metric
from .settings import get_settings
from .training import (
    configure_lm,
    get_instructions,
    load_examples,
    optimize_copro,
    optimize_gepa,
    optimize_simba,
    optimize_srp,
)


def main() -> None:
    settings = get_settings()
    lm_refine = configure_lm(settings)

    trainset = load_examples(settings.train_path, settings.max_train, settings.seed)
    valset = load_examples(settings.dev_path, settings.max_dev, settings.seed)
    testset = load_examples(settings.test_path, settings.max_test, settings.seed)

    initial_instructions: str | None = None
    if settings.initial_prompt_path is not None:
        initial_instructions = settings.initial_prompt_path.read_text(
            encoding="utf-8"
        ).strip()

    run_dir = Path("runs") / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train: {len(trainset)} | Val: {len(valset)} | Test: {len(testset)}")
    print(f"Optimizer: {settings.optimizer}")
    print(f"Log dir: {run_dir}")

    match settings.optimizer:
        case "copro":
            optimized = optimize_copro(
                trainset=trainset,
                metric=domain_metric,
                prompt_model=lm_refine,
                initial_instructions=initial_instructions,
                breadth=settings.copro.breadth,
                depth=settings.copro.depth,
                init_temperature=settings.copro.init_temperature,
                num_threads=settings.num_threads,
            )
        case "simba":
            optimized = optimize_simba(
                trainset=trainset,
                metric=domain_metric,
                prompt_model=lm_refine,
                initial_instructions=initial_instructions,
                max_steps=settings.simba.max_steps,
                bsize=settings.simba.bsize,
                num_candidates=settings.simba.num_candidates,
                num_threads=settings.num_threads,
                seed=settings.seed,
            )
        case "gepa":
            optimized = optimize_gepa(
                trainset=trainset,
                valset=valset,
                metric=domain_metric,
                reflection_lm=lm_refine,
                initial_instructions=initial_instructions,
                gepa_auto=settings.gepa.auto,
                num_threads=settings.num_threads,
                log_dir=str(run_dir / "gepa"),
            )
        case "srp":
            optimized = optimize_srp(
                trainset=trainset,
                valset=valset,
                metric=domain_metric,
                prompt_model=lm_refine,
                initial_instructions=initial_instructions,
                max_iters=settings.srp.max_iters,
                patience=settings.srp.patience,
                max_examples=settings.srp.max_examples,
                max_error_cases=settings.srp.max_error_cases,
                num_threads=settings.num_threads,
                seed=settings.seed,
            )

    evaluator = dspy.Evaluate(
        devset=testset,
        metric=domain_metric,
        num_threads=settings.num_threads,
        display_progress=True,
    )
    result = evaluator(optimized)
    print(f"\nTest score: {result.score:.1f}%")

    final_instructions = get_instructions(optimized)
    print(f"\nFinal instructions:\n{final_instructions}")

    optimized.save(str(run_dir / "program.json"))
    (run_dir / "final_instructions.txt").write_text(
        final_instructions + "\n", encoding="utf-8"
    )

    trial_logs = None
    if settings.optimizer == "srp":
        trial_logs = getattr(optimized, "trial_logs", None)
        candidates = getattr(optimized, "candidate_programs", None)
        if trial_logs is not None:
            srp_data = {
                "trial_logs": trial_logs,
                "candidates": [
                    {
                        "iteration": c["iteration"],
                        "score": c["score"],
                        "accepted": c["accepted"],
                        "rules": c["rules"],
                        "instruction": c["instruction"],
                    }
                    for c in (candidates or [])
                ],
            }
            (run_dir / "srp_candidates.json").write_text(
                json.dumps(srp_data, indent=2, ensure_ascii=False) + "\n"
            )

    metadata = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "dspy_version": dspy.__version__,
        "optimizer": settings.optimizer,
        "test_score": result.score,
        "train_size": len(trainset),
        "val_size": len(valset),
        "test_size": len(testset),
        "final_instruction_words": len(final_instructions.split()),
        "initial_prompt_path": str(settings.initial_prompt_path)
        if settings.initial_prompt_path
        else None,
        "settings": {
            "num_threads": settings.num_threads,
            "seed": settings.seed,
            "eval_deployment": settings.eval.deployment,
            "refine_deployment": settings.refine.deployment,
        },
    }
    if trial_logs:
        metadata["trial_logs"] = trial_logs
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()

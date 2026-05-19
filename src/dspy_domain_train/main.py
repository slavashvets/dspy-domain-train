import json
from datetime import UTC, datetime
from pathlib import Path

import dspy

from .domain_task import domain_metric, load_text
from .settings import get_settings
from .training import (
    configure_lm,
    get_instructions,
    load_examples,
    optimize_copro,
    optimize_gepa,
    optimize_simba,
    select_best_candidate,
)


def main() -> None:
    settings = get_settings()
    lm_eval, lm_refine = configure_lm(settings)

    trainset = load_examples(settings.train_path, settings.max_train, settings.seed)
    valset = load_examples(settings.dev_path, settings.max_dev, settings.seed)
    testset = load_examples(settings.test_path, settings.max_test, settings.seed)
    p0 = load_text(settings.prompt_path)

    run_dir = Path("runs") / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train: {len(trainset)} | Val: {len(valset)} | Test: {len(testset)}")
    print(f"Optimizer: {settings.optimizer}")
    print(f"Log dir: {run_dir}")

    if settings.optimizer == "copro":
        optimized = optimize_copro(
            trainset=valset,
            metric=domain_metric,
            initial_instructions=p0,
            prompt_model=lm_refine,
            breadth=settings.copro.breadth,
            depth=settings.copro.depth,
            init_temperature=settings.copro.init_temperature,
            num_threads=settings.num_threads,
        )
    elif settings.optimizer == "simba":
        optimized = optimize_simba(
            trainset=trainset,
            metric=domain_metric,
            initial_instructions=p0,
            prompt_model=lm_refine,
            max_steps=settings.simba.max_steps,
            bsize=settings.simba.bsize,
            num_candidates=settings.simba.num_candidates,
            num_threads=settings.num_threads,
            seed=settings.seed,
        )
    else:
        optimized = optimize_gepa(
            trainset=trainset,
            valset=valset,
            metric=domain_metric,
            initial_instructions=p0,
            reflection_lm=lm_refine,
            gepa_auto=settings.gepa.auto,
            num_threads=settings.num_threads,
            log_dir=str(run_dir / "gepa"),
        )

    optimized = select_best_candidate(
        optimized=optimized,
        evalset=valset,
        metric=domain_metric,
        max_instruction_words=settings.max_instruction_words,
        num_threads=settings.num_threads,
    )

    evaluator = dspy.Evaluate(
        devset=testset,
        metric=domain_metric,
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

    metadata = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "dspy_version": dspy.__version__,
        "optimizer": settings.optimizer,
        "test_score": result.score,
        "train_size": len(trainset),
        "val_size": len(valset),
        "test_size": len(testset),
        "final_instruction_words": len(final_instructions.split()),
        "settings": {
            "num_threads": settings.num_threads,
            "seed": settings.seed,
            "max_instruction_words": settings.max_instruction_words,
            "eval_deployment": settings.eval.deployment,
            "refine_deployment": settings.refine.deployment,
        },
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()

import json
from datetime import UTC, datetime
from pathlib import Path

import dspy

from .domain_task import domain_metric, load_text
from .settings import get_settings
from .training import configure_lm, load_examples, optimize, split_dataset


def main() -> None:
    settings = get_settings()
    lm_eval, lm_refine = configure_lm(settings)

    examples = load_examples(settings.devset_path)
    trainset, valset = split_dataset(examples, settings.val_ratio, settings.seed)
    p0 = load_text(settings.prompt_path)

    run_dir = Path("runs") / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train: {len(trainset)} | Val: {len(valset)}")
    print(f"Log dir: {run_dir}")

    optimized = optimize(
        trainset=trainset,
        valset=valset,
        metric=domain_metric,
        initial_instructions=p0,
        reflection_lm=lm_refine,
        gepa_auto=settings.gepa_auto,
        num_threads=settings.num_threads,
        log_dir=str(run_dir / "gepa"),
    )

    evaluator = dspy.Evaluate(
        devset=valset,
        metric=domain_metric,
        display_progress=True,
    )
    result = evaluator(optimized)
    print(f"\nVal score: {result.score:.1f}%")

    optimized.save(str(run_dir / "program.json"))

    metadata = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "dspy_version": dspy.__version__,
        "val_score": result.score,
        "train_size": len(trainset),
        "val_size": len(valset),
        "settings": {
            "gepa_auto": settings.gepa_auto,
            "num_threads": settings.num_threads,
            "seed": settings.seed,
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

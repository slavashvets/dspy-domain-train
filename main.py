from domain_task import (
    DomainClassifier,
    build_error_report,
    load_devset,
    load_text,
    make_auto_refiner,
    parse_prediction,
)
from settings import get_settings
from training import BestResult, configure_lm, offline_srp


def main() -> None:
    settings = get_settings()
    lms = configure_lm(settings)

    devset = load_devset(settings.devset_path)
    p0 = load_text(settings.prompt_path)

    best: BestResult = offline_srp(
        devset=devset,
        instructions=p0,
        make_classifier=DomainClassifier,
        parse_pred=parse_prediction,
        build_error_report=build_error_report,
        make_auto_refiner=make_auto_refiner,
        max_iters=settings.max_iters,
        tol=settings.tol,
        eval_lm=lms["eval"],
        refiner_lm=lms["refine"],
        instr_max_len=settings.instr_max_len,
        refiner_candidates=settings.refiner_candidates,
        refiner_retries=settings.refiner_retries,
    )

    print(f"\nFinal dev accuracy: {best.acc:.3f} on {len(devset)} examples")
    print("\nFinal instructions (P*):\n" + best.instr)
    if best.errors:
        print("\nRemaining errors:")
        for e in best.errors:
            print(f"- Ut: {e['turn']} | GT: {e['gt']} | Pred: {e['pred']}")


if __name__ == "__main__":
    main()

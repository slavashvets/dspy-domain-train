import logging
from dataclasses import dataclass
from typing import Callable, Sequence, TypedDict

import dspy

from instrumentation import add_usage, format_usage
from settings import AzureOpenAIModelSettings, Settings


class Example(TypedDict):
    history: str
    turn: str
    gt: list[str]


class ErrorCase(TypedDict):
    history: str
    turn: str
    gt: list[str]
    pred: list[str]


@dataclass
class BestResult:
    instr: str
    acc: float
    errors: list[ErrorCase]


def _make_lm(cfg: AzureOpenAIModelSettings) -> dspy.LM:
    """Build a dspy.LM instance from a single model configuration."""
    return dspy.LM(
        model=f"azure/{cfg.deployment}",
        model_type=cfg.model_type,
        api_base=str(cfg.endpoint),
        api_version=cfg.api_version,
        api_key=cfg.api_key,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


def configure_lm(settings: Settings) -> tuple[dspy.LM, dspy.LM]:
    """Configure all LMs via strongly typed settings instead of ad-hoc env reads."""
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    logging.getLogger("dspy").setLevel(logging.INFO)

    # Eval / classifier LM
    lm_eval = _make_lm(settings.eval)

    # Enable JSONAdapter globally so signatures get JSON Schema structured outputs
    json_adapter = dspy.JSONAdapter()

    dspy.configure(
        lm=lm_eval,
        adapter=json_adapter,
        track_usage=True,
    )

    # Refiner LM
    lm_refine = _make_lm(settings.refine)

    return lm_eval, lm_refine


def evaluate(
    program: dspy.Module,
    instructions: str,
    dataset: Sequence[Example],
    parse_pred: Callable[[dspy.Prediction], list[str]],
    verbose: bool = True,
) -> tuple[float, list[ErrorCase]]:
    """Generic evaluation hook used by tasks and refiner reward functions."""
    correct = 0
    errors: list[ErrorCase] = []
    usage_total: dict = {}

    n = len(dataset)
    print_every = max(1, n // 4) if n else 1

    for i, ex in enumerate(dataset, 1):
        out = program(instructions, ex["history"], ex["turn"])
        pred = parse_pred(out)
        gt = [d.lower() for d in ex["gt"]]

        # Accumulate usage when available; never fail on telemetry.
        try:
            add_usage(usage_total, out.get_lm_usage())
        except Exception:
            pass

        if set(pred) == set(gt):
            correct += 1
        else:
            errors.append(
                ErrorCase(history=ex["history"], turn=ex["turn"], gt=gt, pred=pred)
            )

        if verbose and (i % print_every == 0 or i == n):
            print(
                f"  progress {i}/{n} | usage {format_usage(usage_total)}",
                flush=True,
            )

    acc = correct / n if n else 0.0
    return acc, errors


def offline_srp(
    devset: Sequence[Example],
    instructions: str,
    make_classifier: Callable[[], dspy.Module],
    parse_pred: Callable[[dspy.Prediction], list[str]],
    build_error_report: Callable[[Sequence[ErrorCase]], str],
    make_auto_refiner: Callable[
        [Sequence[Example], dspy.LM, int, int, int], dspy.Module
    ],
    max_iters: int = 6,
    tol: float = 5e-3,
    eval_lm: dspy.LM | None = None,
    refiner_lm: dspy.LM | None = None,
    instr_max_len: int = 50,
    refiner_candidates: int = 5,
    refiner_retries: int = 1,
) -> BestResult:
    """SRP loop agnostic of specific classification task."""
    if eval_lm is None:
        raise ValueError("eval_lm must be provided for offline_srp")

    clf = make_classifier()
    ref = make_auto_refiner(
        devset,
        eval_lm,
        instr_max_len,
        refiner_candidates,
        refiner_retries,
    )
    instr = instructions
    best = BestResult(instr=instr, acc=0.0, errors=[])

    for it in range(1, max_iters + 1):
        # Evaluate current instructions with normal tracking
        pre_acc, pre_errs = evaluate(clf, instr, devset, parse_pred)
        print(f"[Iter {it} pre]  accuracy={pre_acc:.3f}  errors={len(pre_errs)}")

        prev_best = best.acc
        if pre_acc >= prev_best:
            best.instr = instr
            best.acc = pre_acc
            best.errors = list(pre_errs)

        report = build_error_report(pre_errs)

        # Propose a revision
        with dspy.context(lm=refiner_lm, adapter=dspy.ChatAdapter(), track_usage=False):
            revised = ref(
                current_instructions=instr,
                error_report=report,
            ).revised_instructions.strip()

        if not revised or revised == instr:
            # No effective update; stop if we're already good enough.
            if pre_acc == 1.0 or (pre_acc - prev_best < tol and not pre_errs):
                break
            continue

        # Re-evaluate the revised instructions with tracking back on
        post_acc, post_errs = evaluate(clf, revised, devset, parse_pred)
        print(f"[Iter {it} post] accuracy={post_acc:.3f}  errors={len(post_errs)}")

        if post_acc >= pre_acc:
            instr = revised
            if post_acc >= best.acc:
                best.instr = instr
                best.acc = post_acc
                best.errors = list(post_errs)

        improved = best.acc - prev_best
        if best.acc == 1.0 or (improved < tol and not best.errors):
            break

    # Final pass using the best instructions found.
    final_acc, final_errs = evaluate(make_classifier(), best.instr, devset, parse_pred)
    best.acc = final_acc
    best.errors = list(final_errs)
    return best

# dspy-domain-train

Multilabel domain classifier optimized with [DSPy](https://dspy.ai/) and
Azure OpenAI.

The project includes a custom **SRP (Self-Reflective Prompting)** optimizer
that iteratively appends concise rules to a predictor's instruction by
analyzing error cases. It also supports DSPy's built-in GEPA for evolutionary
instruction optimization.

## Requirements

The runtime is managed by [mise](https://mise.jdx.dev/). The project pins both
Python and uv in `mise.toml`:

- Python 3.13.9
- uv 0.11.15

Install the tools and locked dependencies:

```bash
mise install
mise run sync
```

## Configuration

Configuration is loaded from TOML files with pydantic-settings:

```bash
cp settings.local.example.toml settings.local.toml
```

`settings.toml` is the tracked base configuration: runtime knobs, data paths,
and shared defaults. `settings.local.toml` is ignored by git and contains
local Azure OpenAI endpoints, API keys, deployments, and model-specific details.

The default profile is `local`, so the loader deep-merges:

1. `settings.toml`
2. `settings.local.toml`

Use `DSPY_PROFILE=<name>` to load `settings.<name>.toml` instead of the local
overlay. Environment variables with the `DSPY_` prefix still override TOML
values, for example `DSPY_NUM_THREADS=8` or `DSPY_EVAL__DEPLOYMENT=gpt-4-1`.
Relative paths are resolved from the directory containing `settings.toml`.

For DSPy/LiteLLM Azure usage, `DSPY_*__DEPLOYMENT` is the Azure deployment name,
`DSPY_*__API_VERSION = "v1"` selects Azure's OpenAI-compatible `/openai/v1/`
route, and `DSPY_*__MODEL_TYPE` should match the deployment API style (`chat`
or `responses`).

## Commands

```bash
mise run srp       # run SRP optimization loop (default)
mise run gepa      # run GEPA optimization loop
mise run srp:demo  # SRP on small sample (smoke-test)
mise run gepa:demo # GEPA on small sample (smoke-test)
mise run format    # format Python files with Ruff
mise run lint      # run Ruff and mypy
mise run test      # run unit tests that do not call Azure
mise run check     # run all static checks plus compileall
```

## Data Format

`data/dev.json` is a list of examples:

```json
{
  "dialogue_context": "Previous dialogue context, or an empty string.",
  "turn": "User turn to classify.",
  "domains": ["hotel", "taxi"]
}
```

Allowed labels are `restaurant`, `attraction`, `hotel`, `taxi`, `train`, `bus`,
`hospital`, `police`, and `none`.

## SRP Optimizer

SRP (Self-Reflective Prompting) is a custom DSPy `Teleprompter` that:

1. Evaluates the current program on a working set
2. Collects error cases (inputs, gold, prediction, feedback)
3. Uses a refiner LM to propose 1-3 concise rules addressing those errors
4. Appends rules to the predictor instruction
5. Accepts the candidate if score improves, otherwise increments patience counter
6. Stops on perfect score, patience exhaustion, or max iterations

```python
from dspy_domain_train.srp import SRP

optimizer = SRP(
    metric=domain_metric,
    prompt_model=refine_lm,
    max_iters=6,
    patience=2,
    max_error_cases=12,
    num_threads=4,
)
optimized = optimizer.compile(student, trainset=trainset, valset=valset)
```

The returned module carries `candidate_programs` (chronological list with
iteration, score, rules, accepted) and `trial_logs` (best_score,
stopped_reason).

# dspy-domain-train

Offline self-refinement loop for a small multilabel domain classifier built with
[DSPy](https://dspy.ai/) and Azure OpenAI.

The packaged application evaluates a prompt on a development set, asks a separate
refinement model to propose better general instructions, rejects dataset-specific
rewrites, and keeps the best prompt by dev accuracy.

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

`settings.toml` is the tracked base configuration: runtime knobs, prompt/data
paths, and shared defaults. `settings.local.toml` is ignored by git and contains
local Azure OpenAI endpoints, API keys, deployments, and model-specific details.

The default profile is `local`, so the loader deep-merges:

1. `settings.toml`
2. `settings.local.toml`

Use `DSPY_PROFILE=<name>` to load `settings.<name>.toml` instead of the local
overlay. Environment variables with the `DSPY_` prefix still override TOML
values, for example `DSPY_MAX_ITERS=2` or `DSPY_EVAL__DEPLOYMENT=gpt-4-1`.
Relative paths are resolved from the directory containing `settings.toml`.

For DSPy/LiteLLM Azure usage, `DSPY_*__DEPLOYMENT` is the Azure deployment name,
`DSPY_*__API_VERSION = "v1"` selects Azure's OpenAI-compatible `/openai/v1/`
route, and `DSPY_*__MODEL_TYPE` should match the deployment API style (`chat`
or `responses`).

## Commands

```bash
mise run dev      # run one offline SRP training loop
# or: uv run --locked dspy-domain-train
mise run format   # format Python files with Ruff
mise run lint     # run Ruff and mypy
mise run test     # run unit tests that do not call Azure
mise run check    # run all static checks plus compileall
```

## Data Format

`data/dev.json` is a list of examples:

```json
{
  "history": "Previous dialogue context, or an empty string.",
  "turn": "User turn to classify.",
  "gt": ["hotel", "taxi"]
}
```

Allowed labels are `restaurant`, `attraction`, `hotel`, `taxi`, `train`, `bus`,
`hospital`, `police`, and `none`.

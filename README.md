# dspy-domain-train

Multilabel domain classifier optimized with [DSPy](https://dspy.ai/) GEPA and
Azure OpenAI.

The application uses DSPy's GEPA optimizer to evolve classification instructions
through reflective prompt optimization: a cheap task model classifies dialogue
turns, while a stronger reflection model analyzes errors and proposes instruction
mutations. The best instructions are selected via Pareto frontier on a validation
split.

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
values, for example `DSPY_NUM_THREADS=8` or `DSPY_EVAL__DEPLOYMENT=gpt-4-1`.
Relative paths are resolved from the directory containing `settings.toml`.

For DSPy/LiteLLM Azure usage, `DSPY_*__DEPLOYMENT` is the Azure deployment name,
`DSPY_*__API_VERSION = "v1"` selects Azure's OpenAI-compatible `/openai/v1/`
route, and `DSPY_*__MODEL_TYPE` should match the deployment API style (`chat`
or `responses`).

## Commands

```bash
mise run dev      # run GEPA optimization loop
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
  "dialogue_context": "Previous dialogue context, or an empty string.",
  "turn": "User turn to classify.",
  "domains": ["hotel", "taxi"]
}
```

Allowed labels are `restaurant`, `attraction`, `hotel`, `taxi`, `train`, `bus`,
`hospital`, `police`, and `none`.

## DSPy Optimizer Cheat Sheet

DSPy optimizers tune two knobs: **instructions** (task description text) and
**demos** (few-shot examples baked into the prompt).

| Optimizer | Optimizes | Mechanism |
|-----------|-----------|-----------|
| BootstrapFewShot | demos only | Teacher solves examples, successful traces become few-shot demos |
| COPRO | instructions only | Generate N instruction variants, score each, pick the best |
| MIPROv2 | instructions + demos | Bootstrap demos + generate instruction candidates + Bayesian search over combinations |
| GEPA | instructions only | Evolutionary: error traces + textual feedback → LLM mutates instructions → Pareto frontier selection |
| SIMBA | instructions + demos | Finds unstable examples, generates targeted rules or adds demos for failure cases |

### When to use what

| Scenario | Pick | Why |
|----------|------|-----|
| Few examples (10-50) | BootstrapFewShot | Demos help most when data is scarce |
| Medium dataset (50-500) | MIPROv2 `auto="light"` | Enough data for Bayesian search without heavy budget |
| Large dataset (500+) | MIPROv2 `auto="heavy"` | Full search pays off |
| Minimal token budget for optimization | BootstrapFewShot or COPRO | Fewest LLM calls (~70-200) |
| Final prompt must be short (no demos) | GEPA or COPRO | Only rewrite instructions, don't inflate prompt with examples |
| Best result, cost doesn't matter | MIPROv2 heavy, then GEPA on top | Joint optimization of everything |
| Classification / structured output | BootstrapFewShot → MIPROv2 | Demos strongly help classifiers |
| Intent detection / production runtime | GEPA | Short prompt with precise rules, no demo overhead per call |
| Agents / multi-step / ReAct | SIMBA | Designed for unstable long pipelines |
| Subtle edge-case errors | GEPA | Reflection on failures catches nuanced patterns |
| Prompt already decent, squeezing last % | GEPA or SIMBA | Focus on failure cases, not broad exploration |
| Need something in 5 minutes | COPRO | One round: generate variants → pick winner |

### Resource comparison

| | BootstrapFewShot | COPRO | MIPROv2 light | MIPROv2 heavy | GEPA | SIMBA |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| LLM calls | ~70-150 | ~100-200 | ~200-400 | ~1000+ | ~300-600 | ~500-1000 |
| Min examples | 10 | 20 | 50 | 200+ | 20 | 32 |
| Needs teacher LM | yes | no | optional | optional | no | no |
| Needs strong reflector LM | no | no | no | no | yes | no |
| Makes prompt longer | yes (demos) | no | yes (demos) | yes (demos) | no | slightly |

### Quick decision tree

```
Need short production prompt (no demos)?
├─ Yes → GEPA (or COPRO if budget is tight)
└─ No, prompt size is fine
   ├─ < 50 examples → BootstrapFewShot
   └─ 50+ examples
      ├─ Prompt works OK, need to fix edge cases → GEPA
      ├─ Unstable results across runs → SIMBA
      └─ Starting fresh or want full optimization → MIPROv2
```

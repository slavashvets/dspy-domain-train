# dspy-train-classification

Offline SRP loop for a small domain classifier built with DSPy and Azure OpenAI.

## Prerequisites

- [mise](https://mise.jdx.dev/installing-mise.html)
- Azure OpenAI with:
  - one deployment for eval/classification
  - one deployment for refinement

## Quickstart

1. Copy env template and fill Azure settings:

   ```bash
   cp .env.example .env
   ```

2. From the project root:

   ```bash
   mise run
   ```

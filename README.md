# Biomedical analysis agent

Given a biomedical dataset and a free-text objective, the agent profiles the data, lists
the analytical challenges it poses (the *challenge register*), plans an analysis across a
registry of tools -- including the dimension-reduction (Project 1), generative-modelling
(Project 2) and multimodal (Project 3) agents -- runs it, lets a critic order at most one
revision, and writes a report.

This README grows with the code; the full version (results, design, sources of every
reused component) comes with the finished agent.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it if missing).

```bash
uv sync                       # creates .venv from uv.lock
uv run bioagent hardware      # detected device, disk and compute budgets
```

`requirements.txt` pins the same versions for `pip install -r requirements.txt`.

LLM calls go through the locally installed, logged-in `claude` CLI (Claude Code) via
`claude-agent-sdk`; no API key is used. `--no-llm` runs everything with the rule-based
planner, critic and writer instead.

## Data

```bash
uv run python -I scripts/prepare_data.py --only quick   # small subsets for smoke tests (~0.3 GB)
uv run python -I scripts/prepare_data.py                # full data and the quick subsets (~7 GB download)
```

`prepare_data.py --help` lists the sizes and options, including `--dataset` to prepare one
dataset, `--reuse-from` to copy verified files from an earlier download instead of fetching
them again, and `--delete-archives` to remove the image archives after extraction.

## Tests

```bash
uv run pytest -q
```

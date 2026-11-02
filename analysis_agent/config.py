"""Run configuration: budgets, seeds, LLM settings and reporting thresholds -- never analysis choices.

A config file may say *how much* compute a run gets, *which* LLM to ask, *what*
seed to use and *what counts as a finding* (the FDR level, the bar for "reliably
predicted"). It may not say which tool, method, number of clusters or correction
to use; those come only from the data profile, the challenge register and the
objective. Keeping the two apart is what stops a per-dataset config from becoming
a dataset-specific branch in disguise (from Project 3, adapted there from Project 2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

# Multiplier on the per-fit training budget, relative to the reference machine (an
# M2 Pro on MPS). A CPU-only run gets less wall-clock per fit, so it trains for fewer
# epochs rather than taking twice as long; the run-level and step-level caps stay,
# and the planner subsamples when its cost estimate does not fit them.
DEVICE_BUDGET_SCALE = {"mps": 1.0, "cuda": 1.0, "cpu": 0.5}


class Budget(BaseModel):
    """Compute caps for one mode (full or quick)."""
    total_seconds: float = 10_800          # estimated cost of the whole plan, critic revision included
    step_seconds: float = 3_600            # estimated cost of any one step
    train_seconds_per_fit: float = 240     # one model fit (count VAE, density model, prediction head)
    max_epochs: int = 300
    max_folds: int = 5                     # outer folds of grouped cross-validation
    inner_folds: int = 3                   # inner folds that choose a penalty
    bootstrap_reps: int = 20               # resamples for cluster stability
    permutations: int = 999                # permutation tests
    max_samples: int | None = None         # cap on samples used at all (quick mode)
    eval_max_samples: int = 5_000          # rows used by O(n^2) metrics
    screening_train: int = 2_000           # samples used to screen candidate encoders
    screening_val: int = 500
    max_candidates: int = 4                # candidate encoders screened per modality
    feature_batch_size: int = 64
    device: str = "mps"                    # set by RunConfig.budget(); planners read it


class Criteria(BaseModel):
    """What counts as a finding. Fixed before any result is seen and printed in the report."""
    fdr: float = 0.05                      # Benjamini-Hochberg level for every family of tests
    ci_level: float = 0.95                 # confidence intervals (DeLong, bootstrap)
    min_abs_log2fc: float = 1.0            # differential tests: effect size on top of significance
    # A target counts as reliably predicted only with enough positives in the held-out
    # test set to judge it, a lower confidence bound on AUROC at or above this, and an
    # FDR-adjusted p-value against chance (AUROC 0.5) below `fdr`. Below the support
    # minimum a target is reported as "too rare to assess", not as a failure.
    reliable_min_test_positives: int = 20
    reliable_min_auroc_lower: float = 0.80


class LLMConfig(BaseModel):
    """Calls go through the local `claude` CLI (Claude Code) and its login; see llm.py."""
    enabled: bool = True
    # Passed to the CLI; the model that actually answered is read back from each
    # response and recorded, so a substitution is visible.
    model: str = "claude-opus-5-5"
    max_output_tokens: int = 16_000        # hard cap per call
    effort: dict[str, str] = Field(default_factory=lambda: {
        "objective": "medium", "planner": "high", "critic": "high", "writer": "medium"})
    max_budget_usd: float = 3.0            # per run, as reported by the CLI; later calls fall back to rules
    structured_output: bool = True         # CLI JSON-schema mode; plain text + JSON extraction if it fails
    timeout_seconds: float = 600
    max_retries: int = 2


class RunConfig(BaseModel):
    seed: int = 774
    device: str = "auto"                   # auto | cuda | mps | cpu (feature extraction)
    # Model fits on tabular inputs (count VAE, density model, prediction heads) run on
    # the CPU by default: slower than an Apple GPU at this size but bit-for-bit
    # reproducible, which GPU kernels are not. "auto" uses `device`.
    fit_device: str = "cpu"
    quick: bool = False
    replay: str | None = None              # directory of cached LLM responses to replay
    runs_dir: str = "runs"
    cache_dir: str = "runs/cache"          # feature and patch caches, shared across runs
    # Column roles, when automatic detection should be overridden. These describe
    # the data's format, not how to analyse it.
    label_column: str | None = None
    group_column: str | None = None
    max_critic_revisions: int = 1
    full: Budget = Field(default_factory=Budget)
    quick_budget: Budget = Field(default_factory=lambda: Budget(
        total_seconds=900, step_seconds=300, train_seconds_per_fit=20, max_epochs=5,
        max_folds=2, inner_folds=2, bootstrap_reps=3, permutations=99, max_samples=2_000,
        eval_max_samples=1_000, screening_train=300, screening_val=100, max_candidates=2,
        feature_batch_size=32,
    ))
    criteria: Criteria = Field(default_factory=Criteria)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    def budget(self, device: str) -> Budget:
        """The active budget, scaled for the detected device."""
        base = self.quick_budget if self.quick else self.full
        scale = DEVICE_BUDGET_SCALE.get(device, 1.0)
        return base.model_copy(update={
            "device": device, "train_seconds_per_fit": base.train_seconds_per_fit * scale})


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> RunConfig:
    """Load default.yaml, layer an optional file on top, then keyword overrides."""
    data: dict = {}
    if DEFAULT_CONFIG.exists():
        data = yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}
    if path is not None:
        data = _deep_update(data, yaml.safe_load(Path(path).read_text()) or {})
    data = _deep_update(data, {k: v for k, v in overrides.items() if v is not None})
    return RunConfig(**data)

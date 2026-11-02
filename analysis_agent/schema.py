"""Typed records passed between the agent's steps.

Every object here is JSON-serialisable, because each one is shown to the LLM,
written to the run directory and quoted in the decision log. Profile fields are
measurements, not judgements: the profile says "silhouette by slide 0.21", never
"strong batch effect". Turning numbers into challenges is the challenge
register's job, and turning challenges into steps is the planner's.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Decision log (from Project 3)
# ---------------------------------------------------------------------------

# "subagent": a choice a ported Project 1-3 agent made by its own rules, inside a step.
DecisionSource = Literal["heuristic", "llm", "validator", "critic", "subagent", "system"]


class Alternative(BaseModel):
    option: str
    reason: str


class Decision(BaseModel):
    step: str                     # profile | objective | challenges | plan | <step id> | critic | ...
    decision: str                 # what was being decided
    chosen: str
    rationale: str
    alternatives_rejected: list[Alternative] = Field(default_factory=list)
    evidence: dict[str, float | int | str | bool | None | list | dict] = Field(default_factory=dict)
    source: DecisionSource = "heuristic"


# ---------------------------------------------------------------------------
# Data splits (from Project 3)
# ---------------------------------------------------------------------------


class SplitSpec(BaseModel):
    strategy: Literal["predefined", "grouped_kfold", "random"]
    group_column: str | None = None
    n_folds: int = 1                              # grouped_kfold: folds over groups (1 = one held-out set)
    val_fraction: float = 0.1                     # random / carved from train when no val split
    test_fraction: float = 0.2                    # random, and grouped with one fold
    stratify_column: str | None = None

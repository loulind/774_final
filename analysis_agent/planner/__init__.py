"""Planning: from the objective, the profile and the challenge register to a validated plan.

``heuristic.py`` writes the rule-based plan, ``validator.py`` holds the guardrails every plan
passes (repairing what can be repaired once), ``refs.py`` resolves the references between
steps. The LLM planner of chunk 6 proposes edits to the rule-based plan and goes through the
same validator; when its plan stays invalid after one repair round, the rule-based plan is used.
"""

from __future__ import annotations

from ..config import Budget
from ..schema import AnalysisPlan, ChallengeRegister, DataProfile, Decision, Objective, ValidationReport
from .heuristic import heuristic_plan
from .validator import validate


def plan_analysis(profile: DataProfile, register: ChallengeRegister, objective: Objective,
                  budget: Budget) -> tuple[AnalysisPlan, ValidationReport, list[Decision]]:
    """The rule-based plan, validated (and repaired once)."""
    plan, decisions = heuristic_plan(profile, register, objective)
    checked, report = validate(plan, profile, register, objective, budget)
    return checked or plan, report, decisions


def _cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_plan_markdown(plan: AnalysisPlan, report: ValidationReport | None = None,
                         decisions: list[Decision] | None = None) -> str:
    lines = ["# Analysis plan", "", f"Objective: {plan.objective}", "",
             f"Questions: {', '.join(plan.questions)}. Source: {plan.source}.", "",
             "| # | step | tool | inputs | parameters | purpose | handles |", "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(plan.steps, 1):
        ins = ", ".join(f"{k}={v}" for k, v in s.inputs.items())
        prm = ", ".join(f"{k}={v}" for k, v in s.params.items()) or "defaults"
        tag = " (validation only)" if s.validation_only else ""
        lines.append(f"| {i} | {s.id}{tag} | {s.tool} | {_cell(ins)} | {_cell(prm)} | {_cell(s.purpose)} | "
                     f"{_cell(', '.join(s.addresses)) or '-'} |")
    if plan.accepted:
        lines += ["", "## Accepted limitations", ""] + [f"- {a.challenge}: {a.reason}" for a in plan.accepted]
    if decisions:
        lines += ["", "## Planning decisions", ""]
        for d in decisions:
            rej = "; ".join(f"{a.option}: {a.reason}" for a in d.alternatives_rejected)
            lines.append(f"- **{d.decision}**: {d.chosen}. {d.rationale}" + (f" Rejected: {rej}." if rej else ""))
    if report is not None:
        lines += ["", "## Validation", "",
                  f"{'Valid' if report.valid else 'Invalid'}; estimated {report.estimated_seconds:,.0f} s in all "
                  f"(longest step {max(report.step_seconds.values(), default=0):,.0f} s)."]
        lines += [f"- {i.severity}{' (repaired)' if i.repaired else ''} [{i.rule}{', ' + i.step if i.step else ''}]: "
                  f"{i.message}" for i in report.issues]
    if plan.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in plan.notes]
    return "\n".join(lines) + "\n"

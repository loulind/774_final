"""LLM planner: profile, objective, challenge register, tool menu and the rule-based plan in;
a validated plan out (the flow of Projects 2-3, adapted to tool steps; every branch is logged).

  1. The rule-based planner has written a complete, validated proposal; its decisions are the
     baseline reasoning the LLM sees.
  2. The LLM may adopt, modify or replace any step, with a decision for each change. It sees
     measurements only - the profile holds no dataset name or location - so its choices must
     come from them. The objective text cannot be changed; question types the reading of the
     objective missed can be added, never removed.
  3. The reply goes through the same validator as any plan (mechanical repairs applied once).
     Remaining errors are sent back once; if the repaired reply still fails, the rule-based
     plan is used and the log says why.
"""

from __future__ import annotations

import json

from ..challenges import assess
from ..config import REPO_ROOT, Budget
from ..decision_log import DecisionLog
from ..llm import LLMClient, LLMUnavailable
from ..schema import (AnalysisPlan, ChallengeRegister, DataProfile, Decision, Objective, PlannerReply, Question,
                      ValidationReport)
from ..taxonomy import QUESTION_TYPES
from ..tools import registry
from .validator import validate

PROMPTS = REPO_ROOT / "prompts"
MAX_REPLY_ECHO = 30_000                  # characters of a failed reply sent back for repair


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text()


def tool_menu() -> list[dict]:
    """Every tool's declaration, without the long documentation strings."""
    out = []
    for spec in registry.all_tools():
        d = spec.describe()
        out.append({"name": d["name"], "origin": d["origin"], "summary": d["summary"],
                    "consumes": [{"port": p["port"], "kinds": p["kinds"], "required": p["required"]}
                                 for p in d["consumes"]],
                    "produces": [{"port": p["port"], "kinds": p["kinds"]} for p in d["produces"]],
                    "params": {k: {x: v[x] for x in ("default", "range", "choices") if x in v}
                               for k, v in d["params"].items()},
                    "answers": d["answers"], "addresses": d["addresses"],
                    "criteria": [{"metric": c["metric"], "goal": c["goal"], "selection": c["selection"]}
                                 for c in d["criteria"]]})
    return out


def register_summary(register: ChallengeRegister) -> dict:
    detected = [{"id": c.id, "type": c.type, "severity": c.severity, "relevant": c.relevant, "summary": c.summary,
                 "evidence": [{"metric": e.metric, "value": e.value, "rule": e.rule} for e in c.evidence],
                 "candidate_tools": c.mitigations} for c in register.entries if c.status == "detected"]
    rest = [{"id": c.id, "status": c.status, "summary": c.summary} for c in register.entries
            if c.status != "detected"]
    return {"questions_judged_against": register.questions, "detected": detected, "not_detected": rest}


def budget_summary(budget: Budget) -> dict:
    return {"device": budget.device, "step_seconds": budget.step_seconds, "total_seconds": budget.total_seconds,
            "max_samples": budget.max_samples, "rule": "every step's declared cost <= step_seconds; their sum <= "
                                                      "total_seconds (leave room for one critic revision)"}


def build_user_message(profile: DataProfile, objective: Objective, register: ChallengeRegister, budget: Budget,
                       proposal: AnalysisPlan, decisions: list[Decision]) -> str:
    payload = {
        "objective": objective.text,
        "objective_reading": objective.model_dump(exclude={"text", "source"}),
        "question_types": QUESTION_TYPES,
        "data_profile": profile.model_dump(exclude={"seconds"}),
        "challenge_register": register_summary(register),
        "compute_budget": budget_summary(budget),
        "tools": tool_menu(),
        "proposal": {"plan": proposal.model_dump(exclude={"source"}),
                     "reasoning": [{"decision": d.decision, "chosen": d.chosen, "rationale": d.rationale,
                                    "alternatives_rejected": [a.model_dump() for a in d.alternatives_rejected]}
                                   for d in decisions]},
    }
    return "Plan the analysis of this dataset for this objective.\n\n" + json.dumps(payload, default=str)


def plan_diff(a: AnalysisPlan, b: AnalysisPlan) -> list[str]:
    """Steps added, removed or changed (tool, inputs, parameters) between two plans."""
    sa, sb = {s.id: s for s in a.steps}, {s.id: s for s in b.steps}
    out = [f"removed step {k} ({sa[k].tool})" for k in sa if k not in sb]
    out += [f"added step {k} ({sb[k].tool})" for k in sb if k not in sa]
    for k in sa:
        if k in sb:
            for field in ("tool", "inputs", "params"):
                x, y = getattr(sa[k], field), getattr(sb[k], field)
                if x != y:
                    out.append(f"{k}.{field}: {x} -> {y}")
    if [s.id for s in a.steps if s.id in sb] != [s.id for s in b.steps if s.id in sa]:
        out.append("step order changed")
    if {c.challenge for c in a.accepted} != {c.challenge for c in b.accepted}:
        out.append(f"accepted limitations: {[c.challenge for c in b.accepted]}")
    return out


def _candidate(raw, objective: Objective) -> tuple[dict, list[str], bool]:
    reply = raw if isinstance(raw, dict) else {}
    plan = dict(reply.get("plan") or {}) if isinstance(reply.get("plan"), dict) else {}
    tampered = plan.get("objective") != objective.text
    plan["objective"] = objective.text
    plan.pop("source", None)
    added = [q for q in reply.get("added_questions") or [] if isinstance(q, str)]
    return plan, added, tampered


def _with_questions(objective: Objective, added: list[str]) -> Objective:
    new = [q for q in added if q in QUESTION_TYPES and q not in objective.types()]
    if not new:
        return objective
    out = objective.model_copy(deep=True)
    out.questions += [Question(type=q, phrase="(added by the LLM planner)", evidence="LLM") for q in new]
    out.source = "llm"
    return out


def _log_llm_decisions(log: DecisionLog, decisions) -> None:
    for d in decisions or []:
        if not isinstance(d, dict):
            continue
        log.add(step=str(d.get("step") or "plan"), decision=str(d.get("decision", "")), chosen=str(d.get("chosen", "")),
                rationale=str(d.get("rationale", "")),
                alternatives=[(str(a.get("option", "")), str(a.get("reason", "")))
                              for a in d.get("alternatives_rejected") or [] if isinstance(a, dict)], source="llm")


def choose_plan(profile: DataProfile, register: ChallengeRegister, objective: Objective, budget: Budget,
                proposal: AnalysisPlan, proposal_decisions: list[Decision], client: LLMClient | None,
                log: DecisionLog) -> tuple[AnalysisPlan, ValidationReport, Objective, ChallengeRegister]:
    """The plan to run, its validation report, and the objective and register it was judged
    against (the LLM may have added question types)."""
    attempted = False

    def fallback(reason: str):
        log.add("plan", "planner used", "rule-based (fallback)" if attempted else "rule-based", reason, [], {},
                source="system")
        plan, report = validate(proposal, profile, register, objective, budget)
        plan = plan or proposal
        plan.source = "heuristic"
        return plan, report, objective, register

    if client is None or not client.available:
        why = client.disabled if client is not None and client.disabled else "--no-llm or no LLM backend"
        return fallback(f"No LLM available ({why}); the rule-based plan is used.")
    system = load_prompt("planner")
    user = build_user_message(profile, objective, register, budget, proposal, proposal_decisions)
    schema = PlannerReply.model_json_schema()
    attempted = True
    try:
        resp = client.complete("planner", system, user, schema)
        raw = resp.json()
    except (LLMUnavailable, ValueError) as err:
        return fallback(f"LLM planning failed: {err}")

    def check(raw_reply):
        cand, added, tampered = _candidate(raw_reply, objective)
        obj = _with_questions(objective, added)
        reg = register if obj is objective else assess(profile, obj.types())
        plan, report = validate(cand, profile, reg, obj, budget)
        return plan, report, obj, reg, tampered

    plan, report, obj, reg, tampered = check(raw)
    source = "llm"
    if not report.valid:
        errors = [i.message for i in report.errors()]
        log.add("validator", "LLM plan validation", "failed; one repair round", "; ".join(errors[:12]),
                source="validator")
        repair = (user + "\n\nYour previous answer:\n" + resp.text[:MAX_REPLY_ECHO]
                  + "\n\nIt failed validation with these errors:\n- " + "\n- ".join(errors)
                  + "\n\nReturn the corrected JSON object.")
        try:
            resp = client.complete("planner_repair", system, repair, schema)
            raw = resp.json()
        except (LLMUnavailable, ValueError) as err:
            return fallback(f"LLM plan invalid ({'; '.join(errors[:6])}) and the repair failed: {err}")
        plan, report, obj, reg, tampered = check(raw)
        source = "repaired"
        if not report.valid:
            return fallback("LLM plan still invalid after one repair: "
                            + "; ".join(i.message for i in report.errors()[:8]))
    plan.source = source
    _log_llm_decisions(log, raw.get("decisions") if isinstance(raw, dict) else None)
    if tampered:
        log.add("validator", "objective text", "restored", "The LLM changed the objective text; the original was "
                "restored.", source="validator")
    for i in report.issues:
        if i.repaired:
            log.add("validator", f"plan repair ({i.rule})", i.step or "plan", i.message, source="validator")
    if obj is not objective:
        log.add("objective", "question types", "added by the LLM: "
                + ", ".join(q for q in obj.types() if q not in objective.types()),
                "The LLM planner added question types the rule-based reading missed; the register's relevance "
                "was judged again with them.", source="llm")
    changes = plan_diff(proposal, plan)
    log.add("plan", "planner used", source, ("LLM revised the rule-based plan: " + "; ".join(changes[:25])
                                             + (f"; ... ({len(changes)} changes)" if len(changes) > 25 else ""))
            if changes else "LLM adopted the rule-based plan unchanged.", [],
            {"llm_model": resp.model_used, "n_changes": len(changes), "changes": changes[:60]}, source="llm")
    return plan, report, obj, reg

"""Guardrails every plan passes before it runs, whoever wrote it (the rules or the LLM).

Checks, each reported as an error or a warning:

  schema          the plan parses as an AnalysisPlan and step ids are unique
  tools           every tool is in the registry; question types are known
  params          parameters exist and lie in the declared ranges (repaired once: unknown keys
                  dropped, numbers clipped, invalid values reset to the default)
  inputs          every required port is bound; each reference is a first artifact or an output
                  of an earlier step; its kind is one the port accepts
  labels          external labels the objective does not ask to predict or test are reserved for
                  validation: a step that lets them shape its result is marked validation-only
                  (repaired), and no later step may consume a validation-only output
  grouping        when samples share groups, prediction keeps the independent unit whole and marker
                  tests use it as the replicate (repaired: the unit is bound)
  double_dipping  markers of groups discovered in counts are tested on the held-out half of a count
                  split whose other half found the groups
  testing         a family of tests the objective asks for is run by a tool that controls the FDR
  challenges      every relevant high-severity challenge is handled by a step (its tool addresses
                  the type and its inputs derive from the challenge's data) or accepted with a
                  reason; the steps' ``addresses`` are recomputed rather than trusted (repaired)
  questions       every question of the objective is answered by a step (warning)
  cost            every step's declared cost fits the step budget, and the plan the total budget

Repairs are applied once; errors that remain make the plan invalid. An LLM plan then gets one
repair round with these messages, and after that the rule-based plan is used.
"""

from __future__ import annotations

from pydantic import ValidationError

from ..config import Budget
from ..schema import (AnalysisPlan, ChallengeRegister, DataProfile, Objective, ValidationIssue, ValidationReport)
from ..taxonomy import QUESTION_TYPES
from ..tools import registry
from .refs import Trace, is_ref, split_ref, step_costs

# Ports where external labels only describe or are compared with, never shape the output.
REPORT_LABEL_PORTS = {("batch_assess", "labels"), ("batch_correct", "labels"), ("label_association", "reference"),
                      ("label_imbalance", "target"), ("label_agreement", "labels"), ("label_agreement", "reference"),
                      ("caption_label_consistency", "reference"), ("label_consistency", "labels")}
DISCOVERY_TOOLS = ("spatial_domains", "cluster")


def _issue(issues, rule, severity, message, step=None, repaired=False):
    issues.append(ValidationIssue(rule=rule, severity=severity, message=message, step=step, repaired=repaired))


def objective_labels(objective: Objective) -> set[str]:
    """Label artifacts the objective asks to predict or to test: these may be learned from."""
    out = set()
    for q in objective.questions:
        if q.type == "predictive_modelling":
            out |= set(q.targets.get("labels", []))
        if q.type == "differential_analysis":
            out |= {g for g in q.targets.get("groups", []) if g != "discovered"}
        if q.type == "data_quality_audit":
            out |= set(q.targets.get("labels", []))
    return out


def validate(plan_like, profile: DataProfile, register: ChallengeRegister, objective: Objective, budget: Budget,
             repair: bool = True) -> tuple[AnalysisPlan | None, ValidationReport]:
    issues: list[ValidationIssue] = []
    if isinstance(plan_like, AnalysisPlan):          # re-validated: in-memory edits are not checked by pydantic
        plan_like = plan_like.model_dump(warnings=False)
    try:
        plan = AnalysisPlan.model_validate(plan_like)
    except ValidationError as err:
        msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors()[:8])
        _issue(issues, "schema", "error", f"the plan does not match the schema: {msg}")
        return None, ValidationReport(valid=False, issues=issues)
    ids = [s.id for s in plan.steps]
    for sid in sorted({i for i in ids if ids.count(i) > 1}):
        _issue(issues, "schema", "error", f"step id '{sid}' is used more than once", sid)

    tools = registry.load()
    known = []
    for s in plan.steps:
        if s.tool not in tools:
            _issue(issues, "tools", "error", f"unknown tool '{s.tool}'", s.id)
            continue
        known.append(s)
        bad = [a for a in s.answers if a not in QUESTION_TYPES]
        if bad:
            s.answers = [a for a in s.answers if a in QUESTION_TYPES] if repair else s.answers
            _issue(issues, "tools", "warning", f"unknown question types {bad} dropped", s.id, repair)
        spec = tools[s.tool]
        clean, notes = spec.repair_params(s.params)
        for note in notes:
            _issue(issues, "params", "warning" if repair else "error", note, s.id, repair)
        if repair:
            s.params = clean
        try:
            spec.parse_params(s.params)
        except ValueError as err:
            _issue(issues, "params", "error", str(err), s.id)

    # inputs, resolved in order
    trace = Trace(AnalysisPlan(objective=plan.objective), profile)
    produced: dict[str, set[str]] = {}
    for s in known:
        spec = tools[s.tool]
        ports = {p.name: p for p in spec.consumes}
        for name in list(s.inputs):
            if name not in ports:
                _issue(issues, "inputs", "warning" if repair else "error",
                       f"'{s.tool}' has no input port '{name}' (dropped)", s.id, repair)
                if repair:
                    s.inputs.pop(name)
        for p in spec.consumes:
            ref = s.inputs.get(p.name)
            if ref is None:
                if p.required:
                    _issue(issues, "inputs", "error", f"required input '{p.name}' of '{s.tool}' is not bound", s.id)
                continue
            if is_ref(ref):
                sid, port = split_ref(ref)
                if sid not in produced:
                    _issue(issues, "inputs", "error", f"'{ref}' refers to no earlier step", s.id)
                    continue
                if port not in produced[sid]:
                    _issue(issues, "inputs", "error", f"step '{sid}' has no output '{port}'", s.id)
                    continue
            elif ref not in trace.kind:
                _issue(issues, "inputs", "error", f"no artifact '{ref}' in the data", s.id)
                continue
            kind = trace.kind.get(ref)
            if kind not in p.kinds:
                _issue(issues, "inputs", "error", f"'{ref}' is {kind}, but '{s.tool}.{p.name}' takes "
                                                  f"{'/'.join(p.kinds)}", s.id)
        trace.add_step(s)
        produced[s.id] = {p.name for p in spec.produces}

    # labels reserved for validation
    allowed = objective_labels(objective)
    external = {a.name for a in profile.artifacts if a.kind in ("labels", "multilabel") and a.detail.get("external")}
    reserved = external - allowed
    for s in known:
        shaping = [p for p, ref in s.inputs.items() if ref in reserved and (s.tool, p) not in REPORT_LABEL_PORTS]
        if shaping and not s.validation_only:
            _issue(issues, "labels", "warning" if repair else "error",
                   f"'{s.tool}' uses labels reserved for validation ({', '.join(s.inputs[p] for p in shaping)}); "
                   "marked validation-only", s.id, repair)
            if repair:
                s.validation_only = True
    only = {s.id for s in known if s.validation_only}
    for s in known:
        used = [ref for ref in s.inputs.values() if is_ref(ref) and split_ref(ref)[0] in only]
        if used:
            _issue(issues, "labels", "error", f"consumes the output of a validation-only step ({', '.join(used)}): "
                                              "validation results must not shape the analysis", s.id)

    # grouping: the independent unit
    nested = any(c.status == "detected" for c in register.of_type("nested_sampling"))
    unit = min(profile.groups, key=lambda g: g.n_levels).artifact if profile.groups else None
    if nested and unit:
        for s in known:
            if s.tool == "predict_targets" and not s.inputs.get("groups"):
                _issue(issues, "grouping", "warning" if repair else "error",
                       f"samples share {unit}, but prediction folds were not grouped; bound '{unit}'", s.id, repair)
                if repair:
                    s.inputs["groups"] = unit
            if s.tool == "differential_expression" and s.inputs.get("replicate") not in (None, unit):
                _issue(issues, "grouping", "warning" if repair else "error",
                       f"marker tests replicate on '{s.inputs['replicate']}', but the independent unit is "
                       f"'{unit}'; rebound", s.id, repair)
                if repair:
                    s.inputs["replicate"] = unit

    # double dipping
    trace = Trace(plan, profile)
    steps = {s.id: s for s in known}
    for s in known:
        if s.tool != "differential_expression" or not is_ref(s.inputs.get("labels", "")):
            continue
        src = steps.get(split_ref(s.inputs["labels"])[0])
        if src is None or src.tool not in DISCOVERY_TOOLS:
            continue
        counts = s.inputs.get("counts", "")
        sid = split_ref(counts)[0] if is_ref(counts) else None
        splitter = steps.get(sid) if sid else None
        ok = (splitter is not None and splitter.tool == "count_split" and counts.endswith(".test")
              and trace.derives_from(src.inputs.get("embedding", ""), [f"@{sid}.fit"])
              and not trace.derives_from(src.inputs.get("embedding", ""), [f"@{sid}.test"]))
        if not ok:
            _issue(issues, "double_dipping", "error", f"markers of groups found by '{src.id}' are not tested on the "
                                                      "held-out half of a count split that found them", s.id)

    # multiple testing
    if "differential_analysis" in objective.types(include_implied=False):
        testers = [s for s in known if "differential_analysis" in s.answers]
        if testers and not any("multiple_testing" in tools[s.tool].addresses for s in testers):
            _issue(issues, "testing", "error", "no step answering differential analysis controls the FDR")

    # challenges: recompute what each step addresses, then check coverage
    detected = [c for c in register.entries if c.status == "detected"]
    for s in known:
        spec = tools[s.tool]
        computed = [c.id for c in detected if c.type in spec.addresses and trace.handles(s, c)]
        claimed = set(s.addresses)
        if claimed - set(computed):
            _issue(issues, "challenges", "warning", f"claims to address {sorted(claimed - set(computed))}, which its "
                                                    "tool or inputs do not; corrected", s.id, True)
        s.addresses = computed
    accepted = {a.challenge: a.reason for a in plan.accepted}
    for c in register.detected("high", relevant_only=True):
        if any(c.id in s.addresses for s in known):
            continue
        if accepted.get(c.id, "").strip():
            _issue(issues, "challenges", "warning", f"high-severity '{c.id}' accepted as a limitation: "
                                                    f"{accepted[c.id]}")
            continue
        _issue(issues, "challenges", "error", f"high-severity challenge '{c.id}' is neither handled by a step nor "
                                              f"accepted with a reason (candidate tools: {', '.join(c.mitigations)})")
    for q in objective.questions:
        if not q.implied and not any(q.type in s.answers for s in known):
            _issue(issues, "questions", "warning", f"no step answers '{q.type}' ({q.phrase})")

    # cost
    costs = step_costs(plan, profile, budget) if known else {}
    total = float(sum(costs.values()))
    for sid, sec in costs.items():
        if sec > budget.step_seconds:
            _issue(issues, "cost", "error", f"estimated {sec:,.0f} s, over the {budget.step_seconds:,.0f} s step "
                                            "budget", sid)
    if total > budget.total_seconds:
        _issue(issues, "cost", "error", f"estimated {total:,.0f} s in all, over the {budget.total_seconds:,.0f} s "
                                        "budget")
    report = ValidationReport(valid=not any(i.severity == "error" and not i.repaired for i in issues),
                              issues=issues, estimated_seconds=total, step_seconds=costs)
    return plan, report

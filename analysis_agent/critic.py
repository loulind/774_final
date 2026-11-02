"""Step 6: the critic - deterministic triggers from the step results, at most one revision.

Triggers are computed from the metrics each step reported (``StepResult.to_record()``):

  step_failed            a step raised instead of returning results
  groups_confined        discovered groups are confined to single levels of the independent unit
                         (Cramér's V with it >= 0.5), so they cannot be compared across replicates
  markers_untestable     no group has enough replicates on both sides to test markers
  unstable_groups        no clustering candidate was stable
  k_at_edge              the number of groups chosen is the largest searched
  k_at_lower_edge        only the coarsest split was found (k = the smallest searched): undo a correction
                         across the outer grouping (unless markers need groups that recur across it), or
                         smooth less over spatial neighbours
  batch_undercorrected   the assessment found a batch effect but no correction was kept
  markers_one_replicate  markers flip direction when one replicate is left out (stability < bar)
  no_reliable_targets    targets could be judged but none met the reliability rule
  leakage_gap            the leak-inclusive fit scores clearly higher (reported; the plan already
                         keeps related samples apart, so nothing to revise)
  criterion_failed       a declared, label-free criterion missed its threshold

Most triggers carry a suggested revision: new parameters for a step (the batch-correction
scope, a wider range of groups, more clustering methods, more replicates per group, the other
model family). With an LLM, the critic reads the triggers and results and picks one revision
or none; its choice must name a step of the plan, parameters inside the tool's ranges (or a
tool that accepts the same inputs), and a label-free, non-test criterion to judge it by -
otherwise the first suggested revision is used. The executor applies the revision, re-runs
the changed step and everything downstream, and ``keep_revision`` keeps it only if the
criterion improved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .decision_log import DecisionLog
from .llm import LLMClient, LLMUnavailable
from .planner.llm import load_prompt
from .planner.refs import is_ref, split_ref
from .schema import AnalysisPlan, ChallengeRegister, CriticReply, Objective, Revision
from .tools import registry

CONFINED_V = 0.5                      # Cramér's V of groups with the independent unit
LEAKAGE_GAP = 0.05                    # AUROC gained by letting related samples straddle the split
DISCOVERY = ("spatial_domains", "cluster")
MIN_GAIN = 0.01                       # smallest improvement that counts: absolute on a 0-1 scale, else relative


@dataclass
class Trigger:
    kind: str
    step: str
    message: str
    suggestion: Revision | None = None
    evidence: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {"kind": self.kind, "step": self.step, "message": self.message, "evidence": self.evidence,
                "suggestion": self.suggestion.model_dump() if self.suggestion else None}


def _metric(results: dict, sid: str | None, name: str):
    r = results.get(sid or "") or {}
    return (r.get("metrics") or {}).get(name)


def _steps(plan: AnalysisPlan, tool: str):
    return [s for s in plan.steps if s.tool == tool]


def _range(tool: str, param: str):
    p = registry.get(tool).describe()["params"].get(param, {})
    return p.get("range", [None, None]), p.get("choices"), p.get("default")


def find_triggers(plan: AnalysisPlan, results: dict[str, dict], register: ChallengeRegister | None = None,
                  objective: Objective | None = None) -> list[Trigger]:
    out: list[Trigger] = []
    for s in plan.steps:
        err = (results.get(s.id) or {}).get("error")
        if err:
            out.append(Trigger("step_failed", s.id, f"{s.id} ({s.tool}) failed: {str(err)[:200]}"))
    correct = next(iter(_steps(plan, "batch_correct")), None)
    markers = next(iter(_steps(plan, "differential_expression")), None)
    unit = markers.inputs.get("replicate") if markers else None
    for d in [s for s in plan.steps if s.tool in DISCOVERY]:
        v = _metric(results, d.id, "cramers_v_groups")
        if v is not None and v >= CONFINED_V and d.inputs.get("groups") and correct is not None \
                and correct.params.get("within_outer") != "no" and correct.inputs.get("outer"):
            out.append(Trigger("groups_confined", d.id,
                               f"groups found by {d.id} are confined to single {d.inputs['groups']} levels "
                               f"(Cramér's V {v:.2f} >= {CONFINED_V}); correcting across them lets shared "
                               "groups come forward",
                               Revision(action="set_params", step=correct.id, params={"within_outer": "no"},
                                        criterion_step=d.id, criterion="cramers_v_groups", goal="min",
                                        trigger="groups_confined",
                                        reason="correct batches across the outer grouping too"), {"cramers_v": v}))
        if _metric(results, d.id, "stable") is False:
            methods = list(d.params.get("methods") or ["kmeans", "gmm", "leiden"])
            missing = [m for m in ("kmeans", "gmm", "leiden") if m not in methods]
            out.append(Trigger("unstable_groups", d.id, f"no candidate of {d.id} kept every group stable "
                                                        f"(least stable {_metric(results, d.id, 'stability')})",
                               Revision(action="set_params", step=d.id, params={"methods": methods + missing},
                                        criterion_step=d.id, criterion="stability", goal="max",
                                        trigger="unstable_groups", reason="more clustering methods")
                               if missing else None))
        k, edge = _metric(results, d.id, "k"), _metric(results, d.id, "k_at_edge")
        (lo, hi), _, default_max = _range(d.tool, "k_max")
        k_max = d.params.get("k_max", default_max)
        k_min = d.params.get("k_min", _range(d.tool, "k_min")[2])
        if edge and k is not None and k_min is not None and k <= k_min:
            # the scope stays across the outer grouping when markers use it as the replicate (the reason it was set)
            replicate = (markers is not None and correct is not None
                         and markers.inputs.get("replicate") == correct.inputs.get("outer"))
            if correct is not None and correct.params.get("within_outer") == "no" and not replicate:
                fix = Revision(action="set_params", step=correct.id, params={"within_outer": "yes"},
                               criterion_step=d.id, criterion="k_at_edge", goal="min", trigger="k_at_lower_edge",
                               reason="correcting across the outer grouping may have removed structure with it")
            elif d.tool == "spatial_domains":
                lam = d.params.get("neighbour_weight", _range(d.tool, "neighbour_weight")[2])
                fix = Revision(action="set_params", step=d.id, params={"neighbour_weight": round(lam / 2, 3)},
                               criterion_step=d.id, criterion="k_at_edge", goal="min", trigger="k_at_lower_edge",
                               reason="less neighbourhood smoothing lets finer domains separate") if lam else None
            else:
                fix = None
            out.append(Trigger("k_at_lower_edge", d.id, f"{d.id} found only k={k}, the smallest number searched",
                               fix))
        if edge and k is not None and k_max is not None and k >= k_max and hi is not None and k_max < hi:
            out.append(Trigger("k_at_edge", d.id, f"{d.id} chose k={k}, the largest searched",
                               Revision(action="set_params", step=d.id, params={"k_max": int(min(hi, k_max + 6))},
                                        criterion_step=d.id, criterion="k_at_edge", goal="min",
                                        trigger="k_at_edge", reason="search a wider range of k")))
    if markers is not None and _metric(results, markers.id, "testable") is False and correct is not None \
            and correct.params.get("within_outer") != "no" and correct.inputs.get("outer") \
            and not any(t.kind == "groups_confined" for t in out):
        out.append(Trigger("markers_untestable", markers.id, f"no group has enough {unit} levels on both sides "
                                                             "to test markers",
                           Revision(action="set_params", step=correct.id, params={"within_outer": "no"},
                                    criterion_step=markers.id, criterion="n_markers", goal="max",
                                    trigger="markers_untestable", reason="groups must recur across replicates")))
    assess_ = next(iter(_steps(plan, "batch_assess")), None)
    if correct is not None and _metric(results, correct.id, "method") == "none" and assess_ is not None \
            and _metric(results, assess_.id, "correction_warranted") == 1:
        (lo, _), _, default = _range("batch_correct", "min_gain")
        cur = correct.params.get("min_gain", default)
        if lo is not None and cur is not None and cur > lo:
            out.append(Trigger("batch_undercorrected", correct.id, "a batch effect was found but no correction "
                                                                   "improved mixing enough to be kept",
                               Revision(action="set_params", step=correct.id,
                                        params={"min_gain": round(max(lo, cur / 2), 4)}, criterion_step=correct.id,
                                        criterion="mixing", goal="max", trigger="batch_undercorrected",
                                        reason="accept a smaller mixing gain")))
    if markers is not None:
        loo = _metric(results, markers.id, "loo_stable_fraction")
        if loo is not None and loo < 0.8:
            (_, hi), _, default = _range("differential_expression", "min_replicates")
            cur = markers.params.get("min_replicates", default)
            out.append(Trigger("markers_one_replicate", markers.id, f"{1 - loo:.0%} of markers change direction "
                                                                    "when one replicate is left out",
                               Revision(action="set_params", step=markers.id, params={"min_replicates": cur + 1},
                                        criterion_step=markers.id, criterion="loo_stable_fraction", goal="max",
                                        trigger="markers_one_replicate", reason="require one more replicate")
                               if hi is not None and cur is not None and cur < hi else None))
    for p in _steps(plan, "predict_targets"):
        if (_metric(results, p.id, "n_assessed") or 0) > 0 and _metric(results, p.id, "n_reliable") == 0:
            model = _metric(results, p.id, "model")
            other = {"ridge": "logistic", "logistic": "ridge"}.get(model)
            out.append(Trigger("no_reliable_targets", p.id, "targets could be judged, but none met the reliability "
                                                            "rule",
                               Revision(action="set_params", step=p.id, params={"model": other},
                                        criterion_step=p.id, criterion="cv_auroc_median", goal="max",
                                        trigger="no_reliable_targets", reason=f"the {other} model family")
                               if other else None))
        gap = _metric(results, p.id, "leakage_gap_median")
        if gap is not None and gap > LEAKAGE_GAP:
            out.append(Trigger("leakage_gap", p.id, f"fitting with related samples gains a median {gap:.3f} AUROC: "
                                                    "results without the separation would be inflated",
                               evidence={"leakage_gap_median": gap}))
    flagged = {(t.step) for t in out}
    for s in plan.steps:
        for c in (results.get(s.id) or {}).get("criteria") or []:
            if c.get("selection") and c.get("passed") is False and s.id not in flagged:
                out.append(Trigger("criterion_failed", s.id, f"{s.id}: {c['metric']} = {c.get('value')} misses its "
                                                             f"threshold {c.get('threshold')}"))
    return out


def validate_revision(rev: Revision, plan: AnalysisPlan) -> tuple[Revision | None, str]:
    """A revision the executor can apply, or None and why not."""
    step = plan.step(rev.step)
    if step is None:
        return None, f"no step '{rev.step}'"
    judge = plan.step(rev.criterion_step)
    if judge is None:
        return None, f"no step '{rev.criterion_step}' to judge by"
    crit = next((c for c in registry.get(judge.tool).criteria if c.metric == rev.criterion), None)
    if crit is None:
        return None, f"'{judge.tool}' declares no criterion '{rev.criterion}'"
    if not crit.selection:
        return None, f"'{rev.criterion}' uses external labels or the test split: it may not judge a revision"
    if rev.criterion_step != rev.step and rev.criterion_step not in downstream(plan, rev.step):
        return None, f"'{rev.criterion_step}' does not depend on '{rev.step}'"
    if rev.action == "set_params":
        spec = registry.get(step.tool)
        clean, notes = spec.repair_params(rev.params)
        if notes or not clean:
            return None, "; ".join(notes) or "no parameter given"
        try:
            spec.parse_params({**step.params, **clean})
        except ValueError as err:
            return None, str(err)
        return rev.model_copy(update={"params": clean}), ""
    if rev.action == "swap_tool":
        try:
            spec = registry.get(rev.tool or "")
        except KeyError as err:
            return None, str(err)
        ports = {p.name: p for p in spec.consumes}
        missing = [p.name for p in spec.consumes if p.required and p.name not in step.inputs]
        if missing or any(k not in ports for k in step.inputs):
            return None, f"'{rev.tool}' does not take the inputs of '{rev.step}'"
        if [p.name for p in spec.produces] != [p.name for p in registry.get(step.tool).produces]:
            return None, f"'{rev.tool}' does not produce the outputs later steps use"
        return rev, ""
    if rev.new_step is None or rev.new_step.tool not in registry.load():
        return None, "add_step needs a new step with a known tool"
    return rev, ""


def downstream(plan: AnalysisPlan, sid: str) -> list[str]:
    """Steps that consume, directly or through others, the outputs of step ``sid``."""
    hit = {sid}
    for s in plan.steps:
        if any(is_ref(r) and split_ref(r)[0] in hit for r in s.inputs.values()):
            hit.add(s.id)
    return [s.id for s in plan.steps if s.id in hit and s.id != sid]


def apply_revision(plan: AnalysisPlan, rev: Revision) -> AnalysisPlan:
    new = plan.model_copy(deep=True)
    step = new.step(rev.step)
    if rev.action == "set_params":
        step.params = {**step.params, **rev.params}
    elif rev.action == "swap_tool":
        step.tool = rev.tool
    else:
        pos = next(i for i, s in enumerate(new.steps) if s.id == rev.step)
        new.steps.insert(pos + 1, rev.new_step)
    return new


def keep_revision(before, after, goal: str) -> tuple[bool, str]:
    """Kept only when the criterion moved in its direction by more than noise (``MIN_GAIN``:
    absolute for values on a 0-1 scale, relative otherwise); a value where there was none counts."""
    if after is None or isinstance(after, bool) and goal == "max" and not after:
        return False, f"the criterion is {after} after the revision"
    if before is None:
        return True, f"the criterion now has a value ({after}) where it had none"
    b, a = float(before), float(after)
    margin = MIN_GAIN if 0 <= min(a, b) and max(a, b) <= 1 else MIN_GAIN * max(abs(b), 1e-9)
    gain = a - b if goal == "max" else b - a
    better = gain > margin
    return better, (f"{before:.4g} -> {after:.4g} ({'better' if better else 'not better'} by more than {margin:.3g}, "
                    f"goal {goal})")


def critique(plan: AnalysisPlan, results: dict[str, dict], objective: Objective, register: ChallengeRegister,
             client: LLMClient | None, log: DecisionLog, budget_left: float | None = None) -> Revision | None:
    """The one revision to try, or None; triggers, the LLM's assessment and the choice are logged."""
    triggers = find_triggers(plan, results, register, objective)
    log.add("critic", "triggers", f"{len(triggers)} found", "; ".join(t.message for t in triggers) or
            "No trigger fired.", [], {"triggers": [t.to_record() for t in triggers]}, source="critic")
    suggested = next((t.suggestion for t in triggers if t.suggestion is not None), None)
    rule_choice = None
    if suggested is not None:
        rule_choice, why = validate_revision(suggested, plan)
        if rule_choice is None:
            log.add("critic", "suggested revision", "dropped", why, source="critic")
    if client is not None and client.available:
        payload = {"objective": objective.text, "questions": objective.types(),
                   "plan": plan.model_dump(exclude={"source"}),
                   "results": {sid: {k: r.get(k) for k in ("tool", "metrics", "criteria", "error")}
                               for sid, r in results.items()},
                   "triggers": [t.to_record() for t in triggers],
                   "editable": {s.id: {"tool": s.tool, "params": registry.get(s.tool).describe()["params"]}
                                for s in plan.steps},
                   "high_challenges_accepted": [a.model_dump() for a in plan.accepted],
                   "compute_seconds_left": budget_left}
        try:
            resp = client.complete("critic", load_prompt("critic"), "Judge this run.\n\n" + json.dumps(payload,
                                   default=str), CriticReply.model_json_schema())
            reply = CriticReply.model_validate(resp.json())
            log.add("critic", "assessment", "revise" if reply.revise else "no revision", reply.assessment,
                    [(a.option, a.reason) for a in reply.alternatives_rejected], source="llm")
            if not reply.revise or reply.revision is None:
                return None
            choice, why = validate_revision(reply.revision, plan)
            if choice is not None:
                log.add("critic", "revision", f"{choice.action} on {choice.step}", choice.reason or choice.trigger,
                        [], {"revision": choice.model_dump()}, source="llm")
                return choice
            log.add("critic", "LLM revision", "rejected", why, source="validator")
        except (LLMUnavailable, ValueError) as err:
            log.add("critic", "LLM critic", "unavailable", str(err)[:300], source="system")
    if rule_choice is not None:
        log.add("critic", "revision", f"{rule_choice.action} on {rule_choice.step}",
                f"{rule_choice.trigger}: {rule_choice.reason}", [], {"revision": rule_choice.model_dump()},
                source="heuristic")
    return rule_choice

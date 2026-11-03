"""LLM planner, critic and writer with a scripted backend: adoption, edits, repair, fallback,
replay, the critic's triggers and revision checks, and the writer's number check."""

import json

import pytest
from test_llm import Scripted

from analysis_agent.challenges import assess
from analysis_agent.config import load_config
from analysis_agent.critic import apply_revision, critique, downstream, find_triggers, keep_revision, validate_revision
from analysis_agent.decision_log import DecisionLog
from analysis_agent.llm import LLMClient, LLMUnavailable, OfflineBackend
from analysis_agent.objective import interpret
from analysis_agent.planner import choose_plan, plan_analysis
from analysis_agent.schema import Revision
from analysis_agent.writer import SECTIONS, check_numbers, write_sections

TISSUE = ("Characterize spatial tissue heterogeneity: identify spatial domains, relate them to histology, find "
          "marker genes, and assess slide-level batch effects.")
BUDGET = load_config(quick=True).budget("cpu")


@pytest.fixture
def setup(registers):
    prof = registers["tissue", True][0]
    obj = interpret(TISSUE, prof)
    reg = assess(prof, obj.types())
    proposal, _, decisions = plan_analysis(prof, reg, obj, BUDGET)
    return prof, obj, reg, proposal, decisions


def _client(tmp_path, replies, **kw):
    backend = Scripted(replies)
    return LLMClient(backend, tmp_path / "llm_cache", **kw), backend


def _reply(plan, decisions=(), **extra):
    return {"plan": plan.model_dump(exclude={"source"}) if hasattr(plan, "model_dump") else plan,
            "decisions": list(decisions), **extra}


def _run(setup, client, log=None):
    prof, obj, reg, proposal, decisions = setup
    log = log or DecisionLog()
    return (*choose_plan(prof, reg, obj, BUDGET, proposal, decisions, client, log), log)


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------

def test_llm_adopts_or_edits_the_proposal(setup, tmp_path):
    proposal = setup[3]
    client, backend = _client(tmp_path, [_reply(proposal)])
    plan, report, _, _, log = _run(setup, client)
    assert report.valid and plan.source == "llm" and len(backend.calls) == 1
    assert "unchanged" in log.by_step("plan")[-1].rationale
    assert "domains" in backend.calls[0]["user"] and backend.calls[0]["schema"] is not None
    edited = proposal.model_copy(deep=True)
    edited.step("domains").params = {"k_max": 8}
    why = {"decision": "domain range", "chosen": "k_max 8", "rationale": "few patients"}
    client, _ = _client(tmp_path / "b", [_reply(edited, [why])])
    plan, report, _, _, log = _run(setup, client)
    assert plan.step("domains").params == {"k_max": 8}
    assert any(d.source == "llm" and d.decision == "domain range" for d in log.entries)
    assert any("domains.params" in c for c in log.by_step("plan")[-1].evidence["changes"])


def test_invalid_plan_is_repaired_once_then_falls_back(setup, tmp_path):
    proposal = setup[3]
    bad = proposal.model_copy(deep=True)
    bad.step("pca").tool = "magic_pca"
    client, backend = _client(tmp_path, [_reply(bad), _reply(proposal)])
    plan, report, _, _, log = _run(setup, client)
    assert report.valid and plan.source == "repaired" and len(backend.calls) == 2
    assert "failed validation" in backend.calls[1]["user"] and log.by_step("validator")
    client, backend = _client(tmp_path / "b", [_reply(bad), _reply(bad)])
    plan, report, _, _, log = _run(setup, client)
    assert plan.source == "heuristic" and report.valid
    assert "still invalid" in log.by_step("plan")[-1].rationale
    client, _ = _client(tmp_path / "c", [LLMUnavailable("claude CLI failed: boom")])
    plan, _, _, _, log = _run(setup, client)
    assert plan.source == "heuristic" and "LLM planning failed" in log.by_step("plan")[-1].rationale
    plan, _, _, _, log = _run(setup, LLMClient(OfflineBackend(), tmp_path / "d"))
    assert plan.source == "heuristic" and "No LLM available" in log.by_step("plan")[-1].rationale


def test_objective_text_is_restored_and_questions_can_be_added(setup, tmp_path):
    proposal = setup[3]
    tampered = proposal.model_dump(exclude={"source"})
    tampered["objective"] = "something easier"
    client, _ = _client(tmp_path, [_reply(tampered, added_questions=["generative_modelling", "made_up"])])
    plan, report, obj, reg, log = _run(setup, client)
    assert plan.objective == TISSUE and any(d.decision == "objective text" for d in log.entries)
    assert "generative_modelling" in obj.types() and "made_up" not in obj.types()
    assert reg.questions == obj.types() and any(i.rule == "questions" for i in report.issues)


def test_replay_reproduces_the_plan_without_calls(setup, tmp_path):
    proposal = setup[3]
    edited = proposal.model_copy(deep=True)
    edited.step("svg").params = {"min_fraction": 0.05}
    client, _ = _client(tmp_path / "run", [_reply(edited)])
    first = _run(setup, client)[0]
    replay = LLMClient(OfflineBackend("fake-model"), tmp_path / "again", replay_from=tmp_path / "run" / "llm_cache")
    second = _run(setup, replay)[0]
    assert second == first and replay.calls[0]["cached"]


# ---------------------------------------------------------------------------
# critic
# ---------------------------------------------------------------------------

def _results(**metrics):
    return {sid: {"tool": sid, "metrics": m, "criteria": []} for sid, m in metrics.items()}


def test_triggers_suggest_valid_revisions(setup):
    plan = setup[3].model_copy(deep=True)
    plan.step("batch_correct").params = {}                         # correction within each patient
    results = _results(domains={"cramers_v_groups": 0.9, "stable": True, "k": 12, "k_at_edge": 1},
                       markers={"testable": True, "loo_stable_fraction": 0.5})
    kinds = {t.kind: t for t in find_triggers(plan, results)}
    assert {"groups_confined", "k_at_edge", "markers_one_replicate"} <= set(kinds)
    rev = kinds["groups_confined"].suggestion
    assert rev.step == "batch_correct" and rev.params == {"within_outer": "no"}
    assert validate_revision(rev, plan, allow_report=True)[0] is not None   # as critique() checks triggers
    assert kinds["k_at_edge"].suggestion.params == {"k_max": 18}
    assert {"validate_label", "enrichment", "markers"} <= set(downstream(plan, "domains"))
    assert "domains" in downstream(plan, "batch_correct")
    revised = apply_revision(plan, rev)
    assert revised.step("batch_correct").params["within_outer"] == "no" and plan.step("batch_correct").params == {}


def test_revisions_must_be_judged_on_label_free_criteria(setup):
    plan = setup[3]
    bad = Revision(action="set_params", step="domains", params={"k_max": 8}, criterion_step="validate_label",
                   criterion="ari", goal="max")
    assert validate_revision(bad, plan)[0] is None                 # agreement with external labels
    out_of_range = Revision(action="set_params", step="domains", params={"k_max": 999}, criterion_step="domains",
                            criterion="stability", goal="max")
    assert validate_revision(out_of_range, plan)[0] is None
    unrelated = Revision(action="set_params", step="svg", params={"min_fraction": 0.05}, criterion_step="domains",
                         criterion="stability", goal="max")
    assert validate_revision(unrelated, plan)[0] is None            # domains do not depend on svg
    undirected = Revision(action="set_params", step="domains", params={"k_max": 8}, criterion_step="domains",
                          criterion="cramers_v_groups", goal="min")
    assert validate_revision(undirected, plan)[0] is None           # declared report-only: no direction
    assert validate_revision(undirected, plan, allow_report=True)[0] is not None   # a trigger may choose one
    reversed_ = Revision(action="set_params", step="domains", params={"k_max": 8}, criterion_step="domains",
                         criterion="stability", goal="min")
    assert validate_revision(reversed_, plan)[0] is None            # stability is declared 'max'


def test_critic_uses_the_llm_choice_or_the_rule(setup, tmp_path):
    plan = setup[3].model_copy(deep=True)
    plan.step("batch_correct").params = {}
    results = _results(domains={"cramers_v_groups": 0.9, "stable": True, "k": 5, "k_at_edge": 0})
    obj, reg = setup[1], setup[2]
    rev = critique(plan, results, obj, reg, None, DecisionLog())
    assert rev.step == "batch_correct" and rev.trigger == "groups_confined"
    wrong = {"assessment": "domains follow patients", "revise": True,
             "revision": {"action": "set_params", "step": "domains", "params": {"k_max": 6},
                          "criterion_step": "validate_label", "criterion": "ari", "goal": "max"}}
    client, _ = _client(tmp_path, [wrong])
    log = DecisionLog()
    rev = critique(plan, results, obj, reg, client, log)
    assert rev.trigger == "groups_confined"                          # the invalid LLM choice was rejected
    assert any(d.decision == "LLM revision" and d.chosen == "rejected" for d in log.entries)
    client, _ = _client(tmp_path / "b", [{"assessment": "fine", "revise": False, "revision": None}])
    assert critique(plan, results, obj, reg, client, DecisionLog()) is None


def test_keep_revision():
    assert keep_revision(0.6, 0.7, "max")[0] and not keep_revision(0.6, 0.6, "max")[0]
    assert keep_revision(0.9, 0.3, "min")[0] and not keep_revision(0.3, 0.9, "min")[0]
    assert keep_revision(None, 3, "max")[0] and not keep_revision(0.5, None, "max")[0]


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------

def _template(record):
    return {k: f"{k}: {record['metrics']['n']} samples" for k in SECTIONS}


def test_writer_checks_every_number(tmp_path):
    record = {"metrics": {"n": 13620, "ari": 0.1805, "share": 0.664}}
    check = check_numbers({"s": "ARI 0.181 on 13,620 spots; 66.4% between patients; 47 invented; ResNet-50"},
                          record)
    assert check["n_checked"] == 4 and [u["number"] for u in check["unmatched"]] == ["47"]
    sections = {k: "ARI 0.181" for k in SECTIONS}
    sections["limitations"] = ""
    client, _ = _client(tmp_path, [sections])
    log = DecisionLog()
    out = write_sections(record, client, log, _template)
    assert out["source"].startswith("LLM") and "template for limitations" in out["source"]
    assert out["sections"]["limitations"] == "limitations: 13620 samples" and out["number_check"]["n_unmatched"] == 0
    client, _ = _client(tmp_path / "b", [LLMUnavailable("down")])
    out = write_sections(record, client, DecisionLog(), _template)
    assert out["source"] == "template"
    assert json.dumps(out["sections"])

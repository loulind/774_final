"""End to end on small synthetic data: the executor, the critic loop, every output file, and an
LLM run (scripted backend) that replays decision for decision."""

import json

import numpy as np
import pytest
from conftest import _cells, _table, _tissue
from test_agents import fake_encoders  # noqa: F401  (fixture: no model downloads)

import analysis_agent.orchestrator as orch
from analysis_agent import run_agent
from analysis_agent.config import load_config
from analysis_agent.data.loaders import load_path
from analysis_agent.decision_log import DecisionLog
from analysis_agent.executor import Executor, subsample
from analysis_agent.llm import LLMClient, LLMResponse
from analysis_agent.schema import AnalysisPlan, PlanStep
from analysis_agent.tools.ingest import workspace_from_dataset

TISSUE = ("Characterize spatial tissue heterogeneity: identify spatial domains, relate them to histology, find "
          "marker genes, and assess slide-level batch effects.")
TABLE = ("Determine which clinical concepts can be reliably predicted from images, and identify data-quality "
         "issues such as class imbalance, near-duplicates, and noisy captions.")
OUTPUTS = ("report.md", "report.html", "methods_summary.md", "analysis_plan.json", "analysis_plan.md",
           "challenge_register.json", "challenge_register.md", "profile.json", "objective.json", "decision_log.json",
           "evaluation.json", "run_manifest.json", "llm_transcript.json", "results_index.json")


def _settings(tmp):
    return {"runs_dir": str(tmp / "runs"), "cache_dir": str(tmp / "cache")}


@pytest.fixture(scope="module")
def tissue_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e_tissue")
    path = _tissue(tmp, True, side=12)
    out = tmp / "out"
    summary = run_agent(str(path), TISSUE, out, quick=True, no_llm=True, **_settings(tmp))
    return summary, out


def test_tissue_run_writes_every_output(tissue_run):
    summary, out = tissue_run
    assert summary["failed"] == [] and summary["planner"] == "heuristic" and summary["numbers_unmatched"] == 0
    for name in OUTPUTS:
        assert (out / name).stat().st_size > 0, name
    assert any((out / "figures").glob("*.png")) and any((out / "tables").glob("*.csv"))
    assert any((out / "results").glob("*.csv"))
    ev = json.loads((out / "evaluation.json").read_text())
    assert ev["markers"]["tool"] == "differential_expression" and ev["markers"]["error"] is None
    reg = json.loads((out / "challenge_register.json").read_text())
    handled = [c for c in reg["entries"] if c["status"] == "detected" and c["handled_by"]]
    assert handled and all(c["handling"] and c["residual_risk"] for c in handled)
    sources = {d["source"] for d in json.loads((out / "decision_log.json").read_text())}
    assert {"heuristic", "subagent", "critic", "system"} <= sources
    report = (out / "report.md").read_text()
    for section in ("## Objective", "## Executive summary", "## Data and challenges", "## Plan", "## Results",
                    "## Critic and revision", "## Limitations", "## Reproducibility",
                    "## Numbers not found in the outputs"):
        assert section in report


def test_two_matrices_without_replicates_end_to_end(tmp_path):
    out = tmp_path / "out"
    summary = run_agent(str(_cells(tmp_path)), "Discover populations of cells using both count matrices, find "
                        "marker genes for each population, and assess how well the two modalities agree.", out,
                        quick=True, no_llm=True, **_settings(tmp_path))
    assert summary["failed"] == [] and summary["numbers_unmatched"] == 0
    ev = json.loads((out / "evaluation.json").read_text())
    assert ev["markers"]["metrics"]["test_unit"] == "sample" and ev["markers"]["metrics"]["n_markers"] > 0
    assert ev["modality_groups"]["metrics"]["ari"] > 0.5                    # the planted populations, in both
    assert ev["modality_pairs"]["metrics"]["val_recall_at_10"] > ev["modality_pairs"]["metrics"][
        "chance_recall_at_10"]
    assert (out / "figures" / "markers.png").exists() and (out / "figures" / "map.png").exists()
    assert "no replicate grouping" in (out / "report.md").read_text()


def test_critic_revision_is_recorded(tissue_run):
    summary, out = tissue_run
    log = json.loads((out / "decision_log.json").read_text())
    critic = [d for d in log if d["step"] == "critic"]
    assert critic and critic[0]["decision"] == "triggers"
    if summary["critic"].startswith(("kept", "discarded")):           # a revision ran and was judged
        assert any(d["decision"] in ("revision kept", "revision discarded") for d in critic)


def test_executor_skips_steps_that_need_a_failed_one(registers, tmp_path):
    ds = load_path(registers["tissue", True][2] / "spots.h5ad")
    ws = workspace_from_dataset(ds)
    cfg = load_config(quick=True, cache_dir=str(tmp_path))
    plan = AnalysisPlan(objective="x", steps=[
        PlanStep(id="norm", tool="normalize_counts", inputs={"counts": "counts.expression"}),
        PlanStep(id="pca", tool="dimreduce_pca", inputs={"matrix": "@norm.normalized"}),
        PlanStep(id="dom", tool="spatial_domains", inputs={"embedding": "@pca.embedding",
                                                           "coords": "coords.spatial"}),   # no library: raises
        PlanStep(id="enrich", tool="spatial_enrichment", inputs={"labels": "@dom.labels", "coords": "coords.spatial"}),
        PlanStep(id="qc", tool="qc_counts", inputs={"counts": "counts.expression"})])
    log = DecisionLog()
    records = Executor(ws, cfg, cfg.budget("cpu"), "cpu", log, echo=lambda m: None).run(plan)
    assert records["pca"]["error"] is None and "per library" in records["dom"]["error"]
    assert records["enrich"]["skipped"] == "needs dom"
    assert records["qc"]["error"] is None and records["qc"]["metrics"]["n_removed"] > 0
    assert any(d.decision == "step failed" for d in log.entries)


def test_quick_subsample_keeps_whole_libraries_and_splits(registers):
    ds = load_path(registers["tissue", True][2] / "spots.h5ad")
    ws = workspace_from_dataset(ds)
    small = subsample(ws, 900, 0, DecisionLog())
    lib = small.get("groups.slide")
    kept = set(np.asarray(lib.data))
    full = ws.get("groups.slide")
    assert len(lib.rows) <= 900 and all((np.asarray(full.data) == k).sum() == (np.asarray(lib.data) == k).sum()
                                       for k in kept)                 # libraries are kept whole
    assert len({str(p) for p in np.asarray(small.get("groups.patient").data)}) >= 2
    t = load_path(registers["table", True][2])
    ws_t = workspace_from_dataset(t)
    st = subsample(ws_t, 100, 0, DecisionLog()).get("split.split")
    shares = {s: float(np.mean(np.asarray(st.data) == s)) for s in ("train", "val", "test")}
    assert abs(shares["train"] - 0.6) < 0.05 and abs(shares["test"] - 0.2) < 0.05


class Smart:
    """A scripted backend that answers by role: adopts the proposal, does not revise, writes prose."""
    name, model, max_output_tokens = "fake", "fake-model", 100

    def __init__(self):
        self.calls = []

    def complete(self, system, user, schema=None, effort=None, budget_usd=None):
        self.calls.append(system[:40])
        payload = json.loads(user.split("\n\n", 1)[1])
        if system.startswith("You are the planning component"):
            reply = {"plan": payload["proposal"]["plan"], "decisions": [], "added_questions": []}
        elif system.startswith("You are the critic"):
            reply = {"assessment": "The results answer the objective.", "revise": False, "revision": None}
        else:
            n = payload["data"]["n_samples"]
            reply = {k: f"Prose for {k}: {n} samples." for k in ("executive_summary", "data_and_challenges",
                                                                   "plan_explained", "results_interpretation",
                                                                   "critic_and_revision", "limitations")}
        return LLMResponse(text=json.dumps(reply), backend="fake", model_requested="fake-model",
                           model_used="fake-model", seconds=0.01, cost_usd=0.01, data=reply, structured=True)


def test_llm_run_replays_decision_for_decision(tmp_path, monkeypatch, fake_encoders):  # noqa: F811
    path = _table(tmp_path, True)
    backend = Smart()
    real = orch.make_client

    def fake_client(cfg, cache_dir, replay_from=None, enabled=None):
        if replay_from:
            return real(cfg, cache_dir, replay_from, enabled)
        return LLMClient(backend, cache_dir, max_budget_usd=3.0)

    monkeypatch.setattr(orch, "make_client", fake_client)
    first = run_agent(str(path), TABLE, tmp_path / "first", quick=True, **_settings(tmp_path))
    assert first["planner"] == "llm" and first["llm_calls"] == 3 and first["numbers_unmatched"] == 0
    assert "Prose for executive_summary" in (tmp_path / "first" / "report.md").read_text()
    again = run_agent(str(path), TABLE, tmp_path / "again", quick=True, replay=str(tmp_path / "first"),
                      **_settings(tmp_path))
    assert len(backend.calls) == 3                                     # replay called no model
    plan_a = json.loads((tmp_path / "first" / "analysis_plan.json").read_text())
    plan_b = json.loads((tmp_path / "again" / "analysis_plan.json").read_text())
    assert plan_a == plan_b and again["llm_calls"] == 3 and again["llm_cost_usd"] == 0

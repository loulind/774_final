"""The common tool interface: artifacts, workspace alignment, step binding, caching, parameter
repair, criteria, the registry, ingest and ``bioagent tools``."""

import json

import numpy as np
import pytest
from pydantic import Field

from analysis_agent.cli import main
from analysis_agent.config import load_config
from analysis_agent.tools import registry
from analysis_agent.tools.base import (Artifact, Criterion, Port, StepContext, StepResult, ToolParams, ToolSpec,
                                       Workspace, evaluate_criteria, run_step)
from analysis_agent.tools.ingest import parse_multilabel, workspace_from_dataset


class ScaleParams(ToolParams):
    factor: float = Field(2.0, ge=0.5, le=10.0)
    mode: str = "plain"


CALLS = []


def _scale(ctx, inputs, p):
    a = inputs["x"]
    CALLS.append(p.factor)
    return StepResult(artifacts={"y": Artifact("embedding", a.data * p.factor, a.rows)},
                      metrics={"total": float((a.data * p.factor).sum()), "nan": float("nan")})


SCALE = ToolSpec(name="scale", origin="new", source="test", summary="multiply",
                 consumes=(Port("x", ("embedding",)), Port("g", ("groups",), required=False)),
                 produces=(Port("y", ("embedding",)),), params=ScaleParams, answers=("visualization",),
                 addresses=(), criteria=(Criterion("total", "max", 10.0), Criterion("nan", "min", 1.0),
                                         Criterion("missing", "report")), run=_scale)


def _ctx(step="s"):
    return StepContext(step_id=step, seed=0, budget=load_config(quick=True).budget("cpu"), log=lambda m: None)


def test_artifacts_check_rows():
    with pytest.raises(ValueError, match="increasing"):
        Artifact("embedding", np.zeros((2, 2)), np.array([3, 1]))
    with pytest.raises(ValueError, match="data rows"):
        Artifact("embedding", np.zeros((3, 2)), np.array([0, 1]))
    with pytest.raises(ValueError, match="unknown artifact kind"):
        Artifact("matrix", np.zeros((2, 2)), np.array([0, 1]))
    assert Artifact("table", {"a": 1}).rows is None                      # tables are not per sample
    a = Artifact("labels", np.array(["a", "b", "c"], object), np.array([2, 5, 9]))
    assert list(a.take([9, 2])) == ["c", "a"]
    with pytest.raises(KeyError):
        a.locate([4])


def test_workspace_aligns_rows_across_artifacts():
    ws = Workspace(10)
    ws.put(Artifact("embedding", np.arange(6.0)[:, None], np.array([0, 2, 4, 6, 8, 9])), "embedding.e")
    ws.put(Artifact("labels", np.array(list("abcde"), object), np.array([1, 2, 3, 8, 9])), "labels.l")
    rows, (E, L) = ws.aligned("embedding.e", "labels.l")
    assert rows.tolist() == [2, 8, 9] and E.ravel().tolist() == [1.0, 4.0, 5.0] and list(L) == ["b", "d", "e"]
    rows, _ = ws.aligned("embedding.e", "labels.l", within=[8, 9])
    assert rows.tolist() == [8, 9]
    with pytest.raises(ValueError, match="outside"):
        ws.put(Artifact("mask", np.ones(1, bool), np.array([10])), "mask.bad")


def test_run_step_binds_checks_and_caches():
    ws = Workspace(4)
    ws.put(Artifact("embedding", np.ones((4, 2)), np.arange(4)), "embedding.a")
    ws.put(Artifact("labels", np.array(list("abcd"), object), np.arange(4)), "labels.l")
    with pytest.raises(ValueError, match="unknown input ports"):
        run_step(SCALE, ws, _ctx(), {"x": "embedding.a", "z": "labels.l"})
    with pytest.raises(TypeError, match="must be embedding"):
        run_step(SCALE, ws, _ctx(), {"x": "labels.l"})
    with pytest.raises(ValueError, match="required input"):
        run_step(SCALE, ws, _ctx(), {})
    with pytest.raises(ValueError, match="invalid parameters"):
        run_step(SCALE, ws, _ctx(), {"x": "embedding.a"}, {"factor": 99})
    CALLS.clear()
    r1 = run_step(SCALE, ws, _ctx("s1"), {"x": "embedding.a"})
    assert r1.outputs == {"y": "embedding.s1"} and ws.get("embedding.s1").produced_by == "s1"
    r2 = run_step(SCALE, ws, _ctx("s2"), {"x": "embedding.a"})                    # same inputs: cached
    assert r2.cached and CALLS == [2.0] and "embedding.s2" in ws
    r3 = run_step(SCALE, ws, _ctx("s3"), {"x": "embedding.a"}, {"factor": 3.0})   # new params: re-run
    assert not r3.cached and CALLS == [2.0, 3.0]
    r4 = run_step(SCALE, ws, _ctx("s4"), {"x": "embedding.s3"})                   # downstream of s3
    assert ws.get("embedding.s4").data.sum() == pytest.approx(4 * 2 * 3 * 2)
    assert ws.get("embedding.s4").provenance != ws.get("embedding.s1").provenance and not r4.cached


def test_criteria_are_judged_only_with_a_goal_threshold_and_a_finite_value():
    ws = Workspace(4)
    ws.put(Artifact("embedding", np.ones((4, 2)), np.arange(4)), "embedding.a")
    r = run_step(SCALE, ws, _ctx(), {"x": "embedding.a"})
    c = {x.metric: x for x in r.criteria}
    assert c["total"].passed is True and c["nan"].passed is None and c["missing"].passed is None
    low = evaluate_criteria([Criterion("total", "max", 100.0)], {"total": 16.0})[0]
    assert low.passed is False


def test_repair_params_drops_unknown_keys_and_clips_ranges():
    clean, notes = SCALE.repair_params({"factor": 50, "colour": "red", "mode": 3})
    assert clean == {"factor": 10.0} and len(notes) == 3
    assert SCALE.parse_params(clean).factor == 10.0


def test_tool_specs_reject_unknown_taxonomy_terms():
    with pytest.raises(ValueError, match="unknown question types"):
        ToolSpec(name="bad", origin="new", source="", summary="", consumes=(), produces=(), params=ScaleParams,
                 answers=("astrology",), addresses=(), criteria=(), run=_scale)


def test_registry_has_the_ported_subagents_with_complete_declarations():
    tools = registry.load()
    assert {"dimreduce_pca", "dimreduce_map", "generative_batch_vae", "generative_density",
            "generative_impute", "multimodal_features", "multimodal_predictability",
            "multimodal_agreement"} <= set(tools)
    for spec in tools.values():
        d = spec.describe()
        assert d["consumes"] and d["produces"] and d["answers"] and d["criteria"] and d["source"]
        assert spec.cost(1000, 50, spec.parse_params({}), load_config().budget("cpu")) > 0
    assert {t.origin for t in tools.values()} >= {"project1", "project2", "project3"}
    assert "multimodal_agreement" in [t.name for t in registry.addressing("text_noise")]


def test_ingest_maps_roles_to_artifacts(spatial_dir):
    from analysis_agent.data.loaders import load_path

    ws = workspace_from_dataset(load_path(spatial_dir))
    kinds = {a["name"]: a["kind"] for a in ws.describe()}
    assert kinds["counts.expression"] == "counts" and kinds["groups.slide"] == "groups"
    assert {"images.spot_patches", "coords.spatial", "labels.label", "groups.patient"} <= set(kinds)
    assert ws.get("groups.slide").meta["nested_in"] == ["groups.patient"]
    assert ws.get("labels.label").n == 92                                 # 'undetermined' spots dropped
    assert ws.get("images.spot_patches").meta["dataset_key"]


def test_parse_multilabel_picks_the_delimiter():
    import pandas as pd

    M, vocab, annotated = parse_multilabel(pd.Series(["C1;C2", "C2", None, "C3;C2;C2"]))
    assert vocab[0] == "C2" and M.shape == (4, 3) and M.sum() == 5
    assert annotated.tolist() == [True, True, False, True]


def test_tools_command(capsys):
    main(["tools"])
    out = capsys.readouterr().out
    assert "multimodal_agreement  [project3]" in out and "dimreduce_pca  [project1]" in out
    main(["tools", "--name", "generative_density"])
    d = json.loads(capsys.readouterr().out)
    assert d["origin"] == "project2" and d["params"]["flag_quantile"]["range"] == [0.9, 0.9999]

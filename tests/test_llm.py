"""LLM client (from Project 3): cache, replay, spend cap, failure handling -- with a fake backend."""

import json

import pytest

from analysis_agent.config import LLMConfig
from analysis_agent.llm import LLMClient, LLMResponse, LLMUnavailable, OfflineBackend, extract_json, make_client


class Scripted:
    """A fake CLI backend: answers in order (dicts are JSON replies), records every call."""
    name, model, max_output_tokens = "fake", "fake-model", 100

    def __init__(self, replies, cost=0.1, used="fake-model"):
        self.replies, self.cost, self.used, self.calls = list(replies), cost, used, []

    def complete(self, system, user, schema=None, effort=None, budget_usd=None):
        self.calls.append({"system": system, "user": user, "schema": schema, "effort": effort,
                           "budget": budget_usd})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        structured = isinstance(r, dict) and schema is not None
        return LLMResponse(text=r if isinstance(r, str) else json.dumps(r), backend=self.name,
                           model_requested=self.model, model_used=self.used, seconds=0.01, cost_usd=self.cost,
                           data=r if structured else None, structured=structured)


def client_for(tmp_path, replies, **kw):
    backend = Scripted(replies, **{k: kw.pop(k) for k in ("cost", "used") if k in kw})
    return LLMClient(backend, tmp_path / "llm_cache", effort={"planner": "high", "critic": "high",
                                                              "writer": "medium"}, **kw), backend


def test_cache_replay_and_role_position_matching(tmp_path):
    client, backend = client_for(tmp_path, [{"a": 1}, {"b": 2}])
    assert client.complete("planner", "sys", "user 1", {"type": "object"}).json() == {"a": 1}
    assert client.complete("critic", "sys", "user 2").json() == {"b": 2}
    assert backend.calls[0]["effort"] == "high" and backend.calls[1]["schema"] is None
    rec = json.loads(next((tmp_path / "llm_cache").glob("planner_*.json")).read_text())
    assert rec["sequence"] == 0 and rec["model_used"] == "fake-model" and rec["user"] == "user 1"

    replay = LLMClient(OfflineBackend("fake-model"), tmp_path / "run2", replay_from=tmp_path / "llm_cache")
    assert replay.available
    assert replay.complete("planner", "sys", "user 1", {"type": "object"}).json() == {"a": 1}
    # a prompt that differs in incidental digits is matched by role and position
    resp = replay.complete("critic", "sys", "user 2 with other digits 0.1234")
    assert resp.json() == {"b": 2} and "role and position" in resp.note
    with pytest.raises(LLMUnavailable):
        replay.complete("writer", "sys", "never cached")
    assert len(list((tmp_path / "run2").glob("*.json"))) == 2       # the new run's cache is complete


def test_replay_from_a_run_directory_that_has_only_its_transcript(tmp_path):
    # A clone has a run's llm_transcript.json and other outputs, but not its llm_cache/.
    client, _ = client_for(tmp_path, [{"a": 1}])
    client.complete("planner", "sys", "user 1")
    run_dir = tmp_path / "outputs" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "llm_transcript.json").write_text(json.dumps({"model_requested": "fake-model",
                                                             "calls": client.records}))
    (run_dir / "decision_log.json").write_text("[]")
    (run_dir / "run_manifest.json").write_text("{}")
    replay = LLMClient(OfflineBackend("fake-model"), tmp_path / "run2", replay_from=run_dir)
    assert replay.complete("planner", "sys", "user 1").json() == {"a": 1}


def test_spend_cap_passes_remaining_budget_then_stops(tmp_path):
    client, backend = client_for(tmp_path, [{"x": 1}, {"x": 2}, {"x": 3}], cost=0.1, max_budget_usd=0.15)
    client.complete("planner", "s", "u1")
    client.complete("critic", "s", "u2")
    assert backend.calls[0]["budget"] == 0.15 and backend.calls[1]["budget"] == pytest.approx(0.05)
    with pytest.raises(LLMUnavailable, match="spend cap"):
        client.complete("writer", "s", "u3")
    assert len(backend.calls) == 2


def test_fatal_error_stops_further_calls(tmp_path):
    client, backend = client_for(tmp_path, [LLMUnavailable("claude CLI failed: Failed to authenticate"), {"x": 1}])
    with pytest.raises(LLMUnavailable):
        client.complete("planner", "s", "u")
    assert not client.available and "authenticate" in client.disabled
    with pytest.raises(LLMUnavailable, match="disabled"):
        client.complete("critic", "s", "u")
    assert len(backend.calls) == 1 and client.summary()["disabled"]


def test_model_substitution_is_noted(tmp_path):
    client, _ = client_for(tmp_path, [{"x": 1}], used="other-model")
    client.complete("planner", "s", "u")
    assert any("answered by other-model" in n for n in client.notes)
    assert client.summary()["models_used"] == ["other-model"]


def test_make_client_modes(tmp_path):
    cfg = LLMConfig()
    assert not make_client(cfg, tmp_path, enabled=False).available
    assert make_client(cfg, tmp_path, replay_from=tmp_path).available


def test_extract_json_variants():
    assert extract_json('Here:\n```json\n{"a": 1}\n```\nDone.') == {"a": 1}
    assert extract_json('prefix {"a": {"b": 2}} suffix') == {"a": {"b": 2}}
    assert extract_json('```json\n{"a": 1}') == {"a": 1}                      # unterminated fence
    assert extract_json('```\n{"a": 1}\n```\n```\n{"b": 2}\n```') == {"a": 1, "b": 2}
    with pytest.raises(ValueError):
        extract_json("no json here")

"""Decision log (from Project 3): append-only, written after every entry."""

import json

from analysis_agent.decision_log import DecisionLog


def test_entries_are_written_as_they_are_added(tmp_path):
    log = DecisionLog(tmp_path / "decision_log.json")
    log.add("plan", "batch correction", "harmony", "best mixing without losing within-slide structure",
            alternatives=[("combat", "lower mixing"), ("none", "slide effect is strong")],
            evidence={"silhouette_by_slide": 0.21}, source="heuristic")
    log.add("domains", "number of clusters", "7", "most stable", source="subagent")
    saved = json.loads((tmp_path / "decision_log.json").read_text())
    assert [e["step"] for e in saved] == ["plan", "domains"]
    assert saved[0]["alternatives_rejected"][1] == {"option": "none", "reason": "slide effect is strong"}
    assert saved[1]["source"] == "subagent" and log.by_step("domains")[0].chosen == "7"

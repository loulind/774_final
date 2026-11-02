"""The agent loop, from a dataset and an objective to a finished report.

    load -> profile -> objective -> challenge register -> plan -> execute and
    evaluate each step -> critic (at most one revision) -> figures -> report

Not built yet: the steps arrive one by one (tools, profiler and challenge register,
planner, LLM layer), and this module joins them once they exist.
"""

from __future__ import annotations


def run(data_path, objective: str, out_dir, **overrides) -> dict:
    raise NotImplementedError(
        "the end-to-end run is not built yet; `bioagent profile --data <path>` loads a dataset")

"""The tool registry: every tool the planner may use, as data.

Tools register themselves when their module is imported; ``load()`` imports the
Project 1-3 sub-agent adapters and the general tools. Adapters keep heavy imports
(torch, transformers) inside their run functions, so listing the registry is cheap.
The planner's menu, the validator's guardrails and ``bioagent tools`` all read the
same declarations, so they cannot drift apart.
"""

from __future__ import annotations

import importlib

from .base import ToolSpec

REGISTRY: dict[str, ToolSpec] = {}

TOOL_MODULES = (
    "analysis_agent.agents.dimreduce.adapter",
    "analysis_agent.agents.generative.adapter",
    "analysis_agent.agents.multimodal.adapter",
    "analysis_agent.tools.qc",
    "analysis_agent.tools.text",
    "analysis_agent.tools.normalize",
    "analysis_agent.tools.batch",
    "analysis_agent.tools.cluster",
    "analysis_agent.tools.spatial",
    "analysis_agent.tools.association",
    "analysis_agent.tools.de",
    "analysis_agent.tools.predict",
    "analysis_agent.tools.quality",
)
_loaded = False


def register(spec: ToolSpec) -> ToolSpec:
    if spec.name in REGISTRY and REGISTRY[spec.name] is not spec:
        raise ValueError(f"tool '{spec.name}' is registered twice")
    REGISTRY[spec.name] = spec
    return spec


def load() -> dict[str, ToolSpec]:
    global _loaded
    if not _loaded:
        for module in TOOL_MODULES:
            importlib.import_module(module)
        _loaded = True
    return REGISTRY


def get(name: str) -> ToolSpec:
    load()
    if name not in REGISTRY:
        raise KeyError(f"unknown tool '{name}'; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def all_tools() -> list[ToolSpec]:
    return list(load().values())


def describe_all() -> list[dict]:
    return [t.describe() for t in all_tools()]


def answering(question_type: str) -> list[ToolSpec]:
    return [t for t in all_tools() if question_type in t.answers]


def addressing(challenge: str) -> list[ToolSpec]:
    return [t for t in all_tools() if challenge in t.addresses]


def producing(kind: str) -> list[ToolSpec]:
    return [t for t in all_tools() if any(kind in p.kinds for p in t.produces)]


def consuming(kind: str) -> list[ToolSpec]:
    return [t for t in all_tools() if any(kind in p.kinds for p in t.consumes)]


def table() -> list[dict]:
    """One line per tool for ``bioagent tools``."""
    rows = []
    for t in all_tools():
        ins = ", ".join(f"{p.name}:{'/'.join(p.kinds)}{'' if p.required else '?'}" for p in t.consumes)
        outs = ", ".join(f"{p.name}:{'/'.join(p.kinds)}" for p in t.produces)
        rows.append({"tool": t.name, "origin": t.origin, "consumes": ins, "produces": outs,
                     "answers": ", ".join(t.answers), "addresses": ", ".join(t.addresses)})
    return rows

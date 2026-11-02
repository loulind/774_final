"""The common tool interface: typed artifacts, a workspace, tool specifications, step results.

Every tool, whether ported from Projects 1-3 or new, is described by a ``ToolSpec``:

    name, origin, source      what it is and where its code came from
    consumes / produces       typed ports (count matrix, embedding, labels, ...)
    params                    a pydantic model whose field bounds are the allowed ranges
    answers / addresses       the question types it answers and the challenges it handles
    criteria                  the metrics it reports, declared before it runs, each with a
                              goal (max / min / report) and an optional threshold
    cost                      a rough estimate of its run time from n, p and its parameters
    run(ctx, inputs, params)  does the work and returns a StepResult

The planner and the validator see only these declarations (``describe()``), never the
implementation. A plan is therefore a chain of ports: a step can run once the data or
an earlier step has supplied an artifact of every kind it consumes.

Artifacts are per-sample, except result tables. Each per-sample artifact carries
``rows``, the sorted dataset positions its rows belong to. Quality control removes
rows, subsampling selects them, and a tool that combines two artifacts calls
``Workspace.aligned`` to take the rows they share, so rows never drift apart.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pydantic import BaseModel, ConfigDict, ValidationError

from ..config import Budget, Criteria
from ..schema import Alternative, Decision, DecisionSource
from ..taxonomy import CHALLENGE_TYPES, QUESTION_TYPES

# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------

ARTIFACT_KINDS: dict[str, str] = {
    "counts": "non-negative integer matrix (sparse), samples x features; meta: feature_names",
    "normalized": "normalised or continuous matrix, samples x features; meta: feature_names, method",
    "embedding": "dense samples x dimensions representation; meta: method, source",
    "labels": "one categorical label per sample (None where missing); meta: external (from the "
              "data rather than made by the agent), source",
    "multilabel": "sparse boolean samples x labels; meta: labels (vocabulary), annotated (mask)",
    "groups": "one group id per sample (slide, patient, article, duplicate cluster); meta: "
              "column, role, nested_in",
    "coords": "samples x 2 spatial coordinates, comparable within a library; meta: library",
    "images": "an image source indexed by dataset position; rows are the samples with an image",
    "text": "one string per sample",
    "split": "one of train / val / test per sample (a predefined split)",
    "tabular": "per-sample covariates (a DataFrame aligned with rows)",
    "mask": "one boolean per sample (e.g. passed quality control); meta: reason",
    "scores": "per-sample numbers (n or n x k); meta: columns, direction",
    "table": "a result table (not per sample)",
}
PER_SAMPLE = frozenset(ARTIFACT_KINDS) - {"table"}
Origin = Literal["project1", "project2", "project3", "new"]


def _length(data) -> int | None:
    if data is None:
        return None
    if sp.issparse(data) or isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
        return int(data.shape[0])
    if isinstance(data, (list, tuple)):
        return len(data)
    return None


@dataclass
class Artifact:
    kind: str
    data: Any
    rows: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
    name: str = ""
    produced_by: str = "data"
    provenance: str = ""

    def __post_init__(self):
        if self.kind not in ARTIFACT_KINDS:
            raise ValueError(f"unknown artifact kind '{self.kind}'; known: {sorted(ARTIFACT_KINDS)}")
        if self.kind not in PER_SAMPLE:
            self.rows = None
            return
        if self.rows is None:
            raise ValueError(f"a '{self.kind}' artifact is per-sample and needs rows")
        rows = np.asarray(self.rows, dtype=np.int64).ravel()
        if len(rows) > 1 and np.any(np.diff(rows) <= 0):
            raise ValueError("rows must be strictly increasing dataset positions")
        self.rows = rows
        n = _length(self.data)
        if self.kind != "images" and n is not None and n != len(rows):
            raise ValueError(f"'{self.kind}' artifact has {n} data rows but {len(rows)} row positions")

    @property
    def n(self) -> int | None:
        return None if self.rows is None else len(self.rows)

    def locate(self, rows) -> np.ndarray:
        """Positions within this artifact of the given dataset rows (all must be present)."""
        rows = np.asarray(rows, dtype=np.int64)
        pos = np.searchsorted(self.rows, rows)
        ok = (pos < len(self.rows)) & (self.rows[np.minimum(pos, len(self.rows) - 1)] == rows)
        if not ok.all():
            raise KeyError(f"{int((~ok).sum())} requested rows are not in artifact '{self.name}'")
        return pos

    def take(self, rows):
        """This artifact's data for the given dataset rows, in that order."""
        if self.kind == "images":
            return self.data                       # indexed by dataset position already
        pos = self.locate(rows)
        if isinstance(self.data, (pd.DataFrame, pd.Series)):
            return self.data.iloc[pos]
        return self.data[pos]

    def shape(self) -> tuple[int, ...] | None:
        d = self.data
        if self.kind == "images":
            return (len(self.rows),)
        if hasattr(d, "shape"):
            return tuple(int(s) for s in d.shape)
        return None if _length(d) is None else (_length(d),)

    def describe(self) -> dict:
        """Kind, size and the scalar metadata, for plans, prompts and reports."""
        meta = {k: v for k, v in self.meta.items()
                if isinstance(v, (str, int, float, bool)) or v is None
                or (isinstance(v, (list, tuple)) and len(v) <= 12
                    and all(isinstance(x, (str, int, float)) for x in v))}
        return {"name": self.name, "kind": self.kind, "n": self.n, "shape": self.shape(),
                "produced_by": self.produced_by, "meta": to_jsonable(meta)}


def digest(obj) -> str:
    return hashlib.sha1(json.dumps(to_jsonable(obj), sort_keys=True, default=str).encode()).hexdigest()[:16]


class Workspace:
    """The typed artifact store of one run, plus a cache of step results."""

    def __init__(self, n_samples: int, sample_ids=None, dataset=None):
        self.n_samples = int(n_samples)
        self.sample_ids = None if sample_ids is None else np.asarray(sample_ids)
        self.dataset = dataset
        self._artifacts: dict[str, Artifact] = {}
        self.results: dict[str, StepResult] = {}

    def put(self, artifact: Artifact, name: str | None = None) -> Artifact:
        if name is not None:
            artifact.name = name
        if not artifact.name:
            raise ValueError("an artifact needs a name to be stored")
        if artifact.rows is not None and len(artifact.rows) and \
                (artifact.rows[0] < 0 or artifact.rows[-1] >= self.n_samples):
            raise ValueError(f"artifact '{artifact.name}' has rows outside 0..{self.n_samples - 1}")
        if not artifact.provenance:
            artifact.provenance = digest({"data": artifact.name, "n": artifact.n})
        self._artifacts[artifact.name] = artifact
        return artifact

    def get(self, name: str) -> Artifact:
        if name not in self._artifacts:
            raise KeyError(f"no artifact '{name}'; available: {sorted(self._artifacts)}")
        return self._artifacts[name]

    def __contains__(self, name: str) -> bool:
        return name in self._artifacts

    def names(self, kind: str | None = None) -> list[str]:
        return [n for n, a in self._artifacts.items() if kind is None or a.kind == kind]

    def aligned(self, *names: str, within=None) -> tuple[np.ndarray, list]:
        """The dataset rows every named per-sample artifact has (optionally within ``within``),
        and each artifact's data for exactly those rows, in the same order."""
        arts = [self.get(n) for n in names]
        rows = None if within is None else np.asarray(within, np.int64)
        for a in arts:
            if a.rows is not None:
                rows = a.rows if rows is None else np.intersect1d(rows, a.rows, assume_unique=True)
        if rows is None:
            rows = np.arange(self.n_samples)
        return rows, [a.take(rows) if a.rows is not None else a.data for a in arts]

    def describe(self) -> list[dict]:
        return [a.describe() for a in self._artifacts.values()]


# ---------------------------------------------------------------------------
# tool specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Port:
    name: str
    kinds: tuple[str, ...]
    required: bool = True
    doc: str = ""

    def __post_init__(self):
        bad = [k for k in self.kinds if k not in ARTIFACT_KINDS]
        if bad:
            raise ValueError(f"port '{self.name}': unknown artifact kinds {bad}")

    def describe(self) -> dict:
        return {"port": self.name, "kinds": list(self.kinds), "required": self.required, "doc": self.doc}


@dataclass(frozen=True)
class Criterion:
    """A metric a step reports, declared before it runs.

    ``goal`` says which direction is better ("report" = shown, not judged). ``selection``
    marks criteria that may decide between alternatives or judge a critic revision: only
    label-free, non-test quantities qualify. Metrics computed against external labels or
    the test split are reported for validation and never used to choose.
    """

    metric: str
    goal: Literal["max", "min", "report"]
    threshold: float | None = None
    doc: str = ""
    selection: bool = True

    def describe(self) -> dict:
        return {"metric": self.metric, "goal": self.goal, "threshold": self.threshold,
                "selection": self.selection, "doc": self.doc}


@dataclass
class CriterionResult:
    metric: str
    goal: str
    threshold: float | None
    value: Any
    passed: bool | None
    selection: bool
    doc: str = ""

    def to_record(self) -> dict:
        return to_jsonable(self.__dict__)


def _finite(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) \
        and math.isfinite(float(v))


def evaluate_criteria(criteria, metrics: dict) -> list[CriterionResult]:
    out = []
    for c in criteria:
        v = metrics.get(c.metric)
        passed = None
        if c.goal != "report" and c.threshold is not None and _finite(v):
            passed = bool(float(v) >= c.threshold) if c.goal == "max" else bool(float(v) <= c.threshold)
        out.append(CriterionResult(c.metric, c.goal, c.threshold, to_jsonable(v), passed, c.selection, c.doc))
    return out


def better(goal: str, new, old, tol: float = 1e-9) -> bool:
    """True when ``new`` strictly improves on ``old`` for a max/min criterion."""
    if goal == "report" or not (_finite(new) and _finite(old)):
        return False
    return float(new) > float(old) + tol if goal == "max" else float(new) < float(old) - tol


class ToolParams(BaseModel):
    """Base class of every tool's parameter model: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")


def _prop_bounds(prop: dict) -> tuple[float | None, float | None, str | None]:
    """(minimum, maximum, json type) of a JSON-schema property, looking inside anyOf."""
    options = prop.get("anyOf", [prop])
    for o in options:
        if o.get("type") in ("integer", "number"):
            return o.get("minimum"), o.get("maximum"), o["type"]
    return None, None, None


CostFn = Callable[[int, int, Any, Budget], float]


def _no_cost(n: int, p: int, params, budget: Budget) -> float:
    return 1.0


@dataclass(frozen=True)
class ToolSpec:
    name: str
    origin: Origin
    source: str
    summary: str
    consumes: tuple[Port, ...]
    produces: tuple[Port, ...]
    params: type[ToolParams]
    answers: tuple[str, ...]
    addresses: tuple[str, ...]
    criteria: tuple[Criterion, ...]
    run: Callable[[StepContext, dict[str, Artifact], Any], StepResult]
    cost: CostFn = _no_cost
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        unknown_q = [q for q in self.answers if q not in QUESTION_TYPES]
        unknown_c = [c for c in self.addresses if c not in CHALLENGE_TYPES]
        if unknown_q or unknown_c:
            raise ValueError(f"tool '{self.name}': unknown question types {unknown_q} "
                             f"or challenge types {unknown_c}")
        for ports in (self.consumes, self.produces):
            names = [p.name for p in ports]
            if len(names) != len(set(names)):
                raise ValueError(f"tool '{self.name}': duplicate port names {names}")
        metrics = [c.metric for c in self.criteria]
        if len(metrics) != len(set(metrics)):
            raise ValueError(f"tool '{self.name}': duplicate criteria {metrics}")

    # --- parameters -----------------------------------------------------------
    def parse_params(self, raw: dict | BaseModel | None) -> ToolParams:
        if isinstance(raw, self.params):
            return raw
        if isinstance(raw, BaseModel):
            raw = raw.model_dump()
        try:
            return self.params.model_validate(raw or {})
        except ValidationError as err:
            parts = [f"{'.'.join(map(str, e['loc'])) or '(params)'}: {e['msg']}" for e in err.errors()[:6]]
            raise ValueError(f"{self.name}: invalid parameters: {'; '.join(parts)}") from None

    def repair_params(self, raw: dict | None) -> tuple[dict, list[str]]:
        """Drop unknown keys and clip numbers into range; returns (params, notes)."""
        schema = self.params.model_json_schema()
        props = schema.get("properties", {})
        clean, notes = {}, []
        for key, value in (raw or {}).items():
            if key not in props:
                notes.append(f"{self.name}: dropped unknown parameter '{key}'")
                continue
            lo, hi, typ = _prop_bounds(props[key])
            if typ and _finite(value) and not isinstance(value, bool):
                v = float(value)
                clipped = min(max(v, lo if lo is not None else v), hi if hi is not None else v)
                if clipped != v:
                    notes.append(f"{self.name}: {key}={value} outside [{lo}, {hi}], clipped to {clipped:g}")
                value = int(round(clipped)) if typ == "integer" else clipped
            clean[key] = value
        try:
            self.params.model_validate(clean)
        except ValidationError as err:
            for e in err.errors():
                key = str(e["loc"][0]) if e["loc"] else ""
                if key in clean:
                    notes.append(f"{self.name}: {key}={clean[key]!r} invalid ({e['msg']}); default used")
                    clean.pop(key)
        return clean, notes

    # --- description for the planner, the validator and the report ------------
    def describe(self) -> dict:
        schema = self.params.model_json_schema()
        params = {}
        for key, prop in schema.get("properties", {}).items():
            lo, hi, typ = _prop_bounds(prop)
            entry = {"type": typ or prop.get("type") or "any", "default": prop.get("default"),
                     "doc": prop.get("description", "")}
            if lo is not None or hi is not None:
                entry["range"] = [lo, hi]
            enum = prop.get("enum") or next((o.get("enum") for o in prop.get("anyOf", []) if o.get("enum")), None)
            if enum is None and "items" in prop:
                enum = prop["items"].get("enum")
            if enum:
                entry["choices"] = enum
            params[key] = entry
        return {"name": self.name, "origin": self.origin, "source": self.source, "summary": self.summary,
                "consumes": [p.describe() for p in self.consumes],
                "produces": [p.describe() for p in self.produces],
                "params": params, "answers": list(self.answers), "addresses": list(self.addresses),
                "criteria": [c.describe() for c in self.criteria], "notes": list(self.notes)}


# ---------------------------------------------------------------------------
# running a step
# ---------------------------------------------------------------------------

@dataclass
class StepContext:
    step_id: str
    seed: int
    budget: Budget
    criteria: Criteria = field(default_factory=Criteria)
    device: str = "cpu"
    fit_device: str = "cpu"
    quick: bool = False
    cache_dir: Path | None = None
    out_dir: Path | None = None
    dataset: Any = None
    log: Callable[[str], None] = print
    decisions: list[Decision] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)

    def rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)

    def seconds_left(self) -> float:
        return self.budget.step_seconds - (time.perf_counter() - self.started)

    def decide(self, decision: str, chosen, rationale: str, rejected=(), evidence: dict | None = None,
               source: DecisionSource = "subagent") -> Decision:
        """Record a decision made inside this step (by a sub-agent's rules, by default)."""
        d = Decision(step=self.step_id, decision=decision, chosen=str(chosen), rationale=rationale,
                     alternatives_rejected=[Alternative(option=str(o), reason=str(r)) for o, r in rejected],
                     evidence=to_jsonable(evidence or {}), source=source)
        self.decisions.append(d)
        return d


@dataclass
class StepResult:
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)
    criteria: list[CriterionResult] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tool: str = ""
    step_id: str = ""
    params: dict = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0
    cached: bool = False
    key: str = ""

    def criterion(self, metric: str) -> CriterionResult | None:
        return next((c for c in self.criteria if c.metric == metric), None)

    def to_record(self) -> dict:
        return to_jsonable({
            "step": self.step_id, "tool": self.tool, "params": self.params, "inputs": self.inputs,
            "outputs": self.outputs, "seconds": round(self.seconds, 2), "cached": self.cached,
            "metrics": self.metrics, "criteria": [c.to_record() for c in self.criteria],
            "tables": {k: list(v.columns) for k, v in self.tables.items()}, "figures": self.figures,
            "notes": self.notes, "warnings": self.warnings,
            "decisions": [d.model_dump() for d in self.decisions]})


def _default_outputs(spec: ToolSpec, step_id: str) -> dict[str, str]:
    kinds = [p.kinds[0] for p in spec.produces]
    return {p.name: f"{p.kinds[0]}.{step_id}" if kinds.count(p.kinds[0]) == 1
            else f"{p.kinds[0]}.{step_id}_{p.name}" for p in spec.produces}


def run_step(spec: ToolSpec, ws: Workspace, ctx: StepContext, inputs: dict[str, str],
             params: dict | BaseModel | None = None, outputs: dict[str, str] | None = None,
             criteria: tuple[Criterion, ...] | None = None, use_cache: bool = True) -> StepResult:
    """Run one tool on workspace artifacts and store what it produces.

    Inputs are bound by artifact name and checked against the tool's ports; outputs are
    stored under the names the plan gave them (``kind.step_id`` by default). A step whose
    tool, parameters, seed and input provenance are unchanged is not re-run: the cached
    result is reused, which is how a critic revision re-runs only the steps downstream of
    its change.
    """
    p = spec.parse_params(params)
    known = {port.name for port in spec.consumes}
    extra = set(inputs) - known
    if extra:
        raise ValueError(f"{spec.name}: unknown input ports {sorted(extra)}; it consumes {sorted(known)}")
    arts: dict[str, Artifact] = {}
    for port in spec.consumes:
        name = inputs.get(port.name)
        if name is None:
            if port.required:
                raise ValueError(f"{spec.name}: required input '{port.name}' ({'/'.join(port.kinds)}) "
                                 "is not bound")
            continue
        a = ws.get(name)
        if a.kind not in port.kinds:
            raise TypeError(f"{spec.name}: input '{port.name}' must be {'/'.join(port.kinds)}, "
                            f"but '{name}' is {a.kind}")
        arts[port.name] = a
    outputs = {**_default_outputs(spec, ctx.step_id), **(outputs or {})}
    key = digest({"tool": spec.name, "params": p.model_dump(), "seed": ctx.seed, "quick": ctx.quick,
                  "inputs": {k: a.provenance for k, a in arts.items()}})

    if use_cache and key in ws.results:
        old = ws.results[key]
        for port, art in old.artifacts.items():
            ws.put(art, outputs[port])
        return StepResult(**{**old.__dict__, "cached": True, "step_id": ctx.step_id,
                             "outputs": {port: outputs[port] for port in old.artifacts}})

    ctx.decisions = []
    ctx.started = time.perf_counter()
    result = spec.run(ctx, arts, p)
    result.seconds = time.perf_counter() - ctx.started
    result.tool, result.step_id, result.key = spec.name, ctx.step_id, key
    result.params, result.inputs = to_jsonable(p.model_dump()), dict(inputs)
    result.decisions = list(ctx.decisions) + list(result.decisions)
    produced = {port.name: port for port in spec.produces}
    unknown = set(result.artifacts) - set(produced)
    if unknown:
        raise RuntimeError(f"{spec.name} returned undeclared outputs {sorted(unknown)}")
    missing = [n for n, port in produced.items() if port.required and n not in result.artifacts]
    if missing:
        raise RuntimeError(f"{spec.name} did not produce required outputs {missing}")
    for port_name, art in result.artifacts.items():
        if art.kind not in produced[port_name].kinds:
            raise RuntimeError(f"{spec.name}: output '{port_name}' is {art.kind}, declared "
                               f"{'/'.join(produced[port_name].kinds)}")
        art.produced_by = ctx.step_id
        art.provenance = digest({"step": key, "port": port_name})
        ws.put(art, outputs[port_name])
    result.outputs = {port: outputs[port] for port in result.artifacts}
    result.criteria = evaluate_criteria(criteria if criteria is not None else spec.criteria, result.metrics)
    ws.results[key] = result
    return result


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def to_jsonable(x):
    """Plain Python values for JSON: numpy scalars and arrays, paths, NaN -> None."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [to_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, BaseModel):
        return to_jsonable(x.model_dump())
    return x


def dense(X) -> np.ndarray:
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def check_time(ctx: StepContext, what: str) -> None:
    """Raise when the step has used its time budget (checked between expensive units)."""
    if ctx.seconds_left() <= 0:
        raise TimeoutError(f"{what}: step time budget of {ctx.budget.step_seconds:.0f}s used up")

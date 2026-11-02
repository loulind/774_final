"""Step 5: run a validated plan step by step, evaluating each step against its declared criteria.

Steps run in plan order. A reference ``@step.port`` resolves to the artifact that step stored;
first artifacts are used by name. A step that raises is recorded with its error, and every step
that needs one of its outputs is skipped with the reason, so one failure does not hide the rest
of the analysis. Each tool's own choices (a sub-agent's model selection, a QC threshold) enter
the decision log the first time the step runs. ``run_step`` caches results by tool, parameters,
seed and input provenance, so running a revised plan re-runs only the changed step and the
steps downstream of it, and running the original plan again restores its artifacts at no cost.

Quick mode on a large dataset first keeps a subsample that preserves its structure: whole
libraries (slides) when samples are spatial, otherwise every split in proportion, and whole
groups when samples share groups.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Budget, RunConfig
from .decision_log import DecisionLog
from .planner.refs import is_ref, split_ref
from .schema import AnalysisPlan
from .tools import registry
from .tools.base import Artifact, StepContext, StepResult, Workspace, run_step


class Executor:
    def __init__(self, ws: Workspace, cfg: RunConfig, budget: Budget, device: str, log: DecisionLog,
                 out_dir: Path | None = None, echo=print):
        self.ws, self.cfg, self.budget, self.device, self.log = ws, cfg, budget, device, log
        self.out_dir, self.echo = out_dir, echo
        self.results: dict[str, StepResult] = {}
        self.logged: set[str] = set()                      # result keys whose decisions are already logged

    def run(self, plan: AnalysisPlan, label: str = "") -> dict[str, dict]:
        """Run every step; returns ``{step id: StepResult.to_record()}`` (plus error / skipped)."""
        records: dict[str, dict] = {}
        refs: dict[str, str] = {}
        failed: set[str] = set()
        for step in plan.steps:
            blocked = sorted({split_ref(r)[0] for r in step.inputs.values() if is_ref(r)} & failed)
            if blocked:
                records[step.id] = {"step": step.id, "tool": step.tool, "skipped": f"needs {', '.join(blocked)}",
                                    "error": None, "metrics": {}, "criteria": []}
                failed.add(step.id)
                continue
            inputs = {port: refs.get(ref, ref) if is_ref(ref) else ref for port, ref in step.inputs.items()}
            ctx = StepContext(step_id=step.id, seed=self.cfg.seed, budget=self.budget, criteria=self.cfg.criteria,
                              device=self.device, fit_device=self.cfg.fit_device, quick=self.cfg.quick,
                              cache_dir=Path(self.cfg.cache_dir), out_dir=self.out_dir, dataset=self.ws.dataset,
                              log=self.echo)
            t0 = time.perf_counter()
            self.echo(f"[{label or 'run'}] {step.id} ({step.tool})")
            try:
                res = run_step(registry.get(step.tool), self.ws, ctx, inputs, step.params)
            except Exception as err:                         # recorded, and its dependants skipped
                msg = f"{type(err).__name__}: {str(err)[:400]}"
                records[step.id] = {"step": step.id, "tool": step.tool, "error": msg, "metrics": {}, "criteria": [],
                                    "seconds": round(time.perf_counter() - t0, 2)}
                failed.add(step.id)
                self.log.add(step.id, "step failed", step.tool, msg, source="system")
                self.echo(f"    failed: {msg}")
                continue
            for port, name in res.outputs.items():
                refs[f"@{step.id}.{port}"] = name
            self.results[step.id] = res
            records[step.id] = {**res.to_record(), "error": None}
            if res.key not in self.logged:
                self.log.extend(res.decisions)
                self.logged.add(res.key)
        return records


def subsample(ws: Workspace, max_samples: int, seed: int, log: DecisionLog) -> Workspace:
    """A structure-preserving subsample of at most ``max_samples`` samples (quick mode)."""
    rng = np.random.default_rng(seed)
    arts = [ws.get(n) for n in ws.names()]
    rows = np.unique(np.concatenate([a.rows for a in arts if a.rows is not None and len(a.rows)]))
    if len(rows) <= max_samples:
        return ws
    groups = [a for a in arts if a.kind == "groups"]
    library = next((g for g in groups if g.meta.get("role") == "library"), None)
    split = next((a for a in arts if a.kind == "split"), None)
    coords = any(a.kind == "coords" for a in arts)
    if coords and library is not None:
        lib = pd.Series(np.asarray(library.data).astype(str), index=library.rows)
        sizes = lib.value_counts()
        coarse = min(groups, key=lambda g: g.meta.get("n_levels") or 0)
        owner = pd.Series(np.asarray(coarse.data).astype(str), index=coarse.rows)
        first = pd.DataFrame({"lib": lib, "unit": owner.reindex(lib.index)}).groupby("lib")["unit"].first()
        order = list(rng.permutation(sizes.index))
        picked, seen_units, total = [], set(), 0
        for pass_ in (0, 1):                               # one library per unit first, then any
            for lv in order:
                if lv in picked or (pass_ == 0 and first.get(lv) in seen_units):
                    continue
                if total + sizes[lv] > max_samples and picked:
                    continue
                picked.append(lv)
                seen_units.add(first.get(lv))
                total += int(sizes[lv])
        keep = lib.index[lib.isin(picked)].to_numpy()
        how = f"{len(picked)} whole {library.meta.get('column', 'library')} levels"
    elif split is not None:
        s = pd.Series(np.asarray(split.data).astype(str), index=split.rows)
        frac = max_samples / len(s)
        keep = np.concatenate([rng.choice(idx.to_numpy(), max(1, int(round(frac * len(idx)))), replace=False)
                               for _, idx in s.groupby(s).groups.items()])
        how = "each split in proportion"
    else:
        keep = rng.choice(rows, max_samples, replace=False)
        how = "at random"
    keep = np.sort(keep.astype(np.int64))
    out = Workspace(ws.n_samples, ws.sample_ids, dataset=ws.dataset)
    for a in arts:
        if a.rows is None:
            out.put(a, a.name)
            continue
        mask = np.isin(a.rows, keep)
        if a.kind == "images":
            data = a.data
        elif isinstance(a.data, (pd.DataFrame, pd.Series)):
            data = a.data.iloc[np.flatnonzero(mask)]
        else:
            data = a.data[np.flatnonzero(mask)]
        meta = dict(a.meta)
        if a.kind == "multilabel" and meta.get("annotated") is not None:
            meta["annotated"] = np.asarray(meta["annotated"])[np.flatnonzero(mask)]
        out.put(Artifact(a.kind, data, a.rows[mask], meta), a.name)
    log.add("data", "quick-mode subsample", f"{len(keep):,} of {len(rows):,} samples ({how})",
            f"Quick mode caps the samples at {max_samples:,}; the subsample keeps the data's structure.",
            [("a random subsample", "would break libraries, splits and groups apart")], source="system")
    return out

"""Step 1: measure the data (the profile).

Every modality and first artifact is measured by fixed conventions and by the same rules
the tools apply later (the default QC rules, the batch measurements, the near-duplicate
hash, the label-support rule), so the challenge register reads numbers the plan will act
on. The profile holds measurements only - "silhouette by slide 0.21", never "strong batch
effect"; judging them is the register's job (``challenges.py``). It never records the
dataset's name or location.
"""

from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd

from ..config import Budget, RunConfig, load_config
from ..data.dataset import is_numeric_column
from ..schema import (ArtifactRecord, ComputeProfile, CostEstimate, DataProfile, ModalityRecord, SplitProfile,
                      TabularProfile)
from ..tools.base import Artifact, Workspace
from ..tools.ingest import workspace_from_dataset
from .labels import agreement_profiles, group_profile, label_profile
from .matrix import group_structure, matrix_profile, spatial_profile
from .media import image_profile, text_profile

EMBEDDING_DIMS = 50            # assumed width of a learned representation in the cost estimates
PATH = re.compile(r"(?:[A-Za-z]:)?(?:[\\/][^\s'\"\\/:]+)+[\\/]?")
TOP_ESTIMATES = 8


def cost_estimates(ws: Workspace, budget: Budget) -> list[CostEstimate]:
    """Declared costs (default parameters) of every tool that can run on the first artifacts,
    or on a representation of them, most expensive first."""
    from ..tools import registry

    first: dict[str, Artifact] = {}
    for name in ws.names():
        first.setdefault(ws.get(name).kind, ws.get(name))
    per_sample = [a.n for a in first.values() if a.rows is not None and a.n]
    n_main = max(per_sample) if per_sample else ws.n_samples
    out = []
    for spec in registry.load().values():
        inputs, n, p, ok = {}, None, None, True
        for port in spec.consumes:
            if not port.required:
                continue
            a = next((first[k] for k in port.kinds if k in first), None)
            if a is not None:
                shp = a.shape()
                pn, pp = a.n or n_main, (shp[1] if shp and len(shp) > 1 else 1)
                inputs[port.name] = a.name
            elif "embedding" in port.kinds:
                pn, pp = n_main, EMBEDDING_DIMS
                inputs[port.name] = f"a {EMBEDDING_DIMS}-dimensional representation"
            else:
                ok = False
                break
            if n is None:
                n, p = pn, pp
        if not ok or n is None:
            continue
        n = min(n, budget.max_samples) if budget.max_samples else n
        out.append(CostEstimate(tool=spec.name, inputs=inputs, seconds=float(spec.cost(n, p, spec.params(), budget))))
    return sorted(out, key=lambda e: -e.seconds)[:TOP_ESTIMATES]


def profile_dataset(ds, ws: Workspace | None = None, config: RunConfig | None = None, budget: Budget | None = None,
                    cache_dir=None, log=print) -> DataProfile:
    """Measure a loaded dataset (and its first artifacts)."""
    t0 = time.perf_counter()
    cfg = config or load_config()
    if budget is None:
        from ..hardware import detect

        budget = cfg.budget(detect().device)
    rng = np.random.default_rng(cfg.seed)
    ws = ws if ws is not None else workspace_from_dataset(ds)
    arts = [ws.get(n) for n in ws.names()]

    def of(kind: str) -> list[Artifact]:
        return [a for a in arts if a.kind == kind]

    groups = of("groups")
    library = next((g for g in groups if g.meta.get("role") == "library"), None)
    split = next(iter(of("split")), None)
    prof = DataProfile(
        n_samples=ds.n, format=ds.format, seed=cfg.seed, notes=[_without_paths(n) for n in ds.notes],
        modalities=[ModalityRecord(name=k, kind=m.kind, n_present=m.n_present,
                                   missing_fraction=float(1 - m.n_present / max(1, ds.n)), source=m.source)
                    for k, m in ds.modalities.items()],
        roles={str(r.column): r.role for r in ds.roles},
        artifacts=[ArtifactRecord(name=a.name, kind=a.kind, n=a.n or 0, detail=_detail(a)) for a in arts])

    caches = {}
    for a in of("counts") + of("normalized"):
        log(f"  profiling {a.name}")
        rec, cache = matrix_profile(a, library, rng, cfg.seed)
        prof.matrices.append(rec)
        caches[a.name] = cache
        prof.group_structure += group_structure(a.name, cache, groups, rng)
    for c in of("coords"):
        lib = ws.get(c.meta["library"]) if c.meta.get("library") in ws else None
        mats = [m for m in of("counts") + of("normalized") if len(np.intersect1d(m.rows, c.rows)) > 0]
        m = mats[0] if mats else None
        log(f"  profiling {c.name}")
        prof.spatial.append(spatial_profile(c, m, caches.get(m.name) if m is not None else None, lib,
                                            cfg.criteria.fdr))
    for a in of("images"):
        grp = ws.get(a.meta["library"]) if a.meta.get("library") in ws else \
            (min(groups, key=lambda g: g.meta.get("n_levels") or 0) if groups else None)
        log(f"  profiling {a.name}")
        prof.images.append(image_profile(a, grp, split, cache_dir, rng, log))
    for a in of("text"):
        prof.texts.append(text_profile(a, split))
    for a in of("tabular"):
        df = a.data
        prof.tabular.append(TabularProfile(artifact=a.name, n_columns=int(df.shape[1]),
                                           n_numeric=int(sum(is_numeric_column(df[c]) for c in df.columns)),
                                           missing_cell_fraction=float(df.isna().to_numpy().mean()) if df.size else 0.0,
                                           columns_with_missing=int(df.isna().any().sum())))
    labels = of("labels") + of("multilabel")
    for a in labels:
        prof.labels.append(label_profile(a, split, groups, caches, ds.n, cfg.criteria.reliable_min_test_positives))
    prof.label_agreement = agreement_profiles(labels)
    prof.groups = [group_profile(g, split) for g in groups]
    if split is not None:
        prof.split = SplitProfile(artifact=split.name,
                                  counts={str(k): int(v) for k, v in pd.Series(split.data).value_counts().items()})
    names = list(ds.modalities)
    prof.pairing = {f"{a}|{b}": int((ds.modalities[a].present & ds.modalities[b].present).sum())
                    for i, a in enumerate(names) for b in names[i + 1:]}
    prof.compute = ComputeProfile(device=budget.device, step_seconds=budget.step_seconds,
                                  total_seconds=budget.total_seconds, estimates=cost_estimates(ws, budget))
    prof.seconds = float(time.perf_counter() - t0)
    return prof


def _without_paths(note: str) -> str:
    """A loader note with file paths cut to their last component: the profile is shown to the
    LLM, which must not learn where (or what) the data are."""
    return PATH.sub(lambda m: m.group(0).rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1], note)


def _detail(a: Artifact) -> dict:
    """The few facts about an artifact a plan binds by: size, columns, levels, labels."""
    shp = a.shape()
    d: dict = {}
    if a.kind in ("counts", "normalized", "embedding") and shp and len(shp) > 1:
        d["n_features"] = int(shp[1])
    for key in ("column", "role", "n_levels", "n_classes", "n_labels", "source_kind", "library", "external"):
        v = a.meta.get(key)
        if isinstance(v, (str, int, float, bool)):
            d[key] = v
    if a.meta.get("nested_in"):
        d["nested_in"] = list(a.meta["nested_in"])
    if a.kind == "split" and a.meta.get("counts"):
        d["levels"] = sorted(a.meta["counts"])
    return d

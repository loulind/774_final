"""Quality control of count matrices and images.

Counts: a sample is removed when it is empty, or when its log library size or its log
number of detected features lies more than ``nmads`` scaled median absolute deviations
below the median (the outlier rule of scater's ``isOutlier``). Depth differs between
libraries (slides, runs), so the thresholds are computed within each library when every
library is large enough to have its own distribution; one global threshold would remove
whole low-depth libraries. Features detected in fewer than ``min_feature_samples`` samples
are dropped.

Images: unreadable, tiny (shorter side below ``min_side`` pixels), an extreme aspect ratio,
or blank (almost no grey-level variation). One pass measures every image's size, grey-level
spread and 64-bit difference hash; the pass is cached per dataset, so the near-duplicate
tool reuses it instead of decoding every image again.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pydantic import Field

from .base import Artifact, Criterion, Port, StepResult, ToolParams, ToolSpec, check_time, digest, values_on
from .registry import register
from .stats import mad_outliers

SOURCE = "new in Project 4 (the hash helper is Project 3's, in agents/multimodal/imaging.py)"


def _by_group(groups: np.ndarray, flags: dict[str, np.ndarray], removed: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"group": groups, "removed": removed, **flags})
    out = df.groupby("group").agg(n=("removed", "size"), removed=("removed", "sum"),
                                  **{k: (k, "sum") for k in flags}).reset_index()
    out["fraction_removed"] = out["removed"] / out["n"]
    return out


# ---------------------------------------------------------------------------
# counts
# ---------------------------------------------------------------------------

class CountQCParams(ToolParams):
    nmads: float = Field(3.0, ge=2.0, le=6.0, description="scaled MADs below the median that flag a sample")
    min_feature_samples: int = Field(10, ge=1, le=1000, description="features detected in fewer samples "
                                                                    "are dropped")
    per_library: Literal["auto", "yes", "no"] = Field(
        "auto", description="thresholds within each library; auto = yes when every library has at least "
                            "min_library_samples samples")
    min_library_samples: int = Field(50, ge=10, le=5000)


def _run_qc_counts(ctx, inputs, p: CountQCParams) -> StepResult:
    a, library, report = inputs["counts"], inputs.get("library"), inputs.get("report_by")
    rows = a.rows
    X = sp.csr_matrix(a.data)
    lib_size = np.asarray(X.sum(1)).ravel().astype(float)
    detected = np.asarray((X > 0).sum(1)).ravel().astype(float)
    lib = values_on(rows, library, "(none)")
    sizes = pd.Series(lib).value_counts() if lib is not None else None
    per_lib = p.per_library == "yes" or (p.per_library == "auto" and sizes is not None and len(sizes) > 1
                                         and int(sizes.min()) >= p.min_library_samples)
    blocks = {lv: np.flatnonzero(lib == lv) for lv in sizes.index} if per_lib else {"all": np.arange(len(rows))}
    empty = lib_size == 0
    low_depth, low_feat = np.zeros(len(rows), bool), np.zeros(len(rows), bool)
    thresholds = []
    for lv, idx in blocks.items():
        ok = idx[~empty[idx]]
        if len(ok) == 0:
            continue
        f1, lo1, _ = mad_outliers(np.log1p(lib_size[ok]), p.nmads)
        f2, lo2, _ = mad_outliers(np.log1p(detected[ok]), p.nmads)
        low_depth[ok], low_feat[ok] = f1, f2
        thresholds.append({"library": lv, "n": len(idx), "min_library_size": float(np.expm1(lo1)),
                           "min_detected": float(np.expm1(lo2)), "median_library_size": float(np.median(lib_size[ok])),
                           "flagged": int((f1 | f2).sum()) + int(empty[idx].sum())})
    removed = empty | low_depth | low_feat
    col = library.meta.get("column", library.name) if library is not None else None
    ctx.decide("count QC thresholds", f"per {col}" if per_lib else "global",
               (f"Depth differs between {col} levels (median library size from "
                f"{min(t['median_library_size'] for t in thresholds):,.0f} to "
                f"{max(t['median_library_size'] for t in thresholds):,.0f}), so each {col} gets its own "
                f"{p.nmads:g}-MAD lower threshold on log library size and log detected features.") if per_lib else
               (f"One {p.nmads:g}-MAD lower threshold on log library size and log detected features"
                + (f": the smallest {col} has {int(sizes.min())} samples, fewer than the {p.min_library_samples} "
                   "needed for its own threshold." if sizes is not None and len(sizes) > 1 else ".")),
               [("global thresholds", "would remove whole low-depth libraries") if per_lib else
                ("per-library thresholds", "too few samples per library" if library is not None
                 else "no library grouping given")],
               {"thresholds": thresholds[:40]})
    keep_rows = rows[~removed]
    Xk = X[~removed]
    feat_n = np.asarray((Xk > 0).sum(0)).ravel()
    keep_feat = feat_n >= p.min_feature_samples
    names = list(a.meta.get("feature_names") or [f"f{j}" for j in range(X.shape[1])])
    ctx.decide("features kept", f"{int(keep_feat.sum()):,} of {X.shape[1]:,}",
               f"Features detected in fewer than {p.min_feature_samples} of the {len(keep_rows):,} kept samples "
               "carry too little information to normalise or test.",
               [("keep all features", "near-empty features add noise and multiple-testing burden")],
               {"dropped": int((~keep_feat).sum())})
    flags = {"empty": empty, "low_library_size": low_depth & ~empty, "low_detected": low_feat & ~empty}
    rep = values_on(rows, report) if report is not None else lib
    table = _by_group(rep if rep is not None else np.full(len(rows), "all"), flags, removed)
    out_counts = Artifact("counts", Xk[:, keep_feat].tocsr(), keep_rows,
                          {**a.meta, "feature_names": [n for n, k in zip(names, keep_feat) if k],
                           "qc": f"passed {ctx.step_id}"})
    mask = Artifact("mask", ~removed, rows, {"reason": "passed count QC", "source": a.name})
    metrics = {"n_input": len(rows), "n_removed": int(removed.sum()), "fraction_removed": float(removed.mean()),
               "max_group_fraction_removed": float(table["fraction_removed"].max()),
               "n_empty": int(empty.sum()), "n_low_library_size": int(flags["low_library_size"].sum()),
               "n_low_detected": int(flags["low_detected"].sum()),
               "n_features_kept": int(keep_feat.sum()), "n_features_dropped": int((~keep_feat).sum()),
               "median_library_size": float(np.median(lib_size[~removed])) if (~removed).any() else None,
               "per_library_thresholds": bool(per_lib)}
    return StepResult(artifacts={"counts": out_counts, "mask": mask}, metrics=metrics,
                      tables={"qc_by_group": table, "qc_thresholds": pd.DataFrame(thresholds)})


def _qc_counts_cost(n, p, prm, budget) -> float:
    return 2e-8 * n * p + 1.0


register(ToolSpec(
    name="qc_counts", origin="new", source=SOURCE,
    summary="Removes empty and low-quality samples of a count matrix (MAD outliers in log library size or "
            "log detected features, per library when libraries are large enough) and near-empty features.",
    consumes=(Port("counts", ("counts",)),
              Port("library", ("groups",), required=False, doc="library / slide / run: thresholds per level"),
              Port("report_by", ("groups",), required=False, doc="grouping for the removal table")),
    produces=(Port("counts", ("counts",), doc="samples and features that passed"),
              Port("mask", ("mask",), doc="True = passed, over the input samples")),
    params=CountQCParams, answers=("data_quality_audit",), addresses=("low_quality_samples",),
    criteria=(Criterion("fraction_removed", "min", 0.20, "share of samples removed"),
              Criterion("max_group_fraction_removed", "min", 0.50, "largest share removed from one group: "
                                                                   "removal concentrated in a group is a warning")),
    run=_run_qc_counts, cost=_qc_counts_cost,
))


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------

STAT_COLUMNS = ("readable", "width", "height", "grey_std", "dhash")


def _measure(source, i: int) -> tuple:
    from ..agents.multimodal.imaging import dhash

    try:
        im = source.load(int(i))
        w, h = im.size
        g = im.convert("L")
        t = g.copy()
        t.thumbnail((64, 64))
        return True, w, h, float(np.asarray(t, np.float32).std()), dhash(g)
    except Exception:                                   # unreadable or missing file
        return False, 0, 0, float("nan"), 0


def image_stats(images: Artifact, rows: np.ndarray, cache_dir=None, log=print, check=None,
                workers: int = 8, chunk: int = 2000) -> pd.DataFrame:
    """Size, grey-level spread of a 64-pixel thumbnail and difference hash of every image in
    ``rows``, indexed by row. Cached per dataset under ``cache_dir/image_stats``; rows not in
    the cache are measured and added."""
    source = images.data
    path = None
    if cache_dir is not None and images.meta.get("dataset_key"):
        key = digest({"data": images.meta["dataset_key"], "source": source.describe()})
        path = cache_dir / "image_stats" / f"{key}.npz"
    have = pd.DataFrame(columns=STAT_COLUMNS)
    if path is not None and path.exists():
        z = np.load(path)
        have = pd.DataFrame({c: z[c] for c in STAT_COLUMNS}, index=z["rows"])
    todo = np.setdiff1d(np.asarray(rows, np.int64), have.index.to_numpy(np.int64))
    if len(todo):
        todo = source.order(todo)
        log(f"    measuring {len(todo):,} images")
        found = []
        pool = ThreadPoolExecutor(max_workers=workers) if getattr(source, "kind", "") == "files" else None
        try:
            for s in range(0, len(todo), chunk):
                if check is not None:
                    check()
                part = todo[s:s + chunk]
                res = (list(pool.map(lambda i: _measure(source, i), part)) if pool
                       else [_measure(source, i) for i in part])
                found.append(pd.DataFrame(res, columns=STAT_COLUMNS, index=part))
        finally:
            if pool:
                pool.shutdown()
        new = pd.concat(found)
        new["dhash"] = new["dhash"].astype(np.uint64)
        have = pd.concat([have.astype(new.dtypes.to_dict()), new]) if len(have) else new
        have = have.sort_index()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, rows=have.index.to_numpy(np.int64),
                     **{c: have[c].to_numpy() for c in STAT_COLUMNS})
    out = have.loc[np.asarray(rows, np.int64)]
    return out.astype({"readable": bool, "width": int, "height": int, "grey_std": float, "dhash": np.uint64})


class ImageQCParams(ToolParams):
    min_side: int = Field(32, ge=4, le=1024, description="shorter side below this many pixels = tiny")
    max_aspect: float = Field(5.0, ge=1.5, le=50.0, description="longer / shorter side above this = extreme")
    blank_std: float = Field(2.0, ge=0.0, le=20.0, description="grey-level standard deviation (0-255) of a "
                                                               "64-pixel thumbnail below this = blank")


def _run_qc_images(ctx, inputs, p: ImageQCParams) -> StepResult:
    images, report, split = inputs["images"], inputs.get("report_by"), inputs.get("split")
    rows = images.rows
    st = image_stats(images, rows, ctx.cache_dir, ctx.log, lambda: check_time(ctx, "image QC"))
    ok = st["readable"].to_numpy()
    w, h = st["width"].to_numpy(float), st["height"].to_numpy(float)
    short, long_ = np.minimum(w, h), np.maximum(w, h)
    flags = {"unreadable": ~ok,
             "tiny": ok & (short < p.min_side),
             "extreme_aspect": ok & (long_ > p.max_aspect * np.maximum(short, 1)),
             "blank": ok & (st["grey_std"].to_numpy() < p.blank_std)}
    removed = np.zeros(len(rows), bool)
    for f in flags.values():
        removed |= f
    tables = {}
    for name, art in (("split", split), ("group", report)):
        if art is not None:
            tables[f"image_qc_by_{name}"] = _by_group(values_on(rows, art, "(none)"), flags, removed)
    if not tables:
        tables["image_qc_by_group"] = _by_group(np.full(len(rows), "all"), flags, removed)
    worst = max(float(t["fraction_removed"].max()) for t in tables.values())
    ctx.decide("image QC rules", f"unreadable, shorter side < {p.min_side}px, aspect > {p.max_aspect:g}, "
                                 f"thumbnail grey s.d. < {p.blank_std:g}",
               "Images that cannot be decoded, are too small to show structure, are strips or are blank cannot "
               "be encoded meaningfully.", [("keep every image", "degenerate inputs give meaningless features")],
               {k: int(v.sum()) for k, v in flags.items()})
    metrics = {"n_input": len(rows), "n_removed": int(removed.sum()), "fraction_removed": float(removed.mean()),
               "max_group_fraction_removed": worst, **{f"n_{k}": int(v.sum()) for k, v in flags.items()},
               "width_median": float(np.median(w[ok])) if ok.any() else None,
               "height_median": float(np.median(h[ok])) if ok.any() else None,
               "size_iqr_ratio": float(np.quantile(short[ok], 0.75) / max(1.0, np.quantile(short[ok], 0.25)))
               if ok.any() else None}
    flagged = pd.DataFrame({"row": rows, **flags})[removed]
    tables["image_qc_flagged"] = flagged.head(200)
    arts = {"images": Artifact("images", images.data, rows[~removed], {**images.meta, "qc": f"passed {ctx.step_id}"}),
            "mask": Artifact("mask", ~removed, rows, {"reason": "passed image QC", "source": images.name})}
    return StepResult(artifacts=arts, metrics=metrics, tables=tables)


def _qc_images_cost(n, p, prm, budget) -> float:
    return 4e-3 * n + 2.0


register(ToolSpec(
    name="qc_images", origin="new", source=SOURCE,
    summary="Flags unreadable, tiny, strip-shaped and blank images; one cached pass also measures sizes and "
            "difference hashes for later steps.",
    consumes=(Port("images", ("images",)),
              Port("split", ("split",), required=False, doc="report removal per split"),
              Port("report_by", ("groups",), required=False, doc="report removal per group")),
    produces=(Port("images", ("images",), doc="images that passed"),
              Port("mask", ("mask",), doc="True = passed, over the input images")),
    params=ImageQCParams, answers=("data_quality_audit",), addresses=("low_quality_samples", "image_heterogeneity"),
    criteria=(Criterion("fraction_removed", "min", 0.10, "share of images removed"),
              Criterion("max_group_fraction_removed", "min", 0.50, "largest share removed from one split/group"),
              Criterion("size_iqr_ratio", "report", doc="spread of image sizes (Q3/Q1 of the shorter side)")),
    run=_run_qc_images, cost=_qc_images_cost,
))

"""Matrix measurements: depth, sparsity, the default count-QC rule, principal components,
how groupings and labels structure those components, and spatial autocorrelation.

Conventions (ways of measuring, not modelling choices): at most ``MAX_ROWS`` sampled rows;
counts scaled to the median library size and log1p-transformed; the ``N_VARIABLE`` most
variable features, standardised; ``N_PCS`` principal components, of which the first
``STRUCTURE_PCS`` carry the group measurements; Moran's I on the ``MORAN_FEATURES`` most
variable features over all rows, on the same neighbour graph the spatial tools use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..schema import GroupStructureProfile, MatrixProfile, SpatialProfile
from ..tools.base import Artifact, values_on
from ..tools.batch import measure_groups
from ..tools.normalize import library_sizes, pc1_depth
from ..tools.qc import CountQCParams, count_outliers
from ..tools.spatial import GraphParams
from ..tools.stats import bh_adjust, moran_libraries, morans_i, row_standardise, spatial_knn

MAX_ROWS = 5000
N_VARIABLE = 2000
N_PCS = 50
STRUCTURE_PCS = 20
MORAN_FEATURES = 100
MIN_LEVEL = 10                 # samples a group level needs to enter the structure measurements


def _sample(n: int, k: int, rng) -> np.ndarray:
    return np.sort(rng.choice(n, k, replace=False)) if n > k else np.arange(n)


def log_values(X: sp.csr_matrix, lib: np.ndarray, target: float) -> sp.csr_matrix:
    """Counts scaled to ``target`` total, then log1p (sparse)."""
    scale = np.divide(target, lib, out=np.zeros_like(lib, dtype=float), where=lib > 0)
    Y = sp.csr_matrix(sp.diags(scale) @ X, dtype=np.float32)
    Y.data = np.log1p(Y.data)
    return Y


def variance_order(Y: sp.csr_matrix) -> np.ndarray:
    """Features from most to least variable."""
    mean = np.asarray(Y.mean(0)).ravel()
    var = np.asarray(Y.multiply(Y).mean(0)).ravel() - mean ** 2
    return np.argsort(-var, kind="stable")


def matrix_profile(a: Artifact, library: Artifact | None, rng, seed: int) -> tuple[MatrixProfile, dict]:
    """Measurements of one matrix artifact, plus what the structure and spatial measurements
    reuse (the sampled rows, their component scores, features ranked by variability)."""
    from sklearn.decomposition import PCA

    X = sp.csr_matrix(a.data)
    n, p = X.shape
    counts = a.kind == "counts"
    rec = MatrixProfile(artifact=a.name, is_counts=counts, n_samples=n, n_features=p,
                        sparsity=float(1 - X.nnz / max(1, n * p)), features_per_sample=p / max(1, n))
    lib = library_sizes(X) if counts else None
    ok = np.arange(n) if lib is None else np.flatnonzero(lib > 0)
    target = None
    if counts:
        libv = values_on(a.rows, library, "(none)")
        qc = count_outliers(X, libv, **CountQCParams().model_dump(exclude={"min_feature_samples"}))
        pos = lib[lib > 0]
        rec.library_size_median = float(np.median(lib))
        rec.library_size_log2_iqr = float(np.subtract(*np.percentile(np.log2(pos), [75, 25]))) if len(pos) else None
        if libv is not None:
            med = pd.Series(lib).groupby(libv).median()
            rec.library_median_range = [float(med.min()), float(med.max())]
        rec.n_empty = int(qc["empty"].sum())
        rec.qc_flagged_fraction = float(qc["removed"].mean())
        rec.qc_per_library = bool(qc["per_library"])
        rec.detected_features_median = float(np.median(qc["detected"]))
        target = float(np.median(pos)) if len(pos) else 1.0
    det = np.asarray((X > 0).mean(0)).ravel()
    rec.n_features_detected_5pct = int((det >= 0.05).sum())
    sub = ok[_sample(len(ok), MAX_ROWS, rng)]
    Xs = X[sub]
    if counts:
        d5 = det >= 0.05
        if d5.any():
            mean = np.asarray(Xs[:, d5].mean(0)).ravel()
            var = np.asarray(Xs[:, d5].multiply(Xs[:, d5]).mean(0)).ravel() - mean ** 2
            rec.overdispersed_fraction = float(np.mean(var > mean))
        Y = log_values(Xs, lib[sub], target)
    else:
        Y = sp.csr_matrix(Xs, dtype=np.float32)
    ranked = variance_order(Y)
    top = np.sort(ranked[:N_VARIABLE])
    M = Y[:, top].toarray().astype(np.float64)
    M -= M.mean(0)
    sd = M.std(0)
    M /= np.where(sd > 0, sd, 1.0)
    k = int(min(N_PCS, M.shape[0] - 1, M.shape[1]))
    pca = PCA(k, svd_solver="randomized", random_state=seed).fit(M)
    S = pca.transform(M)
    cum = np.cumsum(pca.explained_variance_ratio_)
    rec.components_50 = int(np.searchsorted(cum, 0.5) + 1) if cum[-1] >= 0.5 else None
    rec.components_90 = int(np.searchsorted(cum, 0.9) + 1) if cum[-1] >= 0.9 else None
    rec.variance_top_components = float(cum[-1])
    if counts:
        logd = np.log(lib[sub])
        rec.pc1_depth_rho_scaled = pc1_depth(Y[:, top].toarray(), logd, True, seed)[0]
        raw = sp.csr_matrix(Xs[:, top], dtype=np.float32)
        raw.data = np.log1p(raw.data)
        rec.pc1_depth_rho_raw = pc1_depth(raw.toarray(), logd, True, seed)[0]
    rec.measured_on = (f"{len(sub):,} sampled rows; "
                       + ("counts scaled to the median library size, log1p; " if counts else "values as stored; ")
                       + f"the {len(top):,} most variable features, standardised; {k} principal components")
    return rec, {"rows": a.rows[sub], "scores": S[:, :STRUCTURE_PCS], "ranked": ranked, "lib": lib,
                 "target": target}


def measurable(rows: np.ndarray, groups: list[Artifact]) -> dict[str, tuple[Artifact, np.ndarray, set]]:
    """Groupings whose levels with >= MIN_LEVEL samples (at least two of them) hold at least
    half of ``rows``: name -> (artifact, values on rows, the large levels)."""
    out = {}
    for g in groups:
        v = values_on(rows, g, "")
        sizes = pd.Series(v[v != ""]).value_counts()
        big = sizes[sizes >= MIN_LEVEL]
        if len(big) >= 2 and big.sum() >= 0.5 * len(rows):
            out[g.name] = (g, v, set(big.index))
    return out


def group_structure(matrix: str, cache: dict, groups: list[Artifact], rng) -> list[GroupStructureProfile]:
    rows, S = cache["rows"], cache["scores"]
    meas = measurable(rows, groups)
    out = []
    for name, (g, v, big) in meas.items():
        outer = next((o for o in g.meta.get("nested_in", []) if o in meas), None)
        keep = np.isin(v, list(big))
        o = None
        if outer is not None:
            o = meas[outer][1]
            keep &= o != ""
            o = o[keep]
        m = measure_groups(S[keep], v[keep], o, 30, MAX_ROWS, rng)
        rec = GroupStructureProfile(matrix=matrix, group=name, column=str(g.meta.get("column", name)),
                                    n_levels=m["n_batches"], silhouette=m["batch_silhouette"],
                                    variance_share=m["variance_share_batch"], mixing=m["mixing"],
                                    measured_on=f"first {S.shape[1]} principal components of {int(keep.sum()):,} "
                                                f"sampled rows (levels with >= {MIN_LEVEL} samples)")
        if m["batch_nested_in_outer"]:
            rec.outer = outer
            rec.within_outer_silhouette = m.get("within_outer_batch_silhouette")
            rec.within_outer_mixing = m.get("within_outer_mixing")
            rec.share_between_outer = m.get("share_between_outer")
            rec.share_between_inner_within_outer = m.get("share_between_inner_within_outer")
            rec.n_outer_with_several_levels = m.get("n_outer_with_several_batches")
        out.append(rec)
    return out


def spatial_profile(coords: Artifact, matrix: Artifact | None, cache: dict | None, library: Artifact | None,
                    fdr: float) -> SpatialProfile:
    from sklearn.neighbors import NearestNeighbors

    rows = coords.rows if matrix is None else np.intersect1d(coords.rows, matrix.rows, assume_unique=True)
    libs = values_on(rows, library, "(none)") if library is not None else np.full(len(rows), "all")
    xy = np.asarray(coords.take(rows), float)
    sizes, nn = [], []
    for lv in np.unique(libs):
        sel = np.flatnonzero(libs == lv)
        sizes.append(len(sel))
        if len(sel) > 2:
            d, _ = NearestNeighbors(n_neighbors=2).fit(xy[sel]).kneighbors(xy[sel])
            nn.append(float(np.median(d[:, 1])))
    rec = SpatialProfile(artifact=coords.name, matrix=None if matrix is None else matrix.name,
                         library=None if library is None else library.name, n_libraries=len(sizes),
                         samples_per_library=[float(min(sizes)), float(np.median(sizes)), float(max(sizes))],
                         nn_distance_median=float(np.median(nn)) if nn else None)
    if matrix is None or cache is None:
        return rec
    gp = GraphParams()
    W = row_standardise(spatial_knn(xy, libs, gp.n_neighbors, gp.max_distance_factor))
    feats = np.sort(cache["ranked"][:MORAN_FEATURES])
    X = sp.csr_matrix(matrix.take(rows))[:, feats]
    Y = (log_values(X, cache["lib"][matrix.locate(rows)], cache["target"]) if matrix.kind == "counts"
         else sp.csr_matrix(X, dtype=np.float32)).toarray()
    res = morans_i(Y, moran_libraries(W, libs))
    q = bh_adjust(res["p"])
    ok = np.isfinite(q)
    rec.features_tested = int(ok.sum())
    rec.fraction_autocorrelated = float(np.mean(q[ok] < fdr)) if ok.any() else None
    rec.median_morans_i = float(np.nanmedian(res["I"])) if ok.any() else None
    rec.measured_on = (f"Moran's I of the {len(feats)} most variable features on each library's "
                       f"{gp.n_neighbors}-nearest-neighbour graph, centred per library, BH across features")
    return rec

"""Spatial tools: domains, spatially variable features, neighbourhood enrichment.

All three use a k-nearest-neighbour graph built inside each library (coordinates are only
comparable within one slide), with neighbours farther than ``max_distance_factor`` times the
library's median nearest-neighbour distance dropped.

Domains (after BANKSY, Singhal et al. 2024): every sample's standardised features are
joined by the mean features of its spatial neighbours, weighted sqrt(1 - lambda) and
sqrt(lambda); lambda = 0.8 is BANKSY's recommendation for domains (0.2 for cell types).
The joined features are clustered with the stability rule of ``cluster``. Spatial
coherence is the share of neighbour pairs in the same domain, corrected for chance like
kappa: (observed - expected) / (1 - expected), expected from the domain sizes per library.

Spatially variable features: Moran's I per feature, centred within each library and pooled
over libraries (see ``stats.morans_i``), Benjamini-Hochberg across features. The test's
calibration is checked on the same data: features whose positions are permuted within each
library should give p < 0.05 about 5% of the time.

Neighbourhood enrichment: edges between every pair of labels counted and compared with
label permutations inside each library (z-scores, two-sided permutation p-values, BH).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pydantic import Field

from .base import Artifact, Criterion, Port, StepResult, ToolParams, ToolSpec, check_time, dense, values_on
from .cluster import Method, choose_clustering, cluster_result
from .normalize import library_sizes
from .registry import register
from .stats import bh_adjust, moran_libraries, morans_i, row_standardise, spatial_knn

SOURCE = "new in Project 4 (domain features after BANKSY; Moran's I as in Project 3's profiler, extended with " \
         "per-library centring and the randomisation variance)"


def _libraries(rows: np.ndarray, library) -> np.ndarray:
    lib = values_on(rows, library, "(none)")
    return np.full(len(rows), "all", object) if lib is None else lib


def _check_coords(coords: Artifact, library) -> None:
    if library is None and coords.meta.get("library"):
        raise ValueError(f"coordinates '{coords.name}' are per library ({coords.meta['library']}); bind it as "
                         "'library' so neighbours are never taken across libraries")


def coherence(labels: np.ndarray, A: sp.csr_matrix, libraries: np.ndarray) -> float | None:
    """Chance-corrected share of neighbour pairs with the same label (kappa-like)."""
    U = sp.triu(((A + A.T) > 0).astype(np.int8), k=1).tocoo()
    if U.nnz == 0:
        return None
    same = labels[U.row] == labels[U.col]
    obs = float(same.mean())
    lib_edges = pd.Series(libraries[U.row]).value_counts()
    exp = 0.0
    for lib, n_edges in lib_edges.items():
        share = pd.Series(labels[libraries == lib]).value_counts(normalize=True)
        exp += n_edges * float((share ** 2).sum())
    exp /= U.nnz
    return (obs - exp) / (1 - exp) if exp < 1 else None


class GraphParams(ToolParams):
    n_neighbors: int = Field(6, ge=3, le=30, description="spatial neighbours per sample")
    max_distance_factor: float = Field(1.5, ge=1.0, le=5.0, description="drop neighbours farther than this x the "
                                                                        "median nearest-neighbour distance")


# ---------------------------------------------------------------------------
# domains
# ---------------------------------------------------------------------------

class DomainParams(GraphParams):
    neighbour_weight: float = Field(0.8, ge=0.0, le=0.95, description="lambda: weight of the neighbourhood "
                                                                      "mean (BANKSY: 0.8 for domains)")
    methods: list[Method] = Field(default_factory=lambda: ["kmeans", "gmm", "leiden"])
    k_min: int = Field(2, ge=2, le=50)
    k_max: int = Field(12, ge=2, le=60)
    min_stability: float = Field(0.75, ge=0.5, le=0.95)
    min_cluster_fraction: float = Field(0.005, ge=0.0, le=0.2)
    graph_neighbors: int = Field(15, ge=5, le=100)
    split_alpha: float = Field(0.05, ge=0.001, le=0.2)


def _run_domains(ctx, inputs, p: DomainParams) -> StepResult:
    emb, coords, library = inputs["embedding"], inputs["coords"], inputs.get("library")
    _check_coords(coords, library)
    rows = np.intersect1d(emb.rows, coords.rows, assume_unique=True)
    if library is not None:
        rows = np.intersect1d(rows, library.rows, assume_unique=True)
    Z = dense(emb.take(rows)).astype(np.float64)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    lib = _libraries(rows, library)
    A = spatial_knn(coords.take(rows), lib, p.n_neighbors, p.max_distance_factor)
    W = row_standardise(A)
    isolated = np.asarray(A.sum(1)).ravel() == 0
    M = W @ Z
    M[isolated] = Z[isolated]
    lam = p.neighbour_weight
    F = np.hstack([np.sqrt(1 - lam) * Z, np.sqrt(lam) * M])
    ctx.decide("domain features", f"own + neighbourhood mean, lambda {lam:g}",
               f"Each sample's features are joined by the mean of its {p.n_neighbors} nearest spatial neighbours "
               f"in the same library (BANKSY), weighted sqrt(1-lambda) : sqrt(lambda); lambda {lam:g} favours "
               "contiguous domains over cell-type-like clusters.",
               [("cluster own features only (lambda 0)", "ignores that neighbouring samples share a domain"),
                ("lambda 0.2", "BANKSY's setting for cell typing, not domains")],
               {"isolated_samples": int(isolated.sum()), "mean_neighbours": float(A.sum(1).mean())})
    lo, hi = min(p.k_min, p.k_max), max(p.k_min, p.k_max)
    res = choose_clustering(ctx, F, p.methods, lo, hi, p.min_stability, p.min_cluster_fraction,
                            p.graph_neighbors, what="spatial domains", split_alpha=p.split_alpha)
    out = cluster_result(ctx, res, rows, emb.name, kind="domains", groups=inputs.get("groups"))
    out.metrics["spatial_coherence"] = coherence(res["labels"], A, lib)
    plain = choose_coherence_reference(Z, res["k"], ctx.seed)
    out.metrics["spatial_coherence_own_features"] = coherence(plain, A, lib)
    out.metrics["isolated_samples"] = int(isolated.sum())
    out.artifacts["labels"].meta.update({"spatial": True, "neighbour_weight": lam})
    return out


def choose_coherence_reference(Z: np.ndarray, k: int, seed: int) -> np.ndarray:
    """k-means on the samples' own features with the same k, as a coherence reference."""
    from sklearn.cluster import KMeans

    return KMeans(k, n_init=4, random_state=seed).fit_predict(Z)


register(ToolSpec(
    name="spatial_domains", origin="new", source=SOURCE,
    summary="Spatial domains: features joined with their spatial neighbourhood mean (BANKSY-style) and clustered "
            "with the subsampling-stability rule; reports chance-corrected spatial coherence.",
    consumes=(Port("embedding", ("embedding", "normalized")), Port("coords", ("coords",)),
              Port("library", ("groups",), required=False, doc="library of each sample (neighbours stay inside)"),
              Port("groups", ("groups",), required=False, doc="groups whose overlap with the domains is reported "
                                                              "(e.g. patients)")),
    produces=(Port("labels", ("labels",), doc="domain of every sample (external = False)"),),
    params=DomainParams, answers=("structure_discovery", "spatial_structure"), addresses=("spatial_autocorrelation",),
    criteria=(Criterion("stability", "max", 0.75, "mean subsample Jaccard of the least stable domain"),
              Criterion("spatial_coherence", "max", None, "chance-corrected share of neighbours in the same domain"),
              Criterion("k_at_edge", "min", 0, "1 when k is at the end of the searched range"),
              Criterion("spatial_coherence_own_features", "report", doc="the same for clusters of own features"),
              Criterion("cramers_v_groups", "report", doc="association of domains with the groups (0 = spread "
                                                          "evenly, 1 = one group each)")),
    run=_run_domains, cost=lambda n, p, prm, b: 2 * (8e-6 * (n + b.bootstrap_reps * min(n, b.eval_max_samples))
                                                    * 11 * max(1, p // 10)) + 3,
))


# ---------------------------------------------------------------------------
# spatially variable features
# ---------------------------------------------------------------------------

class SVGParams(GraphParams):
    min_fraction: float = Field(0.01, ge=0.0, le=0.5, description="features detected in fewer samples are "
                                                                  "not tested")
    max_features: int = Field(20_000, ge=50, le=60_000)
    chunk: int = Field(1000, ge=50, le=5000)
    calibration_features: int = Field(200, ge=0, le=2000, description="features re-tested with permuted "
                                                                      "positions to check calibration")


def _feature_block(a: Artifact, X, cols: np.ndarray, lib_size, target) -> np.ndarray:
    sub = X[:, cols]
    if a.kind == "counts":
        scale = np.divide(target, lib_size, out=np.zeros_like(lib_size), where=lib_size > 0)
        sub = sp.diags(scale) @ sp.csr_matrix(sub)
        return np.log1p(np.asarray(sub.toarray(), np.float64))
    return np.asarray(sub.toarray() if sp.issparse(sub) else sub, np.float64)


def _run_svg(ctx, inputs, p: SVGParams) -> StepResult:
    a, coords, library = inputs["matrix"], inputs["coords"], inputs.get("library")
    _check_coords(coords, library)
    rows = np.intersect1d(a.rows, coords.rows, assume_unique=True)
    if library is not None:
        rows = np.intersect1d(rows, library.rows, assume_unique=True)
    X = a.take(rows)
    X = sp.csr_matrix(X) if sp.issparse(X) else np.asarray(X)
    names = np.asarray(a.meta.get("feature_names") or [f"f{j}" for j in range(X.shape[1])], dtype=object)
    lib = _libraries(rows, library)
    det = np.asarray((X > 0).mean(0)).ravel() if a.kind == "counts" else np.ones(X.shape[1])
    test = np.flatnonzero(det >= p.min_fraction)
    if len(test) > p.max_features:
        test = np.sort(test[np.argsort(-det[test])[: p.max_features]])
    lib_size = library_sizes(sp.csr_matrix(X)) if a.kind == "counts" else None
    target = float(np.median(lib_size[lib_size > 0])) if lib_size is not None else None
    A = spatial_knn(coords.take(rows), lib, p.n_neighbors, p.max_distance_factor)
    libs = moran_libraries(row_standardise(A), lib)
    if not libs:
        raise RuntimeError("no library has enough samples with spatial neighbours for Moran's I")
    parts = []
    for s in range(0, len(test), p.chunk):
        check_time(ctx, "spatially variable features")
        cols = test[s:s + p.chunk]
        parts.append(morans_i(_feature_block(a, X, cols, lib_size, target), libs))
    res = {k: np.concatenate([r[k] for r in parts]) for k in parts[0]}
    q = bh_adjust(res["p"])
    fdr = ctx.criteria.fdr
    table = pd.DataFrame({"feature": names[test], "morans_i": res["I"], "z": res["z"], "p": res["p"], "q": q,
                          "n_libraries": res["n_libraries"],
                          "positive_library_share": res["n_positive"] / np.maximum(res["n_libraries"], 1),
                          "detected_fraction": det[test]}).sort_values(["q", "z"], ascending=[True, False])
    calib = None
    if p.calibration_features:
        rng = ctx.rng(13)
        pick = np.sort(rng.choice(test, min(p.calibration_features, len(test)), replace=False))
        block = _feature_block(a, X, pick, lib_size, target)
        perm = np.arange(len(rows))
        for L in libs:
            perm[L["idx"]] = rng.permutation(L["idx"])
        calib = float(np.nanmean(morans_i(block[perm], libs)["p"] < 0.05))
    sig = table["q"] < fdr
    ctx.decide("spatially variable features", f"{int(sig.sum()):,} of {len(test):,} at FDR {fdr:g}",
               f"Moran's I on the {p.n_neighbors}-nearest-neighbour graph of each library, centred within library "
               "(differences between libraries are not spatial structure), randomisation variance pooled over "
               f"{len(libs)} libraries, Benjamini-Hochberg across features."
               + (f" Calibration: {calib:.1%} of {min(p.calibration_features, len(test))} features with permuted "
                  "positions reach p < 0.05." if calib is not None else ""),
               [("global centring", "slide-level mean differences would count as spatial autocorrelation"),
                ("no multiple-testing control", f"{len(test):,} tests")],
               {"tested": len(test), "libraries": len(libs), "permuted_p05_fraction": calib})
    metrics = {"n_tested": len(test), "n_significant": int(sig.sum()), "fraction_significant": float(sig.mean()),
               "n_libraries": len(libs), "permuted_p05_fraction": calib,
               "median_i_significant": float(table.loc[sig, "morans_i"].median()) if sig.any() else None,
               "top_features": table["feature"].head(10).tolist()}
    return StepResult(artifacts={"table": Artifact("table", table)}, metrics=metrics,
                      tables={"spatially_variable": table})


register(ToolSpec(
    name="spatial_svg", origin="new", source=SOURCE,
    summary="Spatially variable features: Moran's I per feature within each library, pooled, BH-corrected, with "
            "a permutation check of the test's calibration.",
    consumes=(Port("matrix", ("counts", "normalized"), doc="counts are log-normalised feature block by block"),
              Port("coords", ("coords",)), Port("library", ("groups",), required=False)),
    produces=(Port("table", ("table",), doc="Moran's I, z, p and q per feature"),),
    params=SVGParams, answers=("spatial_structure", "differential_analysis"),
    addresses=("spatial_autocorrelation", "multiple_testing"),
    criteria=(Criterion("n_significant", "report", doc="features spatially variable at the FDR level"),
              Criterion("permuted_p05_fraction", "min", 0.10, "share of permuted features with p < 0.05 "
                                                              "(about 0.05 when calibrated)")),
    run=_run_svg, cost=lambda n, p, prm, b: 6e-8 * n * min(p, 20_000) * 3 + 2.0,
))


# ---------------------------------------------------------------------------
# neighbourhood enrichment
# ---------------------------------------------------------------------------

class EnrichParams(GraphParams):
    n_permutations: int = Field(999, ge=19, le=9999, description="capped by the budget's permutations")


def _edge_counts(codes: np.ndarray, u: np.ndarray, v: np.ndarray, L: int) -> np.ndarray:
    a, b = codes[u], codes[v]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return np.bincount(lo * L + hi, minlength=L * L)


def _run_enrichment(ctx, inputs, p: EnrichParams) -> StepResult:
    labels, coords, library = inputs["labels"], inputs["coords"], inputs.get("library")
    _check_coords(coords, library)
    rows = np.intersect1d(labels.rows, coords.rows, assume_unique=True)
    if library is not None:
        rows = np.intersect1d(rows, library.rows, assume_unique=True)
    y = np.asarray(labels.take(rows)).astype(str)
    lib = _libraries(rows, library)
    levels, codes = np.unique(y, return_inverse=True)
    L = len(levels)
    A = spatial_knn(coords.take(rows), lib, p.n_neighbors, p.max_distance_factor)
    U = sp.triu(((A + A.T) > 0).astype(np.int8), k=1).tocoo()
    u, v = U.row, U.col
    obs = _edge_counts(codes, u, v, L)
    B = int(min(p.n_permutations, ctx.budget.permutations))
    rng = ctx.rng(17)
    blocks = [np.flatnonzero(lib == lv) for lv in np.unique(lib)]
    null = np.zeros((B, L * L))
    for r in range(B):
        if r % 100 == 0:
            check_time(ctx, "neighbourhood enrichment")
        perm = codes.copy()
        for idx in blocks:
            perm[idx] = codes[rng.permutation(idx)]
        null[r] = _edge_counts(perm, u, v, L)
    mu, sd = null.mean(0), null.std(0, ddof=1)
    pairs = [(i, j) for i in range(L) for j in range(i, L)]
    flat = [i * L + j for i, j in pairs]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(sd > 0, (obs - mu) / sd, 0.0)
    extreme = (np.abs(null - mu) >= np.abs(obs - mu) - 1e-9).sum(0)
    pval = (1 + extreme) / (1 + B)
    table = pd.DataFrame({"label_a": [levels[i] for i, _ in pairs], "label_b": [levels[j] for _, j in pairs],
                          "edges": obs[flat], "expected": mu[flat], "z": z[flat], "p": pval[flat]})
    table["q"] = bh_adjust(table["p"])
    fdr = ctx.criteria.fdr
    sig = table["q"] < fdr
    self_z = table.loc[table["label_a"] == table["label_b"], "z"]
    metrics = {"n_labels": L, "n_edges": int(len(u)), "n_pairs": len(table), "n_permutations": B,
               "n_enriched": int((sig & (table["z"] > 0)).sum()), "n_depleted": int((sig & (table["z"] < 0)).sum()),
               "self_enrichment_median_z": float(self_z.median()) if len(self_z) else None}
    ctx.decide("neighbourhood enrichment test", f"{B} label permutations within each library",
               "Edges between every pair of labels are compared with label permutations that keep each library's "
               f"label counts; two-sided permutation p-values, Benjamini-Hochberg over {len(table)} pairs.",
               [("analytic z-scores", "edge counts are not independent")], {"edges": int(len(u))})
    return StepResult(artifacts={"table": Artifact("table", table)}, metrics=metrics,
                      tables={"neighbourhood_enrichment": table})


register(ToolSpec(
    name="spatial_enrichment", origin="new", source=SOURCE,
    summary="Which labels (domains, cell types) border each other more or less than chance: neighbour-edge counts "
            "against within-library label permutations, BH-corrected.",
    consumes=(Port("labels", ("labels",)), Port("coords", ("coords",)), Port("library", ("groups",), required=False)),
    produces=(Port("table", ("table",), doc="edges, expected, z, p, q per label pair"),),
    params=EnrichParams, answers=("spatial_structure",), addresses=("multiple_testing",),
    criteria=(Criterion("n_enriched", "report"), Criterion("n_depleted", "report"),
              Criterion("self_enrichment_median_z", "report", doc="how strongly labels cluster with themselves")),
    run=_run_enrichment, cost=lambda n, p, prm, b: 2e-6 * n * 6 * min(getattr(prm, "n_permutations", 999),
                                                                      b.permutations) + 1.0,
))

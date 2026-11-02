"""Clustering, with the number of clusters chosen by subsampling stability.

Candidates are k-means and Gaussian mixtures over a range of k and Leiden community
detection on a k-nearest-neighbour graph over a range of resolutions. Every candidate is
fitted on all samples and again on ``bootstrap_reps`` random subsamples (80%, at most
``eval_max_samples``). A cluster's stability is its mean best Jaccard overlap with the
clusters of the subsample fits (Hennig 2007, as in fpc::clusterboot), and a candidate's
stability is that of its least stable cluster. The finest candidate whose every cluster
reaches ``min_stability`` - Hennig's 0.75 bar for a stable cluster - and whose clusters are
separated is kept: a k that cuts one real cluster in two, at a different place in every
subsample, fails the bar, so the data rather than a fixed k decide how fine to go.

Stability alone can pass a k-means cut through one elongated group, because the cut falls
in the same place in every subsample. So every pair of clusters whose centroids are mutual
nearest neighbours is also tested against a single Gaussian (SigClust, Liu et al. 2008; the
same safeguard against over-clustering as sc-SHC, Grabski et al. 2023): the pair's two-means
cluster index is compared with those of Gaussian samples with the pair's covariance, and a
candidate with a pair that is not better separated (p > ``split_alpha``) is passed over.
When no candidate passes, the most stable one is kept and the result is flagged as unstable
(or, if only the separation test fails, as not separated). Candidates with a cluster smaller
than ``min_cluster_fraction`` of the samples are not eligible.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from .base import Artifact, Criterion, Port, StepResult, ToolParams, ToolSpec, check_time, dense, values_on
from .registry import register
from .stats import cramers_v

SOURCE = ("new in Project 4 (stability after Hennig 2007; separation test after Liu et al. 2008; Leiden via "
          "the leidenalg package)")
Method = Literal["kmeans", "gmm", "leiden"]
LEIDEN_RESOLUTIONS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0)
SUBSAMPLE = 0.8
SPLIT_REPS = 99                           # Gaussian samples per separation test
SPLIT_CAP = 2000                          # samples of a cluster pair used by the test
GROUP_PRESENT = 10                        # samples a group needs in a cluster to count as present there


def knn_graph(Z: np.ndarray, k: int = 15):
    import igraph as ig
    from sklearn.neighbors import NearestNeighbors

    kk = int(min(k, len(Z) - 1))
    _, ind = NearestNeighbors(n_neighbors=kk + 1).fit(Z).kneighbors(Z)
    src = np.repeat(np.arange(len(Z)), kk)
    dst = ind[:, 1:].ravel()
    pairs = np.unique(np.sort(np.column_stack([src, dst]), 1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return ig.Graph(n=len(Z), edges=pairs.tolist())


def fit_clusters(method: str, Z: np.ndarray, size: float, seed: int, graph_k: int = 15) -> np.ndarray:
    """Integer cluster labels of one candidate (size = k, or the Leiden resolution)."""
    if method == "kmeans":
        from sklearn.cluster import KMeans

        return KMeans(int(size), n_init=4, random_state=seed).fit_predict(Z)
    if method == "gmm":
        from sklearn.mixture import GaussianMixture

        cov = "full" if Z.shape[1] <= 20 else "diag"
        return GaussianMixture(int(size), covariance_type=cov, reg_covar=1e-4, max_iter=300,
                               random_state=seed).fit(Z).predict(Z)
    if method == "leiden":
        import leidenalg

        part = leidenalg.find_partition(knn_graph(Z, graph_k), leidenalg.RBConfigurationVertexPartition,
                                        resolution_parameter=float(size), seed=int(seed) % (2 ** 31),
                                        n_iterations=-1)
        return np.asarray(part.membership)
    raise ValueError(f"unknown clustering method {method}")


def relabel_by_size(lab: np.ndarray) -> np.ndarray:
    """Clusters renumbered 0..K-1 from largest to smallest."""
    _, inv, counts = np.unique(lab, return_inverse=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inv]


def jaccard_stability(method: str, Z: np.ndarray, size: float, full: np.ndarray, reps: int, seed: int,
                      cap: int, graph_k: int = 15, check=None) -> np.ndarray:
    """Mean over subsample fits of every full-data cluster's best Jaccard overlap."""
    rng = np.random.default_rng(seed)
    n, K = len(Z), int(full.max()) + 1
    m = int(min(round(SUBSAMPLE * n), cap))
    J = np.zeros((reps, K))
    for r in range(reps):
        if check is not None:
            check()
        sub = np.sort(rng.choice(n, m, replace=False))
        lab = fit_clusters(method, Z[sub], size, seed + 1 + r, graph_k)
        ct = pd.crosstab(full[sub], lab).reindex(index=range(K), fill_value=0).to_numpy(float)
        rs, cs = ct.sum(1), ct.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            jac = np.where(rs[:, None] > 0, ct / (rs[:, None] + cs[None, :] - ct), 0.0)
        J[r] = jac.max(1)
    return J.mean(0)


def two_means_index(U: np.ndarray, iters: int = 20) -> float:
    """Within-cluster share of U's total sum of squares after splitting it in two (Lloyd's
    algorithm started from the sign of the first principal component)."""
    A = U - U.mean(0)
    tot = float((A ** 2).sum())
    if tot <= 0 or len(A) < 4:
        return 1.0
    y = A @ np.linalg.eigh(A.T @ A)[1][:, -1] > 0
    for _ in range(iters):
        if y.all() or not y.any():
            return 1.0
        c1, c0 = A[y].mean(0), A[~y].mean(0)
        new = ((A - c1) ** 2).sum(1) < ((A - c0) ** 2).sum(1)
        if np.array_equal(new, y):
            break
        y = new
    if y.all() or not y.any():
        return 1.0
    return float((((A[y] - A[y].mean(0)) ** 2).sum() + ((A[~y] - A[~y].mean(0)) ** 2).sum()) / tot)


def split_p_value(U: np.ndarray, reps: int, rng) -> float:
    """Share of Gaussian samples with U's covariance that split in two at least as well as U;
    small = U holds two groups rather than one group cut in half."""
    obs = two_means_index(U)
    lam = np.clip(np.linalg.eigvalsh(np.atleast_2d(np.cov(U, rowvar=False))), 0, None)
    null = np.array([two_means_index(rng.standard_normal(U.shape) * np.sqrt(lam)) for _ in range(reps)])
    return float((1 + (null <= obs).sum()) / (reps + 1))


def split_support(Z: np.ndarray, lab: np.ndarray, reps: int, seed: int, cap: int = SPLIT_CAP) -> float:
    """Largest separation p-value over the cluster pairs whose centroids are mutual nearest
    neighbours (the pairs most likely to be one group cut in two)."""
    rng = np.random.default_rng(seed)
    K = int(lab.max()) + 1
    C = np.vstack([Z[lab == c].mean(0) for c in range(K)])
    D = ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    worst = 0.0
    for i in range(K):
        j = int(nn[i])
        if i < j and nn[j] == i:
            idx = np.flatnonzero((lab == i) | (lab == j))
            if len(idx) > cap:
                idx = rng.choice(idx, cap, replace=False)
            worst = max(worst, split_p_value(Z[idx], reps, rng))
    return worst


def choose_clustering(ctx, Z: np.ndarray, methods, k_min: int, k_max: int, min_stability: float,
                      min_cluster_fraction: float, graph_k: int, what: str = "clustering",
                      split_alpha: float = 0.05) -> dict:
    """Fit every candidate, score its stability, keep the finest stable and separated one
    (see module doc)."""
    reps = max(2, int(ctx.budget.bootstrap_reps))
    cap = int(ctx.budget.eval_max_samples)
    min_size = max(10, int(np.ceil(min_cluster_fraction * len(Z))))
    rows, fits = [], {}
    for method in methods:
        sizes = range(k_min, k_max + 1) if method != "leiden" else LEIDEN_RESOLUTIONS
        seen_k = set()
        for size in sizes:
            check_time(ctx, what)
            lab = relabel_by_size(fit_clusters(method, Z, size, ctx.seed, graph_k))
            k = int(lab.max()) + 1
            if method == "leiden" and (k < k_min or k > k_max or k in seen_k):
                continue                                   # outside the range, or a k already covered
            seen_k.add(k)
            smallest = int(np.bincount(lab).min())
            jac = jaccard_stability(method, Z, size, lab, reps, ctx.seed + 100, cap, graph_k,
                                    lambda: check_time(ctx, what)) if k > 1 else np.ones(1)
            key = f"{method}:{size}"
            fits[key] = (lab, jac)
            rows.append({"candidate": key, "method": method, "parameter": size, "k": k,
                         "stability_min": float(jac.min()), "stability_mean": float(jac.mean()),
                         "smallest_cluster": smallest, "size_ok": smallest >= min_size})
            ctx.log(f"    {what}: {key} k={k} stability min {jac.min():.2f} mean {jac.mean():.2f}")
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError(f"{what}: no candidate produced between {k_min} and {k_max} clusters")
    table["eligible"] = table["size_ok"] & (table["stability_min"] >= min_stability)
    table["split_p"] = np.nan
    pool = table[table["eligible"]].sort_values(["k", "stability_min", "stability_mean"], ascending=False)
    stable, separated, best = len(pool) > 0, False, None
    split_reps = int(min(ctx.budget.permutations, SPLIT_REPS))
    for _, cand in pool.iterrows():                        # finest first; stop at the first separated one
        check_time(ctx, what)
        pv = split_support(Z, fits[cand["candidate"]][0], split_reps, ctx.seed + 7)
        table.loc[table["candidate"] == cand["candidate"], "split_p"] = pv
        if pv <= split_alpha:
            best, separated = table[table["candidate"] == cand["candidate"]].iloc[0], True
            break
    if best is None:
        fallback = pool if stable else (table[table["size_ok"]] if table["size_ok"].any() else table)
        best = fallback.sort_values(["stability_min", "stability_mean", "k"], ascending=False).iloc[0]
    table["chosen"] = table["candidate"] == best["candidate"]
    lab, jac = fits[best["candidate"]]
    subs = f"{reps} subsamples of {min(round(SUBSAMPLE * len(Z)), cap):,} samples"
    if separated:
        why = (f"The finest candidate whose every cluster has mean subsample Jaccard >= {min_stability:g} ({subs}) "
               f"and whose neighbouring clusters are better separated than one Gaussian group cut in two (largest "
               f"separation p = {best['split_p']:.3f} <= {split_alpha:g}): {best['method']} with k={best['k']} "
               f"(least stable cluster {best['stability_min']:.2f}).")
    elif stable:
        why = (f"Every candidate with all clusters at mean subsample Jaccard >= {min_stability:g} ({subs}) has a pair "
               f"of neighbouring clusters no better separated than one Gaussian group cut in two (p > "
               f"{split_alpha:g}), so the structure may be continuous rather than discrete; the most stable one is "
               f"kept and flagged: {best['method']} with k={best['k']} (least stable cluster "
               f"{best['stability_min']:.2f}).")
    else:
        why = (f"No candidate keeps every cluster at mean subsample Jaccard >= {min_stability:g} ({subs}); the most "
               f"stable one is kept and flagged: {best['method']} with k={best['k']} (least stable cluster "
               f"{best['stability_min']:.2f}).")

    def reason(r) -> str:
        if not r["size_ok"]:
            return f"k={r['k']}: smallest cluster {r['smallest_cluster']} < {min_size}"
        if r["stability_min"] < min_stability:
            return f"k={r['k']}: least stable cluster {r['stability_min']:.2f} < {min_stability:g}"
        if np.isfinite(r["split_p"]) and r["split_p"] > split_alpha:
            return (f"k={r['k']}: two neighbouring clusters no better separated than one group cut in two "
                    f"(p = {r['split_p']:.2f})")
        return f"k={r['k']}: less stable than the chosen candidate"

    finer = table[table["k"] > best["k"]].sort_values(["k", "stability_min"], ascending=[True, False])
    rejected = [(r["candidate"], reason(r)) for _, r in finer.drop_duplicates("k").head(4).iterrows()]
    ctx.decide(f"{what}: method and number of clusters", f"{best['method']}, k={best['k']}", why, rejected,
               {"candidates": table[["candidate", "k", "stability_min", "stability_mean", "split_p"]]
                .round(3).to_dict("records")})
    per_cluster = pd.DataFrame({"cluster": [f"C{i + 1}" for i in range(len(jac))],
                                "size": np.bincount(lab, minlength=len(jac)), "jaccard": jac})
    k = int(best["k"])
    return {"labels": lab, "k": k, "method": best["method"], "parameter": best["parameter"], "stable": stable,
            "separated": separated, "split_p": float(best["split_p"]), "stability": float(best["stability_min"]),
            "stability_mean": float(best["stability_mean"]), "k_at_edge": int(k in (k_min, k_max)), "table": table,
            "per_cluster": per_cluster}


class ClusterParams(ToolParams):
    methods: list[Method] = Field(default_factory=lambda: ["kmeans", "gmm", "leiden"])
    k_min: int = Field(2, ge=2, le=50)
    k_max: int = Field(12, ge=2, le=60)
    min_stability: float = Field(0.75, ge=0.5, le=0.95, description="Jaccard bar every cluster must reach")
    min_cluster_fraction: float = Field(0.005, ge=0.0, le=0.2)
    graph_neighbors: int = Field(15, ge=5, le=100, description="neighbours of Leiden's graph")
    split_alpha: float = Field(0.05, ge=0.001, le=0.2, description="neighbouring clusters must be better "
                                                                  "separated than one Gaussian at this level")


def cluster_result(ctx, res: dict, rows: np.ndarray, source: str, kind: str = "clusters",
                   groups: Artifact | None = None) -> StepResult:
    """Labels artifact, metrics and tables of a chosen clustering; with ``groups`` (e.g. patients),
    also how far the clusters are confined to single groups."""
    names = np.array([f"C{i + 1}" for i in range(res["k"])], dtype=object)
    art = Artifact("labels", names[res["labels"]], rows,
                   {"external": False, "source": f"{kind} from '{source}'", "method": res["method"],
                    "k": res["k"], "n_classes": res["k"], "stability": res["stability"]})
    metrics = {"method": res["method"], "k": res["k"], "stability": res["stability"],
               "stability_mean": res["stability_mean"], "stable": res["stable"], "separated": res["separated"],
               "separation_p": res["split_p"], "k_at_edge": res["k_at_edge"],
               "smallest_cluster": int(res["per_cluster"]["size"].min())}
    tables = {f"{kind}_candidates": res["table"], f"{kind}_stability": res["per_cluster"]}
    g = values_on(rows, groups)
    if g is not None:
        ok = g != ""
        col = groups.meta.get("column", groups.name)
        ct = pd.crosstab(pd.Series(names[res["labels"]][ok], name=kind), pd.Series(g[ok], name=col))
        present = (ct >= GROUP_PRESENT).sum(axis=1)
        metrics.update({"cramers_v_groups": cramers_v(names[res["labels"]][ok], g[ok]),
                        "groups_per_cluster_median": float(present.median()),
                        "clusters_in_one_group": int((present <= 1).sum())})
        tables[f"{kind}_by_{col}"] = ct.reset_index()
    return StepResult(artifacts={"labels": art}, metrics=metrics, tables=tables)


def _run_cluster(ctx, inputs, p: ClusterParams) -> StepResult:
    a = inputs["embedding"]
    Z = dense(a.data).astype(np.float64)
    lo, hi = min(p.k_min, p.k_max), max(p.k_min, p.k_max)
    res = choose_clustering(ctx, Z, p.methods, lo, hi, p.min_stability, p.min_cluster_fraction,
                            p.graph_neighbors, split_alpha=p.split_alpha)
    return cluster_result(ctx, res, a.rows, a.name, groups=inputs.get("groups"))


def _cluster_cost(n, p, prm, budget) -> float:
    ks = (prm.k_max - prm.k_min + 1) if prm is not None else 11
    per = {"kmeans": 2e-6, "gmm": 8e-6, "leiden": 4e-6}
    methods = prm.methods if prm is not None else list(per)
    m = min(n, budget.eval_max_samples)
    return float(sum(per[x] * (n + budget.bootstrap_reps * m) * ks * max(1, p // 10) for x in methods) + 2)


register(ToolSpec(
    name="cluster", origin="new", source=SOURCE,
    summary="Clusters samples (k-means, Gaussian mixtures, Leiden) and keeps the finest candidate whose every "
            "cluster is reproducible across subsamples (Jaccard >= 0.75).",
    consumes=(Port("embedding", ("embedding", "normalized"), doc="representation to cluster"),
              Port("groups", ("groups",), required=False, doc="groups whose overlap with the clusters is reported "
                                                              "(e.g. patients)")),
    produces=(Port("labels", ("labels",), doc="cluster of every sample (external = False)"),),
    params=ClusterParams, answers=("structure_discovery",), addresses=(),
    criteria=(Criterion("stability", "max", 0.75, "mean subsample Jaccard of the least stable cluster"),
              Criterion("k_at_edge", "min", 0, "1 when k is at the end of the searched range"),
              Criterion("k", "report", doc="number of clusters"),
              Criterion("cramers_v_groups", "report", doc="association of clusters with the groups (0 = spread "
                                                          "evenly, 1 = one group each)")),
    run=_run_cluster, cost=_cluster_cost,
))

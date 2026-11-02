"""The dimension-reduction sub-agent (Project 1) behind the common tool interface.

It makes two decisions by its own rules, logged with source "subagent":

  analysis dimension  how many principal components later steps (batch correction,
                      clustering) should use. Horn's parallel analysis: keep the leading
                      components whose variance exceeds the 95th percentile of the same
                      component's variance after every feature has been permuted
                      independently, which destroys correlation but keeps each feature's
                      distribution. Components below that line are what uncorrelated
                      noise of the same marginals would also produce. The 90%-of-variance
                      rule of Project 1's profiler is logged as the rejected alternative:
                      on noisy, high-dimensional data it keeps hundreds of noise components.
  2-D map             which method draws the map. Candidates span Project 1's families
                      (linear, neighbour embedding, graph diffusion); infeasible ones (n too
                      large, run time over budget) are skipped before fitting. Each fitted
                      map is scored with Project 1's metrics on a fixed subsample, and the
                      map with the best mean of trustworthiness and continuity at k = 10 is
                      chosen among those stable under a change of seed (k-NN overlap of two
                      fits >= 0.5). A map is for reading neighbourhoods, so local fidelity
                      decides; the global metrics are reported beside it.

New in Project 4: parallel analysis, the time-aware feasibility filter and the selection
rule (Project 1 compared methods but left the choice to the reader).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .evaluate import evaluate_embedding, knn_overlap
from .methods import run_method
from .registry import REGISTRY, estimated_seconds, feasibility

MAP_CANDIDATES = ("umap", "tsne", "pca", "diffusion_map", "laplacian_eigenmaps", "isomap")
MIN_STABILITY = 0.5
NULL_QUANTILE = 0.95


def _rows(n: int, cap: int, rng) -> np.ndarray:
    return np.arange(n) if n <= cap else np.sort(rng.choice(n, cap, replace=False))


def standardize(X: np.ndarray, clip: float = 10.0) -> np.ndarray:
    """Zero mean, unit variance per feature, clipped at +-clip (as in common practice for
    expression data, so a handful of extreme values cannot dominate a component)."""
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return np.clip((X - mu) / sd, -clip, clip).astype(np.float32)


def parallel_analysis(X: np.ndarray, max_k: int, n_perm: int, seed: int, cap: int = 5000) -> dict:
    """Number of leading components above the permutation null (Horn 1965)."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    Xs = np.asarray(X[_rows(len(X), cap, rng)], dtype=np.float64)
    Xs = Xs - Xs.mean(0)
    k_fit = int(min(max_k + 1, min(Xs.shape) - 1))
    if k_fit < 1:
        return {"k": 1, "observed": [], "null_95": [], "n_used": len(Xs)}
    observed = PCA(k_fit, svd_solver="randomized", random_state=seed).fit(Xs).explained_variance_
    null = []
    for b in range(max(1, n_perm)):
        order = np.argsort(rng.random(Xs.shape), axis=0)
        Xp = np.take_along_axis(Xs, order, axis=0)
        null.append(PCA(k_fit, svd_solver="randomized", random_state=seed + b + 1).fit(Xp).explained_variance_)
    thr = np.quantile(np.array(null), NULL_QUANTILE, axis=0)
    above = observed > thr
    k = int(np.argmin(above)) if not above.all() else k_fit
    return {"k": k, "observed": observed.tolist(), "null_95": thr.tolist(), "n_used": len(Xs),
            "n_permutations": max(1, n_perm)}


def pca_scores(X: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    k = int(min(k, min(X.shape) - 1))
    model = PCA(k, svd_solver="randomized", random_state=seed).fit(X)
    return model.transform(X).astype(np.float32), model.explained_variance_ratio_


def stability(method: str, X: np.ndarray, params: dict, seed: int, cap: int, Z=None) -> float:
    """k-NN overlap between two fits under different seeds (1.0 for deterministic methods).

    With more rows than ``cap`` both fits use the same random subsample of ``cap`` rows,
    which estimates seed sensitivity at a fraction of the cost of refitting everything.
    """
    if REGISTRY[method].deterministic:
        return 1.0
    rng = np.random.default_rng(seed + 17)
    if len(X) <= cap and Z is not None:
        Z1, Xs = Z, X
    else:
        idx = _rows(len(X), cap, rng)
        Xs = X[idx]
        Z1 = run_method(method, Xs, 2, seed, params)
    Z2 = run_method(method, Xs, 2, seed + 1, params)
    if not (np.isfinite(Z1).all() and np.isfinite(Z2).all()):
        return float("nan")
    return knn_overlap(Z1, Z2, k=min(10, len(Xs) - 2))


def map_candidates(n: int, p: int, budget_seconds: float, max_candidates: int,
                   requested: list[str] | None = None) -> tuple[list[str], list[tuple[str, str]]]:
    """Candidates in prior order, with the reason each skipped method was skipped."""
    feas = feasibility(n, p)
    order = list(requested) if requested else list(MAP_CANDIDATES)
    chosen, skipped = [], []
    share = budget_seconds / max(1, max_candidates)
    for m in order:
        if m not in REGISTRY:
            skipped.append((m, "not a known method"))
        elif feas[m]:
            skipped.append((m, feas[m]))
        elif estimated_seconds(m, n, p) * 2 > share:     # the fit plus a stability refit
            skipped.append((m, f"estimated {estimated_seconds(m, n, p):.0f}s per fit exceeds its share "
                               f"({share:.0f}s) of the step budget"))
        elif len(chosen) >= max_candidates:
            skipped.append((m, f"candidate cap ({max_candidates}) reached"))
        else:
            chosen.append(m)
    return chosen, skipped


def method_params(method: str, n: int, n_neighbors: int) -> dict:
    k = int(max(2, min(n_neighbors, n - 1)))
    return {"umap": {"n_neighbors": k, "min_dist": 0.3},
            "tsne": {"perplexity": float(min(30.0, max(5.0, (n - 1) / 3)))},
            "diffusion_map": {"n_neighbors": k, "alpha": 1.0},
            "laplacian_eigenmaps": {"n_neighbors": k},
            "isomap": {"n_neighbors": k}}.get(method, {})


def fit_and_score(method: str, X: np.ndarray, params: dict, seed: int, eval_cap: int) -> dict:
    """Fit one map and score it. Library warnings raised meanwhile (e.g. a disconnected
    neighbour graph) are returned with the result; UMAP's notice that a fixed seed disables
    parallelism is dropped, since the seed is deliberate."""
    import time
    import warnings

    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=".*overridden to 1 by setting random_state")
        Z = run_method(method, X, 2, seed, params)
        seconds = time.perf_counter() - t0
        if not np.isfinite(Z).all():
            return {"method": method, "Z": None, "error": "non-finite embedding", "seconds": seconds}
        m = evaluate_embedding(X, Z, None, ks=(10,), random_state=seed)
        m["stability"] = stability(method, X, params, seed, eval_cap, Z)
    notes = list(dict.fromkeys(str(w.message)[:160] for w in caught))
    return {"method": method, "Z": Z, "metrics": m, "seconds": seconds, "warnings": notes}


def choose_map(results: list[dict]) -> tuple[dict | None, str]:
    """The fitted map with the best mean of trustworthiness and continuity at k = 10,
    among maps whose seed stability is >= MIN_STABILITY (all maps, if none is)."""
    ok = [r for r in results if r.get("Z") is not None]
    if not ok:
        return None, "no candidate produced a valid map"

    def local(r):
        m = r["metrics"]
        vals = [m.get("trustworthiness@10"), m.get("continuity@10")]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else -np.inf

    stable = [r for r in ok if np.isfinite(r["metrics"].get("stability", np.nan))
              and r["metrics"]["stability"] >= MIN_STABILITY]
    pool = stable or ok
    best = max(pool, key=lambda r: (local(r), REGISTRY[r["method"]].deterministic,
                                    r["metrics"].get("distance_spearman") or -1))
    why = (f"highest mean of trustworthiness and continuity at k=10 ({local(best):.3f})"
           + (f" among maps with seed stability >= {MIN_STABILITY}" if stable else
              f"; no map reached seed stability {MIN_STABILITY}, so all were eligible"))
    return best, why


def as_dense(X) -> np.ndarray:
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)

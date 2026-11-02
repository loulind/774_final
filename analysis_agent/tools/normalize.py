"""Normalisation of count matrices, batch-aware variable features, and count splitting.

Normalisation. Three transforms, one chosen from the data:

    log                 counts scaled to the median library size, then log1p
    pearson_residuals   analytic Pearson residuals of a negative binomial with a common
                        overdispersion (Lause, Berens & Kobak 2021), clipped at sqrt(n)
    tfidf               term frequency x inverse document frequency, then log1p, for
                        binary matrices (presence / absence of peaks or terms)

A binary matrix gets TF-IDF. Otherwise both count transforms are computed and judged by
how strongly the first principal component still tracks sequencing depth (|Spearman rho|
with log library size, a label-free check of the technical variation normalisation is
meant to remove); Pearson residuals are kept only if they lower it by more than
``prefer_simple_margin``, because log values give interpretable fold changes.

Variable features are chosen within batches: residual variances are computed inside each
batch (so a feature that differs only between batches does not look variable), features
are ranked in every batch, and those variable in the most batches win (ties broken by the
median rank), as in scanpy's batch-aware selection.

Count splitting (Neufeld et al. 2023) divides every count into two independent halves, one
to discover structure (clusters, domains) and one to test it (markers), so the test does not
re-use the data that generated the hypothesis. Poisson counts are split binomially. For
overdispersed counts binomial thinning leaves the halves correlated, so negative-binomial
thinning is used instead, with each feature's overdispersion estimated by moments.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pydantic import Field
from scipy import stats

from .base import Artifact, Criterion, Port, StepResult, ToolParams, ToolSpec, check_time, values_on
from .registry import register

SOURCE = "new in Project 4"
CHUNK = 1000                              # features per dense block


def library_sizes(X: sp.csr_matrix) -> np.ndarray:
    return np.asarray(X.sum(1)).ravel().astype(float)


def log_normalise(X: sp.csr_matrix, lib: np.ndarray, target: float | None = None) -> np.ndarray:
    """log1p of counts scaled to ``target`` (default: the median library size), dense float32."""
    target = float(np.median(lib[lib > 0])) if target is None else target
    scale = np.divide(target, lib, out=np.zeros_like(lib), where=lib > 0)
    Y = sp.diags(scale) @ X
    return np.log1p(np.asarray(Y.toarray() if sp.issparse(Y) else Y, dtype=np.float32))


def pearson_residuals(X: sp.csr_matrix, theta: float, lib: np.ndarray | None = None) -> np.ndarray:
    """Analytic Pearson residuals (x - mu) / sqrt(mu + mu^2 / theta), mu = library size x the
    feature's share of all counts, clipped at +-sqrt(n); dense float32. ``lib`` (library sizes
    over all features) lets a block of features be transformed exactly as in the full matrix."""
    lib = library_sizes(X) if lib is None else lib
    colsum = np.asarray(X.sum(0)).ravel().astype(float)
    mu = np.outer(lib, colsum / max(float(lib.sum()), 1.0))
    D = np.asarray(X.toarray(), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(mu > 0, (D - mu) / np.sqrt(mu + mu ** 2 / theta), 0.0)
    clip = np.sqrt(X.shape[0])
    return np.clip(R, -clip, clip).astype(np.float32)


def residual_variance(X: sp.csr_matrix, theta: float, check=None) -> np.ndarray:
    """Variance of every feature's Pearson residuals, computed in feature blocks."""
    lib = library_sizes(X)
    out = np.zeros(X.shape[1])
    for s in range(0, X.shape[1], CHUNK):
        if check is not None:
            check()
        out[s:s + CHUNK] = pearson_residuals(X[:, s:s + CHUNK], theta, lib).var(0)
    return out


def tfidf(X: sp.csr_matrix) -> np.ndarray:
    """log1p(TF x IDF x 1e4) with TF = x / row total and IDF = n / document frequency."""
    lib = library_sizes(X)
    dfreq = np.asarray((X > 0).sum(0)).ravel().astype(float)
    idf = X.shape[0] / np.maximum(dfreq, 1)
    tf = sp.diags(np.divide(1.0, lib, out=np.zeros_like(lib), where=lib > 0)) @ X
    return np.log1p(np.asarray((tf @ sp.diags(idf)).toarray(), dtype=np.float32) * 1e4)


def pc1_depth(M: np.ndarray, log_depth: np.ndarray, standardise: bool, seed: int) -> tuple[float, float]:
    """|Spearman rho| of PC1 with log depth, and R^2 of log depth on the first 10 PCs."""
    from sklearn.decomposition import PCA

    A = M - M.mean(0)
    if standardise:
        sd = A.std(0)
        A = A / np.where(sd > 0, sd, 1)
    k = int(min(10, min(A.shape) - 1))
    P = PCA(k, svd_solver="randomized", random_state=seed).fit_transform(A)
    rho = abs(float(stats.spearmanr(P[:, 0], log_depth)[0]))
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(P)), P]), log_depth, rcond=None)
    fit = np.column_stack([np.ones(len(P)), P]) @ beta
    r2 = 1 - float(((log_depth - fit) ** 2).sum() / max(1e-12, ((log_depth - log_depth.mean()) ** 2).sum()))
    return rho, r2


def variable_features(X: sp.csr_matrix, batch: np.ndarray | None, n_features: int, theta: float,
                      check=None) -> tuple[np.ndarray, pd.DataFrame]:
    """Indices of the features variable in the most batches (ties: lower median rank), from
    Pearson residual variances computed inside each batch."""
    levels = ["all"] if batch is None else list(pd.unique(batch))
    ranks, rv = [], []
    for lv in levels:
        idx = np.arange(X.shape[0]) if batch is None else np.flatnonzero(batch == lv)
        v = residual_variance(X[idx], theta, check)
        rv.append(v)
        ranks.append(stats.rankdata(-v, method="min"))
    ranks, rv = np.vstack(ranks), np.vstack(rv)
    n_hvg = (ranks <= n_features).sum(0)
    med = np.median(ranks, 0)
    order = np.lexsort((med, -n_hvg))
    keep = np.sort(order[: min(n_features, X.shape[1])])
    table = pd.DataFrame({"feature": np.arange(X.shape[1]), "n_batches_variable": n_hvg,
                          "median_rank": med, "residual_variance_mean": rv.mean(0)})
    return keep, table


class NormalizeParams(ToolParams):
    method: Literal["auto", "log", "pearson_residuals", "tfidf"] = Field(
        "auto", description="auto = TF-IDF for binary matrices, else log vs Pearson residuals by depth correlation")
    n_features: int = Field(2000, ge=200, le=10_000, description="variable features kept")
    theta: float = Field(100.0, ge=1.0, le=10_000.0, description="overdispersion of the Pearson residuals")
    min_batch_samples: int = Field(50, ge=10, le=5000, description="smaller batches are pooled for feature "
                                                                   "selection")
    prefer_simple_margin: float = Field(0.1, ge=0.0, le=1.0, description="Pearson residuals must lower the "
                                                                          "PC1-depth correlation by more")


def _run_normalize(ctx, inputs, p: NormalizeParams) -> StepResult:
    a, batch_art = inputs["counts"], inputs.get("batch")
    rows = a.rows
    X = sp.csr_matrix(a.data).astype(np.float64)
    lib = library_sizes(X)
    if (lib <= 0).any():
        raise RuntimeError(f"{int((lib <= 0).sum())} samples have no counts; run count QC first")
    log_depth = np.log(lib)
    names = list(a.meta.get("feature_names") or [f"f{j}" for j in range(X.shape[1])])
    binary = bool(X.nnz and np.all(X.data == 1))
    batch = values_on(rows, batch_art, "(none)")
    if batch is not None:
        sizes = pd.Series(batch).value_counts()
        small = list(sizes[sizes < p.min_batch_samples].index)
        if small:
            batch = np.where(np.isin(batch, small), "(pooled small batches)", batch)
        if pd.Series(batch).nunique() < 2:
            batch = None
    check = lambda: check_time(ctx, "normalisation")  # noqa: E731

    if binary and p.method in ("auto", "tfidf") or p.method == "tfidf":
        dfreq = np.asarray((X > 0).sum(0)).ravel()
        keep = np.sort(np.argsort(-dfreq)[: p.n_features])
        sel_table = pd.DataFrame({"feature": np.arange(X.shape[1]), "document_frequency": dfreq})
        ctx.decide("variable features", f"{len(keep):,} most frequently detected",
                   "For presence/absence data, features present in more samples carry more information "
                   "(as in TF-IDF / latent semantic indexing pipelines).", [], {"n_features": len(keep)})
    else:
        keep, sel_table = variable_features(X, batch, p.n_features, p.theta, check)
        col = batch_art.meta.get("column", batch_art.name) if batch is not None else None
        ctx.decide("variable features", f"{len(keep):,} by Pearson residual variance"
                                        + (f" within each {col}" if col else ""),
                   (f"Residual variances are computed inside each {col} level and features variable in the most "
                    "levels are kept, so features that differ only between levels are not selected as "
                    "biological variation.") if col else
                   "Residual variance ranks features by variability beyond what their mean predicts.",
                   [("highest log-variance", "favours highly expressed features regardless of noise")] +
                   ([("pooled selection", "between-level differences would be picked as variable features")]
                    if col else []),
                   {"n_batches": int(pd.Series(batch).nunique()) if batch is not None else 1})
    sel_table["feature"] = [names[j] for j in sel_table["feature"]]
    sel_table["selected"] = np.isin(np.arange(X.shape[1]), keep)
    Xs = X[:, keep]
    raw_log = np.log1p(np.asarray(Xs.toarray(), dtype=np.float32))
    before = pc1_depth(raw_log, log_depth, True, ctx.seed)

    cands: dict[str, tuple[np.ndarray, bool]] = {}
    if binary and p.method == "auto" or p.method == "tfidf":
        cands["tfidf"] = (tfidf(X)[:, keep], False)
    else:
        if p.method in ("auto", "log"):
            cands["log"] = (log_normalise(Xs, lib), False)
        if p.method in ("auto", "pearson_residuals"):
            cands["pearson_residuals"] = (pearson_residuals(Xs, p.theta, lib), True)
    depth = {m: pc1_depth(M, log_depth, not scaled, ctx.seed) for m, (M, scaled) in cands.items()}
    if len(cands) == 1:
        method = next(iter(cands))
        why = ("Binary matrix: TF-IDF weights presence by rarity." if method == "tfidf" else "Set by the plan.")
        rejected = []
    else:
        gain = depth["log"][0] - depth["pearson_residuals"][0]
        method = "pearson_residuals" if gain > p.prefer_simple_margin else "log"
        why = (f"|rho(PC1, log depth)| is {depth['log'][0]:.2f} after log normalisation and "
               f"{depth['pearson_residuals'][0]:.2f} with Pearson residuals; "
               + (f"residuals remove more depth signal (by {gain:.2f} > {p.prefer_simple_margin:g})."
                  if method == "pearson_residuals" else
                  f"the difference ({gain:+.2f}) is not above {p.prefer_simple_margin:g}, so the simpler log "
                  "values (interpretable fold changes) are kept."))
        other = "log" if method == "pearson_residuals" else "pearson_residuals"
        rejected = [(other, f"|rho(PC1, log depth)| = {depth[other][0]:.2f}")]
    ctx.decide("normalisation", method, why, rejected,
               {"pc1_depth_rho_before": before[0], **{f"pc1_depth_rho_{m}": d[0] for m, d in depth.items()}})
    M, scaled = cands[method]
    metrics = {"method": method, "n_features": int(len(keep)), "binary_input": binary,
               "depth_corr_pc1_before": before[0], "depth_corr_pc1_after": depth[method][0],
               "depth_r2_top10_before": before[1], "depth_r2_top10_after": depth[method][1],
               **{f"depth_corr_pc1_{m}": d[0] for m, d in depth.items()},
               "batch_aware_selection": batch is not None}
    art = Artifact("normalized", M, rows, {"feature_names": [names[j] for j in keep], "method": method,
                                           "already_scaled": scaled, "source": a.name,
                                           "theta": p.theta if method == "pearson_residuals" else None})
    return StepResult(artifacts={"normalized": art}, metrics=metrics,
                      tables={"variable_features": sel_table.sort_values("selected", ascending=False)})


def _normalize_cost(n, p, prm, budget) -> float:
    return 4e-8 * n * p * 3 + 2.0


register(ToolSpec(
    name="normalize_counts", origin="new", source=SOURCE,
    summary="Normalises counts (log, analytic Pearson residuals or TF-IDF, chosen by how much depth signal is "
            "left in PC1) and selects variable features within batches.",
    consumes=(Port("counts", ("counts",)),
              Port("batch", ("groups",), required=False, doc="batches for batch-aware feature selection")),
    produces=(Port("normalized", ("normalized",), doc="normalised values of the selected features"),),
    params=NormalizeParams, answers=("representation_learning",),
    addresses=("depth_variation", "high_dimensionality"),
    criteria=(Criterion("depth_corr_pc1_after", "min", 0.5, "|Spearman rho| of PC1 with log library size "
                                                             "after normalisation"),
              Criterion("depth_corr_pc1_before", "report", doc="the same for log1p of raw counts")),
    run=_run_normalize, cost=_normalize_cost,
))


# ---------------------------------------------------------------------------
# count splitting
# ---------------------------------------------------------------------------

def overdispersion(X: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Method-of-moments negative-binomial overdispersion phi (Var = mu + phi mu^2) of every
    feature, with mu = size factor x feature rate; returns (phi, mean count)."""
    lib = library_sizes(X)
    s = lib / lib.mean()
    n = X.shape[0]
    colsum = np.asarray(X.sum(0)).ravel().astype(float)
    lam = colsum / s.sum()
    sx2 = np.asarray(X.multiply(X).sum(0)).ravel()
    sxs = np.asarray(X.T @ s).ravel()
    ss2 = float((s ** 2).sum())
    num = sx2 - 2 * lam * sxs + lam ** 2 * ss2 - lam * s.sum()
    den = lam ** 2 * ss2
    phi = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return np.maximum(phi, 0.0), lam * s.sum() / n


def binomial_dependence(phi: np.ndarray, mean: np.ndarray, eps: float) -> np.ndarray:
    """Correlation between the two halves of binomially thinned NB(mean, phi) counts."""
    m = mean
    cov = eps * (1 - eps) * phi * m ** 2
    v1 = eps * m + eps ** 2 * phi * m ** 2
    v2 = (1 - eps) * m + (1 - eps) ** 2 * phi * m ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((v1 > 0) & (v2 > 0), cov / np.sqrt(v1 * v2), 0.0)


def thin(X: sp.csr_matrix, eps: float, phi: np.ndarray | None, rng) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Split counts into two halves: binomial thinning (phi None or 0 for a feature) or
    negative-binomial thinning, x1 | x ~ BetaBinomial(x, eps / phi, (1 - eps) / phi)."""
    X = sp.csr_matrix(X)
    x = X.data.astype(np.int64)
    if phi is None:
        prob = np.full(len(x), eps)
    else:
        b = np.where(phi[X.indices] > 1e-8, 1.0 / np.maximum(phi[X.indices], 1e-6), np.inf)
        b = np.minimum(b, 1e6)
        nb = np.isfinite(b) & (b < 1e6)
        prob = np.full(len(x), eps)
        prob[nb] = rng.beta(eps * b[nb], (1 - eps) * b[nb])
    x1 = rng.binomial(x, prob)
    # own index arrays: eliminate_zeros() works in place and would corrupt shared ones (and X)
    A = sp.csr_matrix((x1, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    B = sp.csr_matrix((x - x1, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    A.eliminate_zeros()
    B.eliminate_zeros()
    return A, B


class CountSplitParams(ToolParams):
    epsilon: float = Field(0.5, ge=0.1, le=0.9, description="expected share of each count in the 'fit' half")
    method: Literal["auto", "poisson", "negative_binomial"] = Field(
        "auto", description="auto = negative binomial when binomial thinning would leave the halves correlated")
    max_dependence: float = Field(0.05, ge=0.0, le=0.5, description="median correlation between binomially "
                                                                    "thinned halves tolerated")
    min_detected_fraction: float = Field(0.05, ge=0.0, le=0.5, description="features judged for overdispersion")


def _run_count_split(ctx, inputs, p: CountSplitParams) -> StepResult:
    a = inputs["counts"]
    X = sp.csr_matrix(a.data)
    phi, mean = overdispersion(X.astype(np.float64))
    det = np.asarray((X > 0).mean(0)).ravel() >= p.min_detected_fraction
    dep = binomial_dependence(phi, mean, p.epsilon)
    med = float(np.median(dep[det])) if det.any() else 0.0
    if p.method == "auto":
        method = "negative_binomial" if med > p.max_dependence else "poisson"
        why = (f"Binomial (Poisson) thinning of counts with the estimated overdispersion would leave a median "
               f"correlation of {med:.3f} between the halves of the {int(det.sum()):,} features detected in at "
               f"least {p.min_detected_fraction:.0%} of samples"
               + (f", above {p.max_dependence:g}; negative-binomial thinning makes them independent."
                  if method == "negative_binomial" else f", within {p.max_dependence:g}."))
    else:
        method, why = p.method, "Set by the plan."
    ctx.decide("count splitting", f"{method} thinning, epsilon {p.epsilon:g}", why,
               [("poisson" if method == "negative_binomial" else "negative_binomial",
                 "halves correlated" if method == "negative_binomial" else "not needed: counts near Poisson"),
                ("no splitting", "markers would be tested on the data that defined the groups (double dipping)")],
               {"median_dependence_binomial": med, "median_phi": float(np.median(phi[det])) if det.any() else None})
    A, B = thin(X, p.epsilon, phi if method == "negative_binomial" else None, ctx.rng(11))
    meta = {k: v for k, v in a.meta.items() if k != "qc"}
    arts = {"fit": Artifact("counts", A, a.rows, {**meta, "split_half": "fit", "split_of": a.name,
                                                   "epsilon": p.epsilon, "thinning": method}),
            "test": Artifact("counts", B, a.rows, {**meta, "split_half": "test", "split_of": a.name,
                                                    "epsilon": 1 - p.epsilon, "thinning": method})}
    metrics = {"method": method, "epsilon": p.epsilon, "median_dependence_binomial": med,
               "median_overdispersion": float(np.median(phi[det])) if det.any() else None,
               "fit_count_share": float(A.sum() / max(1, X.sum()))}
    return StepResult(artifacts=arts, metrics=metrics)


register(ToolSpec(
    name="count_split", origin="new", source=SOURCE,
    summary="Splits every count into two independent halves (Poisson or negative-binomial thinning): one to find "
            "groups, one to test them.",
    consumes=(Port("counts", ("counts",)),),
    produces=(Port("fit", ("counts",), doc="half used to discover structure"),
              Port("test", ("counts",), doc="half used to test it")),
    params=CountSplitParams, answers=("differential_analysis",), addresses=("double_dipping",),
    criteria=(Criterion("median_dependence_binomial", "report", doc="correlation binomial thinning would leave"),
              Criterion("fit_count_share", "report", doc="share of all counts in the fit half")),
    run=_run_count_split, cost=lambda n, p, prm, b: 3e-8 * n * p + 1.0,
))

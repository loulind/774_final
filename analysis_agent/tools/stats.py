"""Statistics shared by the general tools. Each function has a known-answer test.

    bh_adjust                  Benjamini-Hochberg adjusted p-values
    delong, auc_inference      AUROC with its DeLong variance (Sun & Xu's midrank algorithm),
                               interval and one-sided test against AUROC 0.5
    spatial_knn                k-nearest-neighbour graph inside each library, never across
    moran_libraries, morans_i  Moran's I per feature, centred within each library, with the
                               randomisation variance of Cliff & Ord, pooled over libraries
    fit_f_dist, moderated_t    limma's empirical-Bayes variance shrinkage (Smyth 2004)
    column_ranks, rank_sum     Wilcoxon rank-sum tests of one group against the rest for many
                               features at once, with tie and continuity corrections
    cohen_kappa, cramers_v     agreement and association between two labelings
    variance_share (from Project 3's profiler), nested_shares, mixing_entropy,
    knn_preservation, silhouette     batch measurements
    mad_outliers               the median-absolute-deviation outlier rule
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.special import digamma, polygamma

# ---------------------------------------------------------------------------
# multiple testing
# ---------------------------------------------------------------------------


def bh_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; NaN entries stay NaN and are not counted."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv, kind="mergesort")
    scaled = pv[order] * m / np.arange(1, m + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(scaled, 1.0)
    q[ok] = out
    return q


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------


def delong(y, score) -> tuple[float, float]:
    """AUROC and its DeLong variance, from midranks (Sun & Xu 2014). Ties count one half."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, float)
    pos, neg = s[y], s[~y]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    tz = stats.rankdata(np.concatenate([pos, neg]))
    v10 = (tz[:m] - stats.rankdata(pos)) / n             # per positive: share of negatives below it
    v01 = 1.0 - (tz[m:] - stats.rankdata(neg)) / m       # per negative: share of positives above it
    auc = float(v10.mean())
    if m < 2 or n < 2:
        return auc, float("nan")
    return auc, float(v10.var(ddof=1) / m + v01.var(ddof=1) / n)


def stratified_delong(y, score, strata) -> tuple[float, float]:
    """AUROC from positive-negative pairs within each stratum only, e.g. the outer folds of a
    cross-validation, whose models differ in calibration: per-stratum AUROCs weighted by their
    number of pairs, with the variance of that weighted mean from each stratum's DeLong variance."""
    y, s, strata = np.asarray(y).astype(bool), np.asarray(score, float), np.asarray(strata)
    total = weight = var = 0.0
    for k in np.unique(strata):
        sel = strata == k
        m, n = int(y[sel].sum()), int((~y[sel]).sum())
        if m == 0 or n == 0:
            continue
        auc_k, var_k = delong(y[sel], s[sel])
        total, weight, var = total + m * n * auc_k, weight + m * n, var + (m * n) ** 2 * var_k
    if weight == 0:
        return float("nan"), float("nan")
    return float(total / weight), float(var / weight ** 2)


def auc_inference(auc: float, var: float, level: float = 0.95) -> dict:
    """DeLong interval (clipped to [0, 1]) and the one-sided p-value of AUROC > 0.5."""
    if not np.isfinite(auc) or not np.isfinite(var) or var < 0:
        return {"auroc": auc if np.isfinite(auc) else None, "se": None, "ci_low": None,
                "ci_high": None, "p_vs_half": None}
    se = float(np.sqrt(var))
    z = float(stats.norm.ppf(0.5 + level / 2))
    if se > 0:
        p = float(stats.norm.sf((auc - 0.5) / se))
    else:                                                # complete separation (or none at all)
        p = 0.0 if auc > 0.5 else 1.0
    return {"auroc": float(auc), "se": se, "ci_low": max(0.0, auc - z * se),
            "ci_high": min(1.0, auc + z * se), "p_vs_half": p}


def fast_auroc(Y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """AUROC of every column of scores S (n x L) for binary targets Y (n x L), by rank sums.
    Columns without both classes give NaN."""
    Y = np.asarray(Y, bool)
    R = stats.rankdata(S, axis=0)
    m = Y.sum(0).astype(float)
    n = Y.shape[0] - m
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = ((R * Y).sum(0) - m * (m + 1) / 2) / (m * n)
    auc[(m == 0) | (n == 0)] = np.nan
    return auc


# ---------------------------------------------------------------------------
# spatial neighbours and Moran's I
# ---------------------------------------------------------------------------


def spatial_knn(coords, libraries, k: int = 6, max_distance_factor: float | None = None) -> sp.csr_matrix:
    """Directed binary k-nearest-neighbour adjacency within each library (coordinates are
    only comparable inside one library, so there are no edges across libraries).

    With ``max_distance_factor``, neighbours farther than that multiple of the library's
    median nearest-neighbour distance are dropped, so samples at tissue edges or beside gaps
    keep only their true neighbours instead of reaching across the gap.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, float)
    libraries = np.asarray(libraries).astype(str)
    n = len(coords)
    r_all, c_all = [], []
    for lib in np.unique(libraries):
        idx = np.flatnonzero(libraries == lib)
        m = len(idx)
        if m < 2:
            continue
        kk = int(min(k, m - 1))
        dist, ind = NearestNeighbors(n_neighbors=kk + 1).fit(coords[idx]).kneighbors(coords[idx])
        self_hit = ind == np.arange(m)[:, None]
        self_hit[~self_hit.any(1), -1] = True            # duplicate coordinates: drop the farthest
        dist, ind = dist[~self_hit].reshape(m, kk), ind[~self_hit].reshape(m, kk)
        keep = np.ones_like(dist, bool)
        if max_distance_factor is not None:
            keep = dist <= max_distance_factor * float(np.median(dist[:, 0])) + 1e-9
        src = np.repeat(np.arange(m), kk).reshape(m, kk)
        r_all.append(idx[src[keep]])
        c_all.append(idx[ind[keep]])
    r = np.concatenate(r_all) if r_all else np.zeros(0, int)
    c = np.concatenate(c_all) if c_all else np.zeros(0, int)
    return sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))


def row_standardise(A) -> sp.csr_matrix:
    d = np.asarray(A.sum(1)).ravel()
    inv = np.divide(1.0, d, out=np.zeros(len(d)), where=d > 0)
    return sp.csr_matrix(sp.diags(inv) @ A)


def moran_libraries(W, libraries, min_samples: int = 8) -> list[dict]:
    """Per-library pieces of Moran's I that depend only on the weights (S0, S1, S2)."""
    W = sp.csr_matrix(W)
    libraries = np.asarray(libraries).astype(str)
    out = []
    for lib in np.unique(libraries):
        idx = np.flatnonzero(libraries == lib)
        n = len(idx)
        if n < max(min_samples, 4):
            continue
        Wl = W[idx][:, idx]
        S0 = float(Wl.sum())
        if S0 <= 0:
            continue
        Ws = Wl + Wl.T
        S1 = 0.5 * float(Ws.multiply(Ws).sum())
        S2 = float(((np.asarray(Wl.sum(1)).ravel() + np.asarray(Wl.sum(0)).ravel()) ** 2).sum())
        out.append({"library": lib, "idx": idx, "W": Wl, "n": n, "S0": S0, "S1": S1, "S2": S2})
    return out


def morans_i(X, libs: list[dict]) -> dict[str, np.ndarray]:
    """Moran's I of every column of X (samples x features, dense or sparse), each library
    centred on its own mean, so a feature that differs only between libraries is not called
    spatially structured.

    Per library: I, E[I] = -1/(n-1) and the randomisation variance (Cliff & Ord 1981, as in
    PySAL). Pooled: I averaged with library sizes as weights; z = sum of (I - E[I]) over the
    square root of the summed variances (independent libraries); p one-sided (positive
    autocorrelation). Libraries where a feature is constant are skipped for that feature.
    """
    p = X.shape[1]
    dev, var, i_sum, w_sum = np.zeros(p), np.zeros(p), np.zeros(p), np.zeros(p)
    n_lib, n_pos = np.zeros(p, int), np.zeros(p, int)
    for L in libs:
        n, S0, S1, S2 = L["n"], L["S0"], L["S1"], L["S2"]
        Z = X[L["idx"]]
        Z = np.asarray(Z.toarray() if sp.issparse(Z) else Z, dtype=np.float64)
        Z = Z - Z.mean(0)
        m2 = (Z ** 2).sum(0)
        ok = m2 > 1e-12 * max(1.0, float(np.max(m2, initial=0)))
        if not ok.any():
            continue
        Zo, m2o = Z[:, ok], m2[ok]
        I = n / S0 * (Zo * (L["W"] @ Zo)).sum(0) / m2o
        b2 = n * (Zo ** 4).sum(0) / m2o ** 2
        EI = -1.0 / (n - 1)
        A = n * ((n * n - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
        B = b2 * ((n * n - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)
        C = (n - 1) * (n - 2) * (n - 3) * S0 ** 2
        V = (A - B) / C - EI ** 2
        good = V > 0
        cols = np.flatnonzero(ok)[good]
        dev[cols] += I[good] - EI
        var[cols] += V[good]
        i_sum[cols] += n * I[good]
        w_sum[cols] += n
        n_lib[cols] += 1
        n_pos[cols] += I[good] > EI
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(var > 0, dev / np.sqrt(var), np.nan)
        I_pooled = np.where(w_sum > 0, i_sum / w_sum, np.nan)
    pval = np.where(np.isfinite(z), stats.norm.sf(z), np.nan)
    return {"I": I_pooled, "z": z, "p": pval, "n_libraries": n_lib, "n_positive": n_pos}


# ---------------------------------------------------------------------------
# moderated t (limma)
# ---------------------------------------------------------------------------


def trigamma_inverse(x):
    """Solve trigamma(y) = x by Newton's method (limma's trigammaInverse)."""
    x = np.atleast_1d(np.asarray(x, float))
    y = np.empty_like(x)
    big, small = x > 1e7, x < 1e-6
    y[big] = 1 / np.sqrt(x[big])
    y[small] = 1 / x[small]
    mid = ~(big | small)
    if mid.any():
        xm = x[mid]
        ym = 0.5 + 1 / xm
        for _ in range(50):
            tri = polygamma(1, ym)
            dif = tri * (1 - tri / xm) / polygamma(2, ym)
            ym = ym + dif
            if np.max(-dif / ym) < 1e-8:
                break
        y[mid] = ym
    return y


def fit_f_dist(s2, df: float) -> tuple[float, float]:
    """Prior degrees of freedom d0 and prior variance s0^2 for sample variances that all
    have ``df`` degrees of freedom (limma's fitFDist without covariates): the moments of
    the log variances are matched to those of a scaled F distribution. Returns d0 = inf
    when the variances are no more spread out than sampling alone explains, and d0 = 0
    (no shrinkage) with fewer than three usable variances."""
    s2 = np.asarray(s2, float)
    x = s2[np.isfinite(s2) & (s2 >= 0)]
    if len(x) < 3 or df <= 0:
        return 0.0, float("nan")
    med = float(np.median(x))
    x = np.maximum(x, 1e-5 * (med if med > 0 else 1.0))
    e = np.log(x) - digamma(df / 2) + np.log(df / 2)
    emean = float(e.mean())
    evar = float(e.var(ddof=1) - polygamma(1, df / 2))
    if evar > 0:
        d0 = float(2 * trigamma_inverse(evar)[0])
        return d0, float(np.exp(emean + digamma(d0 / 2) - np.log(d0 / 2)))
    return float("inf"), float(np.exp(emean))


def moderated_t(mean, s2, n: int, d0: float, s02: float) -> dict[str, np.ndarray]:
    """Moderated one-sample t-test of ``mean`` (e.g. mean paired differences of n replicates),
    each feature's variance shrunk towards the prior: s~^2 = (d0 s0^2 + d s^2) / (d0 + d)
    with d = n - 1; the t statistic has d0 + d degrees of freedom."""
    mean, s2 = np.asarray(mean, float), np.asarray(s2, float)
    d = n - 1
    if np.isinf(d0):
        post, dof = np.full_like(s2, s02), np.inf
    elif d0 > 0:
        post, dof = (d0 * s02 + d * s2) / (d0 + d), d0 + d
    else:
        post, dof = s2, float(d)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = mean / np.sqrt(post / n)
    p = 2 * (stats.norm.sf(np.abs(t)) if np.isinf(dof) else stats.t.sf(np.abs(t), dof))
    return {"t": t, "p": p, "df": dof, "s2_post": post}


# ---------------------------------------------------------------------------
# rank-sum tests
# ---------------------------------------------------------------------------


def column_ranks(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average ranks of every column, and each column's tie term sum(t^3 - t) over tie groups
    (computed as sum of t_i^2 over samples, t_i the size of sample i's tie group, minus n)."""
    R = stats.rankdata(X, axis=0)
    t = stats.rankdata(X, method="max", axis=0) - stats.rankdata(X, method="min", axis=0) + 1
    return R, (t ** 2).sum(0) - X.shape[0]


def rank_sum(R: np.ndarray, tie: np.ndarray, mask) -> dict[str, np.ndarray]:
    """Two-sided Wilcoxon rank-sum test of samples in ``mask`` against the rest, per column,
    with the normal approximation, tie correction and continuity correction (as scipy's
    asymptotic Mann-Whitney test); also U / (n1 n2), the probability that a sample of the
    group ranks above one of the rest."""
    mask = np.asarray(mask, bool)
    n = R.shape[0]
    n1 = int(mask.sum())
    n2 = n - n1
    U = R[mask].sum(0) - n1 * (n1 + 1) / 2
    d = U - n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(sigma > 0, (d - 0.5 * np.sign(d)) / sigma, 0.0)
    return {"z": z, "p": np.minimum(1.0, 2 * stats.norm.sf(np.abs(z))), "auc": U / max(1, n1 * n2)}


# ---------------------------------------------------------------------------
# agreement and association between labelings
# ---------------------------------------------------------------------------


def cohen_kappa(a, b) -> float:
    """Cohen's kappa between two labelings of the same samples (NaN when chance agreement is 1)."""
    a, b = np.asarray(a).astype(str), np.asarray(b).astype(str)
    if len(a) == 0:
        return float("nan")
    levels = np.union1d(a, b)
    M = np.zeros((len(levels), len(levels)))
    np.add.at(M, (np.searchsorted(levels, a), np.searchsorted(levels, b)), 1)
    n = M.sum()
    po = np.trace(M) / n
    pe = float(M.sum(1) @ M.sum(0)) / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def cramers_v(a, b) -> float | None:
    """Bias-corrected Cramer's V (Bergsma 2013) between two categorical variables."""
    table = pd.crosstab(pd.Series(np.asarray(a).astype(str)), pd.Series(np.asarray(b).astype(str))).to_numpy()
    n = table.sum()
    r, k = table.shape
    if r < 2 or k < 2 or n < 3:
        return None
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    den = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / den)) if den > 0 else None


# ---------------------------------------------------------------------------
# batch measurements
# ---------------------------------------------------------------------------


def variance_share(scores: np.ndarray, groups: np.ndarray) -> float | None:
    """Share of the variance of ``scores`` (n x k) explained by a categorical grouping,
    summed over columns (a variance-weighted R^2). From Project 3's profiler, unchanged."""
    keep = pd.notna(groups)
    if keep.sum() < 10 or pd.Series(groups[keep]).nunique() < 2:
        return None
    S, g = scores[keep], pd.Series(groups[keep]).astype(str).to_numpy()
    S = S - S.mean(0)
    total = float((S ** 2).sum())
    between = 0.0
    for level in np.unique(g):
        rows = S[g == level]
        between += len(rows) * float((rows.mean(0) ** 2).sum())
    return between / total if total > 0 else None


def nested_shares(Z: np.ndarray, inner, outer) -> dict[str, float]:
    """Sum-of-squares shares of Z between outer groups, between inner groups within their
    outer group, and within inner groups. With inner nested in outer the three add to 1."""
    Z = np.asarray(Z, float)
    Z = Z - Z.mean(0)
    total = float((Z ** 2).sum())
    df = pd.DataFrame(Z)
    mo = df.groupby(np.asarray(outer).astype(str)).transform("mean").to_numpy()
    mi = df.groupby(np.asarray(inner).astype(str)).transform("mean").to_numpy()
    if total <= 0:
        return {"between_outer": float("nan"), "between_inner_within_outer": float("nan"),
                "within_inner": float("nan")}
    return {"between_outer": float((mo ** 2).sum()) / total,
            "between_inner_within_outer": float(((mi - mo) ** 2).sum()) / total,
            "within_inner": float(((Z - mi) ** 2).sum()) / total}


def _neighbours(Z: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Indices (into Z) of the k nearest neighbours of Z[query], excluding the point itself."""
    from sklearn.neighbors import NearestNeighbors

    kk = int(min(k, len(Z) - 1))
    _, ind = NearestNeighbors(n_neighbors=kk + 1).fit(Z).kneighbors(Z[query])
    self_hit = ind == np.asarray(query)[:, None]
    self_hit[~self_hit.any(1), -1] = True
    return ind[~self_hit].reshape(len(query), kk)


def mixing_entropy(Z: np.ndarray, batch, k: int = 30, within=None, cap: int | None = None,
                   rng=None) -> float | None:
    """Mean over samples of the entropy of batch labels among each sample's k nearest
    neighbours, divided by the entropy of the batch proportions (1 = neighbourhoods as mixed
    as the batches overall, 0 = every neighbour from the sample's own batch; per-sample values
    capped at 1). With ``within``, neighbours are searched only inside the sample's own group
    (e.g. slides of one patient) and groups holding a single batch are skipped. ``cap``
    limits the query samples per group (neighbours are still searched among all)."""
    Z = np.asarray(Z, float)
    batch = np.asarray(batch).astype(str)
    blocks = [np.arange(len(Z))] if within is None else \
        [np.flatnonzero(np.asarray(within).astype(str) == w) for w in np.unique(np.asarray(within).astype(str))]
    rng = rng if rng is not None else np.random.default_rng(0)
    vals = []
    for idx in blocks:
        levels, codes = np.unique(batch[idx], return_inverse=True)
        if len(levels) < 2 or len(idx) <= 2:
            continue
        prop = np.bincount(codes) / len(codes)
        h_ref = float(-(prop * np.log(prop)).sum())
        q = np.arange(len(idx)) if cap is None or len(idx) <= cap else np.sort(rng.choice(len(idx), cap, replace=False))
        nb = codes[_neighbours(Z[idx], q, k)]
        counts = np.zeros((len(q), len(levels)))
        np.add.at(counts, (np.repeat(np.arange(len(q)), nb.shape[1]), nb.ravel()), 1)
        p = counts / nb.shape[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -(np.where(p > 0, p * np.log(p), 0.0)).sum(1)
        vals.append(np.minimum(1.0, h / h_ref))
    return float(np.mean(np.concatenate(vals))) if vals else None


def knn_preservation(Z_ref: np.ndarray, Z_new: np.ndarray, batch, k: int = 15, cap: int = 2000,
                     rng=None) -> float | None:
    """Mean Jaccard overlap of each sample's k nearest same-batch neighbours before and after a
    correction. A correction should move batches relative to each other, not rearrange the
    samples inside a batch; a low value means it did (over-correction)."""
    batch = np.asarray(batch).astype(str)
    rng = rng if rng is not None else np.random.default_rng(0)
    vals = []
    for level in np.unique(batch):
        idx = np.flatnonzero(batch == level)
        if len(idx) <= k + 1:
            continue
        if len(idx) > cap:
            idx = np.sort(rng.choice(idx, cap, replace=False))
        q = np.arange(len(idx))
        a, b = _neighbours(Z_ref[idx], q, k), _neighbours(Z_new[idx], q, k)
        for ra, rb in zip(a, b):
            sa, sb = set(ra.tolist()), set(rb.tolist())
            vals.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(vals)) if vals else None


def silhouette(Z: np.ndarray, labels, cap: int = 5000, rng=None) -> float | None:
    """Silhouette width of ``labels`` in Z (Euclidean) on at most ``cap`` samples."""
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels).astype(str)
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = np.arange(len(Z)) if len(Z) <= cap else np.sort(rng.choice(len(Z), cap, replace=False))
    if len(np.unique(labels[idx])) < 2 or len(idx) < 3:
        return None
    return float(silhouette_score(np.asarray(Z)[idx], labels[idx]))


# ---------------------------------------------------------------------------
# outliers
# ---------------------------------------------------------------------------


def mad_outliers(x, nmads: float = 3.0, tail: str = "lower", min_diff: float = 0.0) -> tuple[np.ndarray, float, float]:
    """Values more than ``nmads`` scaled MADs (1.4826 x MAD, the normal-consistent scale) from
    the median, and at least ``min_diff`` from it (scater's ``min.diff``: a very narrow
    distribution otherwise flags ordinary values); returns (flags, lower threshold, upper
    threshold). A zero width flags nothing."""
    x = np.asarray(x, float)
    med = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - med)))
    width = max(nmads * mad, min_diff)
    lo, hi = med - width, med + width
    if width == 0:
        return np.zeros(len(x), bool), lo, hi
    out = x < lo
    if tail == "both":
        out |= x > hi
    return out, lo, hi

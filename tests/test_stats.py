"""Known answers for the shared statistics: BH, DeLong, Moran's I, limma's moderated t,
rank-sum tests, kappa, Cramer's V and the batch measurements."""

import numpy as np
import pytest
from scipy import stats as st

from analysis_agent.tools import stats as S


def test_bh_matches_hand_computed_values():
    q = S.bh_adjust([0.01, 0.04, 0.03, 0.005, np.nan])
    assert np.allclose(q[:4], [0.02, 0.04, 0.04, 0.02]) and np.isnan(q[4])
    assert np.all(S.bh_adjust(np.ones(5)) == 1.0)


def test_delong_equals_the_brute_force_structural_components():
    rng = np.random.default_rng(0)
    y = rng.random(80) < 0.4
    s = np.round(rng.normal(size=80) + y, 1)                       # rounding creates ties
    auc, var = S.delong(y, s)
    pos, neg = s[y], s[~y]
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    assert auc == pytest.approx(psi.mean())
    assert var == pytest.approx(psi.mean(1).var(ddof=1) / len(pos) + psi.mean(0).var(ddof=1) / len(neg))
    assert S.fast_auroc(y[:, None], s[:, None])[0] == pytest.approx(auc)


def test_delong_interval_covers_the_true_auroc():
    rng = np.random.default_rng(1)
    true_auc = st.norm.cdf(1 / np.sqrt(2))                          # N(1,1) vs N(0,1)
    hits = []
    for _ in range(300):
        y = np.r_[np.ones(60, bool), np.zeros(90, bool)]
        s = rng.normal(size=150) + y
        r = S.auc_inference(*S.delong(y, s), 0.95)
        hits.append(r["ci_low"] <= true_auc <= r["ci_high"])
    assert 0.91 <= np.mean(hits) <= 0.98
    null = S.auc_inference(*S.delong(rng.random(200) < 0.5, rng.normal(size=200)))
    assert null["p_vs_half"] > 0.01


def _grid(n=10):
    xy = np.array([(i, j) for i in range(n) for j in range(n)], float)
    A = S.spatial_knn(xy, np.zeros(len(xy)), k=4, max_distance_factor=1.01)    # rook neighbours
    return xy, S.moran_libraries(S.row_standardise(A), np.zeros(len(xy)))


def test_morans_i_known_patterns_and_null_calibration():
    rng = np.random.default_rng(2)
    xy, libs = _grid()
    checker = ((xy[:, 0] + xy[:, 1]) % 2)[:, None]
    gradient = xy[:, :1] + 0.01 * rng.normal(size=(100, 1))
    res = S.morans_i(np.hstack([checker, gradient]), libs)
    assert res["I"][0] == pytest.approx(-1.0) and res["I"][1] > 0.9 and res["p"][1] < 1e-6
    null = S.morans_i(rng.normal(size=(100, 2000)), libs)
    assert 0.03 <= np.mean(null["p"] < 0.05) <= 0.07
    assert np.mean(null["I"]) == pytest.approx(-1 / 99, abs=0.005)


def test_morans_i_centres_each_library_so_offsets_are_not_spatial():
    rng = np.random.default_rng(3)
    xy = np.array([(i, j) for i in range(12) for j in range(12)], float)
    xy2 = np.vstack([xy, xy])
    lib = np.repeat(["a", "b"], len(xy))
    A = S.spatial_knn(xy2, lib, k=4, max_distance_factor=1.01)
    assert A[: len(xy), len(xy):].nnz == 0                          # no edges across libraries
    offset = (lib == "b") * 5.0 + rng.normal(size=len(lib))
    res = S.morans_i(offset[:, None], S.moran_libraries(S.row_standardise(A), lib))
    assert res["p"][0] > 0.01 and res["n_libraries"][0] == 2


def test_limma_prior_is_recovered_and_moderated_t_is_calibrated():
    rng = np.random.default_rng(4)
    d0, s02, G, n = 6.0, 0.5, 4000, 4
    sig2 = d0 * s02 / rng.chisquare(d0, G)                          # scaled inverse chi-square prior
    x = rng.normal(size=(G, n)) * np.sqrt(sig2)[:, None]
    d0_hat, s02_hat = S.fit_f_dist(x.var(1, ddof=1), n - 1)
    assert 4.0 < d0_hat < 9.0 and s02_hat == pytest.approx(s02, rel=0.15)
    t = S.moderated_t(x.mean(1), x.var(1, ddof=1), n, d0_hat, s02_hat)
    assert 0.035 <= np.mean(t["p"] < 0.05) <= 0.065 and t["df"] == pytest.approx(d0_hat + n - 1)
    same = rng.normal(size=(2000, 5))                               # equal true variances: heavy shrinkage
    assert S.fit_f_dist(same.var(1, ddof=1), 4)[0] > 20


def test_rank_sum_matches_scipy_with_ties():
    rng = np.random.default_rng(5)
    X = np.round(rng.normal(size=(70, 6)), 1)
    X[:, 0] = np.where(rng.random(70) < 0.6, 0, X[:, 0])            # many zeros, as in counts
    m = rng.random(70) < 0.3
    R, tie = S.column_ranks(X)
    ours = S.rank_sum(R, tie, m)["p"]
    ref = [st.mannwhitneyu(X[m, j], X[~m, j], method="asymptotic").pvalue for j in range(6)]
    assert np.allclose(ours, ref)


def test_kappa_and_cramers_v():
    a = ["x"] * 25 + ["y"] * 25
    b = ["x"] * 20 + ["y"] * 5 + ["x"] * 10 + ["y"] * 15
    assert S.cohen_kappa(a, b) == pytest.approx(0.4)                # p_o 0.7, p_e 0.5
    rng = np.random.default_rng(6)
    g = rng.integers(0, 4, 2000)
    assert S.cramers_v(g, g) > 0.99 and S.cramers_v(g, rng.integers(0, 3, 2000)) < 0.05


def test_batch_measurements():
    rng = np.random.default_rng(7)
    outer = np.repeat(["p1", "p2"], 400)
    inner = np.repeat(["s1", "s2", "s3", "s4"], 200)
    Z = rng.normal(size=(800, 5)) + (outer == "p2")[:, None] * 4.0 + (inner == "s2")[:, None] * 2.0
    sh = S.nested_shares(Z, inner, outer)
    assert sum(sh.values()) == pytest.approx(1.0)
    assert sh["between_outer"] > sh["between_inner_within_outer"] > 0.05
    mixed = S.mixing_entropy(rng.normal(size=(600, 3)), np.tile(["a", "b", "c"], 200), k=20)
    split = S.mixing_entropy(Z, inner, k=20, within=outer)
    assert mixed > 0.9 and split < 0.6
    assert S.knn_preservation(Z, Z + 3.0, inner) == pytest.approx(1.0)
    assert S.knn_preservation(Z, rng.normal(size=Z.shape), inner) < 0.2
    x = np.r_[rng.normal(10, 1, 200), [1.0, 2.0]]
    flags, lo, _ = S.mad_outliers(x, 3)
    assert flags[-2:].all() and flags[:-2].sum() <= 2

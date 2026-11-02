"""NB versus ZINB for count data, by per-gene maximum likelihood (from Project 2's profiler).

Moment matching cannot separate zero-inflation from overdispersion: extra zeros simply
inflate the variance, the fitted NB gets a smaller theta, and its predicted zero
fraction rises to meet the observed one. A likelihood fit can, because ZINB places the
extra mass exactly at zero while NB must spread it across the whole distribution. All
genes are fitted jointly (their parameters are independent, so the summed loss
decouples) with L-BFGS, on a sample of genes and samples.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

COUNT_MIN_DETECTION = 0.05   # genes detected in >= 5% of samples are assessed
MAX_GENES_LIKELIHOOD = 300
MAX_CELLS_LIKELIHOOD = 3_000


def nb_vs_zinb(Y: np.ndarray) -> tuple[float, float]:
    """(fraction of genes where ZINB has the lower AIC, median fitted zero-inflation pi)."""
    import torch

    y = torch.as_tensor(Y, dtype=torch.float64)
    m = y.mean(0).clamp_min(1e-3)

    def nb_logpmf(log_mu, log_theta):
        theta = log_theta.exp()
        return (torch.lgamma(y + theta) - torch.lgamma(theta) - torch.lgamma(y + 1)
                + theta * (log_theta - torch.logaddexp(log_theta, log_mu))
                + y * (log_mu - torch.logaddexp(log_theta, log_mu)))

    def fit(zero_inflated: bool):
        log_mu = m.log().clone().requires_grad_(True)
        log_theta = torch.zeros_like(m, requires_grad=True)
        logit_pi = torch.full_like(m, -3.0, requires_grad=True)
        params = [log_mu, log_theta] + ([logit_pi] if zero_inflated else [])
        opt = torch.optim.LBFGS(params, max_iter=200, line_search_fn="strong_wolfe")

        def loglik():
            ll = nb_logpmf(log_mu, log_theta.clamp(-8, 8))
            if zero_inflated:
                log_pi = torch.nn.functional.logsigmoid(logit_pi)
                log_1mpi = torch.nn.functional.logsigmoid(-logit_pi)
                ll = torch.where(y == 0, torch.logaddexp(log_pi, log_1mpi + ll), log_1mpi + ll)
            return ll.sum(0)

        def closure():
            opt.zero_grad()
            loss = -loglik().sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            return loglik(), torch.sigmoid(logit_pi)

    ll_nb, _ = fit(False)
    ll_zinb, pi = fit(True)
    aic_gain = 2 * (ll_zinb - ll_nb) - 2          # one extra parameter
    return float((aic_gain > 0).double().mean()), float(pi.median())


def count_evidence(X, seed: int = 0) -> dict:
    """Overdispersion and the NB-vs-ZINB comparison on a sample of expressed genes."""
    X = sp.csr_matrix(X)
    n = X.shape[0]
    mu = np.asarray(X.mean(axis=0)).ravel()
    sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    det = np.asarray((X != 0).sum(axis=0)).ravel() / max(n, 1)
    var = sq - mu ** 2
    expressed = (mu > 0) & (det >= COUNT_MIN_DETECTION)
    if expressed.sum() < 10:
        expressed = mu > 0
    rng = np.random.default_rng(seed)
    genes = np.flatnonzero(expressed)
    genes = np.sort(rng.choice(genes, min(MAX_GENES_LIKELIHOOD, len(genes)), replace=False))
    cells = np.arange(n) if n <= MAX_CELLS_LIKELIHOOD else \
        np.sort(rng.choice(n, MAX_CELLS_LIKELIHOOD, replace=False))
    out = {"n_genes_assessed": int(len(genes)), "n_samples_assessed": int(len(cells)),
           "median_dispersion_ratio": float(np.median(var[expressed] / mu[expressed])),
           "fraction_overdispersed": float((var[expressed] > mu[expressed]).mean())}
    try:
        frac, pi = nb_vs_zinb(X[cells][:, genes].toarray())
        out.update(fraction_genes_zinb_preferred=frac, median_zero_inflation_pi=pi)
    except Exception as err:          # an awkward fit leaves the comparison undecided, not the run
        out.update(fraction_genes_zinb_preferred=None, median_zero_inflation_pi=None,
                   error=f"{type(err).__name__}: {err}")
    return out

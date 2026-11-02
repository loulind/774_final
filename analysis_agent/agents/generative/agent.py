"""The generative-modelling sub-agent (Project 2) behind the common tool interface.

Its rules decide, for each use, whether a generative model is warranted at all, which
family and likelihood, and its hyperparameters; every decision is logged with source
"subagent" and the numbers that triggered it. Thresholds are named constants.

  batch-conditioned count VAE   (batch correction; scVI-style). Warranted for raw
      integer counts with enough samples to train an encoder; NB or ZINB by per-gene
      likelihood (Project 2's rule); hyperparameters by Project 2's scVI rules. The batch
      label conditions encoder and decoder, so the latent means describe what the batch
      does not explain.
  density model                 (atypical-sample flags). A masked autoregressive flow on
      the leading principal components of an embedding (Project 2 chose MAF for anomaly
      scoring); kept only if its held-out log-likelihood beats a Gaussian fitted to the
      same components -- otherwise the Gaussian is used, as Project 2 kept baselines when
      the deep model did not beat them.
  imputation                    (missing tabular values). MIWAE against mean/mode, k-NN
      and MICE on held-out observed entries; adopted only when it beats the best baseline.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .registry import MAX_FLOW_DIM, REGISTRY, clip_hyperparameters

# A deep encoder-decoder below this many samples mostly memorises; linear corrections
# (Harmony, ComBat) remain the better tools there.
MIN_SAMPLES_DEEP = 500
# Batches with fewer samples than this cannot be conditioned on reliably (Project 2's
# minimum class size for a conditional model); they are pooled.
MIN_BATCH_SAMPLES = 10
# Project 2's NB-vs-ZINB rule.
ZINB_PI = 0.1
ZINB_GENE_FRACTION = 0.3
# Flow sizing: the flow sees at most this many training samples per dimension's worth
# of parameters; below 20 samples per dimension the Gaussian baseline is used alone.
MIN_SAMPLES_PER_DIM = 20
DENSITY_VARIANCE = 0.90
DUPLICATE_TOLERANCE = 0.01
DEQUANTISE_STD = 0.05


def _pow2(x: float, lo: int, hi: int) -> int:
    return int(min(hi, max(lo, 2 ** round(math.log2(max(x, 1))))))


# ---------------------------------------------------------------------------
# batch-conditioned count VAE
# ---------------------------------------------------------------------------

def warrant_batch_vae(n: int, is_counts: bool, batch_sizes: dict[str, int]) -> tuple[bool, str]:
    if not is_counts:
        return False, ("the matrix is not non-negative integer counts, so a count likelihood "
                       "does not apply")
    if n < MIN_SAMPLES_DEEP:
        return False, (f"only {n} samples (< {MIN_SAMPLES_DEEP}); an encoder-decoder would mostly "
                       "memorise them, and linear corrections remain")
    big = [b for b, s in batch_sizes.items() if s >= MIN_BATCH_SAMPLES]
    if len(big) < 2:
        return False, f"fewer than two batches with >= {MIN_BATCH_SAMPLES} samples to condition on"
    return True, (f"{n} samples of raw counts in {len(big)} batches with >= {MIN_BATCH_SAMPLES} "
                  "samples each: enough to train a count VAE conditioned on the batch")


def pool_small_batches(batch: np.ndarray) -> tuple[np.ndarray, list[str]]:
    s = pd.Series(batch).astype(str)
    sizes = s.value_counts()
    small = sorted(sizes.index[sizes < MIN_BATCH_SAMPLES])
    if small:
        s = s.where(~s.isin(small), "__pooled__")
    return s.to_numpy(), small


def choose_count_likelihood(ev: dict) -> tuple[str, str, list[tuple[str, str]]]:
    pi, frac = ev.get("median_zero_inflation_pi"), ev.get("fraction_genes_zinb_preferred")
    if (pi is not None and pi > ZINB_PI) or (frac is not None and frac > ZINB_GENE_FRACTION):
        return ("zinb_vae", f"per-gene MLE prefers ZINB for {100 * (frac or 0):.0f}% of genes (median "
                            f"fitted zero-inflation {pi if pi is not None else float('nan'):.2f}): more "
                            "zeros than a negative binomial can explain",
                [("NB", "under-predicts zeros on these data")])
    return ("nb_vae", f"genes are overdispersed (median var/mean {ev.get('median_dispersion_ratio', float('nan')):.2f}) "
                      f"but ZINB beats NB by AIC for only {100 * (frac or 0):.0f}% of genes (median "
                      f"fitted zero-inflation {pi if pi is not None else float('nan'):.3f}); the extra "
                      "parameter is not supported",
            [("ZINB", "zero-inflation not supported by the per-gene likelihood comparison"),
             ("Poisson", "ignores overdispersion")])


def count_vae_hyperparameters(n: int, latent: int | None, max_epochs: int | None,
                              epoch_cap: int) -> tuple[dict, list[str]]:
    """Project 2's scVI rules: latent 10 (5 below 1,000 samples), 128 hidden units, one
    layer, min(400, 20000/n x 400) epochs, batch size ~n/20 as a power of two in [32, 256]."""
    batch = _pow2(n / 20, 32, 256)
    lat = latent or (10 if n >= 1000 else 5)
    epochs = max_epochs or int(min(400, round(20_000 / max(n, 1) * 400)))
    epochs = int(min(epochs, epoch_cap))
    hp, notes = clip_hyperparameters("nb_vae", {"latent_dim": lat, "hidden_dim": 128, "n_layers": 1,
                                                "max_epochs": epochs, "lr": 1e-3, "batch_size": batch})
    why = [f"latent_dim={hp['latent_dim']} ({'set by the plan' if latent else 'scVI default; 5 if n<1000'})",
           f"max_epochs={hp['max_epochs']} ({'set by the plan' if max_epochs else 'scVI rule min(400, 20000/n x 400)'}"
           f"{', capped by the budget' if epochs == epoch_cap else ''})",
           f"batch_size={hp['batch_size']} (~n/20, power of two in [32, 256])"]
    return hp, why + notes


def fit_count_vae(name: str, X: sp.csr_matrix, batch_codes: np.ndarray, n_batches: int, hp: dict,
                  seconds: float, seed: int, device: str, val_fraction: float = 0.1):
    from .models.base import ModelSpec
    from .registry import build

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    lib = np.asarray(X.sum(1)).ravel().astype(np.float32)
    usable = np.flatnonzero(lib > 0)
    perm = rng.permutation(usable)
    n_val = max(1, int(round(val_fraction * len(usable))))
    val, tr = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    Xd = X.toarray().astype(np.float32)
    spec = ModelSpec(name=name, conditional=True, representation="counts_hvg", hyperparameters=hp,
                     train_seconds=seconds)
    model = build(spec, info={"kind": "counts", "shape": (X.shape[1],)}, n_classes=n_batches,
                  device=device, seed=seed)
    hist = model.fit(Xd[tr], batch_codes[tr], Xd[val], batch_codes[val], extras={"library": lib[tr]},
                     extras_val={"library": lib[val]})
    Z = model.encode(Xd, batch_codes)
    val_elbo = float(np.mean(model.score(Xd[val], batch_codes[val], extras={"library": lib[val]})))
    return Z.astype(np.float32), {**{k: v for k, v in hist.items() if k != "curve"},
                                  "val_elbo": val_elbo, "n_train": int(len(tr)), "n_val": int(len(val)),
                                  "n_zero_library": int(n - len(usable)),
                                  "median_inverse_dispersion_theta":
                                      model.summary().get("median_inverse_dispersion_theta"),
                                  "curve": hist.get("curve", [])}


# ---------------------------------------------------------------------------
# density model for atypical-sample flags
# ---------------------------------------------------------------------------

def density_dimension(var_ratio: np.ndarray, n_train: int, max_dim: int) -> tuple[int, str]:
    cum = np.cumsum(var_ratio)
    hit = np.flatnonzero(cum >= DENSITY_VARIANCE)
    k90 = int(hit[0] + 1) if hit.size else len(var_ratio)
    cap = int(min(max_dim, MAX_FLOW_DIM, max(2, n_train // MIN_SAMPLES_PER_DIM)))
    k = int(max(2, min(k90, cap)))
    why = (f"{k90} principal components hold {DENSITY_VARIANCE:.0%} of the variance"
           + (f"; capped at {cap} (max_dim {max_dim}, flow limit {MAX_FLOW_DIM}, and at least "
              f"{MIN_SAMPLES_PER_DIM} training samples per dimension)" if k < k90 else ""))
    return k, why


NEAR_DUPLICATE_RATIO = 0.2   # nearest-neighbour distance < 20% of the median one (Project 2)


def near_duplicate_fraction(X: np.ndarray, rng, cap: int = 3000) -> float:
    """Share of samples whose nearest neighbour is far closer than typical (Project 2's
    profiler): on such points a continuous density can concentrate without bound."""
    from sklearn.neighbors import NearestNeighbors

    if len(X) < 10:
        return 0.0
    Xs = X if len(X) <= cap else X[rng.choice(len(X), cap, replace=False)]
    dist, _ = NearestNeighbors(n_neighbors=2).fit(Xs).kneighbors(Xs)
    r1 = dist[:, 1]
    med = float(np.median(r1))
    return float(np.mean(r1 < NEAR_DUPLICATE_RATIO * med)) if med > 0 else float(np.mean(r1 == 0))


def flow_hyperparameters(n: int, p: int, dup_fraction: float, epoch_cap: int) -> tuple[dict, list[str]]:
    """Project 2's flow rules: 5 blocks from 2,000 samples (else 3), hidden = 8p in [32, 256],
    dequantisation noise when more than 1% of rows are (near-)duplicates."""
    blocks = 5 if n >= 2000 else 3
    hidden = int(min(256, max(32, 8 * p)))
    noise = DEQUANTISE_STD if dup_fraction > DUPLICATE_TOLERANCE else 0.0
    hp, notes = clip_hyperparameters("maf", {"n_blocks": blocks, "hidden_dim": hidden, "lr": 1e-3,
                                             "noise_std": noise, "batch_size": _pow2(n / 20, 32, 256),
                                             "weight_decay": 1e-5 if n >= 2000 else 1e-4,
                                             "max_epochs": epoch_cap})
    why = [f"n_blocks={blocks}, hidden_dim={hidden} (8 x p, in [32, 256])",
           f"noise_std={noise}" + (f" ({100 * dup_fraction:.1f}% of rows are near-duplicates, on which a "
                                   "continuous density is degenerate)" if noise else "")]
    return hp, why + notes


def gaussian_loglik(Ztr: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Per-sample log-density under a Gaussian fitted to Ztr (Ledoit-Wolf shrinkage)."""
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(Ztr)
    P = lw.precision_
    _, logdet = np.linalg.slogdet(lw.covariance_)
    D = Z - lw.location_
    maha = np.einsum("ij,jk,ik->i", D, P, D)
    return -0.5 * (Z.shape[1] * math.log(2 * math.pi) + logdet + maha)


def fit_flow(Ztr: np.ndarray, Zval: np.ndarray, Zall: np.ndarray, hp: dict, seconds: float, seed: int,
             device: str):
    from .models.base import ModelSpec
    from .registry import build

    spec = ModelSpec(name="maf", representation="standardized", hyperparameters=hp, train_seconds=seconds)
    model = build(spec, info={"kind": "tabular", "shape": (Ztr.shape[1],)}, device=device, seed=seed)
    hist = model.fit(Ztr, X_val=Zval)
    return model.score(Zall), {k: v for k, v in hist.items() if k != "curve"}


# ---------------------------------------------------------------------------
# imputation
# ---------------------------------------------------------------------------

class TableEncoding:
    """Numeric columns standardised, categorical columns one-hot; spans as in Project 2."""

    def fit(self, df: pd.DataFrame) -> TableEncoding:
        self.columns = list(df.columns)
        self.numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                        and not pd.api.types.is_bool_dtype(df[c])]
        self.categorical = [c for c in df.columns if c not in self.numeric]
        self.mean = {c: float(df[c].mean()) for c in self.numeric}
        self.std = {c: float(df[c].std()) or 1.0 for c in self.numeric}
        self.levels = {c: sorted(df[c].dropna().astype(str).unique()) for c in self.categorical}
        self.spans, a = [], 0
        for c in self.numeric:
            self.spans.append((a, a + 1, "cont"))
            a += 1
        for c in self.categorical:
            k = max(1, len(self.levels[c]))
            self.spans.append((a, a + k, "cat"))
            a += k
        self.dim = a
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Z = np.zeros((len(df), self.dim), np.float32)
        M = np.zeros((len(df), self.dim), np.float32)
        for (a, b, _), c in zip(self.spans, self.numeric + self.categorical):
            v = df[c]
            ok = v.notna().to_numpy()
            M[ok, a:b] = 1.0
            if c in self.mean:
                Z[ok, a] = ((v[ok].astype(float) - self.mean[c]) / self.std[c]).to_numpy()
            else:
                idx = {lv: j for j, lv in enumerate(self.levels[c])}
                codes = v[ok].astype(str).map(idx)
                known = codes.notna().to_numpy()
                rows = np.flatnonzero(ok)[known]
                Z[rows, a + codes[known].astype(int).to_numpy()] = 1.0
                M[np.flatnonzero(ok)[~known], a:b] = 0.0
        return Z, M

    def inverse(self, Z: np.ndarray, index) -> pd.DataFrame:
        out = {}
        for (a, b, _), c in zip(self.spans, self.numeric + self.categorical):
            if c in self.mean:
                out[c] = Z[:, a] * self.std[c] + self.mean[c]
            else:
                lv = self.levels[c] or [None]
                out[c] = [lv[j] for j in Z[:, a:b].argmax(1)]
        return pd.DataFrame(out, index=index)[self.columns]


def miwae_hyperparameters(n: int, d: int, epoch_cap: int) -> dict:
    """Project 2's MIWAE rules: latent = p/4 in [2, 16], hidden 64 below 1,000 rows (else 128),
    10 importance samples."""
    hp, _ = clip_hyperparameters("masked_vae", {
        "latent_dim": int(min(16, max(2, d // 4))), "hidden_dim": 64 if n < 1000 else 128,
        "n_importance": 10, "lr": 1e-3, "weight_decay": 1e-4 if n < 1000 else 1e-5,
        "batch_size": _pow2(n / 20, 32, 256), "max_epochs": epoch_cap})
    return hp


def harden(Z: np.ndarray, spans) -> np.ndarray:
    Z = Z.copy()
    for a, b, kind in spans:
        if kind == "cat":
            Z[:, a:b] = np.eye(b - a)[np.nan_to_num(Z[:, a:b], nan=-1).argmax(1)]
    return Z


def masked_scores(Zhat, Ztrue, hide, spans) -> dict:
    cont = [a for a, b, k in spans if k == "cont"]
    cats = [(a, b) for a, b, k in spans if k == "cat"]
    out = {"masked_rmse": None, "masked_categorical_accuracy": None}
    if cont:
        h = hide[:, cont]
        if h.any():
            out["masked_rmse"] = float(np.sqrt(np.mean(((Zhat[:, cont] - Ztrue[:, cont])[h]) ** 2)))
    hits = total = 0
    for a, b in cats:
        rows = hide[:, a]
        hits += int((Zhat[rows, a:b].argmax(1) == Ztrue[rows, a:b].argmax(1)).sum())
        total += int(rows.sum())
    if total:
        out["masked_categorical_accuracy"] = hits / total
    return out

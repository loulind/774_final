"""Project 2 as three tools: batch-conditioned count VAE, density flags, imputation."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pydantic import Field

from ...tools.base import Artifact, Criterion, Port, StepResult, ToolParams, ToolSpec, dense
from ...tools.ingest import is_count_matrix
from ...tools.registry import register
from . import agent

SOURCE = ("Project 2 (genmodel_agent): models/base.py, train.py, vae.py, flows.py, registry.py and the "
          "NB-vs-ZINB fit copied; warrant, likelihood and hyperparameter rules adapted from "
          "planner/heuristic.py; tool wiring in agents/generative/ is new")


def _torch_threads():
    import torch

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


# ---------------------------------------------------------------------------
# batch-conditioned count VAE
# ---------------------------------------------------------------------------

class BatchVAEParams(ToolParams):
    likelihood: Literal["auto", "nb", "zinb"] = Field("auto", description="auto = per-gene NB-vs-ZINB "
                                                                          "likelihood comparison")
    latent_dim: int | None = Field(None, ge=2, le=64, description="None = scVI rule (10; 5 below 1,000 "
                                                                  "samples)")
    n_genes: int = Field(2000, ge=100, le=5000, description="genes modelled when no normalised input "
                                                            "names them (most variable log counts)")
    max_epochs: int | None = Field(None, ge=1, le=2000, description="None = scVI rule, capped by the "
                                                                    "budget")


def _top_variable_genes(X: sp.csr_matrix, k: int) -> np.ndarray:
    Y = X.astype(np.float32).copy()
    Y.data = np.log1p(Y.data)
    mean = np.asarray(Y.mean(0)).ravel()
    sq = np.asarray(Y.multiply(Y).mean(0)).ravel()
    return np.sort(np.argsort(-(sq - mean ** 2))[: min(k, X.shape[1])])


def _run_batch_vae(ctx, inputs, p: BatchVAEParams) -> StepResult:
    counts, batch = inputs["counts"], inputs["batch"]
    feats = inputs.get("features")
    names = ["counts", "batch"] + (["features"] if feats is not None else [])
    rows = counts.rows
    for k in names[1:]:
        rows = np.intersect1d(rows, inputs[k].rows, assume_unique=True)
    X = sp.csr_matrix(counts.take(rows))
    b_raw = np.asarray(batch.take(rows)).astype(str)
    genes_all = list(counts.meta.get("feature_names") or [f"f{j}" for j in range(X.shape[1])])
    if feats is not None and feats.meta.get("feature_names"):
        index = {g: j for j, g in enumerate(genes_all)}
        cols = np.array([index[g] for g in feats.meta["feature_names"] if g in index])
        gene_rule = f"the {len(cols)} features of '{feats.name}'"
    else:
        cols = _top_variable_genes(X, p.n_genes)
        gene_rule = f"the {len(cols)} most variable genes (log counts)"
    X = X[:, cols]
    pooled, small = agent.pool_small_batches(b_raw)
    sizes = pd.Series(pooled).value_counts().to_dict()
    ok, why = agent.warrant_batch_vae(len(rows), is_count_matrix(X), sizes)
    ctx.decide("use a batch-conditioned count VAE?", "yes" if ok else "no", why,
               [("no", "linear corrections only")] if ok else [("train it anyway", why)],
               {"n": len(rows), "n_batches": len(sizes), "pooled_batches": small})
    if not ok:
        raise RuntimeError(f"generative_batch_vae not warranted: {why}")
    if small:
        ctx.decide("small batches", "pooled", f"{len(small)} batches have fewer than "
                   f"{agent.MIN_BATCH_SAMPLES} samples and share one condition level.",
                   [("one level each", "too few samples to learn a batch offset")], {"pooled": small})

    if p.likelihood == "auto":
        from .likelihood import count_evidence

        ev = count_evidence(X, ctx.seed)
        name, why_l, rej = agent.choose_count_likelihood(ev)
    else:
        ev, name = {}, {"nb": "nb_vae", "zinb": "zinb_vae"}[p.likelihood]
        why_l, rej = "set by the plan", []
    ctx.decide("count likelihood", "ZINB" if name == "zinb_vae" else "NB", why_l, rej, ev)

    hp, why_h = agent.count_vae_hyperparameters(len(rows), p.latent_dim, p.max_epochs, ctx.budget.max_epochs)
    seconds = ctx.budget.train_seconds_per_fit
    ctx.decide("count VAE hyperparameters", ", ".join(f"{k}={v}" for k, v in hp.items()),
               "; ".join(why_h) + f"; training capped at {seconds:.0f}s.", [],
               {"n": len(rows), "train_seconds": seconds})
    levels = sorted(set(pooled))
    codes = np.array([levels.index(v) for v in pooled])
    _torch_threads()
    ctx.log(f"    training {name} on {X.shape[0]:,} x {X.shape[1]:,} counts, {len(levels)} batches")
    Z, hist = agent.fit_count_vae(name, X, codes, len(levels), hp, seconds, ctx.seed, ctx.fit_device)
    curve = hist.pop("curve")
    metrics = {"model": name, "latent_dim": int(Z.shape[1]), "n_genes": int(X.shape[1]),
               "n_batches": len(levels), "epochs_run": hist["epochs_run"],
               "stopped_because": hist["stopped_because"], "best_epoch": hist["best_epoch"],
               "val_neg_elbo_per_gene": -hist["val_elbo"] / X.shape[1], "train_seconds": hist["seconds"]}
    art = Artifact("embedding", Z, rows, {"method": f"{name} conditioned on '{batch.name}'",
                                         "source": counts.name, "batch": batch.name,
                                         "genes": gene_rule})
    return StepResult(artifacts={"embedding": art}, metrics=metrics,
                      tables={"training_curve": pd.DataFrame(curve)},
                      notes=[f"modelled {gene_rule}"])


def _batch_vae_cost(n, p, prm, budget) -> float:
    return float(budget.train_seconds_per_fit + 30)


register(ToolSpec(
    name="generative_batch_vae", origin="project2", source=SOURCE,
    summary="Count VAE (NB or ZINB, chosen by per-gene likelihood) conditioned on a batch label; its "
            "latent means are a batch-corrected representation (scVI-style), a non-linear "
            "alternative to linear corrections.",
    consumes=(Port("counts", ("counts",), doc="raw counts"),
              Port("batch", ("groups",), doc="batch to condition on (e.g. slide)"),
              Port("features", ("normalized",), required=False, doc="its feature names select the genes")),
    produces=(Port("embedding", ("embedding",), doc="latent means"),),
    params=BatchVAEParams, answers=("generative_modelling", "representation_learning", "batch_assessment"),
    addresses=("batch_effect", "depth_variation", "high_dimensionality"),
    criteria=(Criterion("val_neg_elbo_per_gene", "min", doc="held-out negative ELBO per gene"),
              Criterion("epochs_run", "report"), Criterion("stopped_because", "report")),
    run=_run_batch_vae, cost=_batch_vae_cost,
))


# ---------------------------------------------------------------------------
# density model: atypical-sample flags
# ---------------------------------------------------------------------------

class DensityParams(ToolParams):
    model: Literal["auto", "maf", "gaussian"] = Field("auto", description="auto = MAF if it beats a "
                                                                          "Gaussian on validation data")
    max_dim: int = Field(64, ge=2, le=200, description="most principal components the density sees")
    flag_quantile: float = Field(0.99, ge=0.9, le=0.9999, description="samples scoring above this "
                                                                      "quantile of validation scores "
                                                                      "are flagged")


def _density_split(ctx, rows, split) -> tuple[np.ndarray, np.ndarray, str]:
    if split is not None:
        s = np.full(len(rows), "", object)
        common = np.intersect1d(rows, split.rows, assume_unique=True)
        s[np.searchsorted(rows, common)] = split.take(common)
        tr, va = np.flatnonzero(s == "train"), np.flatnonzero(s == "val")
        if len(tr) >= 50 and len(va) >= 20:
            return tr, va, f"the predefined split ('{split.name}': train fits, val selects)"
    rng = ctx.rng(5)
    perm = rng.permutation(len(rows))
    n_val = max(20, int(0.2 * len(rows)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val]), "a random 80/20 split of the samples"


def _project(Z: np.ndarray, fit: np.ndarray, k: int, seed: int) -> np.ndarray:
    """k principal components of Z, standardised; both fitted on rows ``fit``."""
    from sklearn.decomposition import PCA

    mu = Z[fit].mean(0)
    P = PCA(k, svd_solver="randomized", random_state=seed).fit(Z[fit] - mu).transform(Z - mu)
    return (P / (P[fit].std(0) + 1e-6)).astype(np.float32)


def _run_density(ctx, inputs, p: DensityParams) -> StepResult:
    from sklearn.decomposition import PCA

    feats, split = inputs["features"], inputs.get("split")
    rows = feats.rows
    Z = dense(feats.data)
    tr, va, how = _density_split(ctx, rows, split)
    ctx.decide("density fit / validation samples", how,
               "The projection and the density are fitted on training samples and compared on validation "
               "samples. Every sample is then scored by a model that was not fitted on it (training samples "
               "by 2-fold cross-fitting) and judged against that model's own validation scores.",
               [("score every sample with one model", "the samples a model was fitted on look more typical "
                 "than new ones (most of all in many dimensions), so flags would differ between splits "
                 "for no reason")],
               {"n_train": len(tr), "n_val": len(va)})
    pca = PCA(min(Z.shape[1], len(tr) - 1, p.max_dim * 4, 256), svd_solver="randomized",
              random_state=ctx.seed).fit(Z[tr] - Z[tr].mean(0))
    k, why_k = agent.density_dimension(pca.explained_variance_ratio_, len(tr), p.max_dim)
    P = _project(Z, tr, k, ctx.seed)
    ctx.decide("density space", f"{k} principal components, standardised", why_k,
               [("all dimensions", f"{Z.shape[1]} dimensions: too many for a full-rank flow")],
               {"k": k, "input_dims": int(Z.shape[1])})

    g_all = agent.gaussian_loglik(P[tr], P)
    g_val = float(np.mean(g_all[va]))
    use_flow = p.model in ("auto", "maf") and len(tr) >= agent.MIN_SAMPLES_PER_DIM * k
    flow_info, f_val, ll, hp = {}, None, g_all, None
    if use_flow:
        dup = agent.near_duplicate_fraction(P[tr], ctx.rng(6))
        hp, why_h = agent.flow_hyperparameters(len(tr), k, dup, ctx.budget.max_epochs)
        _torch_threads()
        ctx.log(f"    training MAF on {len(tr):,} x {k}")
        f_all, flow_info = agent.fit_flow(P[tr], P[va], P, hp, ctx.budget.train_seconds_per_fit, ctx.seed,
                                          ctx.fit_device)
        f_val = float(np.mean(f_all[va])) if np.isfinite(f_all[va]).all() else float("-inf")
        ctx.decide("flow hyperparameters", ", ".join(f"{a}={b}" for a, b in hp.items()), "; ".join(why_h),
                   [], {"n_train": len(tr), "dim": k, "near_duplicate_fraction": dup})
        keep_flow = p.model == "maf" or f_val > g_val
        if keep_flow:
            ll = f_all
        ctx.decide("density model", "MAF" if keep_flow else "Gaussian",
                   f"held-out mean log-likelihood: MAF {f_val:.3f}, Gaussian {g_val:.3f} per sample"
                   + ("" if keep_flow else "; the flow does not beat the Gaussian, so the simpler model "
                      "is used"),
                   [("Gaussian" if keep_flow else "MAF", "lower held-out log-likelihood")],
                   {"val_loglik_maf": f_val, "val_loglik_gaussian": g_val})
    else:
        reason = ("set by the plan" if p.model == "gaussian" else
                  f"{len(tr)} training samples < {agent.MIN_SAMPLES_PER_DIM} per dimension x {k} dimensions")
        ctx.decide("density model", "Gaussian", reason, [("MAF", reason)], {"n_train": len(tr), "dim": k})
    use_maf = ll is not g_all

    def refit(fit):                                   # the chosen model, fitted on other samples
        Pf = _project(Z, fit, k, ctx.seed)
        if not use_maf:
            return agent.gaussian_loglik(Pf[fit], Pf)
        return agent.fit_flow(Pf[fit], Pf[va], Pf, hp, ctx.budget.train_seconds_per_fit, ctx.seed,
                              ctx.fit_device)[0]

    perm = ctx.rng(8).permutation(tr)
    h1, h2 = np.sort(perm[: len(tr) // 2]), np.sort(perm[len(tr) // 2:])
    ctx.log(f"    cross-fitting on two halves of the {len(tr):,} training samples")
    scorers = [(np.setdiff1d(np.arange(len(rows)), tr), ll), (h2, refit(h1)), (h1, refit(h2))]
    score, tail, flags = np.empty(len(rows)), np.empty(len(rows)), np.zeros(len(rows), bool)
    thresholds = []
    for idx, lik in scorers:
        ref = np.sort(-lik[va])                       # this model's validation scores
        thresholds.append(float(np.quantile(ref, p.flag_quantile)))
        score[idx] = -lik[idx]
        tail[idx] = 1 - np.searchsorted(ref, score[idx], side="left") / len(ref)
        flags[idx] = score[idx] > thresholds[-1]
    metrics = {"model": "maf" if use_maf else "gaussian", "dim": k,
               "val_loglik": float(np.mean(ll[va])), "val_loglik_gaussian": g_val,
               "val_loglik_gain": (float(np.mean(ll[va])) - g_val) if use_maf else 0.0,
               "flag_threshold": thresholds[0], "cross_fitted": True,
               "flagged_fraction": float(flags.mean()), "n_flagged": int(flags.sum())}
    if flow_info:
        metrics.update(epochs_run=flow_info.get("epochs_run"), stopped_because=flow_info.get("stopped_because"))
    if split is not None:
        s = pd.Series(np.full(len(rows), "none", object))
        common = np.intersect1d(rows, split.rows, assume_unique=True)
        s.iloc[np.searchsorted(rows, common)] = split.take(common)
        for name, frac in pd.Series(flags).groupby(s.to_numpy()).mean().items():
            metrics[f"flagged_fraction_{name}"] = float(frac)
    return StepResult(
        artifacts={"scores": Artifact("scores", np.column_stack([score, tail]).astype(np.float32), rows,
                                      {"columns": ["neg_log_density", "tail_probability"],
                                       "direction": "higher neg_log_density, lower tail_probability = more "
                                                    "atypical",
                                       "note": "each sample is scored by a model not fitted on it; "
                                               "tail_probability (share of that model's validation samples "
                                               "at least as atypical) is comparable across samples",
                                       "source": feats.name}),
                   "flags": Artifact("mask", flags, rows, {"reason": f"density score above the "
                                                                     f"{p.flag_quantile} quantile of the "
                                                                     "scoring model's validation scores"})},
        metrics=metrics)


def _density_cost(n, p, prm, budget) -> float:
    return float(3 * budget.train_seconds_per_fit + 3e-6 * n * p + 10)


register(ToolSpec(
    name="generative_density", origin="project2", source=SOURCE,
    summary="Density model (masked autoregressive flow, kept only if it beats a Gaussian on held-out "
            "data) on the leading principal components of a representation; flags samples whose "
            "density is unusually low, each scored by a cross-fitted model that was not fitted on it.",
    consumes=(Port("features", ("embedding",), doc="representation to model"),
              Port("split", ("split",), required=False, doc="fit on train, judge on val")),
    produces=(Port("scores", ("scores",), doc="negative log-density"),
              Port("flags", ("mask",), doc="atypical samples")),
    params=DensityParams, answers=("generative_modelling", "data_quality_audit"),
    addresses=("low_quality_samples", "image_heterogeneity"),
    criteria=(Criterion("val_loglik_gain", "max", 0.0, "held-out log-likelihood gain over a Gaussian"),
              Criterion("flagged_fraction", "report", doc="share of samples flagged as atypical")),
    run=_run_density, cost=_density_cost,
))


# ---------------------------------------------------------------------------
# imputation
# ---------------------------------------------------------------------------

class ImputeParams(ToolParams):
    holdout_fraction: float = Field(0.2, ge=0.05, le=0.5, description="share of observed entries "
                                                                      "hidden in held-out rows to score "
                                                                      "imputers")


def _run_impute(ctx, inputs, p: ImputeParams) -> StepResult:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

    tab = inputs["table"]
    df = tab.data
    miss = float(df.isna().to_numpy().mean())
    if miss == 0:
        ctx.decide("impute?", "no", "the table has no missing values", [("impute", "nothing to fill")],
                   {"missing_fraction": 0.0})
        return StepResult(artifacts={"imputed": Artifact("tabular", df.copy(), tab.rows, dict(tab.meta))},
                          metrics={"missing_fraction": 0.0, "method": "none"})
    rng = ctx.rng(7)
    enc = agent.TableEncoding().fit(df)
    Z, M = enc.transform(df)
    perm = rng.permutation(len(df))
    n_te = max(5, int(0.2 * len(df)))
    te, tr = np.sort(perm[:n_te]), np.sort(perm[n_te:])
    hide = np.zeros_like(M, dtype=bool)
    for a, b, _ in enc.spans:
        chosen = M[te, a].astype(bool) & (rng.random(len(te)) < p.holdout_fraction)
        hide[np.ix_(te[chosen], np.arange(a, b))] = True
    M_eval = M.astype(bool) & ~hide
    nan = lambda A, mask: np.where(mask, A, np.nan)  # noqa: E731
    baselines = {"mean_mode": SimpleImputer(strategy="mean"), "knn": KNNImputer(n_neighbors=5),
                 "mice": IterativeImputer(max_iter=10, random_state=ctx.seed)}
    scores = {}
    for name, imp in baselines.items():
        imp.fit(nan(Z[tr], M[tr].astype(bool)))
        scores[name] = agent.masked_scores(agent.harden(imp.transform(nan(Z[te], M_eval[te])), enc.spans),
                                           Z[te], hide[te], enc.spans)
    hp = agent.miwae_hyperparameters(len(tr), enc.dim, ctx.budget.max_epochs)
    from .models.base import ModelSpec
    from .registry import build

    _torch_threads()

    def fit_miwae(rows_fit):
        spec = ModelSpec(name="masked_vae", hyperparameters=hp, train_seconds=ctx.budget.train_seconds_per_fit)
        model = build(spec, info={"kind": "tabular", "shape": (enc.dim,), "spans": enc.spans},
                      device=ctx.fit_device, seed=ctx.seed)
        model.fit(Z[rows_fit] * M[rows_fit], extras={"mask": M[rows_fit]})
        return model

    model = fit_miwae(tr)
    scores["miwae"] = agent.masked_scores(model.impute(Z[te] * M_eval[te], M_eval[te].astype(np.float32)),
                                          Z[te], hide[te], enc.spans)
    key = "masked_rmse" if enc.numeric else "masked_categorical_accuracy"
    sign = 1 if key == "masked_rmse" else -1
    valid = {k: v[key] for k, v in scores.items() if v[key] is not None}
    best_base = min((k for k in valid if k != "miwae"), key=lambda k: sign * valid[k])
    use_miwae = "miwae" in valid and sign * valid["miwae"] < sign * valid[best_base]
    chosen = "miwae" if use_miwae else best_base
    ctx.decide("imputation method", chosen,
               f"held-out {key.replace('_', ' ')}: " + ", ".join(f"{k} {v:.3f}" for k, v in valid.items())
               + ("; MIWAE beats the best baseline" if use_miwae else
                  "; MIWAE does not beat the best baseline, so the baseline is used"),
               [(k, f"{key} {v:.3f}") for k, v in valid.items() if k != chosen],
               {"scores": scores, "n_hidden_entries": int(hide[te].sum())})
    if use_miwae:
        filled = fit_miwae(np.arange(len(df))).impute(Z * M, M)
    else:
        imp = baselines[chosen]
        imp.fit(nan(Z, M.astype(bool)))
        filled = agent.harden(imp.transform(nan(Z, M.astype(bool))), enc.spans)
    out = enc.inverse(filled, df.index)
    out = df.where(df.notna(), out)
    table = pd.DataFrame([{"method": k, **v} for k, v in scores.items()])
    return StepResult(artifacts={"imputed": Artifact("tabular", out, tab.rows, {**tab.meta, "imputed_by": chosen})},
                      metrics={"missing_fraction": miss, "method": chosen, f"{key}_chosen": valid[chosen],
                               f"{key}_best_baseline": valid[best_base], "miwae_beats_best_baseline": use_miwae},
                      tables={"imputation_comparison": table})


register(ToolSpec(
    name="generative_impute", origin="project2", source=SOURCE,
    summary="Missing-value imputation: a masked VAE (MIWAE) is compared with mean/mode, k-NN and MICE "
            "on held-out observed entries, and used only if it beats the best of them.",
    consumes=(Port("table", ("tabular",), doc="covariates with missing values"),),
    produces=(Port("imputed", ("tabular",), doc="completed covariates"),),
    params=ImputeParams, answers=("generative_modelling", "data_quality_audit"), addresses=("missing_data",),
    criteria=(Criterion("missing_fraction", "report"), Criterion("miwae_beats_best_baseline", "report")),
    run=_run_impute, cost=lambda n, p, prm, b: float(2 * b.train_seconds_per_fit + 1e-4 * n * p),
))

"""Supervised prediction of labels from features, estimated without leakage.

Evaluation design. With a predefined split, models are fitted on the other splits and
scored once on the test split; fitting samples that share a group with any test sample are
set aside first, because a predefined split need not keep groups apart. Without a split,
every sample gets an out-of-fold prediction from grouped K-fold cross-validation. Either way
the penalty is chosen by grouped inner cross-validation on training samples only (AUROC per
target). Folds keep groups whole; when several groupings are given (articles and
near-duplicate clusters, say), samples linked by any of them are kept together.

Models. Per-target logistic regression on standardised features for a few targets;
beyond ``logistic_max_targets``, closed-form ridge regression fits every target at once from
one eigendecomposition per fold, and its scores are turned into probabilities by Platt
scaling fitted on the out-of-fold training scores.

Reliability ("which targets can be reliably predicted") is decided by the rule fixed in the
run configuration before any result: enough positives in the evaluation set to judge
(otherwise "too rare to assess", not a failure), the lower DeLong bound of AUROC at or above
the bar, and a Benjamini-Hochberg-adjusted one-sided p-value against AUROC 0.5 below the FDR.
AUPRC (with the prevalence as its chance level), Brier skill and expected calibration error
are reported alongside.

Leakage sensitivity: when fitting samples were set aside, a second model is fitted with
them and scored on the same test samples; its AUROC gain shows how much a split that lets
related samples (same article, near-duplicates) straddle it would inflate the results.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pydantic import Field

from .base import (Artifact, Criterion, Port, StepResult, ToolParams, ToolSpec, check_time, dense,
                   label_matrix, values_on)
from .registry import register
from .stats import auc_inference, bh_adjust, delong, fast_auroc, stratified_delong

SOURCE = "new in Project 4 (DeLong after Sun & Xu 2014)"
RIDGE_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)        # x mean eigenvalue of the training Gram matrix
C_GRID = (0.01, 0.1, 1.0, 10.0)


def merge_groups(n: int, *groupings) -> np.ndarray | None:
    """Connected components of samples linked by sharing a level of any grouping (missing
    values link nothing)."""
    from scipy.sparse.csgraph import connected_components

    gs = [g for g in groupings if g is not None]
    if not gs:
        return None
    r, c, off = [], [], 0
    for g in gs:
        g = np.asarray(g).astype(str)
        g = np.where(g == "", np.char.add("__row", np.arange(n).astype(str)), g)
        levels, codes = np.unique(g, return_inverse=True)
        r.append(np.arange(n))
        c.append(codes + off)
        off += len(levels)
    B = sp.csr_matrix((np.ones(n * len(gs)), (np.concatenate(r), np.concatenate(c))), shape=(n, off))
    _, lab = connected_components(sp.bmat([[None, B], [B.T, None]]), directed=False)
    return lab[:n]


def make_folds(n: int, groups: np.ndarray | None, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold, KFold

    if groups is not None and len(np.unique(groups)) >= k:
        return list(GroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(n), groups=groups))
    return list(KFold(k, shuffle=True, random_state=seed).split(np.zeros(n)))


class RidgePath:
    """Ridge regression of all targets at once for any penalty, from one eigendecomposition."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.mu, sd = X.mean(0), X.std(0)
        self.sd = np.where(sd > 0, sd, 1.0)
        Xs = (X - self.mu) / self.sd
        self.evals, self.V = np.linalg.eigh(Xs.T @ Xs)
        self.ymean = Y.mean(0)
        self.VtXtY = self.V.T @ (Xs.T @ (Y - self.ymean))
        self.scale = max(float(self.evals.mean()), 1e-12)

    def coef(self, rel: float) -> np.ndarray:
        return self.V @ (self.VtXtY / (self.evals + rel * self.scale)[:, None])

    def predict(self, X: np.ndarray, W: np.ndarray, cols: np.ndarray | None = None) -> np.ndarray:
        """Predictions of the targets ``cols`` (default: all), whose coefficients are W."""
        return ((X - self.mu) / self.sd) @ W + (self.ymean if cols is None else self.ymean[cols])


def platt(scores: np.ndarray, Y: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-column logistic calibration p = sigmoid(a z + b) of standardised scores z, fitted by
    Newton's method with a tiny ridge; returns (a, b, score mean, score sd)."""
    m, s = scores.mean(0), scores.std(0)
    s = np.where(s > 0, s, 1.0)
    Z = (scores - m) / s
    Yf = Y.astype(float)
    a, b = np.ones(Z.shape[1]), np.zeros(Z.shape[1])
    prev = Yf.mean(0).clip(1e-6, 1 - 1e-6)
    b[:] = np.log(prev / (1 - prev))
    for _ in range(iters):
        P = 1 / (1 + np.exp(-(Z * a + b)))
        W = P * (1 - P)
        ga = ((P - Yf) * Z).sum(0) + 1e-6 * a
        gb = (P - Yf).sum(0)
        haa, hab, hbb = (W * Z * Z).sum(0) + 1e-6, (W * Z).sum(0), W.sum(0) + 1e-9
        det = haa * hbb - hab ** 2
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        a, b = a - da, b - db
        if max(np.abs(da).max(), np.abs(db).max()) < 1e-8:
            break
    return a, b, m, s


def _calibrate(scores, a, b, m, s) -> np.ndarray:
    return 1 / (1 + np.exp(-(((scores - m) / s) * a + b)))


def fit_predict(X: np.ndarray, Y: np.ndarray, groups: np.ndarray | None, Xnew: np.ndarray, model: str,
                inner_folds: int, seed: int, check=None) -> dict:
    """Probabilities for Xnew (one column per target), the penalty chosen per target by grouped
    inner cross-validation (AUROC), and each target's inner-CV AUROC at that penalty."""
    folds = make_folds(len(X), groups, inner_folds, seed)
    if model == "ridge":
        aucs = np.full((len(folds), len(RIDGE_GRID), Y.shape[1]), np.nan)
        for f, (tr, va) in enumerate(folds):
            if check is not None:
                check()
            path = RidgePath(X[tr], Y[tr].astype(float))
            for g, rel in enumerate(RIDGE_GRID):
                aucs[f, g] = fast_auroc(Y[va], path.predict(X[va], path.coef(rel)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_auc = np.nanmean(aucs, 0)
        best = np.nan_to_num(mean_auc, nan=-1).argmax(0)
        oof = np.zeros(Y.shape, float)
        for tr, va in folds:
            path = RidgePath(X[tr], Y[tr].astype(float))
            for g in np.unique(best):
                cols = np.flatnonzero(best == g)
                oof[np.ix_(va, cols)] = path.predict(X[va], path.coef(RIDGE_GRID[g])[:, cols], cols)
        a, b, m, s = platt(oof, Y)
        path = RidgePath(X, Y.astype(float))
        W = np.column_stack([path.coef(RIDGE_GRID[g])[:, j] for j, g in enumerate(best)])
        probs = _calibrate(path.predict(Xnew, W), a, b, m, s)
        penalty = [RIDGE_GRID[g] for g in best]
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler().fit(X)
        Xs, Xn = sc.transform(X), sc.transform(Xnew)
        mean_auc = np.full((len(C_GRID), Y.shape[1]), np.nan)
        probs, penalty = np.zeros((len(Xnew), Y.shape[1])), []
        for j in range(Y.shape[1]):
            y = Y[:, j]
            for g, C in enumerate(C_GRID):
                vals = []
                for tr, va in folds:
                    if check is not None:
                        check()
                    if y[tr].all() or not y[tr].any() or y[va].all() or not y[va].any():
                        continue
                    lr = LogisticRegression(C=C, max_iter=2000).fit(Xs[tr], y[tr])
                    vals.append(fast_auroc(y[va, None], lr.decision_function(Xs[va])[:, None])[0])
                mean_auc[g, j] = np.mean(vals) if vals else np.nan
            g = int(np.nan_to_num(mean_auc[:, j], nan=-1).argmax())
            penalty.append(C_GRID[g])
            probs[:, j] = LogisticRegression(C=C_GRID[g], max_iter=2000).fit(Xs, y).predict_proba(Xn)[:, 1]
        best = np.nan_to_num(mean_auc, nan=-1).argmax(0)
    cv_auc = mean_auc[best, np.arange(Y.shape[1])]
    return {"probs": probs, "penalty": penalty, "cv_auroc": cv_auc}


def calibration_error(y: np.ndarray, p: np.ndarray, bins: int) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    err = 0.0
    for k in range(bins):
        m = idx == k
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


class PredictParams(ToolParams):
    model: Literal["auto", "logistic", "ridge"] = Field(
        "auto", description="auto = logistic up to logistic_max_targets targets, ridge + Platt beyond")
    logistic_max_targets: int = Field(20, ge=1, le=500)
    min_train_positives: int = Field(10, ge=2, le=10_000, description="targets with fewer training positives "
                                                                      "are not modelled")
    max_targets: int = Field(300, ge=1, le=5000, description="most frequent targets modelled")
    calibration_bins: int = Field(10, ge=5, le=50)
    leakage_check: bool = Field(True, description="refit with the set-aside group-mates of test samples to "
                                                  "measure the leak")


def _run_predict(ctx, inputs, p: PredictParams) -> StepResult:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score

    feats, target = inputs["features"], inputs["target"]
    split, groups, leak = inputs.get("split"), inputs.get("groups"), inputs.get("leakage")
    rows = np.intersect1d(feats.rows, target.rows, assume_unique=True)
    if split is not None:
        rows = np.intersect1d(rows, split.rows, assume_unique=True)
    Ysp, names, ann = label_matrix(target, rows)
    rows, Ysp = rows[ann], Ysp[ann]
    X = dense(feats.take(rows)).astype(np.float64)
    s = values_on(rows, split, "")
    g = merge_groups(len(rows), values_on(rows, groups), values_on(rows, leak))
    cv_groups = g
    gnames = [a.meta.get("column", a.name) for a in (groups, leak) if a is not None]
    has_test = s is not None and (s == "test").any() and (s != "test").any()
    is_test = (s == "test") if has_test else np.zeros(len(rows), bool)
    mates = np.zeros(len(rows), bool)                      # fitting samples sharing a group with test samples
    if has_test and g is not None:
        mates = ~is_test & np.isin(g, np.unique(g[is_test]))
    train = ~is_test & ~mates
    Y = Ysp.toarray()
    pos_train = Y[train].sum(0)
    neg_train = train.sum() - pos_train
    order = np.argsort(-pos_train, kind="stable")
    use = [j for j in order if pos_train[j] >= p.min_train_positives and neg_train[j] >= p.min_train_positives]
    use = np.array(sorted(use[: p.max_targets]), int)
    if len(use) == 0:
        raise RuntimeError(f"no target has {p.min_train_positives} training positives and negatives")
    ctx.decide("targets modelled", f"{len(use)} of {len(names)}",
               f"Targets need at least {p.min_train_positives} positive and negative training samples to fit and "
               f"tune a model; at most the {p.max_targets} most frequent are modelled.",
               [("model every target", "too few positives to estimate anything")],
               {"too_rare_to_train": int(len(names) - len(use))})
    model = p.model if p.model != "auto" else ("logistic" if len(use) <= p.logistic_max_targets else "ridge")
    if p.model == "auto":
        ctx.decide("prediction model", model,
                   f"{len(use)} targets: "
                   + ("per-target logistic regression is affordable." if model == "logistic"
                      else "closed-form ridge fits all targets from one eigendecomposition per fold; Platt scaling "
                           "on out-of-fold scores gives probabilities."),
                   [("ridge" if model == "logistic" else "logistic",
                     "unnecessary approximation" if model == "logistic" else
                     f"{len(use)} separate tuned fits per fold exceed the compute budget")],
                   {"n_targets": int(len(use)), "n_train": int(train.sum()), "dims": int(X.shape[1])})
    if mates.any():
        eval_mates = np.isin(g[is_test], np.unique(g[mates]))
        ctx.decide("fit / test separation", f"{int(mates.sum()):,} fitting samples set aside",
                   f"{int(mates.sum()):,} of {int((~is_test).sum()):,} non-test samples "
                   f"({mates.sum() / (~is_test).sum():.1%}) share a {' / '.join(gnames)} group with test samples "
                   f"({eval_mates.mean():.1%} of the test samples have such a group-mate). They are left out of "
                   "fitting, so every test sample is scored by a model that never saw its group; a second fit that "
                   "keeps them measures how much the leak would inflate AUROC.",
                   [("keep them", "test scores would partly measure recognition of related samples"),
                    ("score only the test samples without group-mates",
                     f"drops {eval_mates.mean():.0%} of the evaluation set")],
                   {"n_set_aside": int(mates.sum()), "test_with_group_mates_fraction": float(eval_mates.mean())})
    design = ("fit on the non-test splits, scored once on the test split" if has_test else
              f"out-of-fold predictions from {ctx.budget.max_folds}-fold cross-validation; AUROC compares positives "
              "with negatives of the same outer fold only")
    ctx.decide("evaluation design", design,
               (f"Folds keep every {' / '.join(gnames)} group whole (samples linked by any grouping stay together); "
                if gnames else "No grouping given: folds are random; ")
               + f"penalties are chosen by {ctx.budget.inner_folds}-fold inner cross-validation on training "
                 "samples only."
               + ("" if has_test else " Each outer fold's model has its own calibration, so pooling scores across "
                  "folds would rank by those offsets as well; AUROCs are computed within folds and combined."),
               [("random folds", "related samples on both sides would inflate the scores")] if gnames else [],
               {"has_test_split": bool(has_test), "groupings": gnames})
    Yu = Y[:, use]
    check = lambda: check_time(ctx, "prediction")  # noqa: E731
    leaky_probs = None
    if has_test:
        tr, te = np.flatnonzero(train), np.flatnonzero(is_test)
        ctx.log(f"    fitting {model} for {len(use)} targets on {len(tr):,} samples ({X.shape[1]} dims)")
        fp = fit_predict(X[tr], Yu[tr], None if cv_groups is None else cv_groups[tr], X[te], model,
                         ctx.budget.inner_folds, ctx.seed, check)
        ev, probs, cv_auc = te, fp["probs"], fp["cv_auroc"]
        if mates.any() and p.leakage_check:
            wide = np.flatnonzero(~is_test)
            ctx.log(f"    leakage check: refitting with the {int(mates.sum()):,} set-aside samples")
            leaky_probs = fit_predict(X[wide], Yu[wide], cv_groups[wide], X[te], model, ctx.budget.inner_folds,
                                      ctx.seed, check)["probs"]
    else:
        ev = np.arange(len(rows))
        probs, cvs, fold_of = np.zeros(Yu.shape), [], np.zeros(len(rows), int)
        for k, (tr, te) in enumerate(make_folds(len(rows), cv_groups, ctx.budget.max_folds, ctx.seed)):
            fold_of[te] = k
            ctx.log(f"    outer fold {k + 1}: {model} for {len(use)} targets on {len(tr):,} samples")
            fp = fit_predict(X[tr], Yu[tr], None if cv_groups is None else cv_groups[tr], X[te], model,
                             ctx.budget.inner_folds, ctx.seed, check)
            probs[te] = fp["probs"]
            cvs.append(fp["cv_auroc"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cv_auc = np.nanmean(np.vstack(cvs), 0)
    crit = ctx.criteria
    Ye = Yu[ev]
    recs = []
    for j, t in enumerate(use):
        y, pr = Ye[:, j], probs[:, j]
        npos = int(y.sum())
        rec = {"target": names[t], "n_train_positives": int(pos_train[t]), "n_eval_positives": npos,
               "eval_prevalence": float(y.mean()), "cv_auroc": cv_auc[j]}
        if 0 < npos < len(y):
            auc, var = delong(y, pr) if has_test else stratified_delong(y, pr, fold_of)
            rec.update(auc_inference(auc, var, crit.ci_level))
            rec["auprc"] = float(average_precision_score(y, pr))
            rec["brier"] = float(np.mean((pr - y) ** 2))
            rec["brier_skill"] = 1 - rec["brier"] / max(1e-12, y.mean() * (1 - y.mean()))
            rec["calibration_error"] = calibration_error(y, pr, p.calibration_bins)
            if leaky_probs is not None:
                rec["auroc_with_group_mates"] = delong(y, leaky_probs[:, j])[0]
        recs.append(rec)
    table = pd.DataFrame(recs)
    for col in ("auroc", "ci_low", "ci_high", "p_vs_half", "auroc_with_group_mates"):
        if col not in table:
            table[col] = np.nan
    assessable = table["n_eval_positives"] >= crit.reliable_min_test_positives
    table["q_vs_half"] = np.nan
    table.loc[assessable, "q_vs_half"] = bh_adjust(table.loc[assessable, "p_vs_half"].to_numpy(float))
    reliable = assessable & (table["ci_low"] >= crit.reliable_min_auroc_lower) & (table["q_vs_half"] < crit.fdr)
    table["status"] = np.where(~assessable, "too rare to assess", np.where(reliable, "reliable", "not reliable"))
    table["leakage_gap"] = table["auroc_with_group_mates"] - table["auroc"]
    ctx.decide("reliability rule", f"{int(reliable.sum())} of {int(assessable.sum())} assessable targets reliable",
               f"Fixed before any result: >= {crit.reliable_min_test_positives} positives in the evaluation set, lower "
               f"{crit.ci_level:.0%} DeLong bound of AUROC >= {crit.reliable_min_auroc_lower:g}, and BH-adjusted "
               f"one-sided p (AUROC > 0.5) < {crit.fdr:g} across the {int(assessable.sum())} assessable targets.",
               [], {"too_rare_to_assess": int((~assessable).sum())})
    a = table[assessable]
    metrics = {"model": model, "evaluation": "test split" if has_test else "cross-validation",
               "n_train": int(train.sum()), "n_eval": int(len(ev)), "n_targets": int(len(use)),
               "n_too_rare_to_train": int(len(names) - len(use)), "n_assessed": int(assessable.sum()),
               "n_reliable": int(reliable.sum()), "n_not_reliable": int((assessable & ~reliable).sum()),
               "n_too_rare_to_assess": int((~assessable).sum()),
               "median_auroc": float(a["auroc"].median()) if len(a) else None,
               "cv_auroc_median": float(np.nanmedian(cv_auc)) if np.isfinite(cv_auc).any() else None,
               "median_auprc_lift": float((a["auprc"] / a["eval_prevalence"]).median()) if len(a) else None,
               "median_brier_skill": float(a["brier_skill"].median()) if len(a) else None,
               "n_set_aside": int(mates.sum()),
               "test_with_group_mates_fraction": float(eval_mates.mean()) if mates.any() else 0.0,
               "leakage_gap_median": float(a["leakage_gap"].median()) if len(a) and a["leakage_gap"].notna().any()
               else None}
    if target.kind == "labels":
        pred = probs.argmax(1)
        truth = Ye.argmax(1)
        metrics["macro_f1"] = float(f1_score(truth, pred, average="macro"))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(truth, pred))
    out = Artifact("scores", probs.astype(np.float32), rows[ev],
                   {"columns": [names[t] for t in use], "direction": "probability of the target",
                    "source": f"{model} on '{feats.name}'", "evaluation": metrics["evaluation"]})
    return StepResult(artifacts={"table": Artifact("table", table), "predictions": out}, metrics=metrics,
                      tables={"target_predictability": table.sort_values(["status", "auroc"], ascending=[True, False])})


def _predict_cost(n, p, prm, budget) -> float:
    folds = budget.inner_folds + 2
    return float(folds * (2e-9 * n * p * p + 1e-7 * n * min(300, getattr(prm, "max_targets", 300)) * 5) + 5)


register(ToolSpec(
    name="predict_targets", origin="new", source=SOURCE,
    summary="Predicts labels or multi-label targets from features (logistic or ridge + Platt), grouped "
            "cross-validation for penalties, a single test-split evaluation (or out-of-fold), DeLong AUROC "
            "intervals, the configured reliability rule, calibration and a leakage sensitivity check.",
    consumes=(Port("features", ("embedding", "normalized")),
              Port("target", ("multilabel", "labels"), doc="what to predict"),
              Port("split", ("split",), required=False, doc="predefined split; the test part is scored once"),
              Port("groups", ("groups",), required=False, doc="independent units kept whole in folds"),
              Port("leakage", ("groups",), required=False, doc="e.g. near-duplicate clusters: kept whole, and "
                                                               "kept apart from the test samples")),
    produces=(Port("table", ("table",), doc="per-target AUROC, CI, q, AUPRC, calibration, status"),
              Port("predictions", ("scores",), required=False, doc="evaluation-set probabilities")),
    params=PredictParams, answers=("predictive_modelling",),
    addresses=("class_imbalance", "near_duplicates", "split_leakage", "nested_sampling", "multiple_testing"),
    criteria=(Criterion("cv_auroc_median", "max", None, "median inner cross-validation AUROC (training data)"),
              Criterion("n_reliable", "report", selection=False, doc="targets meeting the reliability rule"),
              Criterion("median_auroc", "report", selection=False, doc="median evaluation AUROC"),
              Criterion("leakage_gap_median", "report", selection=False,
                        doc="AUROC gain when fitting also uses samples sharing a group with test samples")),
    run=_run_predict, cost=_predict_cost,
))

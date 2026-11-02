"""Batch assessment and correction.

Assessment measures how strongly a technical grouping (slide, site, run) shapes a
representation: silhouette by batch, the share of variance between batches, and the
mixing of batches among each sample's nearest neighbours. When batches are nested in a
coarser group (slides inside patients), the coarse group is biology as much as technique,
so the same measurements are repeated *within* each coarse group - slides of one patient
are expected to look alike unless there is a slide effect - and the variance is split into
between-patient, between-slide-within-patient and within-slide shares. External labels, when
given, are used only to report how batches and labels coincide (Cramer's V).

Correction compares Harmony (via the harmonypy package, a dependency), ComBat's
empirical-Bayes location/scale model (implemented here, after Johnson, Li & Rabinovic 2007)
and any embedding given as a candidate (e.g. a batch-conditioned VAE) with leaving the data
uncorrected. With nested batches each coarse group is corrected on its own, so differences
between patients are not removed as if they were batch effects. The choice is label-free:
the candidate that mixes batches best while keeping at least ``min_preservation`` of every
sample's nearest same-batch neighbours wins, and only if it improves mixing over the
uncorrected data by ``min_gain``; a candidate that rearranges samples within a batch has
over-corrected and is rejected.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from .base import Artifact, Criterion, Port, StepResult, ToolParams, ToolSpec, check_time, dense, values_on
from .registry import register
from .stats import cramers_v, knn_preservation, mixing_entropy, nested_shares, silhouette, variance_share

SOURCE = "new in Project 4 (Harmony through the harmonypy package; ComBat implemented after Johnson et al. 2007; " \
         "variance share from Project 3's profiler)"

# How strong a batch effect is, from the within-group measurements (or global ones without nesting).
SILHOUETTE_WEAK, SILHOUETTE_STRONG = 0.05, 0.15
MIXING_WEAK, MIXING_STRONG = 0.80, 0.50


def _nested(batch: np.ndarray, outer: np.ndarray | None) -> bool:
    if outer is None:
        return False
    per_batch = pd.DataFrame({"b": batch, "o": outer}).groupby("b")["o"].nunique()
    return bool((per_batch == 1).all()) and pd.Series(outer).nunique() > 1


def _within(Z, batch, outer, fn) -> float | None:
    """Mean of fn(Z_g, batch_g) over outer groups holding at least two batches, weighted by size."""
    vals, weights = [], []
    for g in np.unique(outer):
        idx = np.flatnonzero(outer == g)
        if len(np.unique(batch[idx])) >= 2:
            v = fn(Z[idx], batch[idx])
            if v is not None:
                vals.append(v)
                weights.append(len(idx))
    return float(np.average(vals, weights=weights)) if vals else None


def batch_strength(silhouette_value: float | None, mixing_value: float | None) -> str:
    s = -1.0 if silhouette_value is None else silhouette_value
    m = 1.0 if mixing_value is None else mixing_value
    if s >= SILHOUETTE_STRONG or m <= MIXING_STRONG:
        return "strong"
    if s >= SILHOUETTE_WEAK or m <= MIXING_WEAK:
        return "weak"
    return "none"


# ---------------------------------------------------------------------------
# assessment
# ---------------------------------------------------------------------------

class AssessParams(ToolParams):
    n_neighbors: int = Field(30, ge=5, le=200, description="neighbours behind the mixing entropy")


def _run_assess(ctx, inputs, p: AssessParams) -> StepResult:
    emb, batch_art, outer_art, labels = inputs["embedding"], inputs["batch"], inputs.get("outer"), inputs.get("labels")
    rows = np.intersect1d(emb.rows, batch_art.rows, assume_unique=True)
    if outer_art is not None:
        rows = np.intersect1d(rows, outer_art.rows, assume_unique=True)
    Z = dense(emb.take(rows)).astype(np.float64)
    b = np.asarray(batch_art.take(rows)).astype(str)
    o = np.asarray(outer_art.take(rows)).astype(str) if outer_art is not None else None
    nested = _nested(b, o)
    cap, rng = ctx.budget.eval_max_samples, ctx.rng(5)
    bcol = batch_art.meta.get("column", batch_art.name)
    ocol = outer_art.meta.get("column", outer_art.name) if outer_art is not None else None
    m = {"n_samples": len(rows), "n_batches": int(len(np.unique(b))), "batch_nested_in_outer": nested,
         "batch_silhouette": silhouette(Z, b, cap, rng),
         "variance_share_batch": variance_share(Z, b),
         "mixing": mixing_entropy(Z, b, p.n_neighbors, cap=cap, rng=rng)}
    if o is not None:
        m["n_outer"] = int(len(np.unique(o)))
        m["variance_share_outer"] = variance_share(Z, o)
        m["outer_silhouette"] = silhouette(Z, o, cap, rng)
    if nested:
        m.update({f"share_{k}": v for k, v in nested_shares(Z, b, o).items()})
        m["within_outer_batch_silhouette"] = _within(Z, b, o, lambda z, g: silhouette(z, g, cap, rng))
        m["within_outer_mixing"] = mixing_entropy(Z, b, p.n_neighbors, within=o, cap=cap, rng=rng)
        m["n_outer_with_several_batches"] = int((pd.DataFrame({"b": b, "o": o}).groupby("o")["b"].nunique() > 1).sum())
    sil = m.get("within_outer_batch_silhouette") if nested else m["batch_silhouette"]
    mix = m.get("within_outer_mixing") if nested else m["mixing"]
    level = batch_strength(sil, mix)
    m["batch_effect"] = level
    m["correction_warranted"] = int(level != "none")
    where = f"between {bcol} levels of the same {ocol}" if nested else f"between {bcol} levels"
    ctx.decide("batch effect", level,
               f"{'Within each ' + ocol + ': ' if nested else ''}silhouette by {bcol} "
               f"{'n/a' if sil is None else f'{sil:.3f}'} and neighbour mixing "
               f"{'n/a' if mix is None else f'{mix:.2f}'} (1 = fully mixed); thresholds: weak at silhouette >= "
               f"{SILHOUETTE_WEAK} or mixing <= {MIXING_WEAK}, strong at >= {SILHOUETTE_STRONG} or <= {MIXING_STRONG}."
               + (f" Differences between {ocol}s mix biology and technique and are not counted as batch effect."
                  if nested else ""),
               [("judge across all batches", f"{ocol} differences are biological as well as technical")]
               if nested else [],
               {"measured": where, "silhouette": sil, "mixing": mix})
    tables = {}
    if labels is not None:
        lrows = np.intersect1d(rows, labels.rows, assume_unique=True)
        pos = np.searchsorted(rows, lrows)
        y = np.asarray(labels.take(lrows)).astype(str)
        m["cramers_v_batch_label"] = cramers_v(b[pos], y)
        if o is not None:
            m["cramers_v_outer_label"] = cramers_v(o[pos], y)
        m["label_silhouette"] = silhouette(Z[pos], y, cap, rng)
        tables["label_mix_by_batch"] = (pd.crosstab(b[pos], y, normalize="index").round(3)
                                        .rename_axis(index=bcol).reset_index())
    per = pd.DataFrame({"batch": b, **({"outer": o} if o is not None else {})})
    tables["batches"] = per.groupby(list(per.columns)).size().rename("n").reset_index()
    return StepResult(artifacts={"report": Artifact("table", tables["batches"])}, metrics=m, tables=tables)


register(ToolSpec(
    name="batch_assess", origin="new", source=SOURCE,
    summary="Measures batch effects in a representation (silhouette, variance shares, neighbour mixing), within "
            "coarser groups when batches are nested, and how batches coincide with external labels.",
    consumes=(Port("embedding", ("embedding", "normalized")), Port("batch", ("groups",), doc="technical grouping"),
              Port("outer", ("groups",), required=False, doc="coarser grouping the batches may be nested in"),
              Port("labels", ("labels",), required=False, doc="external labels: confounding check, report only")),
    produces=(Port("report", ("table",), doc="batch sizes by outer group"),),
    params=AssessParams, answers=("batch_assessment",),
    addresses=("batch_effect", "batch_confounding", "nested_sampling"),
    criteria=(Criterion("batch_silhouette", "report", doc="silhouette width by batch (0 = no separation)"),
              Criterion("within_outer_batch_silhouette", "report", doc="the same within each outer group"),
              Criterion("mixing", "report", doc="neighbour batch entropy relative to the batch proportions"),
              Criterion("within_outer_mixing", "report"),
              Criterion("share_between_inner_within_outer", "report",
                        doc="variance share between batches of one outer group"),
              Criterion("cramers_v_batch_label", "report", selection=False,
                        doc="association of batch and external label")),
    run=_run_assess, cost=lambda n, p, prm, b: 5e-5 * min(n, b.eval_max_samples) * 30 + 1e-6 * n * p + 2.0,
))


# ---------------------------------------------------------------------------
# correction
# ---------------------------------------------------------------------------

def combat(X: np.ndarray, batch) -> np.ndarray:
    """ComBat (parametric empirical Bayes, no covariates): every feature is standardised, each
    batch's location and scale are shrunk towards priors shared by the features, removed, and
    the pooled scale and grand mean restored."""
    X = np.asarray(X, float)
    levels, b = np.unique(np.asarray(batch).astype(str), return_inverse=True)
    B, (n, G) = len(levels), X.shape
    if B < 2:
        return X.copy()
    nb = np.bincount(b)
    means = np.vstack([X[b == k].mean(0) for k in range(B)])
    grand = (nb / n) @ means
    var = ((X - means[b]) ** 2).mean(0)
    sd = np.sqrt(np.where(var > 0, var, 1.0))
    Z = (X - grand) / sd
    g_hat = np.vstack([Z[b == k].mean(0) for k in range(B)])
    d_hat = np.vstack([Z[b == k].var(0, ddof=1) if nb[k] > 1 else np.ones(G) for k in range(B)])
    d_hat = np.maximum(d_hat, 1e-8)
    g_star, d_star = g_hat.copy(), d_hat.copy()
    if G >= 2:
        g_bar, t2 = g_hat.mean(1), g_hat.var(1, ddof=1)
        mean_d, var_d = d_hat.mean(1), d_hat.var(1, ddof=1)
        for k in range(B):
            if t2[k] <= 0 or var_d[k] <= 0:
                continue
            a_pr = (2 * var_d[k] + mean_d[k] ** 2) / var_d[k]
            b_pr = (mean_d[k] * var_d[k] + mean_d[k] ** 3) / var_d[k]
            zk, g_old, d_old = Z[b == k], g_hat[k], d_hat[k]
            for _ in range(500):
                g_new = (nb[k] * t2[k] * g_hat[k] + d_old * g_bar[k]) / (nb[k] * t2[k] + d_old)
                d_new = (0.5 * ((zk - g_new) ** 2).sum(0) + b_pr) / (nb[k] / 2 + a_pr - 1)
                change = max(np.max(np.abs(g_new - g_old) / np.maximum(np.abs(g_old), 1e-12)),
                             np.max(np.abs(d_new - d_old) / d_old))
                g_old, d_old = g_new, d_new
                if change < 1e-4:
                    break
            g_star[k], d_star[k] = g_old, d_old
    return (Z - g_star[b]) / np.sqrt(d_star[b]) * sd + grand


def harmony(Z: np.ndarray, batch, theta: float, seed: int) -> np.ndarray:
    import harmonypy

    out = harmonypy.run_harmony(np.asarray(Z, np.float64), pd.DataFrame({"batch": np.asarray(batch).astype(str)}),
                                "batch", theta=theta, verbose=False, random_state=seed).Z_corr
    out = np.asarray(out)
    return out if out.shape[0] == len(Z) else out.T


def _per_group(fn, Z: np.ndarray, batch: np.ndarray, outer: np.ndarray | None, min_samples: int = 20) -> np.ndarray:
    """Apply a correction inside each outer group with two or more batches (all samples at once
    without an outer grouping)."""
    if outer is None:
        return fn(Z, batch)
    out = Z.copy()
    for g in np.unique(outer):
        idx = np.flatnonzero(outer == g)
        if len(np.unique(batch[idx])) >= 2 and len(idx) >= min_samples:
            out[idx] = fn(Z[idx], batch[idx])
    return out


class CorrectParams(ToolParams):
    methods: list[Literal["harmony", "combat"]] | None = Field(None, description="None = both")
    within_outer: Literal["auto", "yes", "no"] = Field(
        "auto", description="correct inside each outer group; auto = yes when batches are nested in it")
    min_preservation: float = Field(0.5, ge=0.1, le=0.95, description="share of within-batch neighbours a "
                                                                      "correction must keep")
    min_gain: float = Field(0.05, ge=0.0, le=0.5, description="mixing gain over no correction required")
    theta: float = Field(2.0, ge=0.0, le=10.0, description="Harmony diversity penalty")
    n_neighbors: int = Field(30, ge=5, le=200)


def _run_correct(ctx, inputs, p: CorrectParams) -> StepResult:
    emb, batch_art, outer_art = inputs["embedding"], inputs["batch"], inputs.get("outer")
    extra, labels = inputs.get("candidate"), inputs.get("labels")
    rows = np.intersect1d(emb.rows, batch_art.rows, assume_unique=True)
    for a in (outer_art, extra):
        if a is not None:
            rows = np.intersect1d(rows, a.rows, assume_unique=True)
    Z0 = dense(emb.take(rows)).astype(np.float64)
    b = np.asarray(batch_art.take(rows)).astype(str)
    o = np.asarray(outer_art.take(rows)).astype(str) if outer_art is not None else None
    nested = _nested(b, o)
    within = p.within_outer == "yes" or (p.within_outer == "auto" and nested)
    ocol = outer_art.meta.get("column", outer_art.name) if outer_art is not None else None
    bcol = batch_art.meta.get("column", batch_art.name)
    if o is not None:
        if within:
            why = (f"Every {bcol} belongs to one {ocol}, so {ocol} differences cannot be told apart from batch "
                   f"effects; correcting inside each {ocol} removes {bcol} effects and keeps {ocol} differences.")
            rej = [("across all samples", f"would remove real differences between {ocol}s")]
        elif nested:
            why = (f"Set by the plan: every {bcol} belongs to one {ocol}, but batches are corrected across all "
                   f"samples, so differences between {ocol}s are removed along with {bcol} effects and structure "
                   f"shared by {ocol}s comes forward.")
            rej = [(f"within each {ocol}", f"keeps {ocol} differences, which can dominate the structure")]
        else:
            why = f"{bcol} levels cross {ocol} levels, so batches are corrected jointly."
            rej = [(f"within each {ocol}", "batches are not nested")]
        ctx.decide("correction scope", f"within each {ocol}" if within else "across all samples", why, rej,
                   {"nested": nested})
    scope = o if within else None
    cands = {"none": Z0}
    for m in (p.methods or ["harmony", "combat"]):
        check_time(ctx, "batch correction")
        ctx.log(f"    batch correction candidate {m}")
        if m == "harmony":
            cands[m] = _per_group(lambda z, g: harmony(z, g, p.theta, ctx.seed), Z0, b, scope)
        else:
            cands[m] = _per_group(combat, Z0, b, scope)
    if extra is not None:
        cands[extra.name] = dense(extra.take(rows)).astype(np.float64)
    cap, k = ctx.budget.eval_max_samples, p.n_neighbors
    y = None
    if labels is not None:
        lrows = np.intersect1d(rows, labels.rows, assume_unique=True)
        lpos, y = np.searchsorted(rows, lrows), np.asarray(labels.take(lrows)).astype(str)
    table = []
    for name, Z in cands.items():
        check_time(ctx, "batch correction scoring")
        row = {"method": name,
               "mixing": mixing_entropy(Z, b, k, within=scope, cap=cap, rng=ctx.rng(6)),
               "neighbourhood_preservation": 1.0 if name == "none" else knn_preservation(Z0, Z, b, 15, rng=ctx.rng(7)),
               "batch_silhouette": (_within(Z, b, scope, lambda z, g: silhouette(z, g, cap, ctx.rng(8)))
                                    if scope is not None else silhouette(Z, b, cap, ctx.rng(8)))}
        if y is not None:
            row["label_silhouette"] = silhouette(Z[lpos], y, cap, ctx.rng(9))
        table.append(row)
    table = pd.DataFrame(table)
    base = table.iloc[0]
    ok = table[(table["method"] != "none") & (table["neighbourhood_preservation"] >= p.min_preservation)]
    best = ok.sort_values("mixing", ascending=False).iloc[0] if len(ok) else None
    if best is not None and best["mixing"] - base["mixing"] >= p.min_gain:
        chosen, why = best["method"], (f"Best batch mixing ({best['mixing']:.2f} vs {base['mixing']:.2f} "
                                       f"uncorrected) among candidates keeping >= {p.min_preservation:.0%} of "
                                       f"within-batch neighbours ({best['neighbourhood_preservation']:.2f}).")
    else:
        chosen = "none"
        why = ("No candidate improves mixing by at least "
               f"{p.min_gain:g} while keeping >= {p.min_preservation:.0%} of within-batch neighbours "
               f"(uncorrected mixing {base['mixing']:.2f}); the data are left uncorrected.")
    rejected = []
    for _, r in table.iterrows():
        if r["method"] == chosen:
            continue
        if r["method"] != "none" and r["neighbourhood_preservation"] < p.min_preservation:
            reason = f"over-corrected: keeps {r['neighbourhood_preservation']:.2f} of within-batch neighbours"
        else:
            reason = f"mixing {r['mixing']:.2f}, preservation {r['neighbourhood_preservation']:.2f}"
        rejected.append((r["method"], reason))
    ctx.decide("batch correction", chosen, why, rejected, {"table": table.round(4).to_dict("records")})
    table["chosen"] = table["method"] == chosen
    sel = table[table["chosen"]].iloc[0]
    metrics = {"method": chosen, "mixing": float(sel["mixing"]),
               "neighbourhood_preservation": float(sel["neighbourhood_preservation"]),
               "batch_silhouette": float(sel["batch_silhouette"]), "mixing_uncorrected": float(base["mixing"]),
               "batch_silhouette_uncorrected": float(base["batch_silhouette"]), "within_outer": within,
               "mixing_gain": float(sel["mixing"] - base["mixing"])}
    if o is not None:                                     # how separate the coarse groups (e.g. patients) remain
        metrics["outer_mixing"] = mixing_entropy(cands[chosen], o, k, cap=cap, rng=ctx.rng(10))
        metrics["outer_mixing_uncorrected"] = mixing_entropy(Z0, o, k, cap=cap, rng=ctx.rng(10))
    if y is not None:
        metrics["label_silhouette"] = float(sel["label_silhouette"])
        metrics["label_silhouette_uncorrected"] = float(base["label_silhouette"])
    art = Artifact("embedding", cands[chosen].astype(np.float32), rows,
                   {"method": f"{emb.meta.get('method', emb.name)} + {chosen}" if chosen != "none" else
                    emb.meta.get("method", emb.name), "source": emb.name, "batch": batch_art.name,
                    "correction": chosen, "within": outer_art.name if within and outer_art is not None else None})
    return StepResult(artifacts={"embedding": art}, metrics=metrics, tables={"correction_comparison": table})


register(ToolSpec(
    name="batch_correct", origin="new", source=SOURCE,
    summary="Compares Harmony, ComBat, an optional candidate embedding and no correction; keeps the best batch "
            "mixing that preserves within-batch neighbourhoods (label-free), inside coarse groups when nested.",
    consumes=(Port("embedding", ("embedding",), doc="representation to correct (e.g. principal components)"),
              Port("batch", ("groups",)),
              Port("outer", ("groups",), required=False, doc="coarser grouping; corrected within when nested"),
              Port("candidate", ("embedding",), required=False, doc="another corrected embedding to compare"),
              Port("labels", ("labels",), required=False, doc="external labels: silhouette reported only")),
    produces=(Port("embedding", ("embedding",), doc="the chosen representation"),),
    params=CorrectParams, answers=("batch_assessment",), addresses=("batch_effect",),
    criteria=(Criterion("mixing", "max", None, "neighbour batch mixing of the chosen representation"),
              Criterion("neighbourhood_preservation", "max", 0.5, "share of within-batch neighbours kept"),
              Criterion("batch_silhouette", "min", None, "silhouette by batch after correction"),
              Criterion("label_silhouette", "report", selection=False, doc="external labels (report only)")),
    run=_run_correct, cost=lambda n, p, prm, b: 1e-4 * n + 3e-4 * min(n, b.eval_max_samples) * 4 + 5.0,
))

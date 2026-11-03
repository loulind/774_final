"""Differential expression between groups of samples, with the replicate as the unit.

Samples (spots, cells) of one patient are not independent, so testing them as if they were
inflates significance (pseudoreplication). For each label (domain, cluster) against all
other samples, counts are summed per replicate (patient) on each side - a pseudobulk - and
converted to log2 counts per million with a library-size-scaled prior count (edgeR's
cpm(log = TRUE)). The paired per-replicate differences are tested with limma's moderated t
(variances shrunk towards a common prior, Smyth 2004), Benjamini-Hochberg within each label.
A marker needs q < FDR and |log2 fold change| >= the configured minimum.

Two checks come with every marker: the share of replicates whose difference has the same
sign as the mean, and whether leaving out any one replicate flips the sign (a marker that
hinges on one patient). For comparison, the naive analysis - a Wilcoxon rank-sum test over
all samples, treating them as independent - is run on the same features; the ratio of its
discoveries to the pseudobulk discoveries shows how much pseudoreplication would inflate the
result. When the counts are the held-out half of a count split, the labels were found on the
other half, so the test does not re-use the data that defined the groups.

Without any grouping of the samples into independent units, there is no replicate to sum
over: each label is then tested against the rest with the sample as the unit (the Wilcoxon
test above, BH within each label), and each marker's direction is checked in two random
halves of the samples instead of leaving out replicates. If the samples come from one source
(one library, donor or culture), such p-values describe that source and do not generalise;
the step says so in its decision and its metrics (``test_unit`` "sample").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pydantic import Field

from .base import Artifact, Criterion, Port, StepResult, ToolParams, ToolSpec, check_time
from .normalize import library_sizes, log_normalise
from .registry import register
from .stats import bh_adjust, column_ranks, fit_f_dist, moderated_t, rank_sum

SOURCE = "new in Project 4 (moderated t after limma's eBayes; log-CPM after edgeR)"


def log_cpm(Y: np.ndarray, prior_count: float) -> np.ndarray:
    """log2 counts per million of pseudobulk rows (samples x features), prior count scaled by
    library size as in edgeR."""
    lib = Y.sum(1).astype(float)
    prior = prior_count * lib / lib.mean()
    return np.log2((Y + prior[:, None]) / (lib + 2 * prior)[:, None] * 1e6)


class DEParams(ToolParams):
    min_samples: int = Field(10, ge=1, le=10_000, description="samples a replicate needs on each side of a "
                                                              "contrast")
    min_replicates: int = Field(3, ge=2, le=1000, description="replicates with both sides needed to test a label")
    min_count: int = Field(10, ge=1, le=10_000, description="features need this many pseudobulk counts in at "
                                                            "least half of the pseudobulk samples")
    prior_count: float = Field(2.0, ge=0.1, le=10.0)
    naive_comparison: bool = Field(True, description="also run the sample-level Wilcoxon test")
    max_labels: int = Field(40, ge=1, le=500)


def _run_de_samples(ctx, counts, labels, p: DEParams) -> StepResult:
    """No replicate grouping: every label against the rest with the sample as the unit."""
    rows = np.intersect1d(counts.rows, labels.rows, assume_unique=True)
    X = sp.csr_matrix(counts.take(rows))
    y = np.asarray(labels.take(rows)).astype(str)
    names = np.asarray(counts.meta.get("feature_names") or [f"f{j}" for j in range(X.shape[1])], dtype=object)
    fdr, min_fc = ctx.criteria.fdr, ctx.criteria.min_abs_log2fc
    half = counts.meta.get("split_half")
    ctx.decide("test unit", "sample (no replicate grouping)",
               "No grouping of the samples into independent units (patients, donors, libraries) is available, so "
               "every label is tested against the rest across samples (Wilcoxon rank-sum on log-normalised values, "
               "BH within each label). If the samples come from one source, the p-values describe that source and "
               "do not generalise to others."
               + (" The counts are the held-out half of a count split, so the labels were found on other data."
                  if half == "test" else ""),
               [("replicate pseudobulk", "needs a grouping of the samples into independent units")],
               {"split_half": half, "n_samples": int(len(rows))})
    lib = library_sizes(X)
    levels = list(pd.Series(y).value_counts().index[: p.max_labels])
    detected = np.asarray((X > 0).sum(0)).ravel()
    keep = np.flatnonzero((detected >= p.min_samples) & (np.asarray(X.sum(0)).ravel() >= p.min_count))
    tested = {lv: y == lv for lv in levels
              if p.min_samples <= (y == lv).sum() <= len(y) - p.min_samples and len(keep) >= 3}
    pvals = {lv: np.full(len(keep), np.nan) for lv in tested}
    for s in range(0, len(keep), 500):                     # ranks computed once per feature block
        check_time(ctx, "differential expression")
        block = keep[s:s + 500]
        R, tie = column_ranks(log_normalise(X[:, block], lib))
        for lv, mask in tested.items():
            pvals[lv][s:s + len(block)] = rank_sum(R, tie, mask)["p"]
    halves = ctx.rng(31).permutation(len(rows)) % 2

    def log2fc(a, b):
        cpm_a = np.asarray(X[a][:, keep].sum(0)).ravel() / max(lib[a].sum(), 1.0) * 1e6
        cpm_b = np.asarray(X[b][:, keep].sum(0)).ravel() / max(lib[b].sum(), 1.0) * 1e6
        return np.log2((cpm_a + 1) / (cpm_b + 1))

    results, summary = [], []
    for lv in levels:
        if lv not in tested:
            summary.append({"label": lv, "n_samples": int((y == lv).sum()), "status": f"skipped: fewer than "
                            f"{p.min_samples} samples on one side, or too few expressed features"})
            continue
        mask = tested[lv]
        fc = log2fc(mask, ~mask)
        stable = np.ones(len(keep), bool)
        for h in (0, 1):
            stable &= np.sign(log2fc(mask & (halves == h), ~mask & (halves == h))) == np.sign(fc)
        q = bh_adjust(pvals[lv])
        sig = (q < fdr) & (np.abs(fc) >= min_fc)
        results.append(pd.DataFrame({"label": lv, "feature": names[keep], "log2fc": fc, "p": pvals[lv], "q": q,
                                     "significant": sig, "half_stable": stable, "n_samples": int(mask.sum()),
                                     "test_unit": "sample"}))
        summary.append({"label": lv, "n_samples": int(mask.sum()), "status": "tested", "n_tested": int(len(keep)),
                        "n_markers": int(sig.sum()), "n_markers_half_stable": int((sig & stable).sum())})
    summary = pd.DataFrame(summary)
    cols = ["label", "feature", "log2fc", "p", "q", "significant", "half_stable", "n_samples", "test_unit"]
    table = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=cols)
    marks = table[table["significant"].astype(bool)]
    metrics = {"test_unit": "sample", "n_labels": len(levels), "n_tested_labels": len(results),
               "n_skipped_labels": len(levels) - len(results), "testable": bool(results),
               "n_markers": int(len(marks)), "n_features_with_marker": int(marks["feature"].nunique()),
               "half_stable_fraction": float(marks["half_stable"].mean()) if len(marks) else None,
               "loo_stable_fraction": None, "median_sign_consistency": None, "held_out": half == "test"}
    if not results:
        ctx.decide("markers", "none testable", f"No label has at least {p.min_samples} samples on both sides of "
                   "its contrast with enough expressed features.", [], {})
    top = (marks.sort_values(["label", "q", "log2fc"], ascending=[True, True, False]).groupby("label").head(10))
    return StepResult(artifacts={"table": Artifact("table", table)}, metrics=metrics,
                      tables={"markers": table, "markers_top": top, "de_summary": summary})


def _run_de(ctx, inputs, p: DEParams) -> StepResult:
    counts, labels, rep = inputs["counts"], inputs["labels"], inputs.get("replicate")
    if rep is None:
        return _run_de_samples(ctx, counts, labels, p)
    rows = np.intersect1d(np.intersect1d(counts.rows, labels.rows, assume_unique=True), rep.rows, assume_unique=True)
    X = sp.csr_matrix(counts.take(rows))
    y = np.asarray(labels.take(rows)).astype(str)
    r = np.asarray(rep.take(rows)).astype(str)
    names = np.asarray(counts.meta.get("feature_names") or [f"f{j}" for j in range(X.shape[1])], dtype=object)
    rcol = rep.meta.get("column", rep.name)
    fdr, min_fc = ctx.criteria.fdr, ctx.criteria.min_abs_log2fc
    half = counts.meta.get("split_half")
    ctx.decide("test unit", f"{rcol} (pseudobulk)",
               f"Samples of one {rcol} are not independent; counts are summed per {rcol} on each side of every "
               f"contrast and the paired {rcol}-level differences tested (moderated t, BH within each label)."
               + (" The counts are the held-out half of a count split, so the labels were found on other data."
                  if half == "test" else
                  " Note: if these labels were found on the same counts, the test re-uses the data that defined "
                  "the groups; a count split avoids this."),
               [("sample-level tests", "treat dependent samples as independent; reported as the naive comparison")],
               {"split_half": half, "n_replicates": int(len(np.unique(r)))})
    levels = list(pd.Series(y).value_counts().index[: p.max_labels])
    results, summary, kept = [], [], {}
    reps = np.unique(r)
    for lv in levels:
        check_time(ctx, "differential expression")
        pairs = []
        for rr in reps:
            a = (r == rr) & (y == lv)
            b = (r == rr) & (y != lv)
            if a.sum() >= p.min_samples and b.sum() >= p.min_samples:
                pairs.append((rr, np.asarray(X[a].sum(0)).ravel(), np.asarray(X[b].sum(0)).ravel()))
        if len(pairs) < p.min_replicates:
            summary.append({"label": lv, "n_replicates": len(pairs), "status": f"skipped: fewer than "
                            f"{p.min_replicates} {rcol}s with >= {p.min_samples} samples on both sides"})
            continue
        Y = np.vstack([pa[1] for pa in pairs] + [pa[2] for pa in pairs]).astype(float)
        keep = (Y >= p.min_count).sum(0) >= Y.shape[0] / 2
        if keep.sum() < 3:
            summary.append({"label": lv, "n_replicates": len(pairs), "status": "skipped: too few expressed features"})
            continue
        L = log_cpm(Y[:, keep], p.prior_count)
        n = len(pairs)
        D = L[:n] - L[n:]
        mean, s2 = D.mean(0), D.var(0, ddof=1)
        d0, s02 = fit_f_dist(s2, n - 1)
        mt = moderated_t(mean, s2, n, d0, s02)
        q = bh_adjust(mt["p"])
        sign = np.sign(mean)
        loo = np.ones(len(mean), bool)
        for i in range(n):
            loo &= np.sign((D.sum(0) - D[i]) / (n - 1)) == sign
        sig = (q < fdr) & (np.abs(mean) >= min_fc)
        results.append(pd.DataFrame({"label": lv, "feature": names[keep], "log2fc": mean, "t": mt["t"], "p": mt["p"],
                                     "q": q, "significant": sig, "sign_consistency": (np.sign(D) == sign).mean(0),
                                     "loo_stable": loo, "n_replicates": n}))
        kept[lv] = np.flatnonzero(keep)
        summary.append({"label": lv, "n_replicates": n, "status": "tested", "n_tested": int(keep.sum()),
                        "n_markers": int(sig.sum()), "n_markers_loo_stable": int((sig & loo).sum()),
                        "prior_df": d0, "prior_var": s02})
    if p.naive_comparison and results:
        # sample-level Wilcoxon on log-normalised values: ranks computed once per feature block
        lib_all = library_sizes(X)
        union = np.flatnonzero(np.isin(np.arange(X.shape[1]), np.concatenate(list(kept.values()))))
        naive = {lv: np.full(len(c), np.nan) for lv, c in kept.items()}
        where = {lv: {int(j): i for i, j in enumerate(c)} for lv, c in kept.items()}
        for s in range(0, len(union), 500):
            check_time(ctx, "naive comparison")
            block = union[s:s + 500]
            R, tie = column_ranks(log_normalise(X[:, block], lib_all))
            for lv in kept:
                cols = [i for i, j in enumerate(block) if int(j) in where[lv]]
                if cols:
                    pv = rank_sum(R[:, cols], tie[cols], y == lv)["p"]
                    naive[lv][[where[lv][int(block[i])] for i in cols]] = pv
        for k, res in enumerate(results):
            lv = res["label"].iloc[0]
            in_lv, cols = y == lv, kept[lv]
            mean_a = np.asarray(X[in_lv][:, cols].sum(0)).ravel() / lib_all[in_lv].sum() * 1e6
            mean_b = np.asarray(X[~in_lv][:, cols].sum(0)).ravel() / lib_all[~in_lv].sum() * 1e6
            res["naive_log2fc"] = np.log2((mean_a + 1) / (mean_b + 1))
            res["naive_p"] = naive[lv]
            res["naive_q"] = bh_adjust(naive[lv])
            res["naive_significant"] = (res["naive_q"] < fdr) & (np.abs(res["naive_log2fc"]) >= min_fc)
            for row in summary:
                if row["label"] == lv:
                    row["n_markers_naive"] = int(res["naive_significant"].sum())
    summary = pd.DataFrame(summary)
    if not results:                                       # a finding about the labels, not a failure
        ctx.decide("markers", "none testable",
                   f"No label has at least {p.min_replicates} {rcol}s with >= {p.min_samples} samples on both sides "
                   f"(most for one label: {int(summary['n_replicates'].max())}), so no {rcol}-level comparison is "
                   f"possible: the labels are confined to too few {rcol}s.",
                   [("sample-level tests", f"would treat samples of one {rcol} as independent replicates")],
                   {"replicates_per_label": dict(zip(summary["label"], summary["n_replicates"].astype(int)))})
        empty = pd.DataFrame(columns=["label", "feature", "log2fc", "t", "p", "q", "significant", "sign_consistency",
                                      "loo_stable", "n_replicates"])
        metrics = {"n_labels": len(levels), "n_tested_labels": 0, "n_skipped_labels": len(levels), "testable": False,
                   "n_markers": 0, "loo_stable_fraction": None, "median_sign_consistency": None,
                   "n_replicates_max": int(summary["n_replicates"].max()), "held_out": half == "test"}
        return StepResult(artifacts={"table": Artifact("table", empty)}, metrics=metrics,
                          tables={"markers": empty, "de_summary": summary})
    table = pd.concat(results, ignore_index=True)
    marks = table[table["significant"]]
    n_markers = int(len(marks))
    metrics = {"n_labels": len(levels), "n_tested_labels": len(results), "testable": True,
               "n_skipped_labels": int((summary["status"] != "tested").sum()),
               "n_markers": n_markers, "n_features_with_marker": int(marks["feature"].nunique()),
               "loo_stable_fraction": float(marks["loo_stable"].mean()) if n_markers else None,
               "median_sign_consistency": float(marks["sign_consistency"].median()) if n_markers else None,
               "n_replicates_min": int(summary.loc[summary["status"] == "tested", "n_replicates"].min()),
               "n_replicates_max": int(summary.loc[summary["status"] == "tested", "n_replicates"].max()),
               "held_out": half == "test"}
    if p.naive_comparison:
        metrics["n_markers_naive"] = int(table["naive_significant"].sum())
        metrics["naive_inflation"] = float(metrics["n_markers_naive"] / max(1, n_markers))
    top = (marks.sort_values(["label", "q", "log2fc"], ascending=[True, True, False]).groupby("label").head(10))
    return StepResult(artifacts={"table": Artifact("table", table)}, metrics=metrics,
                      tables={"markers": table, "markers_top": top, "de_summary": summary})


def _de_cost(n, p, prm, budget) -> float:
    return 3e-8 * n * p * (12 if prm is None or prm.naive_comparison else 1) + 2.0


register(ToolSpec(
    name="differential_expression", origin="new", source=SOURCE,
    summary="Markers of every label against the rest with the replicate (patient) as the unit: pseudobulk "
            "log-CPM, paired moderated t, BH; sign consistency, leave-one-replicate-out stability, and the naive "
            "sample-level Wilcoxon test for comparison. Without a replicate grouping: the sample-level test, BH, "
            "and a split-half check of each marker's direction.",
    consumes=(Port("counts", ("counts",), doc="counts to test (the held-out half after a count split)"),
              Port("labels", ("labels",), doc="groups to compare, one against the rest"),
              Port("replicate", ("groups",), required=False,
                   doc="independent unit (patient, donor); without one the sample is the unit")),
    produces=(Port("table", ("table",), doc="all tests: log2fc, t, p, q, flags, naive p/q"),),
    params=DEParams, answers=("differential_analysis",), addresses=("nested_sampling", "multiple_testing"),
    criteria=(Criterion("n_markers", "report", doc="label-feature pairs at the FDR and fold-change bar"),
              Criterion("loo_stable_fraction", "max", 0.8, "markers whose sign survives leaving out any one "
                                                           "replicate"),
              Criterion("median_sign_consistency", "max", 0.75, "share of replicates agreeing in sign"),
              Criterion("naive_inflation", "report", doc="naive discoveries per pseudobulk discovery")),
    run=_run_de, cost=_de_cost,
))

"""Label, grouping and split measurements: how many samples are labelled and in which
groups, class frequencies and the support each label has for modelling and judging,
agreement between two label sources, group sizes and nesting, and how far groups reach
across a predefined split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import GroupProfile, LabelAgreementProfile, LabelProfile
from ..tools.base import Artifact, label_matrix, values_on
from ..tools.quality import ImbalanceParams, label_frequencies, source_agreement
from ..tools.stats import cramers_v, variance_share
from .matrix import measurable


def label_profile(a: Artifact, split: Artifact | None, groups: list[Artifact], caches: dict[str, dict],
                  n_total: int, min_test: int) -> LabelProfile:
    table, m = label_frequencies(a, split, ImbalanceParams().min_train_positives, min_test)
    used = table[table["count"] > 0].sort_values("count", ascending=False)
    _, _, ann = label_matrix(a, a.rows)
    rows = a.rows[ann]
    rec = LabelProfile(artifact=a.name, column=str(a.meta.get("column", a.name)), kind=a.kind,
                       n_annotated=len(rows), annotated_fraction=len(rows) / max(1, n_total),
                       n_classes=int(len(used)), imbalance_ratio=m["imbalance_ratio"],
                       largest={str(k): int(v) for k, v in zip(used["label"].head(10), used["count"].head(10))},
                       smallest={str(k): int(v) for k, v in zip(used["label"].tail(5), used["count"].tail(5))},
                       n_modelable=m["n_modelable"], n_assessable=m["n_assessable"],
                       too_rare_fraction=float(1 - m["n_assessable"] / max(1, len(used))))
    if a.kind == "multilabel":
        rec.labels_per_sample_median = m["labels_per_sample_median"]
        rec.gini = m["gini"]
    for g in groups:
        v = values_on(rows, g, "")
        rec.annotated_groups[g.name] = [int(pd.Series(v[v != ""]).nunique()), int(g.meta.get("n_levels") or
                                                                             pd.Series(g.data).nunique())]
    if a.kind == "labels":
        y = np.asarray(a.take(rows)).astype(str)
        seen = []                                     # partitions of the labelled rows already measured
        meas = measurable(rows, groups)
        coarse_first = sorted(meas.items(), key=lambda kv: (int(kv[1][0].meta.get("n_levels") or 0), kv[0]))
        for name, (g, v, big) in coarse_first:
            keep = np.isin(v, list(big))
            codes = pd.factorize(np.where(keep, v, ""))[0]
            if any(np.array_equal(codes, c) for c in seen):
                continue                              # same split of the labelled samples as a coarser grouping
            seen.append(codes)
            V = cramers_v(v[keep], y[keep])
            if V is not None:
                rec.cramers_v[name] = float(V)
        for mat, cache in caches.items():
            common = np.intersect1d(cache["rows"], rows, assume_unique=True)
            if len(common) >= 10:
                pos = np.searchsorted(cache["rows"], common)
                share = variance_share(cache["scores"][pos], np.asarray(a.take(common)).astype(str))
                if share is not None:
                    rec.variance_share[mat] = float(share)
    return rec


def agreement_profiles(labels: list[Artifact]) -> list[LabelAgreementProfile]:
    """Every pair of label sources (single- or multi-label) that define at least one label in
    common; the source with the larger vocabulary is compared against the other."""
    out = []
    vocab = {a.name: set(label_matrix(a, a.rows[:1])[1]) for a in labels}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            if not vocab[a.name] & vocab[b.name]:
                continue
            first, second = (a, b) if len(vocab[a.name]) >= len(vocab[b.name]) else (b, a)
            try:
                _, m = source_agreement(first, second)
            except RuntimeError:                         # no shared label with enough positives
                continue
            out.append(LabelAgreementProfile(first=first.name, second=second.name, n_samples_both=m["n_samples_both"],
                                             n_shared_labels=m["n_shared_labels"], kappa_micro=m["kappa_micro"],
                                             precision_micro=m["precision_micro"], recall_micro=m["recall_micro"],
                                             first_source_coverage=m["first_source_coverage"]))
    return out


def group_profile(g: Artifact, split: Artifact | None) -> GroupProfile:
    v = pd.Series(np.asarray(g.data).astype(str))
    sizes = v.value_counts()
    rec = GroupProfile(artifact=g.name, column=str(g.meta.get("column", g.name)), role=str(g.meta.get("role", "group")),
                       n_levels=int(len(sizes)), size_min=int(sizes.min()), size_median=float(sizes.median()),
                       size_max=int(sizes.max()), samples_per_level=float(len(v) / max(1, len(sizes))),
                       shared_fraction=float((v.map(sizes) >= 2).mean()), nested_in=list(g.meta.get("nested_in", [])))
    s = values_on(g.rows, split, "") if split is not None else None
    if s is not None:
        vals = v.to_numpy()
        rec.split_overlap = {}
        for lv in sorted(set(s) - {""}):
            mine = s == lv
            others = set(vals[~mine & (s != "")])
            rec.split_overlap[lv] = float(np.isin(vals[mine], list(others)).mean()) if mine.any() else 0.0
    return rec

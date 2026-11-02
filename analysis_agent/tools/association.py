"""Agreement between labels the agent found (clusters, domains) and external labels.

External labels (pathologist annotations, curated classes) are used here once, for
validation; nothing the agent chooses depends on them, so every criterion is report-only.
Agreement is summarised by the adjusted Rand index, adjusted mutual information, NMI,
purity (each found group's majority label share, weighted by size) and inverse purity.
Samples of one patient or slide are not independent, so the same indices are also given
per group, with their spread across groups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field

from .base import Artifact, Criterion, Port, StepResult, ToolParams, ToolSpec, values_on
from .registry import register

SOURCE = "new in Project 4"


def agreement(found: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

    ct = pd.crosstab(found, ref).to_numpy()
    n = ct.sum()
    return {"ari": float(adjusted_rand_score(ref, found)), "ami": float(adjusted_mutual_info_score(ref, found)),
            "nmi": float(normalized_mutual_info_score(ref, found)),
            "purity": float(ct.max(1).sum() / n), "inverse_purity": float(ct.max(0).sum() / n)}


class AssociationParams(ToolParams):
    min_group_samples: int = Field(30, ge=5, le=10_000, description="groups with fewer labelled samples are "
                                                                    "left out of the per-group indices")


def _run_association(ctx, inputs, p: AssociationParams) -> StepResult:
    found_art, ref_art, groups = inputs["found"], inputs["reference"], inputs.get("groups")
    rows = np.intersect1d(found_art.rows, ref_art.rows, assume_unique=True)
    found = np.asarray(found_art.take(rows)).astype(str)
    ref = np.asarray(ref_art.take(rows)).astype(str)
    if len(rows) < 10 or len(np.unique(ref)) < 2:
        raise RuntimeError(f"too few labelled samples to compare ({len(rows)} with "
                           f"{len(np.unique(ref))} reference classes)")
    m = {"n_samples": len(rows), "n_found": int(len(np.unique(found))), "n_reference": int(len(np.unique(ref))),
         **agreement(found, ref)}
    ct = pd.crosstab(pd.Series(found, name="found"), pd.Series(ref, name="reference"))
    major = pd.DataFrame({"found": ct.index, "n": ct.sum(axis=1).to_numpy(),
                          "majority_label": ct.idxmax(axis=1).to_numpy(),
                          "majority_share": (ct.max(axis=1) / ct.sum(axis=1)).to_numpy()})
    tables = {"contingency": ct.reset_index(), "majority_label": major}
    g = values_on(rows, groups)
    if g is not None:
        per = []
        for lv in np.unique(g):
            idx = np.flatnonzero(g == lv)
            if len(idx) >= p.min_group_samples and len(np.unique(ref[idx])) >= 2:
                per.append({"group": lv, "n": len(idx), **agreement(found[idx], ref[idx])})
        if per:
            per = pd.DataFrame(per)
            tables["agreement_by_group"] = per
            m.update({"ari_group_median": float(per["ari"].median()), "ari_group_min": float(per["ari"].min()),
                      "ari_group_max": float(per["ari"].max()), "n_groups": len(per)})
    ctx.decide("validation against external labels", f"{m['n_samples']:,} labelled samples",
               f"'{ref_art.name}' is used only here, after the agent's choices, to measure agreement; it never "
               "informed them.", [], {"ari": m["ari"], "nmi": m["nmi"]})
    return StepResult(artifacts={"table": Artifact("table", major)}, metrics=m, tables=tables)


register(ToolSpec(
    name="label_association", origin="new", source=SOURCE,
    summary="Agreement of found clusters or domains with external labels (ARI, AMI, NMI, purity, contingency), "
            "overall and per group; validation only.",
    consumes=(Port("found", ("labels",), doc="labels made by the agent"),
              Port("reference", ("labels",), doc="external labels (used for validation only)"),
              Port("groups", ("groups",), required=False, doc="independent units for per-group agreement")),
    produces=(Port("table", ("table",), doc="majority external label of each found group"),),
    params=AssociationParams, answers=("structure_discovery", "cross_modal_association"),
    addresses=("partial_labels",),
    criteria=(Criterion("ari", "report", selection=False, doc="adjusted Rand index"),
              Criterion("ami", "report", selection=False, doc="adjusted mutual information"),
              Criterion("purity", "report", selection=False),
              Criterion("ari_group_median", "report", selection=False, doc="median ARI across groups")),
    run=_run_association, cost=lambda n, p, prm, b: 1e-6 * n + 1.0,
))

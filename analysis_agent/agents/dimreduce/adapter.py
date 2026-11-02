"""Project 1 as two tools: the analysis PCA and the 2-D map."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from ...tools.base import (Artifact, Criterion, Port, StepResult, ToolParams, ToolSpec, check_time,
                           dense)
from ...tools.registry import register
from . import agent
from .registry import REGISTRY

SOURCE = ("Project 1 (dimreduce_agent): methods/, evaluate.py, registry.py copied; "
          "sub-agent rules in agents/dimreduce/agent.py are new")

MapMethod = Literal["pca", "truncated_svd", "kernel_pca", "sparse_pca", "mds", "isomap", "lle",
                    "laplacian_eigenmaps", "diffusion_map", "tsne", "umap"]


# ---------------------------------------------------------------------------
# analysis PCA
# ---------------------------------------------------------------------------

class PCAParams(ToolParams):
    n_components: int | None = Field(None, ge=2, le=200,
                                     description="fixed number of components; None lets parallel "
                                                 "analysis choose")
    min_components: int = Field(5, ge=2, le=100)
    max_components: int = Field(50, ge=2, le=200)
    scale: Literal["auto", "yes", "no"] = Field(
        "auto", description="standardise features first; auto = yes for normalised matrices, no for "
                            "embeddings")
    n_permutations: int = Field(20, ge=1, le=100, description="permutations for parallel analysis "
                                                              "(capped by the budget's resampling reps)")


def _components_for(ratio: np.ndarray, share: float) -> int | None:
    cum = np.cumsum(ratio)
    hit = np.flatnonzero(cum >= share)
    return int(hit[0] + 1) if hit.size else None


def _run_pca(ctx, inputs, p: PCAParams) -> StepResult:
    a = inputs["matrix"]
    X = dense(a.data)
    prescaled = bool(a.meta.get("already_scaled"))
    scale = p.scale == "yes" or (p.scale == "auto" and a.kind == "normalized" and not prescaled)
    ctx.decide("standardise features before PCA", "yes" if scale else "no",
               "Expression-type features have very different variances; standardising (clipped at "
               "10 s.d.) lets every feature contribute." if scale else
               (f"The input ({a.meta.get('method')}) is on a common scale by construction; standardising would "
                "give noise features the weight of variable ones." if prescaled else
                "Embedding dimensions are already on a common scale; standardising would inflate "
                "low-variance noise dimensions."),
               [("no" if scale else "yes", "rejected for the reason given")],
               {"input_kind": a.kind, "setting": p.scale, "already_scaled": prescaled})
    Xs = agent.standardize(X) if scale else (X - X.mean(0)).astype(np.float32)
    lo, hi = min(p.min_components, p.max_components), max(p.min_components, p.max_components)
    hi = int(min(hi, min(Xs.shape) - 1))
    lo = int(min(lo, hi))
    k_pa = None
    if p.n_components:
        k = int(min(p.n_components, min(Xs.shape) - 1))
        ctx.decide("analysis dimension", k, "Fixed by the plan.", [], {"n_components": k})
    else:
        n_perm = int(min(p.n_permutations, ctx.budget.bootstrap_reps))
        pa = agent.parallel_analysis(Xs, hi, n_perm, ctx.seed, cap=ctx.budget.eval_max_samples)
        k_pa = pa["k"]
        k = int(min(max(k_pa, lo), hi))
        clamp = "" if k == k_pa else f"; clamped to the allowed range [{lo}, {hi}]"
        sub = agent._rows(len(Xs), ctx.budget.eval_max_samples, ctx.rng(3))
        _, ratio_hi = agent.pca_scores(Xs[sub], hi, ctx.seed)
        k90 = _components_for(ratio_hi, 0.90)
        ctx.decide("analysis dimension", k,
                   f"Parallel analysis: {k_pa} leading components have more variance than the 95th "
                   f"percentile of their permutation null ({pa['n_permutations']} permutations of "
                   f"{pa['n_used']} samples){clamp}.",
                   [("components for 90% of the variance",
                     (f"{k90} components" if k90 else f"more than {hi} components")
                     + "; past the null line the extra components are what uncorrelated noise "
                       "with the same marginals also produces")],
                   {"parallel_analysis_k": k_pa, "k_for_90pct": k90, "range": [lo, hi],
                    "observed_variance_top10": pa["observed"][:10], "null95_top10": pa["null_95"][:10]})
    Z, ratio = agent.pca_scores(Xs, k, ctx.seed)
    table = pd.DataFrame({"component": np.arange(1, len(ratio) + 1), "variance_ratio": ratio,
                          "cumulative": np.cumsum(ratio)})
    metrics = {"n_components": int(Z.shape[1]), "variance_explained": float(ratio.sum()),
               "pc1_variance": float(ratio[0]), "parallel_analysis_k": k_pa}
    art = Artifact("embedding", Z, a.rows, {"method": "pca", "source": a.name, "standardised": scale,
                                            "n_components": int(Z.shape[1])})
    return StepResult(artifacts={"embedding": art}, metrics=metrics, tables={"pca_variance": table})


def _pca_cost(n, p, prm, budget) -> float:
    k = prm.max_components if prm is not None else 50
    perms = min(getattr(prm, "n_permutations", 20), budget.bootstrap_reps)
    return 3e-9 * n * p * k + perms * 3e-9 * min(n, budget.eval_max_samples) * p * k + 1.0


register(ToolSpec(
    name="dimreduce_pca", origin="project1", source=SOURCE,
    summary="Principal components for later steps; the number of components is chosen by parallel "
            "analysis (Horn), with the 90%-variance rule as the logged alternative.",
    consumes=(Port("matrix", ("normalized", "embedding"), doc="features to reduce"),),
    produces=(Port("embedding", ("embedding",), doc="principal component scores"),),
    params=PCAParams, answers=("representation_learning",), addresses=("high_dimensionality",),
    criteria=(Criterion("variance_explained", "report", doc="share of variance in the kept components"),
              Criterion("n_components", "report", doc="analysis dimension")),
    run=_run_pca, cost=_pca_cost,
))


# ---------------------------------------------------------------------------
# 2-D map
# ---------------------------------------------------------------------------

class MapParams(ToolParams):
    methods: list[MapMethod] | None = Field(None, description="candidates to compare; None = the "
                                                              "sub-agent's family-spanning default")
    max_candidates: int = Field(4, ge=1, le=6)
    n_neighbors: int = Field(15, ge=5, le=100)
    input_dims: int = Field(50, ge=2, le=200, description="PCA-reduce wider inputs to this many "
                                                          "dimensions before mapping")


def _run_map(ctx, inputs, p: MapParams) -> StepResult:
    a = inputs["embedding"]
    X = dense(a.data)
    if X.shape[1] > p.input_dims:
        X, _ = agent.pca_scores(X - X.mean(0), p.input_dims, ctx.seed)
        ctx.decide("reduce before mapping", f"PCA to {X.shape[1]} dimensions",
                   f"The input has {a.data.shape[1]} dimensions; neighbour searches and embeddings "
                   "are faster and less noisy on leading principal components.",
                   [("map the full input", "slower, and noise dimensions blur neighbourhoods")],
                   {"input_dims": int(a.data.shape[1])})
    n, d = X.shape
    budget_s = 0.8 * max(1.0, min(ctx.seconds_left(), ctx.budget.step_seconds))
    cap = int(min(p.max_candidates, ctx.budget.max_candidates))
    cands, skipped = agent.map_candidates(n, d, budget_s, cap, p.methods)
    results = []
    for m in cands:
        check_time(ctx, "2-D map")
        ctx.log(f"    map candidate {m} on {n:,} x {d}")
        try:
            results.append(agent.fit_and_score(m, X, agent.method_params(m, n, p.n_neighbors), ctx.seed,
                                               ctx.budget.eval_max_samples))
        except Exception as err:                        # one failed candidate is reported, not fatal
            results.append({"method": m, "Z": None, "error": f"{type(err).__name__}: {str(err)[:200]}"})
    best, why = agent.choose_map(results)
    if best is None:
        raise RuntimeError(f"no 2-D map could be made: {[r.get('error') for r in results]}")

    def summary(r):
        if r.get("Z") is None:
            return f"failed ({r.get('error')})"
        m = r["metrics"]
        return (f"trustworthiness {m.get('trustworthiness@10', float('nan')):.3f}, continuity "
                f"{m.get('continuity@10', float('nan')):.3f}, stability {m.get('stability', float('nan')):.2f}")

    ctx.decide("2-D map method", best["method"], f"{why}: {summary(best)}.",
               [(r["method"], summary(r)) for r in results if r is not best] + skipped,
               {r["method"]: {k: v for k, v in r.get("metrics", {}).items() if k != "n_evaluated"}
                for r in results if r.get("Z") is not None})
    rows = []
    for r in results:
        m = r.get("metrics", {})
        rows.append({"method": r["method"], "family": REGISTRY[r["method"]].family,
                     "seconds": round(r.get("seconds", float("nan")), 1),
                     **{k: m.get(k) for k in ("trustworthiness@10", "continuity@10", "knn_overlap@10",
                                              "distance_spearman", "kruskal_stress", "stability")},
                     "chosen": r is best, "error": r.get("error"),
                     "warnings": "; ".join(r.get("warnings", [])) or None})
    bm = best["metrics"]
    metrics = {"method": best["method"], "n_candidates": len(results),
               **{k: bm.get(k) for k in ("trustworthiness@10", "continuity@10", "distance_spearman",
                                         "stability")}}
    art = Artifact("embedding", best["Z"], a.rows, {"method": best["method"], "source": a.name, "dims": 2,
                                                    "caveats": REGISTRY[best["method"]].caveats})
    return StepResult(artifacts={"map": art}, metrics=metrics, tables={"map_methods": pd.DataFrame(rows)})


def _map_cost(n, p, prm, budget) -> float:
    from .registry import estimated_seconds, feasibility

    methods = (prm.methods if prm is not None and prm.methods else list(agent.MAP_CANDIDATES))
    d = min(p, getattr(prm, "input_dims", 50))
    feas = feasibility(n, d)
    return float(sum(2 * estimated_seconds(m, n, d) for m in methods[:budget.max_candidates]
                     if m in REGISTRY and not feas[m]))


register(ToolSpec(
    name="dimreduce_map", origin="project1", source=SOURCE,
    summary="Two-dimensional map: feasible methods from different families are fitted and scored "
            "(trustworthiness, continuity, seed stability); the most faithful stable map is kept.",
    consumes=(Port("embedding", ("embedding", "normalized"), doc="representation to map"),),
    produces=(Port("map", ("embedding",), doc="two-dimensional coordinates"),),
    params=MapParams, answers=("visualization",), addresses=("high_dimensionality",),
    criteria=(Criterion("trustworthiness@10", "max", 0.80, "share of map neighbours that are true "
                                                           "neighbours (rank-weighted)"),
              Criterion("continuity@10", "max", None, "share of true neighbours kept in the map"),
              Criterion("stability", "max", agent.MIN_STABILITY, "k-NN overlap of two fits with "
                                                                  "different seeds"),
              Criterion("distance_spearman", "report", doc="rank correlation of pairwise distances")),
    run=_run_map, cost=_map_cost,
))

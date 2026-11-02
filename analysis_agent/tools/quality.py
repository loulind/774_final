"""Data-quality audits: near-duplicates, imbalance and agreement between label sources.

Near-duplicates are found two ways. Pixel level: 64-bit difference hashes (from the cached
image pass of ``qc_images``) within ``hash_max_distance`` bits, found exactly by
multi-index hashing (with h allowed differences, two hashes split into h + 1 blocks share at
least one block). Representation level: pairs of mutual nearest neighbours whose embedding
cosine is at least as high as that of the pixel-level duplicates (the 5th percentile of
their cosines), catching crops, overlays and re-scaled copies the hash misses; without
enough pixel-level pairs to calibrate it, the threshold falls back to ``fallback_cosine``.
Hashes of near-uniform images carry no information and are skipped. Pairs from either
method are joined into clusters (connected components), which later steps can use as
cross-validation groups. Clusters spanning splits are leakage; clusters spanning groups
(articles, patients) are reuse across sources.

Imbalance summarises how unevenly labels are used: counts per split, the imbalance ratio,
the Gini coefficient, the effective number of labels (exp of the entropy) and how many
labels have enough training and test samples to be assessed at all.

Label agreement compares two label sources on the samples both annotate - for every label
in both vocabularies, Cohen's kappa and the precision and recall of the first source
against the second - which estimates the label noise of the first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pydantic import Field

from .base import (Artifact, Criterion, Port, StepResult, ToolParams, ToolSpec, check_time, dense, label_matrix,
                   values_on)
from .qc import image_stats
from .registry import register
from .stats import cohen_kappa

SOURCE = "new in Project 4 (difference hash from Project 3's profiler)"


# ---------------------------------------------------------------------------
# near-duplicates
# ---------------------------------------------------------------------------

def popcount(x: np.ndarray) -> np.ndarray:
    return np.bitwise_count(x.astype(np.uint64))


def hash_pairs(h: np.ndarray, max_dist: int, max_bucket: int = 20_000) -> tuple[np.ndarray, int]:
    """All index pairs (i < j) whose 64-bit hashes differ in at most ``max_dist`` bits, by
    multi-index hashing; returns (pairs, number of skipped oversized buckets)."""
    h = h.astype(np.uint64)
    nb = max_dist + 1
    widths = [64 // nb + (1 if b < 64 % nb else 0) for b in range(nb)]
    found, skipped, shift = [], 0, 0
    for w in widths:
        key = (h >> np.uint64(shift)) & np.uint64((1 << w) - 1)
        shift += w
        order = np.argsort(key, kind="stable")
        ks = key[order]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        ends = np.r_[starts[1:], len(ks)]
        for s0, e0 in zip(starts, ends):
            m = e0 - s0
            if m < 2:
                continue
            if m > max_bucket:
                skipped += 1
                continue
            idx = order[s0:e0]
            for c0 in range(0, m - 1, 512):                 # rows of the bucket against later members
                ia = idx[c0:c0 + 512]
                close = popcount(h[ia][:, None] ^ h[idx][None, :]) <= max_dist
                close &= np.arange(m)[None, :] > (c0 + np.arange(len(ia)))[:, None]
                rr, cc = np.nonzero(close)
                if len(rr):
                    a, b = ia[rr], idx[cc]
                    found.append(np.column_stack([np.minimum(a, b), np.maximum(a, b)]))
    if not found:
        return np.zeros((0, 2), int), skipped
    return np.unique(np.vstack(found), axis=0), skipped


def mutual_neighbours(E: np.ndarray, chunk: int = 512, check=None) -> tuple[np.ndarray, np.ndarray]:
    """Each row's nearest neighbour by cosine (exact) and the cosine; E is L2-normalised."""
    nn, cos = np.zeros(len(E), int), np.zeros(len(E))
    for s in range(0, len(E), chunk):
        if check is not None:
            check()
        S = E[s:s + chunk] @ E.T
        S[np.arange(len(S)), np.arange(s, s + len(S))] = -np.inf
        nn[s:s + chunk] = S.argmax(1)
        cos[s:s + chunk] = S.max(1)
    return nn, cos


def components(n: int, pairs: np.ndarray) -> np.ndarray:
    from scipy.sparse.csgraph import connected_components

    A = sp.csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else \
        sp.csr_matrix((n, n))
    return connected_components(A, directed=False)[1]


class DuplicateParams(ToolParams):
    hash_max_distance: int = Field(4, ge=0, le=16, description="differing bits that still count as a duplicate")
    use_embedding: bool = Field(True, description="also look for representation-level duplicates")
    min_calibration_pairs: int = Field(20, ge=5, le=10_000, description="pixel-level pairs needed to set the "
                                                                         "embedding threshold")
    fallback_cosine: float = Field(0.98, ge=0.8, le=0.9999, description="embedding threshold without them")
    max_pairs_reported: int = Field(200, ge=10, le=10_000)


def _run_duplicates(ctx, inputs, p: DuplicateParams) -> StepResult:
    images, emb, split, groups = inputs["images"], inputs.get("embedding"), inputs.get("split"), inputs.get("groups")
    rows = images.rows if emb is None else np.intersect1d(images.rows, emb.rows, assume_unique=True)
    st = image_stats(images, rows, ctx.cache_dir, ctx.log, lambda: check_time(ctx, "near-duplicates"))
    h = st["dhash"].to_numpy(np.uint64)
    bits = popcount(h)
    informative = st["readable"].to_numpy() & (bits > 2) & (bits < 62)
    hi = np.flatnonzero(informative)
    hp, skipped = hash_pairs(h[hi], p.hash_max_distance)
    hp = hi[hp] if len(hp) else hp
    hp_set = {tuple(x) for x in hp.tolist()}
    tau, ep, cos_h = None, np.zeros((0, 2), int), None
    if emb is not None and p.use_embedding:
        E = dense(emb.take(rows)).astype(np.float32)
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        if len(hp):
            cos_h = np.einsum("ij,ij->i", E[hp[:, 0]], E[hp[:, 1]])
        if cos_h is not None and len(cos_h) >= p.min_calibration_pairs:
            tau, how = float(np.quantile(cos_h, 0.05)), (f"the 5th percentile of the embedding cosine of the "
                                                          f"{len(cos_h)} pixel-level duplicate pairs")
        else:
            tau, how = p.fallback_cosine, (f"fallback {p.fallback_cosine:g}: only "
                                           f"{0 if cos_h is None else len(cos_h)} pixel-level pairs to calibrate on")
        nn, cos = mutual_neighbours(E, check=lambda: check_time(ctx, "near-duplicates"))
        mutual = nn[nn] == np.arange(len(E))
        keep = mutual & (cos >= tau)
        ep = np.unique(np.sort(np.column_stack([np.flatnonzero(keep), nn[keep]]), 1), axis=0)
        ctx.decide("embedding duplicate threshold", f"cosine >= {tau:.4f}", f"Mutual nearest neighbours whose "
                   f"cosine reaches {how}.",
                   [("a fixed threshold for every encoder", "cosine scales differ by encoder")],
                   {"n_hash_pairs": len(hp), "threshold": tau})
    ep_set = {tuple(x) for x in ep.tolist()}
    pairs = np.array(sorted(hp_set | ep_set), int).reshape(-1, 2)
    comp = components(len(rows), pairs)
    sizes = np.bincount(comp)
    in_dup = sizes[comp] > 1
    cluster_id = np.array([f"D{c}" if in_dup[i] else f"S{rows[i]}" for i, c in enumerate(comp)], dtype=object)
    m = {"n_images": len(rows), "n_hash_informative": int(informative.sum()), "n_hash_pairs": len(hp_set),
         "n_embedding_pairs": len(ep_set), "n_pairs_both": len(hp_set & ep_set),
         "pair_agreement": (len(hp_set & ep_set) / len(hp_set | ep_set)) if (hp_set | ep_set) else None,
         "embedding_threshold": tau, "n_clusters": int((sizes > 1).sum()), "n_in_clusters": int(in_dup.sum()),
         "fraction_in_clusters": float(in_dup.mean()), "largest_cluster": int(sizes.max()) if len(sizes) else 0,
         "skipped_hash_buckets": skipped,
         "hash_pair_median_cosine": float(np.median(cos_h)) if cos_h is not None and len(cos_h) else None}
    s = values_on(rows, split, "")
    if s is not None:
        dup = pd.DataFrame({"c": comp[in_dup], "s": s[in_dup]})
        per = dup.groupby("c")["s"].nunique()
        m["cross_split_clusters"] = int((per > 1).sum())
        m["cross_split_cluster_fraction"] = float((per > 1).mean()) if len(per) else 0.0
        train_c = set(comp[(s == "train") | (s == "val")].tolist())
        test = s == "test"
        shared = np.array([c in train_c for c in comp]) & in_dup & test
        m["test_with_train_duplicate_fraction"] = float(shared[test].mean()) if test.any() else None
    gv = values_on(rows, groups, "")
    if gv is not None:
        per = pd.DataFrame({"c": comp[in_dup], "g": gv[in_dup]}).groupby("c")["g"].nunique()
        m["cross_group_cluster_fraction"] = float((per > 1).mean()) if len(per) else 0.0
    rep = []
    for a, b in pairs[: p.max_pairs_reported]:
        rep.append({"row_a": int(rows[a]), "row_b": int(rows[b]),
                    "hash_distance": int(popcount(np.array([h[a] ^ h[b]]))[0]),
                    "method": "both" if (a, b) in hp_set and (a, b) in ep_set else
                    ("hash" if (a, b) in hp_set else "embedding"),
                    **({"split_a": s[a], "split_b": s[b]} if s is not None else {})})
    ctx.decide("near-duplicate clusters", f"{m['n_clusters']:,} clusters, {m['n_in_clusters']:,} images",
               f"Pairs within {p.hash_max_distance} bits of difference hash"
               + (" or mutual nearest neighbours above the embedding threshold" if tau is not None else "")
               + " are joined into connected components.",
               [("exact file duplicates only", "re-encoded and resized copies would be missed")],
               {k: m[k] for k in ("n_hash_pairs", "n_embedding_pairs", "n_clusters")})
    art = Artifact("groups", cluster_id, rows, {"column": "near_duplicate_cluster", "role": "group",
                                                "n_levels": int(len(set(cluster_id))), "nested_in": [],
                                                "source": f"near-duplicate clusters of '{images.name}'"})
    return StepResult(artifacts={"clusters": art}, metrics=m, tables={"duplicate_pairs": pd.DataFrame(rep)})


register(ToolSpec(
    name="near_duplicates", origin="new", source=SOURCE,
    summary="Near-duplicate images by difference hash and, calibrated on those, by embedding cosine; clusters "
            "(usable as CV groups), cross-split leakage and cross-group reuse.",
    consumes=(Port("images", ("images",)),
              Port("embedding", ("embedding",), required=False, doc="image features for representation-level pairs"),
              Port("split", ("split",), required=False), Port("groups", ("groups",), required=False)),
    produces=(Port("clusters", ("groups",), doc="duplicate cluster of every image (singletons have their own)"),),
    params=DuplicateParams, answers=("data_quality_audit",), addresses=("near_duplicates",),
    criteria=(Criterion("fraction_in_clusters", "report", doc="share of images with a near-duplicate"),
              Criterion("test_with_train_duplicate_fraction", "report", doc="test images with a training duplicate"),
              Criterion("pair_agreement", "report", doc="Jaccard of hash and embedding pairs"),
              Criterion("cross_group_cluster_fraction", "report", doc="clusters spanning groups")),
    run=_run_duplicates, cost=lambda n, p, prm, b: 4e-3 * n + 2e-9 * n * n * max(p, 1) / 512 + 2.0,
))


# ---------------------------------------------------------------------------
# imbalance
# ---------------------------------------------------------------------------

def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


class ImbalanceParams(ToolParams):
    min_train_positives: int = Field(10, ge=1, le=10_000, description="training samples a label needs to be "
                                                                      "modelled")


def _run_imbalance(ctx, inputs, p: ImbalanceParams) -> StepResult:
    target, split = inputs["target"], inputs.get("split")
    rows = target.rows if split is None else np.intersect1d(target.rows, split.rows, assume_unique=True)
    Y, names, ann = label_matrix(target, rows)
    rows, Y = rows[ann], sp.csc_matrix(Y[ann])
    counts = np.asarray(Y.sum(0)).ravel()
    s = values_on(rows, split, "")
    table = pd.DataFrame({"label": names, "count": counts, "prevalence": counts / max(1, len(rows))})
    min_test = ctx.criteria.reliable_min_test_positives
    if s is not None:
        for lv in sorted(set(s) - {""}):
            table[f"n_{lv}"] = np.asarray(Y[np.flatnonzero(s == lv)].sum(0)).ravel()
        tr = table.get("n_train", 0) + table.get("n_val", 0)
        table["assessable"] = (tr >= p.min_train_positives) & (table.get("n_test", 0) >= min_test)
    else:
        table["assessable"] = table["count"] >= p.min_train_positives + min_test
    used = counts[counts > 0]
    share = used / used.sum() if used.sum() else used
    per_sample = np.asarray(sp.csr_matrix(Y).sum(1)).ravel()
    top = np.sort(used)[::-1]
    m = {"n_samples": len(rows), "n_labels": len(names), "n_labels_used": int((counts > 0).sum()),
         "imbalance_ratio": float(top[0] / top[-1]) if len(top) else None,
         "gini": gini(used), "effective_labels": float(np.exp(-(share * np.log(share)).sum())) if len(share) else None,
         "top10pct_share": float(top[: max(1, len(top) // 10)].sum() / max(1, top.sum())),
         "n_assessable": int(table["assessable"].sum()), "n_below_support": int((~table["assessable"]).sum()),
         "labels_per_sample_median": float(np.median(per_sample)), "unannotated_fraction": float(1 - ann.mean()),
         "multilabel": target.kind == "multilabel"}
    ctx.decide("labels assessable", f"{m['n_assessable']:,} of {len(names):,}",
               f"A label needs >= {p.min_train_positives} training positives to be modelled and >= {min_test} test "
               "positives to be judged; the others are reported as too rare to assess.", [],
               {"imbalance_ratio": m["imbalance_ratio"], "gini": m["gini"]})
    table = table.sort_values("count", ascending=False)
    return StepResult(artifacts={"table": Artifact("table", table)}, metrics=m, tables={"label_frequencies": table})


register(ToolSpec(
    name="label_imbalance", origin="new", source=SOURCE,
    summary="Label frequency audit: counts per split, imbalance ratio, Gini, effective number of labels and how "
            "many labels have the support to be assessed.",
    consumes=(Port("target", ("labels", "multilabel")), Port("split", ("split",), required=False)),
    produces=(Port("table", ("table",), doc="count, prevalence, per-split counts and assessability per label"),),
    params=ImbalanceParams, answers=("data_quality_audit",), addresses=("class_imbalance",),
    criteria=(Criterion("imbalance_ratio", "report"), Criterion("gini", "report"),
              Criterion("n_assessable", "report", doc="labels with enough training and test positives")),
    run=_run_imbalance, cost=lambda n, p, prm, b: 1e-6 * n + 1.0,
))


# ---------------------------------------------------------------------------
# agreement between label sources
# ---------------------------------------------------------------------------

class AgreementParams(ToolParams):
    min_positives: int = Field(5, ge=1, le=10_000, description="labels need this many positives in either source")


def _run_label_agreement(ctx, inputs, p: AgreementParams) -> StepResult:
    a_art, b_art = inputs["labels"], inputs["reference"]
    rows = np.intersect1d(a_art.rows, b_art.rows, assume_unique=True)
    A, an, aa = label_matrix(a_art, rows)
    B, bn, ba = label_matrix(b_art, rows)
    both = aa & ba
    A, B = A[both].toarray(), B[both].toarray()
    shared = [x for x in an if x in set(bn)]
    ia, ib = [an.index(x) for x in shared], [bn.index(x) for x in shared]
    recs = []
    for name, i, j in zip(shared, ia, ib):
        x, y = A[:, i], B[:, j]
        if x.sum() + y.sum() < p.min_positives:
            continue
        tp = int((x & y).sum())
        recs.append({"label": name, "n_first": int(x.sum()), "n_reference": int(y.sum()), "both": tp,
                     "kappa": cohen_kappa(x, y), "precision": tp / x.sum() if x.sum() else np.nan,
                     "recall": tp / y.sum() if y.sum() else np.nan})
    table = pd.DataFrame(recs)
    if table.empty:
        raise RuntimeError("the two label sources share no label with enough positives on co-annotated samples")
    Xa, Xb = A[:, ia].ravel(), B[:, ib].ravel()
    m = {"n_samples_both": int(both.sum()), "n_shared_labels": len(table), "kappa_micro": cohen_kappa(Xa, Xb),
         "kappa_median": float(table["kappa"].median()),
         "precision_micro": float((Xa & Xb).sum() / max(1, Xa.sum())),
         "recall_micro": float((Xa & Xb).sum() / max(1, Xb.sum())),
         "disagreement_rate": float((Xa != Xb).mean())}
    ctx.decide("label noise estimate", f"kappa {m['kappa_micro']:.2f} on {m['n_shared_labels']} shared labels",
               f"'{a_art.name}' is compared with '{b_art.name}' on the {m['n_samples_both']:,} samples both annotate; "
               "precision and recall treat the second source as the reference.", [], {})
    return StepResult(artifacts={"table": Artifact("table", table)}, metrics=m,
                      tables={"label_agreement": table.sort_values("kappa")})


register(ToolSpec(
    name="label_agreement", origin="new", source=SOURCE,
    summary="Agreement between two label sources on co-annotated samples (Cohen's kappa, precision and recall per "
            "shared label): an estimate of label noise.",
    consumes=(Port("labels", ("labels", "multilabel"), doc="source whose noise is estimated"),
              Port("reference", ("labels", "multilabel"), doc="second (e.g. curated) source")),
    produces=(Port("table", ("table",), doc="kappa, precision, recall per shared label"),),
    params=AgreementParams, answers=("data_quality_audit",), addresses=("label_noise",),
    criteria=(Criterion("kappa_micro", "report", doc="kappa over all (sample, shared label) pairs"),
              Criterion("kappa_median", "report"), Criterion("precision_micro", "report"),
              Criterion("recall_micro", "report")),
    run=_run_label_agreement, cost=lambda n, p, prm, b: 1e-6 * n + 1.0,
))

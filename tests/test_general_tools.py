"""Known answers for the general tools on synthetic data with planted structure."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from PIL import Image
from sklearn.metrics import adjusted_rand_score

from analysis_agent.config import load_config
from analysis_agent.data.dataset import ImageSource
from analysis_agent.tools import registry
from analysis_agent.tools.base import Artifact, StepContext, Workspace, run_step
from analysis_agent.tools.ingest import parse_multilabel
from analysis_agent.tools.normalize import variable_features

GENERAL = {"qc_counts", "qc_images", "qc_text", "caption_label_consistency", "normalize_counts", "count_split",
           "batch_assess", "batch_correct", "cluster", "spatial_domains", "spatial_svg", "spatial_enrichment",
           "label_association", "differential_expression", "predict_targets", "near_duplicates",
           "label_imbalance", "label_agreement", "label_consistency"}


def run(tool, arts, params=None, quick=True):
    """Run one tool on fresh artifacts; returns (result, {output port: artifact})."""
    n = max(int(a.rows.max()) + 1 for a in arts.values() if a.rows is not None and len(a.rows))
    ws = Workspace(n)
    for port, a in arts.items():
        ws.put(a, f"{a.kind}.in_{port}")
    cfg = load_config(quick=quick)
    ctx = StepContext(step_id="s", seed=0, budget=cfg.budget("cpu"), criteria=cfg.criteria, quick=quick,
                      log=lambda m: None)
    r = run_step(registry.get(tool), ws, ctx, {port: f"{a.kind}.in_{port}" for port, a in arts.items()}, params)
    return r, {port: ws.get(name) for port, name in r.outputs.items()}


def art(kind, data, meta=None, rows=None):
    n = data.shape[0] if hasattr(data, "shape") else len(data)
    return Artifact(kind, data, np.arange(n) if rows is None else rows, meta or {})


def test_registry_has_the_general_tools():
    tools = registry.load()
    assert GENERAL <= set(tools)
    assert all(tools[t].origin == "new" for t in GENERAL)


# ---------------------------------------------------------------------------
# counts: QC, normalisation, splitting
# ---------------------------------------------------------------------------

def test_qc_counts_uses_per_library_thresholds():
    rng = np.random.default_rng(0)
    lib = np.repeat(["a", "b"], 300)
    base = np.where(lib == "a", 40.0, 10.0) * rng.lognormal(0, 0.25, 600)
    base[:5] = 8.0                                     # low for library a, ordinary for library b
    X = rng.poisson(base[:, None] * rng.uniform(0.5, 1.5, (1, 40)))
    X[5] = 0
    X = np.hstack([X, np.zeros((600, 1), int), (np.arange(600) < 3)[:, None].astype(int)])
    names = [f"g{j}" for j in range(42)]
    r, out = run("qc_counts", {"counts": art("counts", sp.csr_matrix(X), {"feature_names": names}),
                               "library": art("groups", lib, {"column": "slide"})})
    mask = out["mask"].data
    assert r.metrics["per_library_thresholds"] and not mask[:6].any()
    assert (~mask[300:]).sum() <= 3 and r.metrics["n_features_dropped"] == 2
    assert out["counts"].n == int(mask.sum()) and len(out["counts"].meta["feature_names"]) == 40


def test_normalisation_selects_features_within_batches_and_removes_depth():
    rng = np.random.default_rng(1)
    n, p = 600, 300
    batch = np.repeat(["b1", "b2"], 300)
    size = rng.lognormal(0, 0.5, n)
    mu = np.full((n, p), 5.0)
    state = rng.random(n) < 0.5
    mu[:, :5] *= np.where(state, 4.0, 0.5)[:, None]          # variable within every batch
    mu[:, 5] *= np.where(batch == "b2", 8.0, 1.0)             # differs only between batches
    X = sp.csr_matrix(rng.poisson(mu * size[:, None]))
    _, aware = variable_features(X, batch, 10, 100.0)
    _, pooled = variable_features(X, None, 10, 100.0)
    rank = lambda t: t.set_index("feature")["median_rank"]    # noqa: E731
    assert rank(aware)[:5].max() <= 5 and rank(aware)[5] > 10 and rank(pooled)[5] <= 6
    r, out = run("normalize_counts", {"counts": art("counts", X, {"feature_names": [f"g{j}" for j in range(p)]}),
                                      "batch": art("groups", batch, {"column": "batch"})}, {"n_features": 200})
    assert r.metrics["depth_corr_pc1_after"] < r.metrics["depth_corr_pc1_before"]
    assert out["normalized"].data.shape == (n, 200) and "g5" not in out["normalized"].meta["feature_names"][:5]


def _half_correlation(A, B):
    A, B = A.toarray().astype(float), B.toarray().astype(float)
    return float(np.mean([np.corrcoef(A[:, j], B[:, j])[0, 1] for j in range(A.shape[1])]))


def test_count_split_halves_are_independent():
    rng = np.random.default_rng(2)
    pois = sp.csr_matrix(rng.poisson(5.0, (2000, 20)))
    r, out = run("count_split", {"counts": art("counts", pois)})
    assert r.metrics["method"] == "poisson" and ((out["fit"].data + out["test"].data) != pois).nnz == 0
    assert abs(_half_correlation(out["fit"].data, out["test"].data)) < 0.05
    nb = sp.csr_matrix(rng.negative_binomial(1.0, 1 / 11, (2000, 20)))      # mean 10, phi 1
    r, out = run("count_split", {"counts": art("counts", nb)})
    assert r.metrics["method"] == "negative_binomial"
    assert abs(_half_correlation(out["fit"].data, out["test"].data)) < 0.08
    _, binom = run("count_split", {"counts": art("counts", nb)}, {"method": "poisson"})
    assert _half_correlation(binom["fit"].data, binom["test"].data) > 0.6


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

def _nested_batches(shift, seed=3):
    rng = np.random.default_rng(seed)
    patient = np.repeat(["p1", "p2", "p3"], 400)
    slide = np.repeat([f"s{i}" for i in range(6)], 200)
    Z = rng.normal(size=(1200, 8))
    Z[:, 0] += np.repeat([0.0, 4.0, 8.0], 400)                               # patients differ (biology)
    Z[:, 1] += (rng.random(1200) < 0.5) * 3.0                                # two states in every slide
    Z[np.isin(slide, ["s1", "s3", "s5"]), 2:4] += shift                      # slide effect within patient
    return {"embedding": art("embedding", Z), "batch": art("groups", slide, {"column": "slide"}),
            "outer": art("groups", patient, {"column": "patient"})}


def test_batch_assessment_measures_slide_effects_within_patients():
    r, _ = run("batch_assess", _nested_batches(2.5))
    assert r.metrics["batch_nested_in_outer"] and r.metrics["correction_warranted"] == 1
    assert r.metrics["share_between_outer"] > r.metrics["share_between_inner_within_outer"] > 0.05
    clean, _ = run("batch_assess", _nested_batches(0.0))
    assert clean.metrics["batch_effect"] == "none"


def test_batch_correction_within_patients_keeps_patient_differences():
    from analysis_agent.tools.stats import variance_share

    arts = _nested_batches(2.5)
    r, out = run("batch_correct", arts)
    assert r.metrics["method"] in ("harmony", "combat") and r.metrics["within_outer"]
    assert r.metrics["mixing_gain"] >= 0.05 and r.metrics["neighbourhood_preservation"] >= 0.5
    assert variance_share(out["embedding"].data[:, :1].astype(float), arts["outer"].data) > 0.8
    assert r.metrics["outer_mixing"] < 0.5                                   # patients stay apart
    clean, _ = run("batch_correct", _nested_batches(0.0))
    assert clean.metrics["method"] == "none"


# ---------------------------------------------------------------------------
# clusters and space
# ---------------------------------------------------------------------------

def test_cluster_stability_recovers_the_planted_number_of_clusters():
    rng = np.random.default_rng(4)
    lab = np.repeat(np.arange(4), 150)
    Z = (rng.normal(size=(4, 6)) * 6)[lab] + rng.normal(size=(600, 6))
    side = np.where(lab < 2, "g1", "g2").astype(object)                      # clusters 0-1 in g1, 2-3 in g2
    r, out = run("cluster", {"embedding": art("embedding", Z), "groups": art("groups", side, {"column": "site"})},
                 {"methods": ["kmeans", "gmm"], "k_max": 8}, quick=False)
    assert r.metrics["k"] == 4 and r.metrics["stability"] >= 0.75 and r.metrics["k_at_edge"] == 0
    assert r.metrics["separated"] and adjusted_rand_score(lab, out["labels"].data) > 0.95
    assert r.metrics["cramers_v_groups"] > 0.99 and r.metrics["clusters_in_one_group"] == 4


def _stripes(seed=5, side=24):
    rng = np.random.default_rng(seed)
    xy = np.array([(i, j) for i in range(side) for j in range(side)], float)
    xy = np.vstack([xy, xy])
    lib = np.repeat(["L1", "L2"], side * side)
    truth = (xy[:, 0] // (side / 3)).astype(int)
    return rng, xy, lib, truth


def test_spatial_domains_recover_planted_regions():
    rng, xy, lib, truth = _stripes()
    Z = (rng.normal(size=(3, 5)) * 1.2)[truth] + rng.normal(size=(len(xy), 5))
    arts = {"embedding": art("embedding", Z), "coords": art("coords", xy, {"library": "groups.in_library"}),
            "library": art("groups", lib, {"column": "slide"})}
    r, out = run("spatial_domains", arts, {"methods": ["kmeans"], "k_max": 6}, quick=False)
    assert r.metrics["k"] == 3 and adjusted_rand_score(truth, out["labels"].data) > 0.8
    assert r.metrics["spatial_coherence"] > r.metrics["spatial_coherence_own_features"]
    with pytest.raises(ValueError, match="per library"):
        run("spatial_domains", {k: v for k, v in arts.items() if k != "library"})


def test_spatial_svg_finds_the_planted_pattern_and_is_calibrated():
    rng, xy, lib, _ = _stripes(6, side=20)
    n = len(xy)
    F = rng.normal(size=(n, 120))
    F[:, 0] += np.sin(xy[:, 0] / 4) * 2                                      # spatial pattern
    F[:, 1] += (lib == "L2") * 5                                             # library offset only
    names = [f"f{j}" for j in range(120)]
    r, out = run("spatial_svg", {"matrix": art("normalized", F, {"feature_names": names}),
                                 "coords": art("coords", xy, {"library": "groups.in_library"}),
                                 "library": art("groups", lib)})
    t = out["table"].data.set_index("feature")
    assert t.loc["f0", "q"] < 1e-6 and t.loc["f1", "q"] > 0.05
    assert (t.drop(["f0", "f1"])["q"] < 0.05).mean() <= 0.03
    assert r.metrics["permuted_p05_fraction"] <= 0.12


def test_neighbourhood_enrichment_and_label_association():
    _, xy, lib, truth = _stripes(7)
    labels = np.array(["A", "B", "C"], dtype=object)[truth]
    r, out = run("spatial_enrichment", {"labels": art("labels", labels), "coords": art("coords", xy),
                                        "library": art("groups", lib)})
    t = out["table"].data
    assert (t.loc[t["label_a"] == t["label_b"], "z"] > 3).all() and r.metrics["n_enriched"] >= 3
    assert t.loc[(t["label_a"] == "A") & (t["label_b"] == "C"), "edges"].iloc[0] == 0      # not adjacent
    found = np.array(["x", "y", "z"], dtype=object)[truth]
    r, _ = run("label_association", {"found": art("labels", found, {"external": False}),
                                     "reference": art("labels", labels, {"external": True}),
                                     "groups": art("groups", lib)})
    assert r.metrics["ari"] == pytest.approx(1.0) and r.metrics["purity"] == 1.0 and r.metrics["n_groups"] == 2


# ---------------------------------------------------------------------------
# differential expression: pseudobulk is calibrated, the naive test is not
# ---------------------------------------------------------------------------

def test_pseudobulk_is_calibrated_where_the_naive_test_is_inflated():
    rng = np.random.default_rng(8)
    n_pat, per, genes, true = 6, 60, 300, 10
    patient = np.repeat([f"P{i}" for i in range(n_pat)], 2 * per)
    label = np.tile(np.repeat(["A", "B"], per), n_pat).astype(object)
    mu = rng.lognormal(np.log(10), 0.5, genes)
    effect = rng.normal(0, 0.4, (n_pat, 2, genes))                            # patient-specific, no average
    effect[:, 0, :true] += np.log(4.0)                                        # true markers of A: 4-fold
    pi = np.repeat(np.arange(n_pat), 2 * per)
    li = (label == "B").astype(int)
    X = rng.poisson(rng.uniform(0.5, 1.5, (len(label), 1)) * mu * np.exp(effect[pi, li]))
    names = [f"g{j}" for j in range(genes)]
    r, out = run("differential_expression", {"counts": art("counts", sp.csr_matrix(X), {"feature_names": names}),
                                             "labels": art("labels", label), "replicate": art("groups", patient,
                                                                                              {"column": "patient"})})
    t = out["table"].data
    a = t[t["label"] == "A"].set_index("feature")
    null = a.drop(names[:true])
    assert (null["p"] < 0.05).mean() <= 0.10 and (null["naive_p"] < 0.05).mean() >= 0.5
    assert a.loc[names[:true], "significant"].sum() >= 8 and null["significant"].sum() <= 2
    assert a.loc[names[:true], "loo_stable"].all() and r.metrics["n_replicates_min"] == n_pat


def test_labels_confined_to_single_replicates_are_reported_as_untestable():
    rng = np.random.default_rng(12)
    patient = np.repeat(["P0", "P1", "P2", "P3"], 50)
    label = np.where(patient == "P0", "A", "B").astype(object)              # each label inside whole patients
    r, out = run("differential_expression", {"counts": art("counts", sp.csr_matrix(rng.poisson(5.0, (200, 30)))),
                                             "labels": art("labels", label), "replicate": art("groups", patient)})
    assert r.metrics["testable"] is False and r.metrics["n_markers"] == 0 and out["table"].data.empty


# ---------------------------------------------------------------------------
# prediction: reliability rule and leakage
# ---------------------------------------------------------------------------

def _prediction_data(seed=9):
    rng = np.random.default_rng(seed)
    n = 1500
    X = rng.normal(size=(n, 12))
    strong = X[:, 0] + 0.3 * rng.normal(size=n) > 0
    noise = rng.random(n) < 0.3
    rare = X[:, 1] > 2.4
    split = rng.choice(["train", "val", "test"], n, p=[0.6, 0.2, 0.2]).astype(object)
    group = np.array([f"g{i}" for i in range(n)], dtype=object)
    test = np.flatnonzero(split == "test")[:100]
    src = np.flatnonzero(split == "train")[:100]
    X[test], group[test] = X[src], group[src]                                # planted copies across splits
    strong[test], noise[test], rare[test] = strong[src], noise[src], rare[src]
    codes = [";".join(c for c, f in (("STRONG", s), ("NOISE", z), ("RARE", q)) if f) or "OTHER"
             for s, z, q in zip(strong, noise, rare)]
    M, vocab, ann = parse_multilabel(pd.Series(codes))
    return {"features": art("embedding", X), "target": art("multilabel", M, {"labels": vocab, "annotated": ann}),
            "split": art("split", split), "groups": art("groups", group, {"column": "article"})}


@pytest.mark.parametrize("model", ["ridge", "logistic"])
def test_prediction_applies_the_reliability_rule_and_measures_leakage(model):
    r, out = run("predict_targets", _prediction_data(), {"model": model})
    t = out["table"].data.set_index("target")
    assert t.loc["STRONG", "status"] == "reliable" and t.loc["STRONG", "ci_low"] > 0.9
    assert t.loc["NOISE", "status"] == "not reliable"
    assert "RARE" not in t.index or t.loc["RARE", "status"] == "too rare to assess"
    assert r.metrics["evaluation"] == "test split" and r.metrics["n_set_aside"] == 100       # the planted copies
    assert 0.25 < r.metrics["test_with_group_mates_fraction"] < 0.45
    assert np.isfinite(t.loc["STRONG", "leakage_gap"])
    assert out["predictions"].data.shape == (r.metrics["n_eval"], len(t))


# ---------------------------------------------------------------------------
# images: QC and planted near-duplicates
# ---------------------------------------------------------------------------

class ArraySource(ImageSource):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def load(self, i):
        if self.images[i] is None:
            raise OSError("unreadable")
        return self.images[i]


def _smooth(rng, side=64):
    return Image.fromarray((rng.random((8, 8)) * 255).astype(np.uint8)).resize((side, side), Image.BILINEAR)


def test_image_qc_flags_degenerate_images():
    rng = np.random.default_rng(10)
    ims = [_smooth(rng) for _ in range(20)] + [Image.new("L", (64, 64), 128),
                                              Image.fromarray((rng.random((10, 10)) * 255).astype(np.uint8)),
                                              _smooth(rng).resize((320, 40)), None]
    r, out = run("qc_images", {"images": Artifact("images", ArraySource(ims), np.arange(len(ims)), {})})
    m = r.metrics
    assert (m["n_blank"], m["n_tiny"], m["n_extreme_aspect"], m["n_unreadable"]) == (1, 1, 1, 1)
    assert out["mask"].data[:20].all() and not out["mask"].data[20:].any() and out["images"].n == 20


def test_wide_hash_confirms_only_real_copies():
    from analysis_agent.tools.qc import WIDE_HASH
    from analysis_agent.tools.quality import pixel_pairs

    rng = np.random.default_rng(12)
    h = rng.integers(0, 2 ** 62, 4).astype(np.uint64) | np.uint64(0xFFFF)   # informative 64-bit hashes
    h[1], h[3] = h[0], h[2]                                                  # two 64-bit matches ...
    wide = rng.integers(-2 ** 62, 2 ** 62, (4, 4))
    wide[1] = wide[0] ^ 0b111                                               # ... one 3 bits apart at 256 bits
    wide[3] = wide[2] ^ ((1 << 40) - 1)                                     # ... one 40 bits apart
    st = pd.DataFrame({"readable": True, "dhash": h, **{c: wide[:, k] for k, c in enumerate(WIDE_HASH)}})
    pairs, _, _, n_candidates = pixel_pairs(st, 4, 16)
    assert n_candidates == 2 and pairs.tolist() == [[0, 1]]


def test_near_duplicates_finds_planted_copies():
    rng = np.random.default_rng(11)
    originals = [_smooth(rng) for _ in range(60)]
    copies, src = [], []
    for i in range(25):
        im = originals[i]
        if i < 10:
            c = im.copy()
        elif i < 20:
            arr = np.asarray(im, np.int16) + rng.integers(-2, 3, (64, 64))
            c = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        else:
            c = im.resize((96, 96), Image.BILINEAR)
        copies.append(c)
        src.append(i)
    ims = originals + copies
    thumbs = np.array([np.asarray(im.resize((8, 8)), float).ravel() for im in ims])
    split = np.array(["train"] * 60 + ["test"] * 25, dtype=object)
    r, out = run("near_duplicates", {"images": Artifact("images", ArraySource(ims), np.arange(85), {}),
                                     "embedding": art("embedding", thumbs - thumbs.mean(0)),
                                     "split": art("split", split)})
    cl = out["clusters"].data
    assert all(cl[60 + k] == cl[i] for k, i in enumerate(src))
    assert len(set(cl[:60])) == 60                                            # no false pairs among originals
    assert r.metrics["n_clusters"] == 25 and r.metrics["test_with_train_duplicate_fraction"] == 1.0


# ---------------------------------------------------------------------------
# text and labels
# ---------------------------------------------------------------------------

def _captions():
    texts = (["x-ray shows fracture of the femur"] * 40 + ["no fracture seen on this view"] * 10
             + ["normal chest radiograph"] * 5 + ["ct of the abdomen with contrast"] * 100
             + ["ct of the abdomen and fracture of the radius"] * 10 + [""] * 2 + ["ct"]
             + ["(a) ct of the abdomen, panel b mri"])
    labels = (["FRACTURE"] * 55 + ["ABDOMEN"] * 110 + ["ABDOMEN"] * 4)
    curated = (["FRACTURE"] * 40 + ["OTHER"] * 15 + ["ABDOMEN"] * 114)
    return np.array(texts, dtype=object), labels, curated


def _multilabel(codes):
    M, vocab, ann = parse_multilabel(pd.Series(codes))
    return art("multilabel", M, {"labels": vocab, "annotated": ann})


def test_text_qc_flags():
    texts, _, _ = _captions()
    r, out = run("qc_text", {"text": art("text", texts)})
    m = r.metrics
    assert m["fraction_empty"] == pytest.approx(2 / len(texts)) and m["fraction_panel_reference"] > 0
    assert m["fraction_too_short"] == pytest.approx(1 / len(texts)) and out["text"].n == len(texts) - 3
    assert m["n_boilerplate_texts"] >= 3


def test_caption_label_consistency_finds_negated_and_unsupported_labels():
    texts, labels, curated = _captions()
    r, out = run("caption_label_consistency", {"text": art("text", texts), "labels": _multilabel(labels),
                                               "reference": _multilabel(curated)})
    lex = out["flags"].data
    assert lex[40:50, 1].sum() == 10                                          # "no fracture seen": negated
    assert lex[50:55, 0].sum() == 5                                           # "normal chest radiograph"
    assert lex[155:165, 2].sum() == 10 and lex[155:167, 0].sum() == 0        # fracture named, label absent
    assert r.metrics["n_texts_without_words"] == 2
    assert r.metrics["reference_absent_given_flagged"] == 1.0
    assert r.metrics["reference_absent_given_supported"] == 0.0


def test_label_imbalance_and_agreement():
    codes = ["A;B"] * 50 + ["A"] * 40 + ["C"] * 10
    split = np.array((["train"] * 4 + ["test"]) * 20, dtype=object)
    r, out = run("label_imbalance", {"target": _multilabel(codes), "split": art("split", split)})
    t = out["table"].data.set_index("label")
    assert t.loc["A", "count"] == 90 and r.metrics["imbalance_ratio"] == 9.0
    assert t.loc["C", "n_test"] == 2 and not t.loc["C", "assessable"]
    first = ["A"] * 20 + ["B"] * 5 + ["A"] * 10 + ["B"] * 15
    ref = ["A"] * 25 + ["B"] * 25
    r, _ = run("label_agreement", {"labels": art("labels", np.array(first, dtype=object)),
                                   "reference": art("labels", np.array(ref, dtype=object))})
    assert r.metrics["kappa_micro"] == pytest.approx(0.4)


def test_markers_without_replicates_use_the_sample():
    rng = np.random.default_rng(3)
    y = np.repeat(np.array(["A", "B", "C"], dtype=object), 100)
    mu = np.full((300, 60), 5.0)
    mu[y == "A", :5] *= 6                                 # five planted markers of A
    counts = art("counts", sp.csr_matrix(rng.poisson(mu)), {"feature_names": [f"g{j}" for j in range(60)]})
    r, out = run("differential_expression", {"counts": counts, "labels": art("labels", y)})
    m, t = r.metrics, out["table"].data
    assert m["test_unit"] == "sample" and m["testable"] and m["loo_stable_fraction"] is None
    assert set(t[(t["label"] == "A") & t["significant"]]["feature"]) == {f"g{j}" for j in range(5)}
    assert m["half_stable_fraction"] == 1.0 and not m["held_out"]


def test_label_consistency_within_links():
    # ten linked triplets (five labelled A, five B) and ten unlinked samples, which are not judged
    links = np.array([f"D{i // 3}" for i in range(30)] + [f"s{i}" for i in range(10)], dtype=object)
    labels = np.array(["A" if (i // 3) % 2 else "B" for i in range(30)] + ["A"] * 10, dtype=object)
    r, out = run("label_consistency", {"labels": art("labels", labels), "links": art("groups", links)})
    assert r.metrics["n_linked_groups"] == 10 and r.metrics["n_linked_samples"] == 30
    assert r.metrics["kappa"] == pytest.approx(1.0) and r.metrics["disagreeing_fraction"] == 0.0
    labels[0] = "A"                                       # one triplet now disagrees: 56 of 60 pairs agree
    r, _ = run("label_consistency", {"labels": art("labels", labels), "links": art("groups", links)})
    chance = (16 / 30) ** 2 + (14 / 30) ** 2
    assert r.metrics["observed_agreement"] == pytest.approx(56 / 60)
    assert r.metrics["kappa"] == pytest.approx((56 / 60 - chance) / (1 - chance))
    assert r.metrics["disagreeing_fraction"] == pytest.approx(0.1)

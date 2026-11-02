"""The Project 1-3 sub-agents behind the common tool interface: known answers on synthetic data.

Encoders are replaced by small fakes (no downloads): an "informative" image tower whose features
are pooled colours, a noise tower, and a bag-of-words text tower.
"""

import zlib

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from analysis_agent.config import load_config
from analysis_agent.data.dataset import ImageSource
from analysis_agent.tools import registry
from analysis_agent.tools.base import Artifact, StepContext, Workspace, run_step


def _ctx(step="s1", **kw):
    cfg = load_config(quick=True)
    return StepContext(step_id=step, seed=0, budget=cfg.budget("cpu"), criteria=cfg.criteria, quick=True,
                       log=lambda m: None, **kw)


def _ws(n, **arts):
    ws = Workspace(n)
    for name, a in arts.items():
        ws.put(a, name)
    return ws


# ---------------------------------------------------------------------------
# Project 1: dimension reduction
# ---------------------------------------------------------------------------

def _clusters(n=600, p=40, k=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    centres = rng.normal(0, 4, (k, p))
    return (centres[y] + rng.normal(0, 1, (n, p))).astype(np.float32), y


def test_parallel_analysis_finds_the_planted_dimension():
    from analysis_agent.agents.dimreduce import agent

    rng = np.random.default_rng(1)
    X = rng.normal(size=(800, 4)) @ rng.normal(0, 2, (4, 30)) + rng.normal(size=(800, 30))
    pa = agent.parallel_analysis(agent.standardize(X.astype(np.float32)), 20, 10, seed=0)
    assert pa["k"] == 4


def test_pca_and_map_tools_decide_and_report():
    X, _ = _clusters()
    ws = _ws(len(X), **{"normalized.x": Artifact("normalized", X, np.arange(len(X)), {"feature_names": None})})
    r = run_step(registry.get("dimreduce_pca"), ws, _ctx("pca"), {"matrix": "normalized.x"})
    Z = ws.get(r.outputs["embedding"]).data
    assert 5 <= Z.shape[1] <= 50 and r.metrics["parallel_analysis_k"] == 2     # 3 clusters span 2 dims
    assert any(d.decision == "analysis dimension" for d in r.decisions)
    m = run_step(registry.get("dimreduce_map"), ws, _ctx("map"), {"embedding": r.outputs["embedding"]},
                 {"methods": ["pca", "laplacian_eigenmaps"]})
    assert ws.get(m.outputs["map"]).data.shape == (len(X), 2)
    assert set(m.tables["map_methods"]["method"]) == {"pca", "laplacian_eigenmaps"}
    assert m.criterion("trustworthiness@10").passed


# ---------------------------------------------------------------------------
# Project 2: generative models
# ---------------------------------------------------------------------------

def test_batch_vae_is_declined_without_counts_or_samples():
    from analysis_agent.agents.generative import agent

    assert not agent.warrant_batch_vae(5000, False, {"a": 2500, "b": 2500})[0]      # not counts
    assert not agent.warrant_batch_vae(100, True, {"a": 50, "b": 50})[0]             # too few samples
    assert not agent.warrant_batch_vae(5000, True, {"a": 5000})[0]                   # one batch
    assert agent.warrant_batch_vae(5000, True, {"a": 2500, "b": 2500})[0]


def test_density_flags_planted_outliers():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(1500, 8)).astype(np.float32)
    out = rng.choice(len(Z), 20, replace=False)
    Z[out] += 8
    ws = _ws(len(Z), **{"embedding.e": Artifact("embedding", Z, np.arange(len(Z)))})
    r = run_step(registry.get("generative_density"), ws, _ctx(), {"features": "embedding.e"})
    flags = ws.get(r.outputs["flags"]).data
    others = np.setdiff1d(np.arange(len(Z)), out)
    assert flags[out].mean() >= 0.9 and flags[others].mean() < 0.03


def test_density_flags_are_comparable_across_splits():
    """No outliers, many dimensions, few training samples: a model scoring its own training samples
    would make them look typical; cross-fitting keeps every split near the nominal 1%."""
    rng = np.random.default_rng(6)
    n = 900
    Z = rng.normal(size=(n, 150)).astype(np.float32)
    split = np.array(["train"] * 600 + ["val"] * 150 + ["test"] * 150, object)
    ws = _ws(n, **{"embedding.e": Artifact("embedding", Z, np.arange(n)),
                   "split.s": Artifact("split", split, np.arange(n))})
    r = run_step(registry.get("generative_density"), ws, _ctx(), {"features": "embedding.e", "split": "split.s"},
                 {"model": "gaussian"})
    for part in ("train", "val", "test"):
        assert r.metrics[f"flagged_fraction_{part}"] < 0.05
    tail = ws.get(r.outputs["scores"]).data[:, 1]
    assert 0.35 < np.median(tail[:600]) < 0.65 and 0.35 < np.median(tail[750:]) < 0.65


def test_imputation_beats_the_mean_on_correlated_columns():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(400, 1))
    df = pd.DataFrame(z + 0.2 * rng.normal(size=(400, 4)), columns=list("abcd"))
    df["grade"] = np.where(z[:, 0] > 0, "high", "low")
    df = df.mask(rng.random(df.shape) < 0.15)
    ws = _ws(len(df), **{"tabular.t": Artifact("tabular", df, np.arange(len(df)))})
    r = run_step(registry.get("generative_impute"), ws, _ctx(), {"table": "tabular.t"})
    comp = r.tables["imputation_comparison"].set_index("method")
    assert r.metrics["method"] != "mean_mode"
    assert comp.loc[r.metrics["method"], "masked_rmse"] < comp.loc["mean_mode", "masked_rmse"]
    assert not ws.get(r.outputs["imputed"]).data.isna().any().any()


def test_batch_vae_removes_a_batch_shift():
    """Two cell types x two batches; the batch multiplies a third of the genes. The conditioned
    VAE's latent should separate batches less than PCA of log counts does."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(4)
    n, p = 800, 120
    cell, batch = rng.integers(0, 2, n), rng.integers(0, 2, n)
    mu = np.exp(rng.normal(0.5, 1, (2, p)))
    shift = np.ones((2, p))
    shift[1, : p // 3] = 4.0
    X = rng.poisson(mu[cell] * shift[batch] * rng.uniform(0.5, 1.5, (n, 1))).astype(np.int32)
    ws = _ws(n, **{"counts.c": Artifact("counts", sp.csr_matrix(X), np.arange(n)),
                   "groups.b": Artifact("groups", batch.astype(str), np.arange(n), {"column": "batch"})})
    ctx = _ctx()
    ctx.budget = ctx.budget.model_copy(update={"train_seconds_per_fit": 60, "max_epochs": 150})
    r = run_step(registry.get("generative_batch_vae"), ws, ctx, {"counts": "counts.c", "batch": "groups.b"},
                 {"max_epochs": 150})
    latent = ws.get(r.outputs["embedding"]).data
    pca = PCA(latent.shape[1]).fit_transform(np.log1p(X / X.sum(1, keepdims=True) * 1e4))
    assert silhouette_score(latent, batch) < silhouette_score(pca, batch)
    assert silhouette_score(latent, cell) > 0.2


# ---------------------------------------------------------------------------
# Project 3: multimodal (fake encoders)
# ---------------------------------------------------------------------------

CLASS_RGB = np.array([[200, 60, 60], [60, 200, 60], [60, 60, 200]])
WORDS = [["red", "lesion"], ["green", "nodule"], ["blue", "mass"]]


class ColourImages(ImageSource):
    kind = "files"

    def __init__(self, rgb):
        self.rgb = rgb

    def __len__(self):
        return len(self.rgb)

    def load(self, i):
        from PIL import Image

        return Image.fromarray(np.broadcast_to(self.rgb[i], (24, 24, 3)).astype(np.uint8))


class FakeImageTower:
    input_size, mean, std, dim = 16, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25), 12

    def __init__(self, name, informative):
        from analysis_agent.agents.multimodal.encoders.registry import REGISTRY

        self.info, self.device, self.informative = REGISTRY[name], "cpu", informative
        self.rng = np.random.default_rng(zlib.crc32(name.encode()))

    def encode(self, x):
        import torch

        if self.informative:
            return torch.nn.functional.adaptive_avg_pool2d(x, 2).flatten(1)
        return torch.as_tensor(self.rng.normal(size=(len(x), self.dim)), dtype=torch.float32)


class FakeTextTower:
    dim = 24

    def __init__(self, name):
        from analysis_agent.agents.multimodal.encoders.registry import REGISTRY

        self.info, self.device = REGISTRY[name], "cpu"

    def tokenize(self, texts, max_tokens):
        return texts

    def encode(self, texts):
        import torch

        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for w in t.split():
                out[i, zlib.crc32(w.encode()) % self.dim] += 1
        return torch.as_tensor(out)


@pytest.fixture
def fake_encoders(monkeypatch):
    from analysis_agent.agents.multimodal.encoders import pretrained

    def load(name, device):
        kinds = pretrained.REGISTRY[name].kinds
        return FakeTextTower(name) if "text" in kinds else FakeImageTower(name, name == "resnet18")

    monkeypatch.setattr(pretrained, "load_encoder", load)


@pytest.fixture
def image_text():
    """900 coloured images in 3 classes from 6 patients; 15% of captions belong to another class."""
    rng = np.random.default_rng(5)
    n = 900
    y = rng.integers(0, 3, n)
    rgb = np.clip(CLASS_RGB[y] + rng.normal(0, 15, (n, 3)), 0, 255)
    noisy = rng.random(n) < 0.15
    cap_class = np.where(noisy, (y + rng.integers(1, 3, n)) % 3, y)
    filler = ["image", "shows", "a", "the", "scan", "region"]
    captions = [" ".join(WORDS[c] + list(rng.choice(filler, 2))) for c in cap_class]
    rows = np.arange(n)
    ws = _ws(n, **{"images.im": Artifact("images", ColourImages(rgb), rows),
                   "text.caption": Artifact("text", np.array(captions, object), rows),
                   "labels.y": Artifact("labels", np.array([f"c{c}" for c in y], object), rows,
                                        {"external": True}),
                   "groups.patient": Artifact("groups", np.array([f"P{i % 6}" for i in rows], object), rows,
                                              {"column": "patient"})})
    return ws, y, noisy


def test_screening_prefers_the_informative_encoder(fake_encoders, image_text):
    ws, _, _ = image_text
    r = run_step(registry.get("multimodal_features"), ws, _ctx(),
                 {"images": "images.im", "text": "text.caption", "holdout": "groups.patient"})
    assert r.metrics["image_encoder"] == "resnet18" and r.metrics["text_encoder"] == "minilm"
    assert r.metrics["screen_clear_of_chance"]
    scores = r.tables["encoder_screening"].set_index("image_encoder")["alignment_recall_at_10"]
    assert scores["resnet18"] > scores["dinov2_small"] + 0.05
    assert ws.get(r.outputs["image_features"]).data.shape == (900, FakeImageTower.dim)
    assert ws.get(r.outputs["text_features"]).meta["encoder"] == "minilm"
    assert {"image encoder", "text encoder", "colour normalisation"} <= {d.decision for d in r.decisions}


def test_inconclusive_screening_falls_back_to_the_prior(monkeypatch, image_text):
    from analysis_agent.agents.multimodal.encoders import pretrained

    monkeypatch.setattr(pretrained, "load_encoder", lambda name, device: (
        FakeTextTower(name) if "text" in pretrained.REGISTRY[name].kinds else FakeImageTower(name, False)))
    ws, _, _ = image_text
    r = run_step(registry.get("multimodal_features"), ws, _ctx(), {"images": "images.im", "text": "text.caption"})
    assert r.metrics["screen_clear_of_chance"] is False
    assert r.metrics["image_encoder"] == "dinov2_small"                   # first in the registry's order
    assert "cannot tell the candidates apart" in next(d.rationale for d in r.decisions
                                                       if d.decision == "image encoder")


def test_predictability_is_grouped_and_beats_chance(fake_encoders, image_text):
    ws, _, _ = image_text
    f = run_step(registry.get("multimodal_features"), ws, _ctx(),
                 {"images": "images.im", "text": "text.caption"}, {"image_encoders": ["resnet18"]})
    r = run_step(registry.get("multimodal_predictability"), ws, _ctx("pred"),
                 {"target": "labels.y", "groups": "groups.patient", "image": f.outputs["image_features"],
                  "other": f.outputs["text_features"]})
    assert r.metrics["balanced_accuracy_image"] > 0.9 and r.metrics["above_chance_image"]
    assert r.metrics["n_folds"] == 2 and r.metrics["chance_balanced_accuracy"] == pytest.approx(1 / 3)
    assert set(r.tables["predictability"]["features"]) == {"image", "other", "both"}
    assert not r.criterion("balanced_accuracy_image").selection          # external labels never select


def test_agreement_flags_mismatched_captions(fake_encoders, image_text):
    ws, _, noisy = image_text
    f = run_step(registry.get("multimodal_features"), ws, _ctx(),
                 {"images": "images.im", "text": "text.caption"}, {"image_encoders": ["resnet18"]})
    r = run_step(registry.get("multimodal_agreement"), ws, _ctx("agree"),
                 {"image": f.outputs["image_features"], "text": f.outputs["text_features"],
                  "groups": "groups.patient"})
    flags = ws.get(r.outputs["flags"]).data
    assert r.metrics["cross_fitted"] and r.metrics["val_recall_at_10"] > r.metrics["chance_recall_at_10"]
    assert flags[noisy].mean() > 0.6 and flags[~noisy].mean() < 0.15
    assert "zero_shot" not in set(r.tables["aligner_comparison"]["aligner"])   # not a dual encoder


def test_holdout_keeps_groups_apart_and_never_returns_test():
    from analysis_agent.agents.multimodal import agent

    rng = np.random.default_rng(0)
    g = np.array([f"G{i % 7}" for i in range(300)], object)
    tr, va, how = agent.holdout(300, None, g, rng)
    assert not set(g[tr]) & set(g[va]) and "groups" in how
    split = np.array(["train", "val", "test"] * 100, object)
    tr, va, _ = agent.holdout(300, split, g, rng)
    assert set(split[tr]) == {"train"} and set(split[va]) == {"val"}


def test_colour_normalisation_rule():
    from analysis_agent.agents.multimodal import agent

    assert agent.colour_normalisation({"domain_hint": "stained_tissue", "colour_between_group_share": 0.4},
                                      "auto", True)[0]
    assert not agent.colour_normalisation({"domain_hint": "stained_tissue", "colour_between_group_share": 0.1},
                                          "auto", True)[0]
    assert not agent.colour_normalisation({"domain_hint": "greyscale_imaging",
                                           "colour_between_group_share": 0.9}, "auto", True)[0]
    assert not agent.colour_normalisation({"domain_hint": "stained_tissue", "colour_between_group_share": 0.9},
                                          "auto", False)[0]


def test_image_summary_measures_groups_and_domain(spatial_dir):
    from analysis_agent.agents.multimodal import imaging
    from analysis_agent.data.loaders import load_path

    ds = load_path(spatial_dir)
    src = ds.modalities["image"].images
    slides = ds.obs["slide"].astype(str).to_numpy()
    s = imaging.image_summary(src, np.arange(ds.n), slides, np.random.default_rng(0))
    assert s["domain_hint"] == "stained_tissue" and s["greyscale_fraction"] == 0
    assert s["n_sampled"] == ds.n

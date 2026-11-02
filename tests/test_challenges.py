"""The profile and the challenge register on synthetic data: planted challenges are found at the
right severity, and the clean versions of the same designs raise no false alarms."""

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from analysis_agent.challenges import RULES, SEVERITY_ORDER, assess, render_markdown
from analysis_agent.config import REPO_ROOT, load_config
from analysis_agent.data.loaders import load_path
from analysis_agent.profiler import profile_dataset
from analysis_agent.schema import json_schemas

SLIDES = [("s1", "P1"), ("s2", "P1"), ("s3", "P2"), ("s4", "P2"), ("s5", "P3"), ("s6", "P3")]
WORDS = ("left right upper lower lesion mass nodule cyst fluid wall margin contrast axial coronal sagittal "
         "liver lung kidney bone joint vessel artery vein chest abdomen pelvis spine skull brain heart "
         "small large dense faint early late acute chronic").split()


def _profile(path, tmp):
    cfg = load_config(quick=True)
    return profile_dataset(load_path(path), config=cfg, budget=cfg.budget("cpu"), cache_dir=tmp, log=lambda m: None)


def _tissue(d, planted: bool):
    """Six slides of three patients on 20 x 20 grids. Planted: three stripe domains, depth that
    varies by spot and by slide, a slide effect inside every patient, 6% failed spots, labels on
    one slide per patient and a rare class. Clean: none of these."""
    import anndata as ad

    rng = np.random.default_rng(0)
    grid = np.array([(i, j) for i in range(20) for j in range(20)], float) * 100
    G = 200
    base = rng.lognormal(np.log(5), 0.5, G)
    programs = rng.normal(0, 1, (3, G))
    obs, X, xy = [], [], []
    for k, (slide, patient) in enumerate(SLIDES):
        stripe = (grid[:, 0] // 700).astype(int)
        mu = np.tile(base, (len(grid), 1))
        if planted:
            mu *= np.exp(0.8 * programs[stripe])
            if k % 2:
                mu[:, :40] *= 3.0                                      # slide effect within the patient
            size = np.exp(rng.normal(0, 0.8, len(grid))) * (4.0 if k % 2 else 1.0)
        else:
            size = np.exp(rng.normal(0, 0.05, len(grid)))
        lam = mu * size[:, None]
        if planted:
            lam[rng.random(len(grid)) < 0.06] *= 0.02                   # failed spots
        X.append(rng.poisson(lam))
        xy.append(grid)
        if planted:
            lab = np.array(["tumour", "stroma", "stroma"], dtype=object)[stripe]
            lab[(stripe == 2) & (rng.random(len(grid)) < 0.12)] = "immune"   # a rare class
            if k % 2:
                lab[:] = None                                         # one labelled slide per patient
        else:
            lab = rng.choice(["tumour", "stroma", "immune"], len(grid)).astype(object)
        obs += [{"slide": slide, "patient": patient, "label": v} for v in lab]
    obs = pd.DataFrame(obs, index=[f"spot{i}" for i in range(len(obs))])
    for c in obs:
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=sp.csr_matrix(np.vstack(X).astype(np.int32)), obs=obs,
                       var=pd.DataFrame(index=[f"g{j}" for j in range(G)]))
    adata.obsm["spatial"] = np.vstack(xy)
    adata.uns["spatial"] = {s: {"scalefactors": {}} for s, _ in SLIDES}
    path = d / "spots.h5ad"
    adata.write_h5ad(path)
    return path


def _table(d, planted: bool):
    """400 image-caption pairs in a 60/20/20 split, two label sources and articles of 2 images.
    Planted: copied images (some across the split), articles shared by validation and test,
    empty / one-word / repeated captions, a second label source that disagrees on a quarter of
    the labels, a long tail of labels, colour and greyscale images of mixed sizes, missing images.
    Clean: none of these."""
    from PIL import Image

    rng = np.random.default_rng(1)
    (d / "images").mkdir()
    n = 400
    split = np.array((["train"] * 3 + ["valid", "test"]) * (n // 5), dtype=object)
    article = np.array([f"A{i // 2}" for i in range(n)], dtype=object)
    order = np.argsort(split, kind="stable")                          # articles inside one split ...
    article = article[np.argsort(order)]
    if planted:                                                        # ... except 40 straddling val/test
        val, test = np.flatnonzero(split == "valid"), np.flatnonzero(split == "test")
        article[test[:40]] = article[val[:40]]
    rows = []
    for i in range(n):
        side = (32, 32)
        if planted and i % 4 == 0:
            side = (64, 48) if i % 8 == 0 else (40, 120)
        arr = rng.integers(0, 255, side[::-1], dtype=np.uint8)
        im = Image.fromarray(arr)
        if planted and i % 3 == 0:
            im = Image.fromarray(rng.integers(0, 255, (*side[::-1], 3), dtype=np.uint8))
        im.save(d / "images" / f"im{i}.png")
        a, b = ("CT scan", "MR image") if i % 2 else ("X-ray", "Ultrasound")
        rows.append({"id": f"im{i}", "image_path": f"images/im{i}.png",
                     "caption": f"{a} of the " + " ".join(rng.choice(WORDS, 6)),
                     "split": split[i], "article": article[i],
                     "concepts": a if not planted else ";".join([a] + [f"rare{(i * 7) % 90}"] * (i % 2)),
                     "concepts_manual": a})
    t = pd.DataFrame(rows)
    if planted:
        from shutil import copyfile

        train, test, val = (np.flatnonzero(split == s) for s in ("train", "test", "valid"))
        copies = list(zip(train[:30], train[100:130])) + list(zip(train[30:40], test[50:60]))   # 10 cross the split
        for src, dst in copies:
            copyfile(d / "images" / f"im{src}.png", d / "images" / f"im{dst}.png")
        bad = rng.choice(n, 40, replace=False)
        t.loc[bad[:15], "caption"] = ""
        t.loc[bad[15:30], "caption"] = "figure"
        t.loc[bad[30:], "caption"] = "see text for details of this image"
        flip = rng.random(n) < 0.25
        t.loc[flip, "concepts_manual"] = np.where(t.loc[flip, "concepts"].str.startswith("CT"), "MR image", "CT scan")
        t.loc[val[10:30], "image_path"] = None
    t.to_csv(d / "pairs.csv", index=False)
    return d


@pytest.fixture(scope="module")
def registers(tmp_path_factory):
    out = {}
    for name, make in (("tissue", _tissue), ("table", _table)):
        for planted in (True, False):
            d = tmp_path_factory.mktemp(f"{name}_{planted}")
            prof = _profile(make(d, planted), tmp_path_factory.mktemp("cache"))
            out[name, planted] = (prof, assess(prof), d)
    return out


def _sev(reg, kind, subject=None):
    hits = [c for c in reg.of_type(kind) if subject is None or subject in c.subject]
    assert hits, f"no {kind} entry"
    return max((c.severity for c in hits), key=SEVERITY_ORDER.get)


def _at_least(reg, kind, level, subject=None):
    got = _sev(reg, kind, subject)
    assert SEVERITY_ORDER[got] >= SEVERITY_ORDER[level], f"{kind}: {got} < {level}"


def test_planted_tissue_challenges_are_found(registers):
    prof, reg, _ = registers["tissue", True]
    _at_least(reg, "depth_variation", "high")
    _at_least(reg, "low_quality_samples", "medium", "counts.expression")
    _at_least(reg, "batch_effect", "medium", "groups.slide")
    _at_least(reg, "spatial_autocorrelation", "high")
    _at_least(reg, "nested_sampling", "high")
    assert _sev(reg, "partial_labels") == "medium"                     # half the samples labelled
    assert _sev(reg, "class_imbalance") in ("medium", "high")
    g = next(x for x in prof.group_structure if x.group == "groups.slide")
    assert g.outer == "groups.patient" and g.within_outer_silhouette > RULES["group_silhouette"].medium
    lab = prof.labels[0]
    assert lab.annotated_groups["groups.slide"] == [3, 6] and lab.annotated_fraction == pytest.approx(0.5)


def test_clean_tissue_raises_no_false_alarms(registers):
    prof, reg, _ = registers["tissue", False]
    for kind in ("depth_variation", "low_quality_samples", "batch_effect", "spatial_autocorrelation",
                 "partial_labels", "class_imbalance", "batch_confounding", "high_dimensionality", "missing_data"):
        assert _sev(reg, kind) == "none", (kind, [c.summary for c in reg.of_type(kind)])
    _at_least(reg, "nested_sampling", "high")                          # the design is still nested


def test_planted_table_challenges_are_found(registers):
    prof, reg, _ = registers["table", True]
    _at_least(reg, "near_duplicates", "high")
    _at_least(reg, "split_leakage", "high", "groups.article")
    _at_least(reg, "text_noise", "medium")
    _at_least(reg, "label_noise", "medium")
    _at_least(reg, "class_imbalance", "high", "multilabel.concepts")
    _at_least(reg, "image_heterogeneity", "medium")
    _at_least(reg, "missing_data", "low")
    im = prof.images[0]
    assert im.n_duplicate_pairs >= 40 and im.cross_split_duplicate_pairs >= 10
    assert prof.groups[0].split_overlap["test"] == pytest.approx(0.5)


def test_clean_table_raises_no_false_alarms(registers):
    _, reg, _ = registers["table", False]
    for kind in ("near_duplicates", "split_leakage", "text_noise", "label_noise", "class_imbalance",
                 "image_heterogeneity", "missing_data", "low_quality_samples"):
        assert _sev(reg, kind) == "none", (kind, [c.summary for c in reg.of_type(kind)])


def test_relevance_follows_the_questions(registers):
    prof, _, _ = registers["tissue", True]
    reg = assess(prof, ["predictive_modelling"])
    assert reg.get("double_dipping").status == "not_applicable"      # nothing is discovered then tested
    assert reg.of_type("spatial_autocorrelation")[0].relevant         # it threatens random splits
    assert not reg.of_type("depth_variation")[0].relevant
    both = assess(prof, ["structure_discovery", "differential_analysis"])
    assert both.of_type("double_dipping")[0].severity == "high"
    assert "count_split" in both.of_type("double_dipping")[0].mitigations


def test_every_challenge_type_is_checked_and_the_outputs_are_clean(registers):
    from analysis_agent.taxonomy import CHALLENGE_TYPES

    for prof, reg, d in registers.values():
        assert {c.type for c in reg.entries} == set(CHALLENGE_TYPES)
        assert len({c.id for c in reg.entries}) == len(reg.entries)
        text = prof.model_dump_json() + reg.model_dump_json()
        assert str(d) not in text and "/" + d.name not in text           # no locations reach the profile
        md = render_markdown(reg)
        assert md.count("\n| **") == len(reg.detected("low", relevant_only=False))


def test_schema_files_match_the_models():
    for name, schema in json_schemas().items():
        path = REPO_ROOT / "schemas" / f"{name}.schema.json"
        assert json.loads(path.read_text()) == schema, f"run `bioagent schema --write` ({name})"

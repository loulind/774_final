"""Shared synthetic datasets: a spatial AnnData with painted spots and an image-text table."""

import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # tests never reach the model hub

# --- synthetic spatial AnnData: 4 slides, 3 patients, painted spots -------------------

COLOURS = {"tumour": (120, 40, 140), "stroma": (230, 120, 170)}


@pytest.fixture(scope="session")
def spatial_dir(tmp_path_factory):
    import anndata as ad
    from PIL import Image

    d = tmp_path_factory.mktemp("spatial")
    (d / "img").mkdir()
    rng = np.random.default_rng(0)
    obs, xy, counts, uns = [], [], [], {}
    for s, patient in [("s1", "P1"), ("s2", "P1"), ("s3", "P2"), ("s4", "P3")]:
        img = np.full((300, 400, 3), 255, np.uint8)
        for gx in range(6):
            for gy in range(4):
                x, y = 40 + 60 * gx, 40 + 60 * gy
                lab = "tumour" if gx < 3 else "stroma"
                img[y - 15:y + 15, x - 15:x + 15] = COLOURS[lab]
                xy.append((x, y))
                smooth = 5 + 4 * np.sin(gx / 2)
                counts.append(rng.poisson([50 * smooth, 50, *rng.uniform(1, 30, 18)]))
                obs.append({"slide": s, "patient": patient,
                            "label": "undetermined" if (gx, gy) == (0, 0) else lab})
        Image.fromarray(img).save(d / "img" / f"{s}.png")
        uns[s] = {"scalefactors": {"spot_diameter_fullres": 30.0},
                  "metadata": {"source_image_path": f"img/{s}.png"}}
    obs = pd.DataFrame(obs, index=[f"spot{i}" for i in range(len(obs))])
    for c in obs:
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=sp.csr_matrix(np.array(counts, np.int32)), obs=obs,
                       var=pd.DataFrame(index=["smooth", "flat"] + [f"g{i}" for i in range(18)]))
    adata.obsm["spatial"] = np.array(xy, float)
    adata.uns["spatial"] = uns
    adata.write_h5ad(d / "spots.h5ad")
    return d


# --- synthetic image-text table: two label sources, articles as groups ------------------

@pytest.fixture(scope="session")
def table_dir(tmp_path_factory):
    from PIL import Image

    d = tmp_path_factory.mktemp("table")
    (d / "images").mkdir()
    rng = np.random.default_rng(1)
    rows = []
    for i in range(60):
        arr = rng.integers(0, 255, (24, 32), dtype=np.uint8)
        Image.fromarray(arr).save(d / "images" / f"im{i}.png")
        mod = ["CT scan", "MR image"][i % 2]
        rows.append({"id": f"im{i}", "image_path": f"images/im{i}.png",
                     "caption": f"{mod} of patient {i} showing a lesion in region {i % 7}",
                     "split": ["train", "train", "valid", "test"][i % 4],
                     "concepts": f"{mod};Chest" if i % 3 else f"{mod};Abdomen;Lesion",
                     "concepts_manual": f"{mod};Chest" if i % 4 == 1 else (mod if i % 5 else ""),
                     "article": f"PMC{i // 3}"})
    pd.DataFrame(rows).to_csv(d / "pairs.csv", index=False)
    return d


# --- planted-challenge and clean synthetic datasets, profiled once per session ------------------

SLIDES = [("s1", "P1"), ("s2", "P1"), ("s3", "P2"), ("s4", "P2"), ("s5", "P3"), ("s6", "P3")]
WORDS = ("left right upper lower lesion mass nodule cyst fluid wall margin contrast axial coronal sagittal "
         "liver lung kidney bone joint vessel artery vein chest abdomen pelvis spine skull brain heart "
         "small large dense faint early late acute chronic").split()


def _profile(path, tmp):
    from analysis_agent.config import load_config
    from analysis_agent.data.loaders import load_path
    from analysis_agent.profiler import profile_dataset

    cfg = load_config(quick=True)
    return profile_dataset(load_path(path), config=cfg, budget=cfg.budget("cpu"), cache_dir=tmp, log=lambda m: None)


def _tissue(d, planted: bool, side: int = 20):
    """Six slides of three patients on 20 x 20 grids. Planted: three stripe domains, depth that
    varies by spot and by slide, a slide effect inside every patient, 6% failed spots, labels on
    one slide per patient and a rare class. Clean: none of these."""
    import anndata as ad

    rng = np.random.default_rng(0)
    grid = np.array([(i, j) for i in range(side) for j in range(side)], float) * 100
    G = 200
    base = rng.lognormal(np.log(5), 0.5, G)
    programs = rng.normal(0, 1, (3, G))
    obs, X, xy = [], [], []
    for k, (slide, patient) in enumerate(SLIDES):
        stripe = np.minimum((grid[:, 0] // (100 * side / 3)).astype(int), 2)
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


@pytest.fixture(scope="session")
def registers(tmp_path_factory):
    from analysis_agent.challenges import assess

    out = {}
    for name, make in (("tissue", _tissue), ("table", _table)):
        for planted in (True, False):
            d = tmp_path_factory.mktemp(f"{name}_{planted}")
            prof = _profile(make(d, planted), tmp_path_factory.mktemp("cache"))
            out[name, planted] = (prof, assess(prof), d)
    return out

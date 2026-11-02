"""Format-generic loaders and column roles (from Project 3), on synthetic data."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from analysis_agent.data.loaders import load_path
from analysis_agent.data.roles import clean_missing

# --- synthetic spatial AnnData: 4 slides, 3 patients, painted spots -------------------

COLOURS = {"tumour": (120, 40, 140), "stroma": (230, 120, 170)}


@pytest.fixture(scope="module")
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


def test_spatial_anndata_modalities_and_roles(spatial_dir):
    ds = load_path(spatial_dir)
    assert ds.format == "spatial_anndata"
    assert {m.kind for m in ds.modalities.values()} == {"matrix", "spatial", "image"}
    roles = {r.column: r.role for r in ds.roles}
    assert roles == {"slide": "library", "patient": "group", "label": "label"}
    assert ds.obs["label"].isna().sum() == 4          # 'undetermined' -> missing


def test_spot_patches_are_aligned_with_coordinates(spatial_dir):
    ds = load_path(spatial_dir)
    src = ds.modalities["image"].images
    for i in (0, 5, 30, 95):
        patch = np.asarray(src.load(i))
        assert patch.shape == (30, 30, 3)
        lab = "tumour" if ds.obs["label"].iloc[i] == "tumour" or i % 24 == 0 else "stroma"
        assert tuple(patch[15, 15]) == COLOURS[lab]
    wide = np.asarray(src.with_context(2.0).load(0))
    assert wide.shape == (60, 60, 3) and tuple(wide[0, 0]) == (255, 255, 255)   # background ring


def test_summary_reports_structure_only(spatial_dir):
    s = load_path(spatial_dir).summary()
    assert s["format"] == "spatial_anndata" and s["n"] == 96
    assert s["modalities"]["matrix"]["n_features"] == 20 if "matrix" in s["modalities"] else True
    assert {m["kind"] for m in s["modalities"].values()} == {"matrix", "spatial", "image"}
    assert s["roles"]["patient"] == "group"


# --- synthetic image-text table: two label sources, articles as groups ------------------

@pytest.fixture(scope="module")
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


def test_table_roles_and_modalities(table_dir):
    ds = load_path(table_dir)
    assert ds.format == "table"
    roles = {r.column: r.role for r in ds.roles}
    assert roles == {"image_path": "image_path", "caption": "text", "split": "split",
                     "concepts": "multilabel", "concepts_manual": "multilabel", "article": "group"}
    assert set(ds.modalities) == {"image", "text"}
    assert ds.modalities["image"].n_present == 60
    assert ds.obs["concepts_manual"].isna().sum() == 9           # empty -> missing


def test_paired_matrices(tmp_path):
    import anndata as ad

    rng = np.random.default_rng(2)
    rna = ad.AnnData(sp.csr_matrix(rng.poisson(2, (30, 40)).astype(np.float32)),
                     obs=pd.DataFrame(index=[f"c{i}" for i in range(30)]),
                     var=pd.DataFrame(index=[f"G{i}" for i in range(40)]))
    atac = ad.AnnData(sp.csr_matrix(rng.poisson(1, (25, 50)).astype(np.float32)),
                      obs=pd.DataFrame(index=[f"c{i}" for i in range(10, 35)]),
                      var=pd.DataFrame(index=[f"chr1:{1000 * i}-{1000 * i + 500}" for i in range(50)]))
    rna.write_h5ad(tmp_path / "rna.h5ad")
    atac.write_h5ad(tmp_path / "atac.h5ad")
    ds = load_path(tmp_path)
    assert ds.format == "paired_matrices" and set(ds.modalities) == {"rna", "peaks"}
    assert ds.n == 35 and ds.paired_mask(["rna", "peaks"]).sum() == 20


def test_missing_tokens():
    s = pd.Series(["tumour", "Undetermined", "", "NA", "stroma", None])
    assert clean_missing(s).isna().tolist() == [False, True, True, True, False, True]

"""Format-generic loaders and column roles (from Project 3), on synthetic data."""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis_agent.data.loaders import load_path
from analysis_agent.data.roles import clean_missing
from conftest import COLOURS


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
    matrix = next(m for m in s["modalities"].values() if m["kind"] == "matrix")
    assert matrix["n_features"] == 20
    assert {m["kind"] for m in s["modalities"].values()} == {"matrix", "spatial", "image"}
    assert s["roles"]["patient"] == "group"


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

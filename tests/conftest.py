"""Shared synthetic datasets: a spatial AnnData with painted spots and an image-text table."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

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

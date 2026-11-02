"""Data preparation helpers, offline: label tables, image re-encoding and archive extraction."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts import prepare_data as P


def write_raw(d, names):
    pd.DataFrame({"CUI": list(names), "Canonical name": list(names.values())}).to_csv(
        d / "cui_mapping.csv", index=False)
    ids = {s: [f"{s}_{i}" for i in range(3)] for s in P.ROCO_SPLITS}
    every = [i for v in ids.values() for i in v]
    pd.DataFrame({"ID": every, "PMCID": [f"PMC{k // 2}" for k in range(len(every))],
                  "Attribution": "CC BY"}).to_csv(d / "license_information.csv", index=False)
    for s, rows in ids.items():
        pd.DataFrame({"ID": rows, "Caption": [f"caption {r}" for r in rows]}).to_csv(
            d / f"{s}_captions.csv", index=False)
        pd.DataFrame({"ID": rows, "CUIs": ["C1;C2", "C1;C9", "C3"]}).to_csv(d / f"{s}_concepts.csv", index=False)
        pd.DataFrame({"ID": rows[:2], "CUIs": ["C1", "C1;C2"]}).to_csv(
            d / f"{s}_concepts_manual.csv", index=False)


def test_label_tables_use_names_with_code_fallback(tmp_path):
    write_raw(tmp_path, {"C1": "CT", "C2": "Chest", "C3": np.nan})
    table, book = P._roco_tables(tmp_path)
    assert len(table) == 9 and set(table["split"]) == set(P.ROCO_SPLITS)
    first = table.loc["train_0"]
    assert first["concepts"] == "CT;Chest" and first["concepts_manual"] == "CT"
    assert table.loc["train_1", "concepts"] == "CT;C9"             # unknown code -> the code itself
    assert table.loc["train_2", "concepts"] == "C3"                # blank name -> the code
    assert table.loc["train_2", "concepts_manual"] == ""           # no curated concepts
    assert table.loc["valid_1", "article"] == "PMC2"
    flags = book.set_index("code")
    assert flags.loc["C2", "manual"] and not flags.loc["C9", "manual"] and not flags.loc["C9", "named"]


def test_a_name_with_a_list_separator_stops_the_conversion(tmp_path):
    write_raw(tmp_path, {"C1": "CT; contrast", "C2": "Chest", "C3": "Head"})
    with pytest.raises(SystemExit, match="separator"):
        P._roco_tables(tmp_path)


def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def test_images_are_capped_never_upscaled_and_16_bit_is_rescaled(tmp_path):
    big = np.random.default_rng(0).integers(0, 255, (300, 600, 3), dtype=np.uint8)
    assert P._save_image(png_bytes(big), str(tmp_path / "a.jpg")) == (600, 300)
    assert Image.open(tmp_path / "a.jpg").size == (512, 256)
    small = np.zeros((40, 50), np.uint8)
    P._save_image(png_bytes(small), str(tmp_path / "b.jpg"))
    assert Image.open(tmp_path / "b.jpg").size == (50, 40)
    deep = (np.arange(64 * 64).reshape(64, 64) * 15).astype(np.uint16)
    P._save_image(png_bytes(deep), str(tmp_path / "c.jpg"))
    out = np.asarray(Image.open(tmp_path / "c.jpg"))
    assert out.min() < 10 and out.max() > 245                        # full 8-bit range used
    msg = P._save_image(b"not an image", str(tmp_path / "d.jpg"))
    assert isinstance(msg, str) and not (tmp_path / "d.jpg").exists()


@pytest.mark.parametrize("n,workers", [(30, 1), (1100, 2)])
def test_extraction_writes_every_member_and_reports_failures(tmp_path, n, workers):
    zpath = tmp_path / "images.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for i in range(n):
            zf.writestr(f"split/img{i}.png", png_bytes(np.full((8, 8), i % 255, np.uint8)))
        zf.writestr("split/broken.png", b"\x89PNG not really")
    with zipfile.ZipFile(zpath) as zf:
        failed = P._extract(zf, [f"img{i}" for i in range(n)] + ["broken"], tmp_path / "out", workers)
    assert list(failed) == ["broken"]
    assert len(list((tmp_path / "out" / "images").glob("*.jpg"))) == n

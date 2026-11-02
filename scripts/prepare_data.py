#!/usr/bin/env python
"""Download the two primary datasets and convert them to the agent's generic input formats.

This script is the only place in the repository that knows dataset names. It
fetches files (sizes are checked against free disk first, checksums after) and
rewrites the data in formats the agent reads without dataset-specific code:

  spatial AnnData   spots.h5ad: raw counts in X, pixel coordinates in
                    obsm['spatial'], one source image per slide referenced from
                    uns['spatial'][slide], per-spot metadata in obs
  image-text table  pairs.csv: image path, caption, split, concept labels from two
                    sources (automatic and manually curated) and the source
                    article, with the images in images/

It makes no analysis choices: no quality control, normalisation, gene filtering,
patch size or encoder-specific resizing. Images are only capped in size for storage.

    uv run python -I scripts/prepare_data.py --only quick         # small subsets for smoke tests
    uv run python -I scripts/prepare_data.py                      # both datasets, full + quick
    uv run python -I scripts/prepare_data.py --dataset rocov2 --only full
    uv run python -I scripts/prepare_data.py --reuse-from ../elsewhere/data   # copy verified
                                                                  # raw files instead of downloading

Sizes (downloaded -> kept):
    STimage-1K4M HER2 cohort   0.73 GB -> 0.80 GB   (quick: ~0.12 GB, 6 slides)
    ROCOv2 v2.0.1              6.4 GB of image archives + 29 MB tables -> ~2.2 GB of images
                               (quick: ~0.2 GB read remotely through range requests)
The archives are kept in data/raw/rocov2/ after extraction unless --delete-archives is given.

Outputs (all under data/, which is gitignored):
    data/stimage_her2/spots.h5ad        data/stimage_her2_quick/spots.h5ad
    data/rocov2/pairs.csv + images/     data/rocov2_quick/pairs.csv + images/
    data/raw/...                        downloaded source files
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
RAW = DATA / "raw"
SEED = 774

# --- STimage-1K4M (Chen et al. 2024): HER2-positive breast cancer ST cohort -----------
ST_REPO = "jiawennnn/STimage-1K4M"
ST_REVISION = "b2073c6e8bb7e6d8dfd15c0db6134ba4f2a8a84f"      # pinned dataset commit
ST_PREFIX = "Human_Breast_Andersson_10142021_ST_"             # 36 slides, patients A-H
# Quick subset: 4 patients, each with its pathologist-annotated slide, two of them with a
# second slide, so slides nested within patients are present even in a smoke test.
ST_QUICK_SLIDES = ("A1", "A2", "B1", "B2", "C1", "D1")
ST_THUMB_MAX = 600                                            # px, for plotting only

# --- ROCOv2 v2.0.1 (Rückert et al. 2024) -------------------------------------------
ROCO_RECORD = "10821435"
ROCO_SPLITS = ("train", "valid", "test")
ROCO_QUICK_N = {"train": 600, "valid": 150, "test": 250}       # first members of each archive
ROCO_SHORT_SIDE = 256                                         # storage cap; encoders use 224


def say(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def check_disk(total_bytes: int, dest: Path, what: str) -> None:
    """Refuse to start a download that would leave the disk nearly full."""
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    need = int(total_bytes * 1.2) + 2 * 10**9
    say(f"{what}: {total_bytes / 1e9:.2f} GB to fetch; {free / 1e9:.1f} GB free")
    if free < need:
        sys.exit(f"not enough free disk for {what}: need ~{need / 1e9:.1f} GB, have {free / 1e9:.1f} GB")


def _digest(path: Path, algo: str, prefix: bytes = b"") -> str:
    h = hashlib.new(algo)
    h.update(prefix)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def md5sum(path: Path) -> str:
    return _digest(path, "md5")


def git_blob_sha1(path: Path) -> str:
    """The id git (and the Hugging Face Hub) gives a file's contents."""
    return _digest(path, "sha1", f"blob {path.stat().st_size}\0".encode())


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


# ===================================================================================
# Reusing an earlier download
# ===================================================================================

REUSE: Path | None = None        # set by --reuse-from: a data/ folder with the same raw/ layout


def _copy(src: Path, dest: Path) -> None:
    """Copy one file; on macOS a copy-on-write clone, which takes no extra disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin" and subprocess.run(["cp", "-c", str(src), str(dest)],
                                                   capture_output=True).returncode == 0:
        return
    shutil.copy2(src, dest)


def reuse(rel: Path, size: int, verify) -> bool:
    """Copy raw/<rel> from the --reuse-from folder if it is there and passes `verify`."""
    if REUSE is None:
        return False
    src = REUSE / "raw" / rel
    if not src.is_file() or src.stat().st_size != size or not verify(src):
        return False
    _copy(src, RAW / rel)
    return True


# ===================================================================================
# STimage-1K4M -> spatial AnnData
# ===================================================================================

def her2_files() -> dict[str, dict]:
    """Repository path -> size and content hash of every file for the cohort at the pinned commit."""
    from huggingface_hub import HfApi

    api = HfApi()
    files = {}
    for folder in ("ST/image", "ST/coord", "ST/gene_exp", "annotation"):
        for item in api.list_repo_tree(ST_REPO, path_in_repo=folder, repo_type="dataset",
                                       revision=ST_REVISION):
            name = item.path.rsplit("/", 1)[-1]
            if name.startswith(ST_PREFIX):
                lfs = getattr(item, "lfs", None)
                files[item.path] = {"size": int(getattr(item, "size", 0) or 0),
                                    "sha256": getattr(lfs, "sha256", None) if lfs else None,
                                    "blob_id": getattr(item, "blob_id", None)}
    return files


def _her2_verify(info: dict):
    if info["sha256"]:
        return lambda p: _digest(p, "sha256") == info["sha256"]
    return lambda p: info["blob_id"] is not None and git_blob_sha1(p) == info["blob_id"]


def _slide_of(path: str) -> str:
    """Slide id of a cohort file: '.../<prefix>A1_coord.csv' -> 'A1'."""
    return path.rsplit("/", 1)[-1][len(ST_PREFIX):].split("_")[0].split(".")[0]


def download_her2(slides: tuple[str, ...] | None = None) -> Path:
    """Fetch the cohort's files at the pinned commit (only the given slides' files, if any)."""
    from huggingface_hub import hf_hub_download

    raw = RAW / "stimage_her2"
    files = {p: f for p, f in her2_files().items() if slides is None or _slide_of(p) in slides}
    missing = {p: f for p, f in files.items()
               if not ((raw / p).is_file() and (raw / p).stat().st_size == f["size"])}
    reused = [p for p, f in sorted(missing.items())
              if reuse(Path("stimage_her2") / p, f["size"], _her2_verify(f))]
    if reused:
        say(f"STimage HER2: copied {len(reused)} verified files from {REUSE}")
    missing = {p: f for p, f in missing.items() if p not in reused}
    check_disk(sum(f["size"] for f in missing.values()), raw,
               f"STimage HER2 {'slides ' + ', '.join(slides) if slides else 'cohort'} ({len(files)} files)")
    for i, path in enumerate(sorted(missing), 1):
        hf_hub_download(ST_REPO, path, repo_type="dataset", revision=ST_REVISION, local_dir=raw)
        if i % 20 == 0 or i == len(missing):
            say(f"  downloaded {i}/{len(missing)}")
    return raw


def _slide_ids(raw: Path) -> list[str]:
    return sorted(p.name[len(ST_PREFIX):-len("_count.csv")]
                  for p in (raw / "ST" / "gene_exp").glob(f"{ST_PREFIX}*_count.csv"))


def _read_annotation(path: Path) -> pd.Series:
    """Annotation files have two columns (spot id, label) whose header names vary."""
    df = pd.read_csv(path)
    return pd.Series(df.iloc[:, 1].astype(str).to_numpy(), index=df.iloc[:, 0].astype(str))


def build_her2(raw: Path, slides: list[str], out_dir: Path) -> dict:
    """One AnnData for the cohort: counts, coordinates, slide/patient/label, images."""
    import anndata as ad
    import scipy.sparse as sp
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None        # whole-slide PNGs from the pinned source exceed PIL's guard
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, coords, obs, spatial, notes = [], [], [], {}, []
    genes = None
    for s in slides:
        name = ST_PREFIX + s
        counts = pd.read_csv(raw / "ST" / "gene_exp" / f"{name}_count.csv", index_col=0)
        coord = pd.read_csv(raw / "ST" / "coord" / f"{name}_coord.csv", index_col=0)
        common = counts.index.intersection(coord.index)
        if len(common) < len(counts) or len(common) < len(coord):
            notes.append(f"{s}: {len(counts)} expression rows, {len(coord)} coordinate rows, "
                         f"{len(common)} kept")
        counts, coord = counts.loc[common], coord.loc[common]
        here = set(counts.columns)
        genes = list(counts.columns) if genes is None else [g for g in genes if g in here]
        label = pd.Series(np.nan, index=common, dtype=object)
        anno_path = raw / "annotation" / f"{name}_anno.csv"
        if anno_path.exists():
            anno = _read_annotation(anno_path)
            hit = anno.index.intersection(common)
            label.loc[hit] = anno.loc[hit].to_numpy()
            if len(hit) < len(anno):
                notes.append(f"{s}: {len(anno) - len(hit)} annotated spots have no expression row")
        frames.append(counts)
        coords.append(coord[["xaxis", "yaxis"]].to_numpy(float))
        spot_ids = [f"{s}_{i[len(name) + 1:]}" for i in common]
        obs.append(pd.DataFrame({"slide": s, "patient": s[0], "label": label.to_numpy()},
                                index=spot_ids))
        img_path = raw / "ST" / "image" / f"{name}.png"
        with Image.open(img_path) as im:
            w, h = im.size
            scale = ST_THUMB_MAX / max(w, h)
            thumb = np.asarray(im.convert("RGB").resize((max(1, round(w * scale)),
                                                         max(1, round(h * scale))),
                                                        Image.Resampling.BILINEAR))
        diameter = float(2 * coord["r"].median())
        say(f"    {s}: {len(common)} spots")
        spatial[s] = {
            "images": {"lowres": thumb},
            "scalefactors": {"spot_diameter_fullres": diameter, "tissue_lowres_scalef": scale},
            "metadata": {"source_image_path": str(Path("..") / img_path.relative_to(DATA)),
                         "image_width": w, "image_height": h},
        }
    X = sp.vstack([sp.csr_matrix(f[genes].to_numpy()) for f in frames]).tocsr()
    if np.allclose(X.data, np.round(X.data)):
        X = X.astype(np.int32)
    obs_df = pd.concat(obs)
    for col in ("slide", "patient"):
        obs_df[col] = obs_df[col].astype("category")
    obs_df["label"] = obs_df["label"].astype("category")
    adata = ad.AnnData(X=X, obs=obs_df, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    adata.obsm["spatial"] = np.vstack(coords)
    adata.uns["spatial"] = spatial
    adata.write_h5ad(out_dir / "spots.h5ad", compression="gzip")
    summary = {"n_spots": adata.n_obs, "n_genes": adata.n_vars, "n_slides": len(slides),
               "n_patients": int(obs_df["patient"].nunique()),
               "labelled_spots": int(obs_df["label"].notna().sum()),
               "label_counts": obs_df["label"].value_counts().to_dict(), "notes": notes}
    say(f"  wrote {out_dir / 'spots.h5ad'}: {adata.n_obs} spots x {adata.n_vars} genes, "
        f"{len(slides)} slides")
    return summary


def prepare_her2(only: str | None) -> None:
    # The quick subset needs only its own slides' files, not the whole cohort.
    raw = download_her2(ST_QUICK_SLIDES if only == "quick" else None)
    slides = _slide_ids(raw)
    targets = {"full": (DATA / "stimage_her2", slides),
               "quick": (DATA / "stimage_her2_quick", [s for s in slides if s in ST_QUICK_SLIDES])}
    for size, (out_dir, chosen) in targets.items():
        if only and only != size:
            continue
        say(f"STimage HER2 ({size}): converting {len(chosen)} slides")
        summary = build_her2(raw, chosen, out_dir)
        write_json({"dataset": "STimage-1K4M, HER2-positive breast cancer ST cohort",
                    "source": f"https://huggingface.co/datasets/{ST_REPO}",
                    "revision": ST_REVISION, "slides": chosen, "subset": size,
                    "prepared": datetime.now().isoformat(timespec="seconds"),
                    "conversion": "counts as downloaded; genes = intersection across slides; "
                                  "coordinates are full-resolution pixels (x, y); spot diameter = "
                                  "2 x radius from the coordinate files",
                    **summary}, out_dir / "PREPARED.json")


# ===================================================================================
# ROCOv2 -> image-text table
# ===================================================================================

def zenodo_files(record: str) -> dict[str, dict]:
    r = requests.get(f"https://zenodo.org/api/records/{record}", timeout=60)
    r.raise_for_status()
    return {f["key"]: {"size": int(f["size"]), "md5": f["checksum"].split(":", 1)[1],
                       "url": f"https://zenodo.org/records/{record}/files/{f['key']}?download=1"}
            for f in r.json()["files"]}


def download_file(url: str, dest: Path, size: int, md5: str) -> Path:
    """Stream to disk with resume, then verify the published MD5."""
    if dest.exists() and dest.stat().st_size == size and md5sum(dest) == md5:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        start = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                if start and r.status_code != 206:
                    start = 0
                    part.unlink(missing_ok=True)
                r.raise_for_status()
                t, last = time.time(), start
                with open(part, "ab") as fh:
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
                        last += len(chunk)
                        if size > 2e8 and time.time() - t > 60:
                            say(f"    {dest.name}: {last / 1e9:.2f}/{size / 1e9:.2f} GB")
                            t = time.time()
            break
        except requests.RequestException as err:
            say(f"    {dest.name}: attempt {attempt + 1} failed ({err}); resuming")
            time.sleep(10 * (attempt + 1))
    if not part.exists() or part.stat().st_size != size or md5sum(part) != md5:
        sys.exit(f"download of {dest.name} failed its size/MD5 check")
    part.rename(dest)
    return dest


def fetch_roco(files: dict, key: str) -> Path:
    """A local, verified copy of one record file: already present, reused, or downloaded."""
    f, dest = files[key], RAW / "rocov2" / key
    if dest.is_file() and dest.stat().st_size == f["size"] and md5sum(dest) == f["md5"]:
        return dest
    if reuse(Path("rocov2") / key, f["size"], lambda p: md5sum(p) == f["md5"]):
        say(f"  {key}: copied a verified copy from {REUSE}")
        return dest
    return download_file(f["url"], dest, f["size"], f["md5"])


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP Range requests.

    ``zipfile`` only needs seek/tell/read, so this lets the quick subset pull a
    few hundred members out of a multi-GB archive without downloading all of it.
    Blocks are cached, and members are read in archive order, so contiguous
    members cost one request per block.
    """

    def __init__(self, url: str, size: int, block: int = 8 << 20):
        self.url, self.size, self.block, self.pos = url, size, block, 0
        self.session = requests.Session()
        self.cache: dict[int, bytes] = {}
        self.n_requests = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            lo, hi = i * self.block, min(self.size, (i + 1) * self.block) - 1
            for attempt in range(5):
                try:
                    r = self.session.get(self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
                    if r.status_code == 206:
                        break
                except requests.RequestException:
                    pass
                time.sleep(5 * (attempt + 1))
            else:
                raise OSError(f"range request failed for bytes {lo}-{hi}")
            self.n_requests += 1
            if len(self.cache) > 16:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = r.content
        return self.cache[i]

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        out = bytearray()
        while self.pos < end:
            i = self.pos // self.block
            data = self._block(i)
            lo = self.pos - i * self.block
            take = min(len(data) - lo, end - self.pos)
            out += data[lo:lo + take]
            self.pos += take
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def _save_image(raw_bytes: bytes, dest: str) -> tuple[int, int] | str:
    """Re-encode one image, capping its shorter side for storage (never upscaling).
    Returns the original (width, height), or the error message if it cannot be decoded."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw_bytes)) as im:
            im.load()
            if im.mode in ("I", "I;16", "I;16B", "F"):
                arr = np.asarray(im, dtype=np.float64)
                lo, hi = np.percentile(arr, [0.5, 99.5])
                im = Image.fromarray(np.clip(255 * (arr - lo) / max(hi - lo, 1e-9), 0, 255).astype(np.uint8))
            elif im.mode not in ("L", "RGB"):
                im = im.convert("RGB")
            w, h = im.size
            short = min(w, h)
            if short > ROCO_SHORT_SIDE:
                f = ROCO_SHORT_SIDE / short
                im = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.Resampling.BICUBIC)
            im.save(dest, "JPEG", quality=95)
            return w, h
    except Exception as err:           # noqa: BLE001 -- any decoder failure is recorded, not fatal
        Path(dest).unlink(missing_ok=True)
        return f"{type(err).__name__}: {err}"


def _save_batch(batch: list[tuple[bytes, str]]) -> list:
    return [_save_image(b, d) for b, d in batch]


def _roco_tables(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per image across splits (caption, automatic and curated concepts, source
    article), and the concept code book (name -> code, which label sources use it)."""
    names = pd.read_csv(raw / "cui_mapping.csv").set_index("CUI")["Canonical name"].to_dict()
    lic = pd.read_csv(raw / "license_information.csv").set_index("ID")
    rows = []
    for split in ROCO_SPLITS:
        cap = pd.read_csv(raw / f"{split}_captions.csv").set_index("ID")
        auto = pd.read_csv(raw / f"{split}_concepts.csv").set_index("ID")["CUIs"]
        manual = pd.read_csv(raw / f"{split}_concepts_manual.csv").set_index("ID")["CUIs"]
        df = pd.DataFrame({"caption": cap["Caption"]})
        df["split"] = split
        df["cuis"] = auto.reindex(df.index).fillna("").astype(str)
        df["cuis_manual"] = manual.reindex(df.index).fillna("").astype(str)
        df["article"] = lic["PMCID"].reindex(df.index)
        df["attribution"] = lic["Attribution"].reindex(df.index)
        rows.append(df)
    out = pd.concat(rows)
    out.index.name = "id"

    def codes(col: str) -> set[str]:
        return {c for s in out[col] for c in s.split(";") if c}

    auto_codes, manual_codes = codes("cuis"), codes("cuis_manual")
    named = {c: n.strip() for c, n in names.items() if isinstance(n, str) and n.strip()}
    # A code without a canonical name is labelled by its code, so every label stays visible.
    label = {c: named.get(c, c) for c in auto_codes | manual_codes}
    bad = [n for n in label.values() if ";" in n or "|" in n]
    if bad:
        sys.exit(f"concept names contain a list separator (e.g. {bad[:3]}); choose another separator")
    if len(set(label.values())) != len(label):
        sys.exit("two concept codes share a canonical name; names would not identify concepts")
    for col, dest in (("cuis", "concepts"), ("cuis_manual", "concepts_manual")):
        out[dest] = out[col].map(lambda s: ";".join(label[c] for c in s.split(";") if c))
    book = pd.DataFrame({"concept": [label[c] for c in sorted(label)], "code": sorted(label),
                         "automatic": [c in auto_codes for c in sorted(label)],
                         "manual": [c in manual_codes for c in sorted(label)],
                         "named": [c in named for c in sorted(label)]})
    return out, book


def _member_index(zf: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Map image id (file stem) -> archive member."""
    return {Path(m.filename).stem: m for m in zf.infolist()
            if not m.is_dir() and Path(m.filename).suffix.lower() in (".jpg", ".jpeg", ".png")}


def _extract(zf: zipfile.ZipFile, ids: list[str], out_dir: Path, workers: int) -> dict[str, str]:
    """Re-encode the given members into out_dir/images/. Returns {id: error} for images that
    could not be decoded (they are left out of the table)."""
    members = _member_index(zf)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in sorted(ids, key=lambda i: members[i].header_offset)
            if not (img_dir / f"{i}.jpg").exists()]
    failed: dict[str, str] = {}

    def record(batch_ids, results):
        for i, res in zip(batch_ids, results):
            if isinstance(res, str):
                failed[i] = res

    batches = [todo[k:k + 64] for k in range(0, len(todo), 64)]
    if workers <= 1 or len(todo) < 1000:
        for b in batches:
            record(b, _save_batch([(zf.read(members[i]), str(img_dir / f"{i}.jpg")) for i in b]))
        return failed
    # Members are read here, in archive order, and decoded in worker processes. At most a few
    # batches per worker are in flight, so memory stays bounded however large the archive.
    done, report_at = 0, 5000
    with ProcessPoolExecutor(workers) as ex:
        pending: deque = deque()
        for b in batches:
            pending.append((b, ex.submit(_save_batch, [(zf.read(members[i]), str(img_dir / f"{i}.jpg"))
                                                       for i in b])))
            while len(pending) >= 3 * workers:
                b0, fut = pending.popleft()
                record(b0, fut.result())
                done += len(b0)
                if done >= report_at:
                    say(f"    extracted {done}/{len(todo)}")
                    report_at += 5000
        for b0, fut in pending:
            record(b0, fut.result())
    say(f"    extracted {len(todo)}/{len(todo)}")
    return failed


def _write_roco(table: pd.DataFrame, book: pd.DataFrame, out_dir: Path, subset: str, files: dict,
                extra: dict) -> None:
    table = table.copy()
    table["image_path"] = [f"images/{i}.jpg" for i in table.index]
    cols = ["image_path", "caption", "split", "concepts", "concepts_manual", "article"]
    table[cols].to_csv(out_dir / "pairs.csv")
    # Licence attributions are for citing any image shown in a figure, and concept codes are
    # for looking a concept up; neither is data, so they live beside the table, not in it.
    (out_dir / "meta").mkdir(exist_ok=True)
    table[["article", "attribution"]].to_csv(out_dir / "meta" / "attribution.csv")
    book.to_csv(out_dir / "meta" / "concept_codes.csv", index=False)

    def n_concepts(col: str) -> int:
        return len({c for s in table[col] for c in s.split(";") if c})

    write_json({"dataset": "ROCOv2 v2.0.1", "source": f"https://zenodo.org/records/{ROCO_RECORD}",
                "doi": f"10.5281/zenodo.{ROCO_RECORD}", "license": "CC BY-NC 4.0", "subset": subset,
                "prepared": datetime.now().isoformat(timespec="seconds"),
                "n_by_split": table["split"].value_counts().to_dict(),
                "n_concepts": {"automatic": n_concepts("concepts"), "manual": n_concepts("concepts_manual")},
                "columns": {"concepts": "automatically extracted UMLS concepts (canonical names; a code "
                                        "where the code book has no name)",
                            "concepts_manual": "manually curated concepts (same vocabulary)",
                            "article": "PubMed Central id of the source article"},
                "image_storage": f"JPEG q95, shorter side capped at {ROCO_SHORT_SIDE} px, no upscaling",
                "source_md5": {k: v["md5"] for k, v in files.items()}, **extra},
               out_dir / "PREPARED.json")
    say(f"  wrote {out_dir / 'pairs.csv'}: {len(table)} pairs")


def prepare_roco(only: str | None, delete_archives: bool, workers: int) -> None:
    raw = RAW / "rocov2"
    files = zenodo_files(ROCO_RECORD)
    tables = [k for k in files if k.endswith(".csv")]
    check_disk(sum(files[k]["size"] for k in tables if not (raw / k).exists()), raw,
               "ROCOv2 caption/concept tables")
    for k in tables:
        fetch_roco(files, k)
    table, book = _roco_tables(raw)
    say(f"ROCOv2: {len(table)} images in metadata ({table['split'].value_counts().to_dict()})")

    if only in (None, "quick"):
        out_dir = DATA / "rocov2_quick"
        picked, requests_made, failed, how = [], 0, {}, set()
        for split, n in ROCO_QUICK_N.items():
            key = f"{split}_images.zip"
            local = raw / key
            if local.is_file() and local.stat().st_size == files[key]["size"]:
                src, how = open(local, "rb"), how | {"local archive"}
            else:
                src, how = HTTPRangeFile(files[key]["url"], files[key]["size"]), how | {"HTTP range requests"}
            say(f"ROCOv2 (quick): {split}: first {n} members of {key} ({', '.join(sorted(how))})")
            with src, zipfile.ZipFile(src) as zf:
                members = sorted(_member_index(zf).items(), key=lambda kv: kv[1].header_offset)
                ids = [i for i, _ in members if i in table.index][:n]
                failed.update(_extract(zf, ids, out_dir, workers=1))
            requests_made += getattr(src, "n_requests", 0)
            picked.append(table.loc[[i for i in ids if i not in failed]])
        _write_roco(pd.concat(picked), book, out_dir, "quick", files,
                    {"selection": "first members of each split's archive (smoke test only)",
                     "read_from": sorted(how), "http_range_requests": requests_made,
                     "undecodable_images": failed})

    if only in (None, "full"):
        out_dir = DATA / "rocov2"
        zips = [f"{s}_images.zip" for s in ROCO_SPLITS]
        check_disk(sum(files[k]["size"] for k in zips if not (raw / k).exists()), raw,
                   "ROCOv2 image archives")
        picked, failed, notes = [], {}, []
        for split in ROCO_SPLITS:
            rows = table[table["split"] == split]
            key = f"{split}_images.zip"
            say(f"ROCOv2 (full): {split}: {len(rows)} images; fetching {key} "
                f"({files[key]['size'] / 1e9:.2f} GB)")
            path = fetch_roco(files, key)
            with zipfile.ZipFile(path) as zf:
                members = _member_index(zf)
                ids = [i for i in rows.index if i in members]
                if len(ids) < len(rows):
                    notes.append(f"{split}: {len(rows) - len(ids)} images in the tables are not in {key}")
                extra_members = len(set(members) - set(rows.index))
                if extra_members:
                    notes.append(f"{split}: {extra_members} archive members have no table row")
                failed.update(_extract(zf, ids, out_dir, workers))
            picked.append(rows.loc[[i for i in ids if i not in failed]])
            if delete_archives:
                path.unlink()
                say(f"  deleted {key} after extraction")
        _write_roco(pd.concat(picked), book, out_dir, "full", files,
                    {"selection": "every image in the official train/valid/test splits",
                     "undecodable_images": failed, "notes": notes})
        if not delete_archives:
            say(f"  the image archives ({sum(files[k]['size'] for k in zips) / 1e9:.1f} GB) are kept "
                f"in {raw}; remove them to free the space (or rerun with --delete-archives)")


def main(argv=None) -> None:
    global REUSE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["stimage_her2", "rocov2"], help="default: both")
    ap.add_argument("--only", choices=["quick", "full"], help="default: both sizes")
    ap.add_argument("--reuse-from", type=Path,
                    help="a data/ folder from an earlier download (same raw/ layout); files that "
                         "match the published size and checksum are copied instead of downloaded")
    ap.add_argument("--delete-archives", action="store_true",
                    help="delete the ROCOv2 image archives after extraction (default: keep)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="processes that re-encode images (default: all cores but two)")
    args = ap.parse_args(argv)
    REUSE = args.reuse_from.resolve() if args.reuse_from else None
    if args.dataset in (None, "stimage_her2"):
        prepare_her2(args.only)
    if args.dataset in (None, "rocov2"):
        prepare_roco(args.only, args.delete_archives, args.workers)
    say("done")


if __name__ == "__main__":
    main()

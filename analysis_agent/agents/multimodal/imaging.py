"""Image measurements used to choose encoders and preprocessing (from Project 3's profiler).

On a sample of images (stratified by group when groups are large enough to have a mean
colour): the share of greyscale images, the share of tissue pixels with a pink/purple
stain hue, the between-group share of variance in mean colour (one-way ICC), sizes, and
a domain hint matched against the encoder registry:

    stained_tissue      mostly colour images whose tissue pixels carry a stain hue
    greyscale_imaging   mostly greyscale images (radiology and similar)
    colour_other        anything else

Adapted: these are functions of an image source and sample rows instead of parts of a
Project 3 profile object; the hashing helper is kept for the near-duplicate tools.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_IMAGES = 240
MIN_GROUP_IMAGES = 10            # images a group needs before its mean colour is estimated


def icc_oneway(values: np.ndarray, groups: np.ndarray) -> float | None:
    """ICC(1) from a one-way ANOVA with unequal group sizes: the between-group
    share of variance (0 = groups look alike, 1 = all variance is between groups)."""
    df = pd.DataFrame({"y": values, "g": groups}).dropna()
    sizes = df.groupby("g")["y"].size()
    sizes = sizes[sizes >= 2]
    df = df[df["g"].isin(sizes.index)]
    g, N = len(sizes), len(df)
    if g < 2 or N <= g:
        return None
    grand = df["y"].mean()
    means = df.groupby("g")["y"].mean()
    msb = float((sizes * (means - grand) ** 2).sum() / (g - 1))
    msw = float(((df["y"] - df["g"].map(means)) ** 2).sum() / (N - g))
    k0 = (N - float((sizes ** 2).sum()) / N) / (g - 1)
    denom = msb + (k0 - 1) * msw
    return float(np.clip((msb - msw) / denom, 0, 1)) if denom > 0 else None


def thumb(im, side: int = 128) -> np.ndarray:
    im = im.convert("RGB")
    im.thumbnail((side, side))
    return np.asarray(im, dtype=np.uint8)


def dhash(im) -> int:
    """64-bit difference hash: adjacent-pixel brightness comparisons on a 9 x 8 thumbnail."""
    g = np.asarray(im.convert("L").resize((9, 8)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hsv(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = arr.reshape(-1, 3).astype(np.float32) / 255
    mx, mn = a.max(1), a.min(1)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    r, g, b = a[:, 0], a[:, 1], a[:, 2]
    h = np.zeros_like(mx)
    nz = d > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = (60 * ((g[rm] - b[rm]) / d[rm])) % 360
    h[gm] = 60 * ((b[gm] - r[gm]) / d[gm]) + 120
    h[bm] = 60 * ((r[bm] - g[bm]) / d[bm]) + 240
    return h, s


def _sample(idx: np.ndarray, k: int, rng) -> np.ndarray:
    return np.sort(rng.choice(idx, min(k, len(idx)), replace=False)) if len(idx) > k else idx


def sample_rows(rows: np.ndarray, groups: np.ndarray | None, rng, k: int = MAX_IMAGES) -> np.ndarray:
    """Up to k rows, stratified over at most 12 groups that have >= MIN_GROUP_IMAGES images
    when such groups hold most of the rows; otherwise a random sample."""
    if groups is not None:
        sizes = pd.Series(groups).value_counts()
        big = sizes[sizes >= MIN_GROUP_IMAGES]
        if len(big) >= 2 and big.sum() >= 0.5 * len(rows):
            levels = rng.permutation(list(big.index))[:12]
            per = max(1, k // len(levels))
            return np.sort(np.concatenate([_sample(rows[groups == lv], per, rng) for lv in levels]))
    return _sample(rows, k, rng)


def image_summary(source, rows: np.ndarray, groups: np.ndarray | None, rng) -> dict:
    """Domain hint, greyscale and stain shares and the between-group colour share."""
    pick = sample_rows(rows, groups, rng)
    g_of = None if groups is None else dict(zip(rows.tolist(), groups.tolist()))
    pick = source.order(pick)
    sizes, grey, means, stain_px, tissue_px = [], [], [], 0, 0
    for i in pick:
        im = source.load(int(i))
        sizes.append(im.size)
        arr = thumb(im)
        chan = arr.astype(np.int16)
        grey.append(float(np.abs(chan[..., 0] - chan[..., 1]).mean() +
                          np.abs(chan[..., 1] - chan[..., 2]).mean()) < 3)
        means.append(arr.reshape(-1, 3).mean(0))
        lt, dk = (arr > 220).all(-1), (arr < 25).all(-1)
        h, s = hsv(arr)
        t = ~(lt | dk).reshape(-1)
        stain = ((h >= 240) | (h <= 15)) & (s > 0.1)
        stain_px += int((stain & t).sum())
        tissue_px += int(t.sum())
    sizes, means = np.array(sizes, float), np.array(means)
    grey_frac = float(np.mean(grey)) if grey else 0.0
    stain_frac = stain_px / max(1, tissue_px)
    share = None
    if g_of is not None:
        g = np.array([g_of[int(i)] for i in pick])
        iccs = [v for v in (icc_oneway(means[:, c], g) for c in range(3)) if v is not None]
        share = float(np.mean(iccs)) if iccs else None
    if grey_frac >= 0.5 and stain_frac < 0.3:
        hint = "greyscale_imaging"
    elif grey_frac < 0.2 and stain_frac >= 0.5:
        hint = "stained_tissue"
    else:
        hint = "colour_other"
    return {"n_sampled": int(len(pick)), "domain_hint": hint, "greyscale_fraction": grey_frac,
            "stain_hue_fraction": float(stain_frac), "colour_between_group_share": share,
            "width_median": float(np.median(sizes[:, 0])) if len(sizes) else None,
            "height_median": float(np.median(sizes[:, 1])) if len(sizes) else None,
            "evidence": f"{grey_frac:.0%} of sampled images greyscale; {stain_frac:.0%} of tissue pixels "
                        "with a stain hue"}

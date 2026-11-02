"""Image and text measurements, by the same rules the quality tools apply.

Images held as one file each are all measured (size, grey-level spread and difference
hash, in the pass the image-QC and near-duplicate tools reuse from the cache); images cut
from larger images (spot patches) are measured on a stratified sample, and not hashed,
since neighbouring patches are not copies of each other. Colour is summarised on a sample
of thumbnails stratified by group, as the feature tool does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..agents.multimodal.imaging import COLOUR_SEED_OFFSET, MIN_GROUP_IMAGES, image_summary, sample_rows
from ..schema import ImageProfile, TextProfile
from ..tools.base import Artifact, values_on
from ..tools.qc import ImageQCParams, image_flags, image_stats
from ..tools.quality import DuplicateParams, pixel_pairs
from ..tools.text import TextQCParams, text_flags

MAX_PATCHES = 240


def image_profile(a: Artifact, group: Artifact | None, split: Artifact | None, cache_dir, rng,
                  log=print, seed: int = 774) -> ImageProfile:
    rows = a.rows
    kind = str(a.meta.get("source_kind") or getattr(a.data, "kind", "files"))
    files = kind == "files"
    g = values_on(rows, group) if group is not None else None
    if g is not None:                                   # colour per group needs groups with enough images
        sizes = pd.Series(g).value_counts()
        big = sizes[sizes >= MIN_GROUP_IMAGES]
        if len(big) < 2 or big.sum() < 0.5 * len(rows):
            g, group = None, None
    measure = rows if files else sample_rows(rows, g, rng, MAX_PATCHES)
    st = image_stats(a, measure, cache_dir, log)
    flags = image_flags(st, **ImageQCParams().model_dump())
    bad = np.zeros(len(st), bool)
    for f in flags.values():
        bad |= f
    ok = st["readable"].to_numpy()
    w, h = st["width"].to_numpy(float), st["height"].to_numpy(float)
    short = np.minimum(w, h)[ok]
    rec = ImageProfile(artifact=a.name, source_kind=kind, n_images=len(rows), n_measured=len(st),
                       unusable_fraction=float(bad.mean()) if len(st) else None,
                       flag_counts={k: int(v.sum()) for k, v in flags.items()},
                       width_median=float(np.median(w[ok])) if ok.any() else None,
                       height_median=float(np.median(h[ok])) if ok.any() else None,
                       size_iqr_ratio=float(np.quantile(short, 0.75) / max(1.0, np.quantile(short, 0.25)))
                       if ok.any() else None)
    # the same colour sample the feature step draws, so both measure the same between-group colour share
    summ = image_summary(a.data, rows, g, np.random.default_rng(seed + COLOUR_SEED_OFFSET))
    rec.n_colour_sampled = summ["n_sampled"]
    rec.domain_hint = summ["domain_hint"]
    rec.greyscale_fraction = summ["greyscale_fraction"]
    rec.stain_hue_fraction = summ["stain_hue_fraction"]
    rec.colour_between_group_share = summ["colour_between_group_share"]
    rec.colour_group = group.name if group is not None and summ["colour_between_group_share"] is not None else None
    measured = "every image" if files else f"{len(st):,} sampled images"
    if files:
        dp = DuplicateParams()
        pairs, informative, _, n_candidates = pixel_pairs(st, dp.hash_max_distance, dp.wide_hash_max_distance)
        rec.n_hashed = int(informative.sum())
        rec.n_hash_candidates = int(n_candidates)
        rec.n_duplicate_pairs = int(len(pairs))
        rec.duplicate_fraction = float(len(np.unique(pairs)) / max(1, informative.sum()))
        s = values_on(st.index.to_numpy(np.int64), split, "") if split is not None else None
        if s is not None and len(pairs):
            sa, sb = s[pairs[:, 0]], s[pairs[:, 1]]
            rec.cross_split_duplicate_pairs = int(((sa != sb) & (sa != "") & (sb != "")).sum())
        elif s is not None:
            rec.cross_split_duplicate_pairs = 0
        measured += (f"; duplicates: 64-bit difference hashes within {dp.hash_max_distance} bits confirmed by "
                     f"256-bit hashes within {dp.wide_hash_max_distance} bits")
    rec.measured_on = measured + f"; colour on {summ['n_sampled']} thumbnails"
    return rec


def text_profile(a: Artifact, split: Artifact | None) -> TextProfile:
    flags, words, norm = text_flags(pd.Series(a.data, dtype=object), **TextQCParams().model_dump(exclude={"remove"}))
    anyf = np.zeros(len(words), bool)
    for f in flags.values():
        anyf |= f
    cross = None
    s = values_on(a.rows, split, "") if split is not None else None
    if s is not None:
        dup = pd.DataFrame({"t": norm, "s": s})[flags["duplicate"]]
        per = dup.groupby("t")["s"].nunique()
        cross = float((per > 1).mean()) if len(per) else 0.0
    return TextProfile(artifact=a.name, n_texts=len(words),
                       flag_fractions={k: float(v.mean()) for k, v in flags.items()},
                       flagged_fraction=float(anyf.mean()), words_median=float(np.median(words)),
                       words_p95=float(np.quantile(words, 0.95)), cross_split_duplicate_fraction=cross)

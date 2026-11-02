"""Project 3's rules as a sub-agent: encoder screening, frozen features, grouped
predictability and image-text agreement.

Adapted from Project 3: the screening of encoders/screening.py (CCA recall@10 on held-out
pairs, zero-shot recall for pre-aligned dual encoders, a probe when the paired data are
labels), the colour-normalisation, patch and alignment rules of planner/heuristic.py and
the probe of evaluate/probes.py. New here: predictability is estimated by grouped nested
cross-validation with a group-bootstrap interval, and the chosen alignment scores every
image-text pair against random partners, cross-fitted so that no pair is scored by a
model that was trained on it.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from .encoders.registry import REGISTRY
from .linear import CCA
from .metrics import classification_scores, l2n, recall_at_k, stratified_cap

COLOUR_ICC_NORMALISE = 0.20      # between-group share of mean-colour variance that calls for Reinhard
NORMALISER_PER_GROUP = 30        # images per group behind its colour statistics
VAL_FRACTION = 0.2
MIN_SCREEN_PAIRS = 50            # fewer screening pairs make a held-out recall meaningless
MIN_PAIRS_CONTRASTIVE = 500      # fewer pairs give too few negatives per batch
PROBE_MAX_TRAIN = 5000           # probe accuracy saturates well before this
C_GRID = (0.01, 0.1, 1.0, 10.0)
AGREEMENT_FLAG = 0.5             # flagged: the true partner beats fewer than half of random partners


def _pow2(x: float, lo: int, hi: int) -> int:
    return int(min(hi, max(lo, 2 ** math.floor(math.log2(max(x, 1))))))


def cap(idx: np.ndarray, k: int, rng) -> np.ndarray:
    return np.sort(rng.choice(idx, k, replace=False)) if len(idx) > k else np.asarray(idx)


def holdout(n: int, split: np.ndarray | None, groups: np.ndarray | None, rng,
            val_fraction: float = VAL_FRACTION) -> tuple[np.ndarray, np.ndarray, str]:
    """Training and validation positions among ``n`` aligned samples, and how they were chosen.

    A predefined split is used when it has both train and val samples; its test samples are
    never returned. Otherwise whole groups go to validation when there are at least two, so
    validation samples share no group with training ones; otherwise a random split.
    """
    pool = np.arange(n)
    if split is not None:
        tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
        if len(tr) and len(va):
            return tr, va, "the predefined split (train fits, val selects; test unused)"
        pool = np.flatnonzero(split != "test")
    if groups is not None and len(np.unique(groups[pool])) >= 2:
        sizes = pd.Series(groups[pool]).value_counts()
        levels, val, total = rng.permutation(sizes.index.to_numpy()), [], 0
        for lv in levels[:-1]:                       # at least one group stays in training
            if total >= val_fraction * len(pool):
                break
            val.append(lv)
            total += int(sizes[lv])
        is_val = np.isin(groups[pool], val)
        return pool[~is_val], pool[is_val], (f"whole groups held out ({len(val)} of {len(levels)} "
                                             "groups to validation)")
    perm = rng.permutation(pool)
    n_val = max(1, int(round(val_fraction * len(pool))))
    return (np.sort(perm[n_val:]), np.sort(perm[:n_val]),
            f"a random {1 - val_fraction:.0%}/{val_fraction:.0%} split")


# ---------------------------------------------------------------------------
# preprocessing and features
# ---------------------------------------------------------------------------

def colour_normalisation(summary: dict, setting: str, has_groups: bool) -> tuple[bool, str]:
    """Reinhard normalisation per group when stain colour differs between groups."""
    share = summary.get("colour_between_group_share")
    if setting != "auto":
        use = setting == "reinhard" and has_groups
        return use, "set by the plan" + ("" if use or setting == "none" else "; no grouping to normalise by")
    if summary.get("domain_hint") == "greyscale_imaging":
        return False, "mostly greyscale images: there is no stain colour to normalise"
    if not has_groups or share is None:
        return False, "no grouping with enough images per group to measure colour shifts between groups"
    if share >= COLOUR_ICC_NORMALISE:
        return True, (f"{share:.0%} of the variance in mean image colour lies between groups (ICC >= "
                      f"{COLOUR_ICC_NORMALISE:.0%}); a frozen encoder would encode those stain or "
                      "scanner shifts as a batch effect")
    return False, f"between-group colour share {share:.0%} < {COLOUR_ICC_NORMALISE:.0%}"


def fit_normalizer(source, rows: np.ndarray, groups: np.ndarray, rng, per_group: int = NORMALISER_PER_GROUP):
    """Per-group colour statistics from a sample of each group's own images. The target is the
    average group, built from images only: no labels and nothing from the other modality."""
    from .preprocess import ReinhardNormalizer

    samples = {}
    for g in np.unique(groups):
        members = rows[groups == g]
        pick = rng.choice(members, min(per_group, len(members)), replace=False)
        samples[str(g)] = [np.asarray(source.load(int(i)).convert("RGB")) for i in source.order(pick)]
    return ReinhardNormalizer().fit(samples)


def encode_images(encoder: str, source, rows: np.ndarray, *, device: str, batch_size: int, cache=None,
                  key_parts: dict | None = None, normalizer=None, group_of=None, progress=None) -> np.ndarray:
    """Frozen features of the images at dataset positions ``rows`` (cached on disk). ``source``
    may be a function returning the source, called only when the features are not cached."""
    from .encoders.pretrained import image_features, load_encoder
    from .preprocess import ImageTransform

    if cache is not None and key_parts is not None:
        hit = cache.load(cache.key(encoder=encoder, kind="image", **key_parts))
        if hit is not None and len(hit) == len(rows):
            return hit
    source = source() if callable(source) else source
    enc = load_encoder(encoder, device)
    tf = ImageTransform(enc.input_size, enc.mean, enc.std, True, normalizer,
                        square_input=getattr(source, "kind", "") == "spot_patches")
    return image_features(enc, source, rows, tf, group_of, batch_size, cache, key_parts, progress)


def encode_texts(encoder: str, texts: list[str], *, device: str, batch_size: int, cache=None,
                 key_parts: dict | None = None) -> np.ndarray:
    from .encoders.pretrained import load_encoder, text_features
    from .preprocess import clean

    max_tokens = REGISTRY[encoder].max_tokens or 256
    if cache is not None and key_parts is not None:
        hit = cache.load(cache.key(encoder=encoder, kind="text", max_tokens=max_tokens, **key_parts))
        if hit is not None and len(hit) == len(texts):
            return hit
    enc = load_encoder(encoder, device)
    return text_features(enc, [clean(t) for t in texts], max_tokens, batch_size, cache, key_parts)


# ---------------------------------------------------------------------------
# screening
# ---------------------------------------------------------------------------

def alignment_scores(Atr, Ava, Btr, Bva, dual: bool) -> dict:
    """Label-free screening scores of one candidate pair, judged on held-out pairs: recall@10
    after a CCA fitted on the training pairs, and in the pair's own space for dual encoders."""
    cca = CCA(k=max(2, min(32, len(Atr) // 10))).fit(Atr, Btr)
    out = {"cca_recall_at_10": recall_at_k(*cca.transform(Ava, Bva), 10),
           "zero_shot_recall_at_10": recall_at_k(Ava, Bva, 10) if dual and Ava.shape[1] == Bva.shape[1]
           else None}
    out["alignment_recall_at_10"] = max(v for v in out.values() if v is not None)
    return out


def probe_score(Xtr, ytr, Xva, yva) -> float | None:
    """Macro-F1 of a logistic-regression probe (labels as strings, "" = missing)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    tr, va = ytr != "", yva != ""
    if len(set(ytr[tr])) < 2 or va.sum() < 5:
        return None
    sc = StandardScaler().fit(Xtr[tr])
    clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(Xtr[tr]), ytr[tr])
    keep = np.isin(yva[va], clf.classes_)
    return float(f1_score(yva[va][keep], clf.predict(sc.transform(Xva[va][keep])), average="macro"))


def clearly_above(score: float, chance: float, n: int, z: float = 2.0) -> bool:
    """Whether a held-out recall@k beats its chance level by more than ``z`` binomial standard
    errors (approximate: queries are treated as independent)."""
    return bool(score - chance > z * math.sqrt(chance * (1 - chance) / max(n, 1)))


def best_per_candidate(rows: list[dict], key: str, metric: str) -> list[tuple[str, dict]]:
    """Each candidate's best-scoring row (over the partner candidates), best first."""
    best: dict[str, dict] = {}
    for r in rows:
        if r.get(metric) is None:
            continue
        if r[key] not in best or r[metric] > best[r[key]][metric]:
            best[r[key]] = r
    return sorted(best.items(), key=lambda kv: -kv[1][metric])


# ---------------------------------------------------------------------------
# predictability: grouped nested cross-validation
# ---------------------------------------------------------------------------

def grouped_folds(y: np.ndarray, groups: np.ndarray, n_folds: int, seed: int) -> list:
    """Folds that keep every group whole, with class shares balanced across folds."""
    from sklearn.model_selection import StratifiedGroupKFold

    k = int(min(n_folds, len(np.unique(groups))))
    if k < 2:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)        # rare classes in a fold
        return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))


def _logistic(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))


def fit_probe(X, y, groups, inner_folds: int, seed: int):
    """Class-balanced logistic regression on standardised features, its penalty chosen by
    grouped inner cross-validation (macro-F1); at most PROBE_MAX_TRAIN training samples."""
    from sklearn.metrics import f1_score

    keep = stratified_cap(y, PROBE_MAX_TRAIN, seed)
    X, y, groups = X[keep], y[keep], groups[keep]
    scores = {}
    for c in C_GRID:
        f1s = [f1_score(y[va], _logistic(c).fit(X[tr], y[tr]).predict(X[va]), average="macro",
                        labels=np.unique(y[va]), zero_division=0)
               for tr, va in grouped_folds(y, groups, inner_folds, seed + 1) if len(np.unique(y[tr])) >= 2]
        if f1s:
            scores[c] = float(np.mean(f1s))
    C = max(scores, key=scores.get) if scores else 1.0
    return _logistic(C).fit(X, y), C, scores


def group_bootstrap(y, pred, groups, reps: int, seed: int, level: float = 0.95) -> tuple:
    """Interval for balanced accuracy that resamples whole groups (samples within a group are
    not independent)."""
    from sklearn.metrics import balanced_accuracy_score

    levels = np.unique(groups)
    if len(levels) < 3 or reps < 10:
        return None, None
    members = [np.flatnonzero(groups == g) for g in levels]
    rng = np.random.default_rng(seed)
    vals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(reps):
            ii = np.concatenate([members[j] for j in rng.integers(0, len(levels), len(levels))])
            if len(np.unique(y[ii])) >= 2:
                vals.append(balanced_accuracy_score(y[ii], pred[ii]))
    a = (1 - level) / 2
    return (float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))) if vals else (None, None)


def grouped_predictability(X, y, groups, n_folds: int, inner_folds: int, seed: int, reps: int,
                           check=None) -> dict:
    """Out-of-fold predictions of grouped nested cross-validation (y = codes 0..K-1), with
    macro-F1, balanced accuracy and a group-bootstrap interval."""
    K = int(y.max()) + 1
    folds = grouped_folds(y, groups, n_folds, seed)
    pred, chosen = np.full(len(y), -1), []
    for tr, te in folds:
        if check is not None:
            check()
        model, C, _ = fit_probe(X[tr], y[tr], groups[tr], inner_folds, seed)
        pred[te] = model.predict(X[te])
        chosen.append(C)
    ok = pred >= 0
    if not ok.any():
        return {"n_folds": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = classification_scores(y[ok], pred[ok], K)
    out["ba_ci_low"], out["ba_ci_high"] = group_bootstrap(y[ok], pred[ok], groups[ok], reps, seed)
    return {**out, "n_folds": len(folds), "C_per_fold": chosen, "pred": pred}


# ---------------------------------------------------------------------------
# image-text agreement
# ---------------------------------------------------------------------------

def aligner_hyperparameters(n_train: int, residual: bool, max_epochs: int, seconds: float) -> tuple[dict, str]:
    """Project 3's head and training rules for a contrastive alignment of frozen features."""
    hp = {"embed_dim": 128 if n_train < 5000 else 256, "hidden": 512, "n_layers": 2,
          "dropout": 0.2 if n_train < 5000 else 0.1, "residual": residual, "lr": 1e-3, "weight_decay": 1e-4,
          "batch_size": _pow2(n_train / 16, 32, 512), "max_epochs": int(max_epochs),
          "patience": 15 if max_epochs > 20 else 2, "seconds": float(seconds)}
    space = ("identity-initialised (residual) heads in the encoders' shared space" if residual
             else f"a {hp['embed_dim']}-d joint space with 2-layer heads")
    why = (f"about {n_train:,} training pairs: {space}, batch {hp['batch_size']} "
           f"({hp['batch_size'] - 1} negatives per positive), early stopping on validation recall@10 "
           f"(patience {hp['patience']}) and a {seconds:.0f}s cap per fit")
    return hp, why


def fit_alignment(method: str, Atr, Btr, Ava, Bva, hp: dict | None, seed: int, device: str):
    """A function mapping (A, B) features into a shared space, fitted on training pairs only;
    validation pairs are used for early stopping. Returns (transform, training history)."""
    if method == "zero_shot":
        return (lambda A, B: (l2n(A), l2n(B))), {}
    if method == "cca":
        k = max(2, min(32, len(Atr) // 10))
        return CCA(k=k).fit(Atr, Btr).transform, {"k": k}
    from .align import ContrastiveAligner

    al = ContrastiveAligner(Atr.shape[1], Btr.shape[1], seed=seed, device=device, **(hp or {}))
    al.fit(Atr, Btr, Ava, Bva)
    return (lambda A, B: (al.embed("a", A), al.embed("b", B))), al.history


def partner_percentiles(Za, Zb, n_random: int, rng, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """For each pair i: the share of ``n_random`` random partners (other rows of the same set)
    whose cosine similarity to item i is below that of its own partner (ties count half),
    and the own-partner cosine. 0.5 is what an unrelated partner scores on average."""
    a, b = l2n(Za), l2n(Zb)
    n = len(a)
    own = np.einsum("ij,ij->i", a, b)
    k = min(n_random, n - 1)
    pct = np.full(n, np.nan)
    if k < 1:
        return pct, own
    for s in range(0, n, chunk):
        idx = np.arange(s, min(n, s + chunk))
        j = rng.integers(0, n - 1, (len(idx), k))
        j = j + (j >= idx[:, None])                  # never the item itself
        sims = np.einsum("id,ikd->ik", a[idx], b[j])
        ref = own[idx, None]
        pct[idx] = ((sims < ref).sum(1) + 0.5 * (sims == ref).sum(1)) / k
    return pct, own


def halves(idx: np.ndarray, groups: np.ndarray | None, rng) -> tuple[np.ndarray, np.ndarray]:
    """Two halves of ``idx`` for cross-fitting, whole groups kept together when given."""
    if groups is not None and len(np.unique(groups[idx])) >= 2:
        levels = rng.permutation(np.unique(groups[idx]))
        first = np.isin(groups[idx], levels[: len(levels) // 2])
        return idx[first], idx[~first]
    perm = rng.permutation(idx)
    return np.sort(perm[: len(idx) // 2]), np.sort(perm[len(idx) // 2:])

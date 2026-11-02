"""Metric functions for screening, alignment and probes (from Project 3: evaluate/metrics.py,
retrieval.py and probes.py). Only the functions used here are copied; ``classification_scores``
keeps just the summary scores."""

from __future__ import annotations

import numpy as np


def l2n(Z: np.ndarray) -> np.ndarray:
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


def recall_at_k(Za: np.ndarray, Zb: np.ndarray, k: int = 10) -> float:
    """Mean of a->b and b->a recall@k for row-aligned pairs (cosine similarity)."""
    a = Za / np.maximum(np.linalg.norm(Za, axis=1, keepdims=True), 1e-9)
    b = Zb / np.maximum(np.linalg.norm(Zb, axis=1, keepdims=True), 1e-9)
    S = a @ b.T
    diag = np.diag(S)
    r_ab = (S > diag[:, None]).sum(1) < k
    r_ba = (S > diag[None, :]).sum(0) < k
    return float((r_ab.mean() + r_ba.mean()) / 2)


def ridge_r2(Xtr, Ytr, Xva, Yva) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(sc.transform(Xtr), Ytr)
    pred = m.predict(sc.transform(Xva))
    sse = ((Yva - pred) ** 2).sum()
    sst = ((Yva - Ytr.mean(0)) ** 2).sum()
    return float(1 - sse / max(sst, 1e-12))


def partner_ranks(Za: np.ndarray, Zb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """0-based rank of each true partner, a->b and b->a (ties counted in the query's favour)."""
    S = l2n(Za) @ l2n(Zb).T
    d = np.diag(S)
    return (S > d[:, None]).sum(1), (S > d[None, :]).sum(0)


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    return {"macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(n_classes)),
                                       zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "accuracy": float(np.mean(y_true == y_pred)), "n_test": int(len(y_true))}


def stratified_cap(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    idx = np.arange(len(y))
    if len(idx) <= n:
        return idx
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(y):
        m = idx[y == c]
        out.extend(rng.choice(m, max(1, int(round(n * len(m) / len(idx)))), replace=False))
    return np.sort(np.array(out))

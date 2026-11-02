"""Classical manifold-learning methods (from Project 1, unchanged).

All four build a neighbourhood graph or a full distance matrix and then solve an
eigenproblem. That shared structure is why they share a constraint: memory grows
as O(n^2), so every one of them needs the data subsampled before it can touch a
dataset of tens of thousands of samples. The registry records that limit; the planner is
expected to respect it.

What separates them is *which* geometry they try to preserve:

    MDS                  all pairwise distances, globally
    Isomap               pairwise distances measured *along* the manifold
    LLE                  each point's reconstruction from its own neighbours
    Laplacian Eigenmaps  proximity only -- near points stay near, far ones free
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding


def _as_dense(X) -> np.ndarray:
    """These estimators reject sparse input, so densify (callers bound the size)."""
    return np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float64)


def _safe_neighbors(requested: int, n: int, floor: int = 2) -> int:
    """Clamp a neighbour count to something the sample size can support."""
    return int(max(floor, min(requested, n - 1)))


def mds(X, n_components: int = 2, random_state: int = 0, metric: bool = True,
        n_init: int = 1, max_iter: int = 300, **_) -> np.ndarray:
    """Multidimensional scaling.

    Finds a layout whose pairwise distances match the originals as closely as
    possible, minimising a stress criterion by SMACOF. Unlike the neighbour
    methods it treats *all* distances as equally important, which makes it the
    best of these at preserving global structure and the worst at resolving
    fine local detail.

    With ``metric=False`` it fits only the rank ordering of distances, which is
    more robust when the distances themselves are on an arbitrary scale.
    """
    Xd = _as_dense(X)
    model = MDS(
        n_components=n_components,
        metric=metric,
        n_init=n_init,
        max_iter=max_iter,
        dissimilarity="euclidean",
        random_state=random_state,
        normalized_stress="auto",
    )
    return model.fit_transform(Xd)


def isomap(X, n_components: int = 2, random_state: int = 0,
           n_neighbors: int = 10, **_) -> np.ndarray:
    """Isometric feature mapping.

    MDS applied to *geodesic* distances: it builds a k-nearest-neighbour graph
    and measures distance as the shortest path along it, so a Swiss roll is
    unrolled rather than folded flat.

    ``n_neighbors`` is the critical and fragile parameter. Too small and the
    graph fragments into disconnected components; too large and edges
    "short-circuit" across the fold, collapsing the very structure the method
    exists to find.
    """
    Xd = _as_dense(X)
    k = _safe_neighbors(n_neighbors, len(Xd))
    model = Isomap(n_components=n_components, n_neighbors=k)
    return model.fit_transform(Xd)


def lle(X, n_components: int = 2, random_state: int = 0, n_neighbors: int = 10,
        method: str = "standard", reg: float = 1e-3, **_) -> np.ndarray:
    """Locally linear embedding.

    Reconstructs each point as a weighted combination of its neighbours, then
    finds the low-dimensional layout preserving those weights. It never forms a
    global distance matrix, only local ones.

    The ``modified`` variant uses several weight vectors per neighbourhood and
    is markedly more stable than ``standard``, which is prone to collapsing when
    neighbourhoods are close to degenerate. Both variants require enough
    neighbours to determine the local geometry, so the count is raised if the
    requested value is too small for the target dimension.
    """
    Xd = _as_dense(X)
    k = _safe_neighbors(n_neighbors, len(Xd))

    # Each variant has its own minimum neighbourhood size; below it, sklearn
    # raises rather than degrading, so we raise k instead of failing the run.
    if method == "modified":
        k = max(k, n_components + 1)
    elif method == "hessian":
        k = max(k, 1 + n_components * (n_components + 3) // 2)
    elif method == "ltsa":
        k = max(k, n_components + 1)
    k = _safe_neighbors(k, len(Xd))

    model = LocallyLinearEmbedding(
        n_components=n_components,
        n_neighbors=k,
        method=method,
        reg=reg,
        random_state=random_state,
    )
    return model.fit_transform(Xd)


def laplacian_eigenmaps(X, n_components: int = 2, random_state: int = 0,
                        n_neighbors: int = 10, **_) -> np.ndarray:
    """Laplacian eigenmaps, via scikit-learn's SpectralEmbedding.

    Builds a neighbourhood graph and takes the eigenvectors of its graph
    Laplacian belonging to the smallest non-zero eigenvalues. This minimises a
    weighted sum of squared embedded distances between neighbours, so connected
    points are pulled together while unconnected pairs are left unconstrained --
    hence strong cluster separation and unreliable between-cluster distances.

    ``SpectralEmbedding`` with ``affinity="nearest_neighbors"`` *is* Laplacian
    eigenmaps; the name differs, the algorithm does not.
    """
    Xd = _as_dense(X)
    k = _safe_neighbors(n_neighbors, len(Xd))
    model = SpectralEmbedding(
        n_components=n_components,
        affinity="nearest_neighbors",
        n_neighbors=k,
        random_state=random_state,
    )
    return model.fit_transform(Xd)

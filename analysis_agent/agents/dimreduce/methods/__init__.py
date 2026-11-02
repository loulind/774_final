"""Dimension reduction methods behind a single uniform interface (from Project 1).

Every method is a function with the same shape::

    fit_transform(X, n_components=2, random_state=0, **params) -> ndarray

so the sub-agent can run any of them without knowing which it holds. Extra keyword
arguments a method does not recognise are ignored rather than raising.

Adapted: Project 1's GPLVM is not copied (it is limited to about 2,000 samples and
was never the chosen map); the other eleven methods are unchanged.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp

from .diffusion import diffusion_map
from .linear import kernel_pca, pca, sparse_pca, truncated_svd
from .manifold import isomap, laplacian_eigenmaps, lle, mds
from .neighbor import tsne, umap_embed

METHODS: dict[str, Callable[..., np.ndarray]] = {
    # linear / matrix factorisation
    "pca": pca,
    "truncated_svd": truncated_svd,
    "kernel_pca": kernel_pca,
    "sparse_pca": sparse_pca,
    # classical manifold learning
    "mds": mds,
    "isomap": isomap,
    "lle": lle,
    "laplacian_eigenmaps": laplacian_eigenmaps,
    "diffusion_map": diffusion_map,
    # neighbour embeddings
    "tsne": tsne,
    "umap": umap_embed,
}


def ensure_dense(X, limit_mb: float = 4_000):
    """Densify a sparse matrix, refusing if the result would be unreasonable."""
    if not sp.issparse(X):
        return np.asarray(X, dtype=np.float32)
    n, p = X.shape
    needed = n * p * 4 / 1e6
    if needed > limit_mb:
        raise MemoryError(
            f"this method needs dense input, which would take {needed:,.0f} MB "
            f"for a {n:,} x {p:,} matrix (limit {limit_mb:,.0f} MB). "
            f"Reduce the feature count first."
        )
    return np.asarray(X.todense(), dtype=np.float32)


# Owned by the caller, not by any individual method: a parameter dict that names these
# would otherwise collide with the explicit arguments below.
RESERVED_PARAMS = {"n_components", "random_state"}


def run_method(name: str, X, n_components: int = 2, random_state: int = 0,
               params: dict | None = None, **extra) -> np.ndarray:
    """Look up and run a method by name; one output row per input row, padded to
    ``n_components`` columns when the data support fewer."""
    fn = METHODS.get(name)
    if fn is None:
        raise KeyError(f"unknown method '{name}'; known: {sorted(METHODS)}")

    merged = {k: v for k, v in {**(params or {}), **extra}.items()
              if k not in RESERVED_PARAMS}
    Z = np.asarray(fn(X, n_components=n_components, random_state=random_state,
                      **merged), dtype=np.float32)

    if Z.ndim == 1:
        Z = Z[:, None]

    # One row out per row in is the whole contract (SparsePCA, for one, returns two rows
    # for a one-sample input), so check it rather than misalign an embedding silently.
    if len(Z) != X.shape[0]:
        raise ValueError(
            f"{name} returned {len(Z)} rows for {X.shape[0]} input samples; "
            f"the embedding does not correspond to the data"
        )

    # Data can support fewer components than were asked for; pad with zeros, the honest
    # representation of a dimension the data do not contain.
    if Z.shape[1] < n_components:
        padding = np.zeros((len(Z), n_components - Z.shape[1]), dtype=np.float32)
        Z = np.hstack([Z, padding])

    return Z

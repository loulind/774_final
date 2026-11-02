"""Neighbour-embedding methods: t-SNE and UMAP (from Project 1, unchanged).

Both optimise a notion of *neighbourhood* agreement rather than reconstruction
error. That makes them excellent at revealing clusters and unreliable about the
distances between those clusters -- a distinction the evaluation module in Phase
2 measures explicitly, and one the generated report is expected to state.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.manifold import TSNE


def tsne(X, n_components: int = 2, random_state: int = 0,
         perplexity: float = 30.0, learning_rate="auto", init: str = "pca",
         early_exaggeration: float = 12.0, max_iter: int = 1000,
         **_) -> np.ndarray:
    """t-distributed stochastic neighbour embedding.

    Matches a distribution over neighbour pairs in the original space to one in
    the embedding, using a heavy-tailed t-distribution in the latter. The heavy
    tail is what stops distant points crowding into the centre, and it is also
    why between-cluster distances in a t-SNE plot carry little meaning.

    ``perplexity`` is roughly the effective number of neighbours each point
    attends to, and it must stay well below the sample size -- scikit-learn
    requires ``perplexity < n_samples``, but the estimate degrades long before
    that, so we cap it at (n-1)/3.

    ``init="pca"`` rather than the historical random initialisation: PCA
    initialisation makes runs far more reproducible and preserves more of the
    global arrangement, and is now the recommended default.
    """
    Xd = np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float64)
    n = len(Xd)

    # scikit-learn's Barnes-Hut t-SNE *segfaults* on degenerate input -- an
    # all-zero matrix, or any data whose points are all identical -- because the
    # perplexity calibration divides by a zero neighbour distance down in the C
    # extension. Measured: `TSNE().fit_transform(np.zeros((60, 8)))` terminates
    # the interpreter with SIGSEGV.
    #
    # This is the one failure mode the surrounding try/except cannot contain: a
    # segfault is not a Python exception, so it takes the whole run with it
    # rather than costing one method. It has to be prevented here, by raising an
    # ordinary error the executor can catch and record.
    if n < 3:
        raise ValueError(f"t-SNE needs at least 3 samples, got {n}")

    probe = Xd
    if n > 2_000:
        rng = np.random.default_rng(random_state)
        probe = Xd[rng.choice(n, 2_000, replace=False)]
    if len(np.unique(probe, axis=0)) < 3:
        raise ValueError(
            "t-SNE requires at least 3 distinct points; this data is degenerate "
            "(all rows identical or nearly so), which crashes the Barnes-Hut "
            "implementation rather than raising"
        )

    max_perplexity = max(5.0, (n - 1) / 3.0)
    p = float(min(perplexity, max_perplexity))

    model = TSNE(
        n_components=n_components,
        perplexity=p,
        learning_rate=learning_rate,
        init=init,
        early_exaggeration=early_exaggeration,
        max_iter=max_iter,
        random_state=random_state,
    )
    return model.fit_transform(Xd)


def umap_embed(X, n_components: int = 2, random_state: int = 0,
               n_neighbors: int = 15, min_dist: float = 0.1,
               metric: str = "euclidean", **_) -> np.ndarray:
    """Uniform Manifold Approximation and Projection.

    ``n_neighbors`` sets the scale at which structure is preserved: small values
    emphasise fine local detail, large values push towards a more global,
    PCA-like picture. ``min_dist`` controls only how tightly points may pack in
    the embedding, and is cosmetic rather than structural.

    Accepts sparse input directly, so no densification is needed.
    """
    import umap  # imported lazily: it triggers a slow numba compile on first use

    n = X.shape[0]
    # UMAP requires n_neighbors < n_samples, and degrades badly as it approaches.
    k = int(max(2, min(n_neighbors, n - 1)))

    model = umap.UMAP(
        n_components=n_components,
        n_neighbors=k,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return model.fit_transform(X)

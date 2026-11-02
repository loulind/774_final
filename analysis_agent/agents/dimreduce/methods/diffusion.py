"""Diffusion maps (Coifman & Lafon, 2006), implemented in Project 1 (unchanged).

There is no scikit-learn implementation, and the maintained third-party packages
are thin wrappers around roughly this much code. Writing it out is worth more
than importing it: the intermediate quantities are exactly the ones the method
is usually explained in terms of.

The idea is to define a random walk on the data and embed points by how that
walk mixes. Two points are close in the embedding when a walker starting at
either reaches the same places with similar probability after t steps -- a
notion of proximity that follows the shape of the data rather than cutting
across empty space, and that is robust to noise because it averages over all
paths rather than trusting any single edge.

Algorithm:

    1. squared distances                D2
    2. Gaussian kernel                  W = exp(-D2 / eps)
    3. density normalisation            W <- W / (q_i^a q_j^a),  q = row sums
    4. row-normalise to a Markov chain  P = D^-1 W
    5. eigendecompose P
    6. embed                            psi_i = (lam_1^t phi_1(i), ...)

Step 3 is what distinguishes diffusion maps from Laplacian eigenmaps. With
``alpha=1`` the influence of sampling density is divided out, so the embedding
reflects the geometry of the manifold rather than where the samples happen to
be dense -- which for biological data usually matters, since cell abundance and
biological similarity are different things.

Step 5 is done on a symmetric conjugate of P rather than P itself. P is not
symmetric, so its numerical eigendecomposition can return complex values from
rounding error alone; ``S = D^(1/2) P D^(-1/2)`` is symmetric, has identical
eigenvalues, and its eigenvectors map back by a diagonal scaling.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from sklearn.metrics.pairwise import euclidean_distances

# A dense n x n eigendecomposition is O(n^3) in time and O(n^2) in memory.
MAX_SAMPLES = 8_000


def diffusion_map(X, n_components: int = 2, random_state: int = 0,
                  n_neighbors: int = 10, alpha: float = 1.0, t: float = 1.0,
                  epsilon: float | None = None, **_) -> np.ndarray:
    """Embed with diffusion maps.

    Parameters
    ----------
    n_neighbors
        Used only to choose the kernel bandwidth when ``epsilon`` is not given.
    alpha
        Density normalisation exponent. 1.0 recovers the Laplace-Beltrami
        operator on the manifold (density divided out), 0.5 the Fokker-Planck
        diffusion, 0.0 the plain graph Laplacian -- at which point this reduces
        to Laplacian eigenmaps with a Gaussian kernel.
    t
        Diffusion time. Larger values suppress the faster-decaying components,
        emphasising coarser structure.
    epsilon
        Kernel bandwidth (in squared-distance units). Defaults to the median
        squared distance to the ``n_neighbors``-th nearest neighbour, which
        adapts the scale to the data rather than assuming one.
    """
    Xd = np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float64)
    n = len(Xd)

    if n > MAX_SAMPLES:
        raise MemoryError(
            f"diffusion maps need a dense {n:,} x {n:,} eigendecomposition "
            f"({n * n * 8 / 1e9:.1f} GB, O(n^3) time); limit is {MAX_SAMPLES:,} "
            f"samples. Subsample first."
        )
    if n <= n_components + 1:
        raise ValueError(
            f"need more than {n_components + 1} samples to extract "
            f"{n_components} non-trivial components; got {n}"
        )

    # --- 1. squared distances ---------------------------------------------
    D2 = euclidean_distances(Xd, squared=True)

    # --- 2. kernel bandwidth and Gaussian kernel ---------------------------
    if epsilon is None:
        k = int(max(1, min(n_neighbors, n - 1)))
        # Column k of the partially sorted row is the k-th smallest distance;
        # column 0 is the point's zero distance to itself.
        kth = np.partition(D2, k, axis=1)[:, k]
        epsilon = float(np.median(kth))
        if epsilon <= 0:  # heavy duplication collapses the median to zero
            positive = D2[D2 > 0]
            epsilon = float(np.median(positive)) if positive.size else 1.0

    W = np.exp(-D2 / epsilon)

    # --- 3. density normalisation ------------------------------------------
    if alpha:
        q = W.sum(axis=1)
        q[q == 0] = 1.0
        q_alpha = q ** alpha
        W = W / np.outer(q_alpha, q_alpha)

    # --- 4/5. symmetric conjugate of the Markov matrix, then eigh ----------
    d = W.sum(axis=1)
    d[d <= 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(d)
    S = W * np.outer(d_inv_sqrt, d_inv_sqrt)
    S = (S + S.T) / 2.0  # kill residual asymmetry from floating-point error

    k_eig = n_components + 1  # one extra for the trivial constant eigenvector
    vals, vecs = eigh(S, subset_by_index=[n - k_eig, n - 1])
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # Map eigenvectors of S back to those of P.
    phi = vecs * d_inv_sqrt[:, None]

    # --- 6. drop the trivial component and scale by lambda^t ---------------
    # phi[:, 0] is constant with eigenvalue 1: it carries no information, and
    # keeping it would waste an embedding dimension on a straight line.
    lam = np.clip(vals[1:k_eig], 0.0, None)
    return phi[:, 1:k_eig] * (lam ** t)

"""Linear (matrix-factorisation) methods: PCA and its variants (from Project 1, unchanged)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, TruncatedSVD


def pca(X, n_components: int = 2, random_state: int = 0, **_) -> np.ndarray:
    """Principal component analysis.

    Projects onto the directions of greatest variance. Linear, deterministic up
    to sign, and the only method here whose axes carry a direct interpretation
    (a loading per input feature). It is the natural baseline: if PCA already
    separates the structure, a nonlinear method has little left to add.

    Sparse input uses the arpack solver, which centers implicitly and so avoids
    materialising a dense copy.
    """
    n, p = X.shape
    k = min(n_components, min(n, p) - 1)
    solver = "arpack" if sp.issparse(X) else "auto"
    model = PCA(n_components=k, svd_solver=solver, random_state=random_state)
    return model.fit_transform(X)


def truncated_svd(X, n_components: int = 2, random_state: int = 0,
                  **_) -> np.ndarray:
    """PCA variant: truncated SVD without centering (a.k.a. LSA).

    Because it skips centering it operates natively on sparse matrices at full
    scale. The trade-off is that the leading component tends to align with the
    data's mean rather than its greatest variation, so it is preferable to PCA
    only when centering is genuinely unaffordable.
    """
    n, p = X.shape
    k = min(n_components, min(n, p) - 1)
    model = TruncatedSVD(n_components=k, random_state=random_state)
    return model.fit_transform(X)


def _median_heuristic_gamma(Xd: np.ndarray, random_state: int = 0,
                            max_samples: int = 2_000) -> float:
    """Choose an RBF bandwidth from the data's own distance scale.

    scikit-learn defaults ``gamma`` to 1/n_features, which assumes features are
    standardised and distances are therefore O(1). That assumption fails badly
    on real input: on PCA-reduced image data the median squared distance was ~2,400,
    so the default gamma of 0.02 gives kernel entries of exp(-48) ~ 1e-21. The
    kernel matrix becomes numerically the identity, every point is its own sole
    neighbour, and the embedding is noise -- silently, with no warning.

    The median heuristic sets gamma = 1/median(squared distance), putting the
    typical kernel entry near exp(-1) ~ 0.37, where the kernel actually
    discriminates. It is the standard fix and it is data-adaptive.
    """
    from sklearn.metrics.pairwise import euclidean_distances

    Xs = Xd
    if len(Xd) > max_samples:
        rng = np.random.default_rng(random_state)
        Xs = Xd[rng.choice(len(Xd), max_samples, replace=False)]

    d2 = euclidean_distances(Xs, squared=True)
    positive = d2[d2 > 0]
    if positive.size == 0:
        return 1.0
    return float(1.0 / np.median(positive))


def kernel_pca(X, n_components: int = 2, random_state: int = 0,
               kernel: str = "rbf", gamma: float | None = None,
               degree: int = 3, **_) -> np.ndarray:
    """PCA variant: PCA in an implicit feature space.

    Replaces the covariance matrix with a kernel matrix, so it can capture
    curved structure that linear PCA cannot -- while remaining an eigenproblem
    rather than an iterative optimisation.

    The cost is that the kernel matrix is n x n, so this scales like the
    manifold methods rather than like PCA, and that ``gamma`` matters a great
    deal: too large and every point is its own neighbour, too small and the
    kernel degenerates towards linear. When it is not supplied we set it by the
    median heuristic rather than accepting scikit-learn's default, which fails
    silently on unstandardised input (see :func:`_median_heuristic_gamma`).
    """
    Xd = np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float64)

    # "scale" and "auto" are how scikit-learn's *SVM* estimators spell an
    # automatically chosen bandwidth. KernelPCA does not accept them and raises,
    # but the intent is unambiguous, so treat them as a request for the median
    # heuristic rather than failing the run over a sibling estimator's idiom.
    if isinstance(gamma, str):
        if gamma.lower() not in {"scale", "auto"}:
            raise ValueError(
                f"kernel_pca gamma={gamma!r} is not recognised; pass a positive "
                f"float, or None/'scale' to select it from the data"
            )
        gamma = None

    if gamma is None and kernel in {"rbf", "laplacian", "sigmoid"}:
        gamma = _median_heuristic_gamma(Xd, random_state=random_state)

    model = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        random_state=random_state,
    )
    return model.fit_transform(Xd)


def sparse_pca(X, n_components: int = 2, random_state: int = 0,
               alpha: float = 1.0, max_iter: int = 200, **_) -> np.ndarray:
    """PCA variant: components constrained to load on few features.

    An L1 penalty forces most loadings to zero, so each component names a short
    list of features rather than mixing all of them. On expression data that means a
    component can be read as a gene set; on pixel data it means far less.

    Interpretability is bought with time -- this solves a dictionary-learning
    problem iteratively rather than an eigenproblem -- and with variance, since
    the components are no longer the optimal low-rank fit and are not
    orthogonal.
    """
    Xd = np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float64)
    model = SparsePCA(
        n_components=n_components,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
    )
    return model.fit_transform(Xd)

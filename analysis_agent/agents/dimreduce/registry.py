"""What each dimension-reduction method needs, costs, and is good for (from Project 1).

The sub-agent's knowledge of its own methods: which can run at a given n and p, what
each preserves, and its caveats (quoted when a map is interpreted). Adapted: the GPLVM
entry is not copied, feasibility takes n, p and sparsity directly instead of a Project 1
profile, and each method has a rough run-time model so candidates that would not fit
the step's time budget are skipped before anything is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MethodInfo:
    name: str
    family: str
    complexity: str
    accepts_sparse: bool
    deterministic: bool
    preserves: str
    use_when: str
    caveats: str
    hyperparameters: dict[str, str] = field(default_factory=dict)
    max_samples: int | None = None  # None means no practical ceiling
    nxn_matrices: int = field(
        default=0,
        metadata={"doc": "how many dense n x n float64 matrices this method "
                         "holds at peak; 0 means it never forms one"},
    )


REGISTRY: dict[str, MethodInfo] = {
    # ---------------------------------------------------- linear / factorisation
    "pca": MethodInfo(
        name="pca", family="linear", complexity="O(n p k)",
        accepts_sparse=True, deterministic=True, preserves="global",
        max_samples=None,
        hyperparameters={},
        use_when="always, as a baseline; especially when the variance spectrum "
                 "decays quickly, which means the data is near a linear subspace",
        caveats="cannot represent curved structure; a rapidly decaying spectrum "
                "is evidence it will suffice, a flat one that it will not",
    ),
    "truncated_svd": MethodInfo(
        name="truncated_svd", family="linear", complexity="O(n p k)",
        accepts_sparse=True, deterministic=True, preserves="global",
        max_samples=None,
        hyperparameters={},
        use_when="sparse data too large to center, since it skips centering "
                 "and so preserves sparsity at full scale",
        caveats="the leading component tends to align with the data mean rather "
                "than its greatest variation; prefer PCA when centering is affordable",
    ),
    "kernel_pca": MethodInfo(
        name="kernel_pca", family="linear", complexity="O(n^2 p) to O(n^3)",
        accepts_sparse=False, deterministic=True, preserves="global",
        max_samples=6_000,
        nxn_matrices=3,
        hyperparameters={
            "kernel": "'rbf', 'poly', 'cosine', 'sigmoid'",
            "gamma": "float or None; RBF width. None selects it by the median "
                     "heuristic from the data's own distance scale",
            "degree": "int, for the polynomial kernel",
        },
        use_when="a PCA variant is wanted that can capture curvature, and n is "
                 "small enough for an n x n kernel matrix",
        caveats="results depend strongly on gamma -- a badly scaled value makes "
                "the kernel numerically the identity and the output pure noise, "
                "with no error raised; scales with n^2 in memory, unlike ordinary PCA",
    ),
    "sparse_pca": MethodInfo(
        name="sparse_pca", family="linear", complexity="iterative, O(n p k) per pass",
        accepts_sparse=False, deterministic=False, preserves="global",
        max_samples=20_000,
        hyperparameters={"alpha": "float, L1 strength; higher gives sparser loadings",
                         "max_iter": "int"},
        use_when="component interpretability matters and features are named, so "
                 "that a component can be read as a short feature list "
                 "(e.g. a gene set)",
        caveats="much slower than PCA; components are neither orthogonal nor "
                "variance-optimal; near-useless on anonymous features such as pixels",
    ),
    # ------------------------------------------------- classical manifold learning
    "mds": MethodInfo(
        name="mds", family="manifold", complexity="O(n^2) memory, O(n^2) per iteration",
        accepts_sparse=False, deterministic=False, preserves="global",
        max_samples=3_000,
        nxn_matrices=2,
        hyperparameters={"metric": "bool; False fits only the rank order of distances",
                         "n_init": "int, restarts", "max_iter": "int"},
        use_when="faithful *global* distances matter more than crisp clusters",
        caveats="needs a full distance matrix; slow; non-metric variant is more "
                "robust when distances are on an arbitrary scale",
    ),
    "isomap": MethodInfo(
        name="isomap", family="manifold", complexity="O(n^2 log n)",
        accepts_sparse=False, deterministic=True, preserves="global",
        max_samples=6_000,
        nxn_matrices=3,
        hyperparameters={"n_neighbors": "int, 5-20; the single critical parameter"},
        use_when="the data is believed to lie on a curved but connected manifold "
                 "and global geodesic structure is wanted",
        caveats="n_neighbors too small fragments the graph; too large lets edges "
                "short-circuit across folds, destroying the structure it exists to find",
    ),
    "lle": MethodInfo(
        name="lle", family="manifold", complexity="O(n k^3) plus sparse eigensolve",
        accepts_sparse=False, deterministic=True, preserves="local",
        max_samples=20_000,
        hyperparameters={"n_neighbors": "int, 5-30",
                         "method": "'standard', 'modified', 'hessian', 'ltsa'",
                         "reg": "float, regularises the local fits"},
        use_when="local neighbourhood geometry is the object of interest and no "
                 "global distance matrix is affordable",
        caveats="'standard' collapses on near-degenerate neighbourhoods -- prefer "
                "'modified'; distances between separated regions are meaningless",
    ),
    "laplacian_eigenmaps": MethodInfo(
        name="laplacian_eigenmaps", family="manifold",
        complexity="sparse eigensolve on a k-NN graph",
        accepts_sparse=False, deterministic=True, preserves="local",
        max_samples=20_000,
        hyperparameters={"n_neighbors": "int, 5-30"},
        use_when="cluster structure is the question; it separates connected "
                 "groups strongly",
        caveats="leaves unconnected pairs unconstrained, so between-cluster "
                "distances carry no information; can collapse if the graph is disconnected",
    ),
    "diffusion_map": MethodInfo(
        name="diffusion_map", family="manifold", complexity="O(n^2) memory, O(n^3) eigensolve",
        accepts_sparse=False, deterministic=True, preserves="both",
        max_samples=5_000,
        nxn_matrices=3,
        hyperparameters={
            "n_neighbors": "int; sets the kernel bandwidth when epsilon is unset",
            "alpha": "float 0-1; 1.0 divides out sampling density",
            "t": "float, diffusion time; larger emphasises coarser structure",
            "epsilon": "float or None; kernel bandwidth in squared-distance units",
        },
        use_when="sampling density and geometry should be separated (alpha=1), "
                 "or continuous trajectories rather than discrete clusters are expected",
        caveats="dense eigendecomposition limits sample size; sensitive to the "
                "bandwidth epsilon",
    ),
    # ------------------------------------------------------- neighbour embeddings
    "tsne": MethodInfo(
        name="tsne", family="neighbor", complexity="O(n log n) with Barnes-Hut",
        accepts_sparse=False, deterministic=False, preserves="local",
        max_samples=50_000,
        hyperparameters={
            "perplexity": "float 5-50, roughly the effective neighbour count; "
                          "must stay well under n",
            "early_exaggeration": "float, default 12",
            "learning_rate": "'auto' or float",
            "init": "'pca' strongly preferred over 'random'",
        },
        use_when="revealing cluster structure in data with tens of thousands of "
                 "samples or fewer",
        caveats="cluster sizes and between-cluster distances are not meaningful; "
                "different seeds give visibly different layouts; always PCA-reduce first",
    ),
    "umap": MethodInfo(
        name="umap", family="neighbor", complexity="approximately O(n^1.14)",
        accepts_sparse=True, deterministic=False, preserves="local",
        max_samples=None,
        hyperparameters={
            "n_neighbors": "int 5-100; small emphasises local detail, large "
                           "pushes towards a global, PCA-like picture",
            "min_dist": "float 0.0-0.99; cosmetic packing only, not structural",
            "metric": "'euclidean', 'cosine', 'correlation', ...",
        },
        use_when="the default choice for large data; faster than t-SNE and "
                 "retains somewhat more global structure",
        caveats="between-cluster distances are more meaningful than t-SNE's but "
                "still not to be trusted; can invent apparent clusters in uniform noise",
    ),
}


# Rough seconds on one laptop-class CPU, from Project 1 timings; used only to skip
# candidates that cannot finish within the step budget, never to rank them.
COST_SECONDS = {
    "pca": lambda n, p: 1e-8 * n * p + 0.5,
    "truncated_svd": lambda n, p: 1e-8 * n * p + 0.5,
    "kernel_pca": lambda n, p: 2e-10 * n ** 3 + 1e-8 * n * n,
    "sparse_pca": lambda n, p: 1e-5 * n * p + 5,
    "mds": lambda n, p: 4e-6 * n ** 2 + 2,
    "isomap": lambda n, p: 1.5e-6 * n ** 2 + 2,
    "lle": lambda n, p: 2e-3 * n + 2,
    "laplacian_eigenmaps": lambda n, p: 1e-3 * n + 2,
    "diffusion_map": lambda n, p: 2e-10 * n ** 3 + 1e-8 * n * n + 1,
    "tsne": lambda n, p: 5e-3 * n + 5,
    "umap": lambda n, p: 1.2e-3 * n + 10,
}


def estimated_seconds(name: str, n: int, p: int) -> float:
    return float(COST_SECONDS[name](n, p))


def feasibility(n: int, p: int, is_sparse: bool = False) -> dict[str, str | None]:
    """Return ``{method: reason it cannot run}``; ``None`` means it can."""
    out: dict[str, str | None] = {}
    for name, info in REGISTRY.items():
        reason = None
        if info.max_samples is not None and n > info.max_samples:
            reason = f"needs n <= {info.max_samples:,} but the input has {n:,} samples"
        elif not info.accepts_sparse and is_sparse and n * p * 4 / 1e6 > 2_000:
            reason = f"requires dense input, which would be {n * p * 4 / 1e6:,.0f} MB"
        out[name] = reason
    return out

"""Score embeddings against the space they came from (from Project 1, unchanged).

There is no single number that says an embedding is good, and pretending
otherwise is the most common way to misread one. The metrics here deliberately
disagree with each other, and the disagreement is the signal:

    local metrics    (trustworthiness, continuity, k-NN overlap)
        Did each point keep its own neighbours? t-SNE and UMAP optimise
        essentially this, so they should win here, and it means little that they do.

    global metrics   (distance correlation, stress)
        Are distances between *distant* points still meaningful? PCA and MDS
        should win. A UMAP embedding scoring well locally and badly here is
        behaving exactly as designed -- and the report should say so rather than
        declaring a winner.

    label agreement  (silhouette, k-NN accuracy, ARI)
        Does the embedding recover known structure? Only available when labels
        exist; for unlabelled data we cluster the *original* space and use those
        assignments as a reference, which tests preservation rather than truth.

    stability
        Re-run with a different seed: how much of the embedding survives? This
        separates structure the method found in the data from structure it
        invented during optimisation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

# Several metrics need an n x n distance matrix; this bounds that cost.
MAX_SAMPLES_FOR_METRICS = 3_000
N_DISTANCE_PAIRS = 200_000


def _densify(X) -> np.ndarray:
    """Return a 2-D dense array.

    ``np.asarray`` on a SciPy sparse matrix produces a zero-dimensional object
    array rather than raising, so every later index silently becomes an
    IndexError far from the cause. Convert explicitly.
    """
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    return X.reshape(len(X), -1) if X.ndim != 2 else X


def _paired_subsample(X, Z, y=None, limit=MAX_SAMPLES_FOR_METRICS, random_state=0):
    """Subsample X, Z and y together, preserving the row correspondence."""
    n = len(Z)
    if n <= limit:
        return _densify(X), np.asarray(Z), (None if y is None else np.asarray(y))
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=limit, replace=False)
    return (_densify(X[idx]), np.asarray(Z)[idx],
            None if y is None else np.asarray(y)[idx])


def _knn_indices(A, k: int) -> np.ndarray:
    """Indices of each row's k nearest neighbours, excluding itself."""
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(A) - 1)).fit(A)
    return nn.kneighbors(A, return_distance=False)[:, 1:]


def knn_overlap(A, B, k: int = 10) -> float:
    """Mean fraction of each point's k nearest neighbours shared between A and B.

    The most directly interpretable of the local metrics: 0.7 means that on
    average 7 of every 10 neighbours survived the projection. Also used to
    compare two embeddings of the same data, which is how stability is measured.
    """
    ka = _knn_indices(A, k)
    kb = _knn_indices(B, k)
    shared = [len(np.intersect1d(a, b, assume_unique=False)) for a, b in zip(ka, kb)]
    return float(np.mean(shared) / ka.shape[1])


def continuity(X, Z, k: int = 10) -> float:
    """Fraction of original neighbourhoods preserved, penalised by rank.

    The mirror image of trustworthiness. Trustworthiness punishes points pulled
    together that were originally apart (false neighbours); continuity punishes
    points pushed apart that were originally together (missed neighbours).

    Both are needed because the two failure modes are opposite. An embedding
    that *tears* a real cluster in half stays trustworthy -- everything it shows
    as close really was close -- while continuity collapses, because genuine
    neighbours were separated. An embedding that *merges* two distinct clusters
    stays continuous -- no real neighbour was lost -- while trustworthiness
    collapses, because it now asserts adjacencies that do not exist. Either
    metric alone is blind to one of these.

    Formally continuity is trustworthiness with the two spaces exchanged, so
    that is exactly how it is computed.
    """
    return float(trustworthiness(Z, X, n_neighbors=k))


def distance_correlation(X, Z, random_state: int = 0) -> float:
    """Spearman correlation between original and embedded pairwise distances.

    Computed on a random sample of pairs rather than all of them. Rank
    correlation rather than Pearson, because no method preserves absolute
    distance scale and it would be unfair to ask.
    """
    n = len(Z)
    if n < 4:
        return float("nan")
    rng = np.random.default_rng(random_state)
    n_pairs = min(N_DISTANCE_PAIRS, n * (n - 1) // 2)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    dx = np.linalg.norm(np.asarray(X)[i] - np.asarray(X)[j], axis=1)
    dz = np.linalg.norm(np.asarray(Z)[i] - np.asarray(Z)[j], axis=1)
    if np.std(dx) == 0 or np.std(dz) == 0:
        return float("nan")
    return float(spearmanr(dx, dz).statistic)


def kruskal_stress(X, Z, random_state: int = 0) -> float:
    """Kruskal stress-1: normalised disagreement between distance sets.

    Lower is better; 0 is a perfect match. The embedded distances are first
    rescaled by the least-squares optimal factor, because a faithful embedding
    at half scale should not be penalised for the scale.
    """
    n = len(Z)
    if n < 4:
        return float("nan")
    rng = np.random.default_rng(random_state)
    n_pairs = min(N_DISTANCE_PAIRS, n * (n - 1) // 2)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    dx = np.linalg.norm(np.asarray(X)[i] - np.asarray(X)[j], axis=1)
    dz = np.linalg.norm(np.asarray(Z)[i] - np.asarray(Z)[j], axis=1)
    denom = np.sum(dz ** 2)
    if denom == 0 or np.sum(dx ** 2) == 0:
        return float("nan")
    scale = np.sum(dx * dz) / denom
    return float(np.sqrt(np.sum((dx - scale * dz) ** 2) / np.sum(dx ** 2)))


def reference_labels(X, n_neighbors: int = 15, resolution: float = 1.0,
                     random_state: int = 0) -> np.ndarray:
    """Cluster the *original* space, to serve as a reference for unlabelled data.

    Uses Leiden community detection on a k-nearest-neighbour graph, the standard
    approach for expression data. These are emphatically not ground truth: they
    are a summary of the structure present before reduction, so agreement with
    them measures how much of that structure each embedding kept.
    """
    import igraph as ig
    import leidenalg
    from sklearn.neighbors import kneighbors_graph

    k = int(max(2, min(n_neighbors, len(X) - 1)))
    A = kneighbors_graph(X, k, mode="connectivity", include_self=False)
    A = A.maximum(A.T)  # an edge either way makes the pair neighbours
    sources, targets = A.nonzero()
    graph = ig.Graph(n=A.shape[0],
                     edges=[(int(s), int(t)) for s, t in zip(sources, targets)],
                     directed=False)
    graph.simplify()
    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution, seed=random_state,
    )
    return np.asarray(partition.membership)


def evaluate_embedding(X, Z, y=None, ks=(10, 50), random_state: int = 0,
                       label_source: str = "given") -> dict[str, float]:
    """Score one embedding against the space it came from.

    ``X`` is the preprocessed matrix the method was fitted on -- not the raw
    data -- since the question is what the *projection* lost, not what
    preprocessing lost.
    """
    Xs, Zs, ys = _paired_subsample(X, Z, y, random_state=random_state)
    n = len(Zs)
    metrics: dict[str, float] = {"n_evaluated": float(n)}

    if not np.all(np.isfinite(Zs)):
        metrics["invalid_embedding"] = 1.0
        return metrics

    for k in ks:
        if n <= k + 1:
            continue
        try:
            metrics[f"trustworthiness@{k}"] = float(
                trustworthiness(Xs, Zs, n_neighbors=k))
            metrics[f"continuity@{k}"] = continuity(Xs, Zs, k=k)
            metrics[f"knn_overlap@{k}"] = knn_overlap(Xs, Zs, k=k)
        except Exception:
            pass

    metrics["distance_spearman"] = distance_correlation(Xs, Zs, random_state)
    metrics["kruskal_stress"] = kruskal_stress(Xs, Zs, random_state)

    if ys is not None and len(np.unique(ys)) > 1:
        n_classes = len(np.unique(ys))
        try:
            metrics["silhouette"] = float(silhouette_score(Zs, ys))
        except Exception:
            pass
        try:
            knn = KNeighborsClassifier(n_neighbors=min(15, n - 1))
            metrics["knn_accuracy"] = float(
                cross_val_score(knn, Zs, ys, cv=min(5, n_classes)).mean())
        except Exception:
            pass
        try:
            pred = KMeans(n_clusters=n_classes, n_init=10,
                          random_state=random_state).fit_predict(Zs)
            metrics["ari"] = float(adjusted_rand_score(ys, pred))
        except Exception:
            pass
        metrics["n_reference_classes"] = float(n_classes)

    return metrics


def stability(embed_fn, X, seeds=(0, 1), k: int = 10) -> float:
    """Re-run a method under two seeds and measure how much survives.

    ``embed_fn`` takes a seed and returns an embedding. The score is the k-NN
    overlap between the two results: 1.0 means the neighbourhood structure is
    identical, and a low value means the layout depends more on the optimiser's
    starting point than on the data.

    Deterministic methods return 1.0 by construction, which is itself worth
    reporting alongside the stochastic ones.
    """
    try:
        Z1 = embed_fn(seeds[0])
        Z2 = embed_fn(seeds[1])
    except Exception:
        return float("nan")
    if not (np.all(np.isfinite(Z1)) and np.all(np.isfinite(Z2))):
        return float("nan")
    if len(Z1) > MAX_SAMPLES_FOR_METRICS:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(Z1), MAX_SAMPLES_FOR_METRICS, replace=False)
        Z1, Z2 = Z1[idx], Z2[idx]
    return knn_overlap(Z1, Z2, k=k)

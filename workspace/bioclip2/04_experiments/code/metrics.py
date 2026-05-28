"""Embedding geometry & taxonomic metrics for BioCLIP2 hierarchical prompt analysis.

All functions assume L2-normalized embeddings unless otherwise stated.

Metrics
-------
- intra_class_variance     : mean within-class pairwise cosine distance (lower = better)
- inter_class_margin       : nearest-other-centroid distance / within-class std (higher = better)
- silhouette_cosine        : sklearn silhouette with cosine distance
- alignment                : Wang & Isola 2020
- uniformity               : Wang & Isola 2020
- rankme                   : Garrido et al. 2023 (effective rank)
- knn_purity_at_k          : taxonomic kNN purity
- lca_depth_mean           : mean LCA depth of predicted vs gt species
- hierarchy_distance       : Bertinetto-style tree-edge distance
- mutual_information_cluster_rank : I(cluster ; rank-label)

Statistical helpers
-------------------
- paired_permutation_test
- bootstrap_ci
- effect_preservation_ratio
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

# Soft import sklearn so the module can be imported even if sklearn missing.
try:
    from sklearn.metrics import silhouette_score, mutual_info_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    _SKLEARN_OK = True
except Exception as e:  # pragma: no cover
    _SKLEARN_OK = False
    _SKLEARN_ERR = e


def l2_normalize(Z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / (n + eps)


def intra_class_variance(Z: np.ndarray, y: np.ndarray) -> float:
    """Mean over classes of mean within-class pairwise cosine distance.

    Z: (N, D) L2-normalized embeddings
    y: (N,)   integer class labels
    """
    Z = l2_normalize(Z)
    classes = np.unique(y)
    vals: List[float] = []
    for c in classes:
        Zc = Z[y == c]
        if len(Zc) < 2:
            continue
        sim = Zc @ Zc.T  # cosine since Z normalized
        # take upper triangle (i<j)
        iu = np.triu_indices(len(Zc), k=1)
        dist = 1.0 - sim[iu]
        vals.append(float(dist.mean()))
    return float(np.mean(vals)) if vals else float("nan")


def inter_class_margin(Z: np.ndarray, y: np.ndarray) -> float:
    """(nearest-other-centroid cosine distance) / (within-class std), mean over classes."""
    Z = l2_normalize(Z)
    classes = np.unique(y)
    centroids = np.stack([Z[y == c].mean(axis=0) for c in classes])
    centroids = l2_normalize(centroids)
    cent_sim = centroids @ centroids.T
    cent_dist = 1.0 - cent_sim
    np.fill_diagonal(cent_dist, np.inf)

    ratios: List[float] = []
    for ci, c in enumerate(classes):
        Zc = Z[y == c]
        if len(Zc) < 2:
            continue
        # std of within-class cosine distance to centroid
        sim_to_mean = Zc @ centroids[ci]
        dist_to_mean = 1.0 - sim_to_mean
        sigma = float(dist_to_mean.std())
        if sigma < 1e-8:
            sigma = 1e-8
        nearest = float(cent_dist[ci].min())
        ratios.append(nearest / sigma)
    return float(np.mean(ratios)) if ratios else float("nan")


def silhouette_cosine(Z: np.ndarray, y: np.ndarray) -> float:
    if not _SKLEARN_OK:
        raise RuntimeError(f"sklearn not available: {_SKLEARN_ERR}")
    Z = l2_normalize(Z)
    classes, counts = np.unique(y, return_counts=True)
    # silhouette requires >=2 classes, each with >=2 samples; filter degenerate
    if len(classes) < 2 or counts.min() < 2:
        return float("nan")
    return float(silhouette_score(Z, y, metric="cosine"))


def alignment(Z_pos_a: np.ndarray, Z_pos_b: np.ndarray, alpha: float = 2.0) -> float:
    """Wang & Isola alignment: E_{positives} || f(x) - f(x+) ||^alpha."""
    Z_pos_a = l2_normalize(Z_pos_a)
    Z_pos_b = l2_normalize(Z_pos_b)
    d = np.linalg.norm(Z_pos_a - Z_pos_b, axis=1)
    return float((d ** alpha).mean())


def uniformity(Z: np.ndarray, t: float = 2.0) -> float:
    """Wang & Isola uniformity: log E_{x,y} exp(-t || f(x) - f(y) ||^2)."""
    Z = l2_normalize(Z)
    n = len(Z)
    if n < 2:
        return float("nan")
    # pairwise squared distance via expansion
    # |x-y|^2 = 2 - 2 x.y  (for normalized)
    sim = Z @ Z.T
    d2 = 2.0 - 2.0 * sim
    iu = np.triu_indices(n, k=1)
    return float(np.log(np.exp(-t * d2[iu]).mean()))


def rankme(Z: np.ndarray, eps: float = 1e-7) -> float:
    """RankMe effective rank (Garrido et al. 2023)."""
    if Z.shape[0] < 2:
        return float("nan")
    # Use np.linalg.svd; for tall matrices use full_matrices=False
    try:
        s = np.linalg.svd(Z, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    s_sum = s.sum() + eps
    p = s / s_sum + eps
    # exp of Shannon entropy of normalized singular values
    return float(np.exp(-(p * np.log(p)).sum()))


def knn_purity_at_k(Z: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    if not _SKLEARN_OK:
        raise RuntimeError(f"sklearn not available: {_SKLEARN_ERR}")
    Z = l2_normalize(Z)
    n = len(Z)
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine").fit(Z)
    _, idx = nn.kneighbors(Z)
    # exclude self (column 0)
    nbr = idx[:, 1:]
    purity = (y[nbr] == y[:, None]).mean(axis=1)
    return float(purity.mean())


def linear_probe_accuracy(
    Z: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> float:
    if not _SKLEARN_OK:
        raise RuntimeError(f"sklearn not available: {_SKLEARN_ERR}")
    classes, counts = np.unique(y, return_counts=True)
    n_test = int(np.ceil(len(y) * test_size))
    if len(classes) < 2 or counts.min() < 2 or n_test < len(classes):
        return float("nan")
    try:
        Z_train, Z_test, y_train, y_test = train_test_split(
            Z, y, test_size=test_size, stratify=y, random_state=seed
        )
    except ValueError:
        return float("nan")
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(Z_train, y_train)
    return float(clf.score(Z_test, y_test))


def plot_embeddings(
    Z: np.ndarray,
    y: np.ndarray,
    title: str = "",
    method: str = "umap",
    save_path: Optional[str] = None,
    seed: int = 42,
) -> None:
    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise RuntimeError("umap-learn not installed") from e
        reducer = umap.UMAP(n_components=2, random_state=seed)
        coords = reducer.fit_transform(Z)
    elif method == "tsne":
        if not _SKLEARN_OK:
            raise RuntimeError(f"sklearn not available: {_SKLEARN_ERR}")
        from sklearn.manifold import TSNE
        perplexity = min(30, len(Z) // 4)
        if perplexity < 1:
            perplexity = 1
        coords = TSNE(n_components=2, random_state=seed, perplexity=perplexity).fit_transform(Z)
    else:
        raise ValueError(method)

    import matplotlib.pyplot as plt

    classes = np.unique(y)
    cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(7, 5))
    for i, c in enumerate(classes):
        mask = y == c
        label = str(c) if len(classes) <= 20 else None
        plt.scatter(coords[mask, 0], coords[mask, 1], s=18, color=cmap(i % 20), label=label)
    plt.title(title)
    if len(classes) <= 20:
        plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
        plt.close()
    else:
        plt.show()


def lca_depth_mean(
    pred_species: Sequence[str],
    true_species: Sequence[str],
    tax_table: Dict[str, List[str]],
) -> float:
    """Mean LCA depth in Linnaean tree.

    tax_table maps species name -> [kingdom, phylum, class, order, family, genus, species]
    Depth from root: kingdom=1, ..., species=7. LCA depth = #ranks shared from root.
    """
    depths: List[int] = []
    for p, t in zip(pred_species, true_species):
        if p not in tax_table or t not in tax_table:
            continue
        pp, tt = tax_table[p], tax_table[t]
        d = 0
        for a, b in zip(pp, tt):
            if a == b:
                d += 1
            else:
                break
        depths.append(d)
    return float(np.mean(depths)) if depths else float("nan")


def hierarchy_distance(
    pred_species: Sequence[str],
    true_species: Sequence[str],
    tax_table: Dict[str, List[str]],
) -> float:
    """Bertinetto-style: number of tree-edges between pred and true (lower = better).
    For 7-rank flat hierarchy: dist = 2 * (7 - LCA depth)."""
    depths: List[int] = []
    for p, t in zip(pred_species, true_species):
        if p not in tax_table or t not in tax_table:
            continue
        pp, tt = tax_table[p], tax_table[t]
        d = 0
        for a, b in zip(pp, tt):
            if a == b:
                d += 1
            else:
                break
        depths.append(2 * (7 - d))
    return float(np.mean(depths)) if depths else float("nan")


def mutual_information_cluster_rank(cluster_labels: np.ndarray, rank_labels: np.ndarray) -> float:
    if not _SKLEARN_OK:
        raise RuntimeError(f"sklearn not available: {_SKLEARN_ERR}")
    return float(mutual_info_score(rank_labels, cluster_labels))


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def paired_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Two-sided paired permutation test on mean difference (a - b).

    Returns dict: {mean_diff, p_value, ci_lo, ci_hi}.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    obs = diff.mean()
    n = len(diff)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        if abs((signs * diff).mean()) >= abs(obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    # Bootstrap CI on diff
    boots = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.integers(0, n, size=n)
        boots[i] = diff[idx].mean()
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": float(obs), "p_value": float(p),
            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}


def bootstrap_ci(
    values: np.ndarray,
    statistic: callable = np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Generic bootstrap CI. Returns (point_estimate, ci_lo, ci_hi)."""
    if rng is None:
        rng = np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    point = float(statistic(values))
    n = len(values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = statistic(values[idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def effect_preservation_ratio(
    metric_C0: float,
    metric_C1: float,
    metric_Cx: float,
) -> float:
    """ rho_x = (M(Cx) - M(C0)) / (M(C1) - M(C0)); returns NaN if denominator near 0."""
    denom = metric_C1 - metric_C0
    if abs(denom) < 1e-8:
        return float("nan")
    return (metric_Cx - metric_C0) / denom


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a) - np.asarray(b)
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    return float(diff.mean() / sd) if sd > 1e-8 else 0.0


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # toy: 4 classes, 8 samples each, 16-dim
    Z = []
    y = []
    for c in range(4):
        mu = rng.normal(0, 1, size=16)
        for _ in range(8):
            Z.append(mu + 0.3 * rng.normal(0, 1, size=16))
            y.append(c)
    Z = np.array(Z); y = np.array(y)
    print("intra_var :", intra_class_variance(Z, y))
    print("inter_marg:", inter_class_margin(Z, y))
    print("silhouette:", silhouette_cosine(Z, y))
    print("rankme    :", rankme(Z))
    print("uniformity:", uniformity(Z))
    print("knn_pur@3 :", knn_purity_at_k(Z, y, k=3))

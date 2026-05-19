"""Calibration, selective-prediction, and conformal metrics.

All functions accept ``probs`` shape (N, K) and ``labels`` shape (N,)
unless documented otherwise.  Returns Python ``float`` (or numpy / list)
to facilitate JSON serialisation.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# ECE family
# --------------------------------------------------------------------------- #
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    """Equal-width ECE (Guo et al. 2017)."""
    p = _to_numpy(probs)
    y = _to_numpy(labels).astype(int)
    conf = p.max(axis=-1)
    pred = p.argmax(axis=-1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / N) * abs(acc_bin - conf_bin)
    return float(ece)


def adaptive_ece(probs, labels, n_bins: int = 15) -> float:
    """Equal-mass ECE (Nixon 2019)."""
    p = _to_numpy(probs)
    y = _to_numpy(labels).astype(int)
    conf = p.max(axis=-1)
    pred = p.argmax(axis=-1)
    correct = (pred == y).astype(float)
    order = np.argsort(conf)
    conf_s = conf[order]
    correct_s = correct[order]
    N = len(conf_s)
    if N == 0:
        return 0.0
    splits = np.array_split(np.arange(N), n_bins)
    ece = 0.0
    for s in splits:
        if len(s) == 0:
            continue
        acc_bin = correct_s[s].mean()
        conf_bin = conf_s[s].mean()
        ece += (len(s) / N) * abs(acc_bin - conf_bin)
    return float(ece)


def classwise_ece(probs, labels, n_bins: int = 15) -> float:
    """Mean over classes of one-vs-rest ECE."""
    p = _to_numpy(probs)
    y = _to_numpy(labels).astype(int)
    K = p.shape[1]
    eces = []
    for k in range(K):
        pk = p[:, k]
        yk = (y == k).astype(float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece_k = 0.0
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            if i == n_bins - 1:
                mask = (pk >= lo) & (pk <= hi)
            else:
                mask = (pk >= lo) & (pk < hi)
            if mask.sum() == 0:
                continue
            ece_k += (mask.sum() / len(pk)) * abs(yk[mask].mean() - pk[mask].mean())
        eces.append(ece_k)
    return float(np.mean(eces))


def mce(probs, labels, n_bins: int = 15) -> float:
    p = _to_numpy(probs)
    y = _to_numpy(labels).astype(int)
    conf = p.max(axis=-1)
    pred = p.argmax(axis=-1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    gaps = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        gaps.append(abs(correct[mask].mean() - conf[mask].mean()))
    return float(max(gaps)) if gaps else 0.0


def brier_score(probs, labels) -> float:
    p = _to_numpy(probs)
    y = _to_numpy(labels).astype(int)
    K = p.shape[1]
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def nll(probs, labels, eps: float = 1e-12) -> float:
    p = _to_numpy(probs)
    y = _to_numpy(labels).astype(int)
    return float(-np.log(p[np.arange(len(y)), y].clip(min=eps)).mean())


def accuracy(probs, labels) -> float:
    p = _to_numpy(probs)
    y = _to_numpy(labels).astype(int)
    return float((p.argmax(axis=-1) == y).mean())


# --------------------------------------------------------------------------- #
# Selective prediction
# --------------------------------------------------------------------------- #
def aurc(probs, labels) -> float:
    """Area under the risk-coverage curve (Geifman & El-Yaniv 2017)."""
    p = _to_numpy(probs)
    y = _to_numpy(labels).astype(int)
    conf = p.max(axis=-1)
    pred = p.argmax(axis=-1)
    err = (pred != y).astype(float)
    order = np.argsort(-conf)
    err_sorted = err[order]
    cum_err = np.cumsum(err_sorted)
    n = np.arange(1, len(err_sorted) + 1)
    risks = cum_err / n
    coverages = n / len(err_sorted)
    return float(np.trapz(risks, coverages))


def selective_accuracy_at_coverage(probs, labels, coverage: float) -> float:
    p = _to_numpy(probs)
    y = _to_numpy(labels).astype(int)
    conf = p.max(axis=-1)
    pred = p.argmax(axis=-1)
    order = np.argsort(-conf)
    n_keep = max(int(np.round(coverage * len(y))), 1)
    keep = order[:n_keep]
    return float((pred[keep] == y[keep]).mean())


def auroc_misclassification(probs, labels) -> float:
    """AUROC of MSP as a detector for misclassification.

    Positive class = misclassified (so lower confidence -> higher score)."""
    from sklearn.metrics import roc_auc_score
    p = _to_numpy(probs)
    y = _to_numpy(labels).astype(int)
    pred = p.argmax(axis=-1)
    miscl = (pred != y).astype(int)
    if miscl.sum() == 0 or miscl.sum() == len(miscl):
        return float("nan")
    score = -p.max(axis=-1)  # higher score = more likely misclassified
    return float(roc_auc_score(miscl, score))


# --------------------------------------------------------------------------- #
# Conformal prediction (split, non-conformity = 1 − p_y)
# --------------------------------------------------------------------------- #
def split_conformal_threshold(probs_cal, labels_cal, alpha: float = 0.1) -> float:
    """Quantile of non-conformity scores on calibration set."""
    p = _to_numpy(probs_cal)
    y = _to_numpy(labels_cal).astype(int)
    n = len(y)
    score = 1.0 - p[np.arange(n), y]
    q = np.ceil((n + 1) * (1.0 - alpha)) / n
    q = float(min(q, 1.0))
    return float(np.quantile(score, q))


def conformal_sets(probs_test, threshold: float) -> np.ndarray:
    p = _to_numpy(probs_test)
    return (1.0 - p) <= threshold  # boolean (N, K)


def conformal_coverage_and_size(probs_test, labels_test, threshold: float
                                ) -> Tuple[float, float]:
    sets = conformal_sets(probs_test, threshold)
    y = _to_numpy(labels_test).astype(int)
    covered = sets[np.arange(len(y)), y]
    coverage = float(covered.mean())
    set_size = float(sets.sum(axis=1).mean())
    return coverage, set_size


# --------------------------------------------------------------------------- #
# All-in-one
# --------------------------------------------------------------------------- #
def summarize(probs, labels, n_bins: int = 15) -> Dict[str, float]:
    return {
        "acc": accuracy(probs, labels),
        "ece": expected_calibration_error(probs, labels, n_bins=n_bins),
        "ada_ece": adaptive_ece(probs, labels, n_bins=n_bins),
        "cw_ece": classwise_ece(probs, labels, n_bins=n_bins),
        "mce": mce(probs, labels, n_bins=n_bins),
        "brier": brier_score(probs, labels),
        "nll": nll(probs, labels),
    }


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals.
# --------------------------------------------------------------------------- #
def bootstrap_ci(metric_fn, probs, labels, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    N = len(labels)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, N, size=N)
        stats.append(metric_fn(probs[idx], _to_numpy(labels)[idx]))
    stats = np.array(stats)
    return (float(np.mean(stats)),
            float(np.quantile(stats, alpha / 2)),
            float(np.quantile(stats, 1 - alpha / 2)))

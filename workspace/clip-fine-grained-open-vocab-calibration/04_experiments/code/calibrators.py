"""Post-hoc calibrators.

All calibrators consume *logits* of shape (N, K) and (optionally) auxiliary
information (e.g., text embeddings for DAC/CAC, original-CLIP logits for CAC),
and produce calibrated probabilities of shape (N, K).

Convention: ``fit`` is called on calibration data with *base classes only*;
``predict`` may be called on either base- or novel-class logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class Calibrator(nn.Module):
    name: str = "base"

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, **kw) -> "Calibrator":
        return self

    def predict_logits(self, logits: torch.Tensor, **kw) -> torch.Tensor:  # noqa: D401
        return logits

    def predict_proba(self, logits: torch.Tensor, **kw) -> torch.Tensor:
        return F.softmax(self.predict_logits(logits, **kw), dim=-1)


# --------------------------------------------------------------------------- #
# C0 — Raw
# --------------------------------------------------------------------------- #
class Raw(Calibrator):
    name = "raw"


# --------------------------------------------------------------------------- #
# C1 — Temperature Scaling
# --------------------------------------------------------------------------- #
class TemperatureScaling(Calibrator):
    name = "temperature_scaling"

    def __init__(self, T_init: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(float(np.log(T_init))))

    @property
    def T(self) -> torch.Tensor:
        return torch.exp(self.log_T).clamp(min=1e-3, max=1e3)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            max_iter: int = 200, lr: float = 0.05, **_) -> "TemperatureScaling":
        logits = logits.detach().float()
        labels = labels.detach().long()
        opt = torch.optim.LBFGS([self.log_T], lr=lr, max_iter=max_iter,
                                tolerance_grad=1e-7, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits / self.T, labels)
            loss.backward()
            return loss
        opt.step(closure)
        return self

    def predict_logits(self, logits: torch.Tensor, **_) -> torch.Tensor:
        return logits / self.T


# --------------------------------------------------------------------------- #
# C2 — Vector Scaling
# --------------------------------------------------------------------------- #
class VectorScaling(Calibrator):
    name = "vector_scaling"

    def __init__(self, K: int):
        super().__init__()
        self.K = K
        self.w = nn.Parameter(torch.ones(K))
        self.b = nn.Parameter(torch.zeros(K))

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            max_iter: int = 200, lr: float = 0.05, **_) -> "VectorScaling":
        logits = logits.detach().float()
        labels = labels.detach().long()
        opt = torch.optim.LBFGS([self.w, self.b], lr=lr, max_iter=max_iter,
                                tolerance_grad=1e-7, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits * self.w + self.b, labels)
            loss.backward()
            return loss
        opt.step(closure)
        return self

    def predict_logits(self, logits: torch.Tensor, **_) -> torch.Tensor:
        return logits * self.w + self.b


# --------------------------------------------------------------------------- #
# C3 — Dirichlet calibration  (linear layer on log-softmax)
# --------------------------------------------------------------------------- #
class DirichletCalibration(Calibrator):
    name = "dirichlet"

    def __init__(self, K: int, l2: float = 1e-3):
        super().__init__()
        self.K = K
        self.l2 = l2
        self.W = nn.Parameter(torch.eye(K))
        self.b = nn.Parameter(torch.zeros(K))

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            max_iter: int = 300, lr: float = 0.05, **_) -> "DirichletCalibration":
        logits = logits.detach().float()
        labels = labels.detach().long()
        opt = torch.optim.LBFGS([self.W, self.b], lr=lr, max_iter=max_iter,
                                tolerance_grad=1e-7, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            logp = F.log_softmax(logits, dim=-1)
            z = logp @ self.W.T + self.b
            loss = F.cross_entropy(z, labels)
            # Off-diagonal L2 regularisation (Kull 2019 trick).
            off = self.W - torch.diag(torch.diag(self.W))
            loss = loss + self.l2 * (off.pow(2).sum() + self.b.pow(2).sum())
            loss.backward()
            return loss
        opt.step(closure)
        return self

    def predict_logits(self, logits: torch.Tensor, **_) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        return logp @ self.W.T + self.b


# --------------------------------------------------------------------------- #
# C4 — Distance-Aware Calibration (DAC, Wang et al. 2024)
# --------------------------------------------------------------------------- #
class DAC(Calibrator):
    """Sample-wise temperature scaling driven by textual distance to base.

    Implementation follows arXiv:2402.04655 §3.2 in spirit::
        T_i = T * (1 + alpha * mean_top1_distance_to_base(y_pred_i))
    """
    name = "dac"

    def __init__(self, T_init: float = 1.0, alpha_init: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(float(np.log(T_init))))
        self.log_alpha = nn.Parameter(torch.tensor(float(np.log(alpha_init))))
        self.register_buffer("base_text", torch.zeros(1, 1))
        self._has_base = False

    def set_base_text(self, base_text_emb: torch.Tensor) -> None:
        self.base_text = F.normalize(base_text_emb.detach().float(), dim=-1)
        self._has_base = True

    def _distance(self, target_text_emb: torch.Tensor) -> torch.Tensor:
        """Per-class distance to *nearest* base class (1 − max cosine)."""
        x = F.normalize(target_text_emb.float(), dim=-1)
        sim = x @ self.base_text.T  # (K, K_base)
        max_sim, _ = sim.max(dim=-1)
        return (1.0 - max_sim).clamp(min=0.0)  # (K,)

    @property
    def T(self) -> torch.Tensor:
        return torch.exp(self.log_T).clamp(min=1e-3, max=1e3)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha).clamp(min=1e-3, max=1e2)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            text_emb: Optional[torch.Tensor] = None,
            max_iter: int = 200, lr: float = 0.05, **_) -> "DAC":
        assert self._has_base, "Call set_base_text() before fit()."
        assert text_emb is not None, "DAC.fit needs text_emb for predicted labels"
        logits = logits.detach().float()
        labels = labels.detach().long()
        opt = torch.optim.LBFGS([self.log_T, self.log_alpha], lr=lr,
                                max_iter=max_iter, tolerance_grad=1e-7,
                                line_search_fn="strong_wolfe")
        dist = self._distance(text_emb)  # (K,)

        def closure():
            opt.zero_grad()
            pred = logits.argmax(dim=-1)
            T_i = self.T * (1.0 + self.alpha * dist[pred])  # (N,)
            z = logits / T_i.unsqueeze(-1)
            loss = F.cross_entropy(z, labels)
            loss.backward()
            return loss
        opt.step(closure)
        return self

    def predict_logits(self, logits: torch.Tensor,
                       text_emb: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
        assert self._has_base and text_emb is not None
        with torch.no_grad():
            dist = self._distance(text_emb)
        pred = logits.argmax(dim=-1)
        T_i = self.T * (1.0 + self.alpha * dist[pred])
        return logits / T_i.unsqueeze(-1)


# --------------------------------------------------------------------------- #
# C5 — Contrast-Aware Calibration (CAC, Lv et al. 2025)
# --------------------------------------------------------------------------- #
class CAC(Calibrator):
    """Reweight fine-tuned logits by their *contrast* with zero-shot logits.

    Implementation::
        z_calibrated = (z_ft - alpha * (z_zs - z_ft)) / T
    where (z_zs, z_ft) come from the *original* CLIP and the *fine-tuned*
    CLIP respectively (or two views of the same model in our zero-shot case).
    """
    name = "cac"

    def __init__(self, T_init: float = 1.0, alpha_init: float = 0.3):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(float(np.log(T_init))))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    @property
    def T(self) -> torch.Tensor:
        return torch.exp(self.log_T).clamp(min=1e-3, max=1e3)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            zs_logits: Optional[torch.Tensor] = None,
            max_iter: int = 200, lr: float = 0.05, **_) -> "CAC":
        assert zs_logits is not None, "CAC.fit requires zs_logits"
        logits = logits.detach().float()
        zs_logits = zs_logits.detach().float()
        labels = labels.detach().long()
        opt = torch.optim.LBFGS([self.log_T, self.alpha], lr=lr,
                                max_iter=max_iter, tolerance_grad=1e-7,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = (logits - self.alpha * (zs_logits - logits)) / self.T
            loss = F.cross_entropy(z, labels)
            loss.backward()
            return loss
        opt.step(closure)
        return self

    def predict_logits(self, logits: torch.Tensor,
                       zs_logits: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
        assert zs_logits is not None
        return (logits - self.alpha * (zs_logits - logits)) / self.T


# --------------------------------------------------------------------------- #
# C6 — Histogram Binning (per-class one-vs-rest)
# --------------------------------------------------------------------------- #
class HistogramBinning(Calibrator):
    name = "histogram_binning"

    def __init__(self, K: int, n_bins: int = 15):
        super().__init__()
        self.K = K
        self.n_bins = n_bins
        self.register_buffer("edges", torch.linspace(0.0, 1.0, n_bins + 1))
        self.register_buffer("bin_val", torch.zeros(K, n_bins))

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, **_) -> "HistogramBinning":
        probs = F.softmax(logits.float(), dim=-1)  # (N, K)
        N, K = probs.shape
        # one-hot labels
        y = F.one_hot(labels.long(), num_classes=K).float()
        bin_val = torch.zeros(K, self.n_bins)
        for k in range(K):
            for b in range(self.n_bins):
                lo, hi = self.edges[b].item(), self.edges[b + 1].item()
                mask = (probs[:, k] >= lo) & (probs[:, k] < hi if b < self.n_bins - 1
                                              else probs[:, k] <= hi)
                if mask.sum() > 0:
                    bin_val[k, b] = y[mask, k].mean()
                else:
                    bin_val[k, b] = (lo + hi) / 2.0
        self.bin_val = bin_val
        return self

    def predict_proba(self, logits: torch.Tensor, **_) -> torch.Tensor:
        probs = F.softmax(logits.float(), dim=-1)
        N, K = probs.shape
        out = torch.zeros_like(probs)
        for k in range(K):
            idx = torch.bucketize(probs[:, k], self.edges) - 1
            idx = idx.clamp(0, self.n_bins - 1)
            out[:, k] = self.bin_val[k, idx]
        # renormalise so each row sums to 1
        out = out / out.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return out

    def predict_logits(self, logits: torch.Tensor, **_) -> torch.Tensor:
        return torch.log(self.predict_proba(logits).clamp(min=1e-12))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def build_calibrator(name: str, K: int) -> Calibrator:
    name = name.lower()
    if name in {"raw", "c0", "none"}:
        return Raw()
    if name in {"ts", "temperature_scaling", "c1"}:
        return TemperatureScaling()
    if name in {"vs", "vector_scaling", "c2"}:
        return VectorScaling(K=K)
    if name in {"dirichlet", "c3"}:
        return DirichletCalibration(K=K)
    if name in {"dac", "c4"}:
        return DAC()
    if name in {"cac", "c5"}:
        return CAC()
    if name in {"hb", "histogram_binning", "c6"}:
        return HistogramBinning(K=K)
    raise ValueError(f"Unknown calibrator: {name}")


CALIBRATOR_IDS = ["raw", "temperature_scaling", "vector_scaling",
                  "dirichlet", "dac", "cac", "histogram_binning"]

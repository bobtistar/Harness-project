"""CLIP wrapper for zero-shot logits and prompt ensembling.

Uses open_clip when available.  Designed to compute *logits* (pre-softmax)
so downstream calibrators can operate on them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import open_clip
    _OPEN_CLIP = True
except Exception:  # pragma: no cover
    _OPEN_CLIP = False


# OpenAI's 7-template ensemble (subset used by CLIP/DAC papers).
OPENAI_TEMPLATES: List[str] = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a close-up photo of a {}.",
    "a low resolution photo of a {}.",
    "a cropped photo of a {}.",
    "an image of a {}.",
]


@dataclass
class ClipBundle:
    model: torch.nn.Module
    preprocess: callable
    tokenizer: callable
    device: torch.device
    backbone: str
    pretrained: str


def load_clip(backbone: str = "ViT-B-16",
              pretrained: str = "laion2b_s34b_b88k",
              device: Optional[torch.device] = None) -> Optional[ClipBundle]:
    if not _OPEN_CLIP:
        warnings.warn("open_clip not installed; load_clip returning None")
        return None
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        backbone, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(backbone)
    model = model.to(device).eval()
    return ClipBundle(model=model, preprocess=preprocess, tokenizer=tokenizer,
                      device=device, backbone=backbone, pretrained=pretrained)


@torch.no_grad()
def encode_text(bundle: ClipBundle,
                classnames: Sequence[str],
                templates: Sequence[str] = ("a photo of a {}.",),
                normalize: bool = True) -> torch.Tensor:
    """Return text embedding tensor of shape (K, D), averaged across templates."""
    embs: List[torch.Tensor] = []
    for c in classnames:
        prompts = [t.format(c) for t in templates]
        tokens = bundle.tokenizer(prompts).to(bundle.device)
        e = bundle.model.encode_text(tokens)
        if normalize:
            e = F.normalize(e, dim=-1)
        e = e.mean(dim=0)
        if normalize:
            e = F.normalize(e, dim=-1)
        embs.append(e)
    out = torch.stack(embs, dim=0)
    return out  # (K, D)


@torch.no_grad()
def encode_images(bundle: ClipBundle,
                  images: Sequence,
                  batch_size: int = 64,
                  normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode an iterable of (PIL, label) pairs into (N, D) features.

    ``images`` is expected to be a Dataset-like object whose ``__getitem__``
    returns (PIL.Image, int).  Returns (features, labels).
    """
    feats: List[torch.Tensor] = []
    labels: List[int] = []
    buf: List[torch.Tensor] = []
    buf_y: List[int] = []
    for i in range(len(images)):  # type: ignore[arg-type]
        img, y = images[i]
        buf.append(bundle.preprocess(img))
        buf_y.append(int(y))
        if len(buf) == batch_size:
            x = torch.stack(buf, dim=0).to(bundle.device)
            e = bundle.model.encode_image(x)
            if normalize:
                e = F.normalize(e, dim=-1)
            feats.append(e.cpu())
            labels.extend(buf_y)
            buf, buf_y = [], []
    if buf:
        x = torch.stack(buf, dim=0).to(bundle.device)
        e = bundle.model.encode_image(x)
        if normalize:
            e = F.normalize(e, dim=-1)
        feats.append(e.cpu())
        labels.extend(buf_y)
    return torch.cat(feats, dim=0), torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def zero_shot_logits(image_features: torch.Tensor,
                     text_features: torch.Tensor,
                     logit_scale: float = 100.0) -> torch.Tensor:
    """Return logits = scale * (img @ text^T).

    Both inputs assumed L2-normalised, shapes (N, D) and (K, D).
    """
    return logit_scale * image_features @ text_features.T


@torch.no_grad()
def evaluate_clip(bundle: ClipBundle,
                  images,
                  classnames: Sequence[str],
                  templates: Sequence[str] = ("a photo of a {}.",),
                  batch_size: int = 64,
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """High-level helper used by experiments.

    Returns (logits NxK, labels N, text_embeddings KxD).
    """
    text_emb = encode_text(bundle, classnames, templates=templates,
                           normalize=True)
    img_emb, labels = encode_images(bundle, images, batch_size=batch_size,
                                    normalize=True)
    logit_scale = float(bundle.model.logit_scale.exp().detach().cpu().item())
    logits = zero_shot_logits(img_emb, text_emb.cpu(), logit_scale=logit_scale)
    return logits, labels, text_emb.cpu()


# --------------------------------------------------------------------------- #
# Text-embedding manipulations (used by RQ2 interventions).
# --------------------------------------------------------------------------- #
def tau_text(text_emb: torch.Tensor) -> float:
    """Mean off-diagonal cosine similarity between class text embeddings."""
    x = F.normalize(text_emb, dim=-1)
    sim = (x @ x.T).cpu().numpy()
    K = sim.shape[0]
    mask = ~np.eye(K, dtype=bool)
    return float(sim[mask].mean())


def pca_whiten(text_emb: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Whiten the class-text embedding matrix to unit-variance principal axes."""
    x = text_emb - text_emb.mean(dim=0, keepdim=True)
    # Compute whitening matrix from class-text covariance.
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    U, S, _ = torch.linalg.svd(cov)
    W = U @ torch.diag(1.0 / (S.sqrt() + eps)) @ U.T
    y = x @ W
    return F.normalize(y, dim=-1)


def orthogonalize(text_emb: torch.Tensor) -> torch.Tensor:
    """Orthogonalise class embeddings via QR; pads with random if K>D."""
    K, D = text_emb.shape
    if K <= D:
        Q, _ = torch.linalg.qr(text_emb.T)
        y = Q.T[:K]
    else:
        # Fall back: random orthogonalisation of first D, leave rest normalized.
        Q, _ = torch.linalg.qr(text_emb[:D].T)
        y = torch.cat([Q.T[:D], F.normalize(text_emb[D:], dim=-1)], dim=0)
    return F.normalize(y, dim=-1)

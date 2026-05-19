"""C5 (text-free hierarchical contrastive) adapter.

Implements a *light* image-image hierarchical InfoNCE on a frozen image encoder.
Based on Kokilepersaud et al. 2024 ("Taxes Are All You Need") -- the loss
weights pairs of images by their taxonomic distance.

Architecture
------------
  frozen image encoder ──> linear adapter (D -> D) ──> L2 normalize ──> InfoNCE

For real BioCLIP2 we would use LoRA on attention blocks; this skeleton uses
a single linear adapter for simplicity & CPU-friendliness.

The hierarchical positive weighting: for anchor with 7-rank labels y and
neighbor y', the similarity weight is the LCA depth normalized to [0,1].
"""
from __future__ import annotations

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)  # start as identity for stability

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.proj(z)
        z = F.normalize(z, dim=-1)
        return z


def lca_matrix(tax_labels_per_sample: List[List[str]]) -> np.ndarray:
    """N x N matrix of LCA depths in [0, 7]."""
    n = len(tax_labels_per_sample)
    M = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 7
                continue
            d = 0
            for a, b in zip(tax_labels_per_sample[i], tax_labels_per_sample[j]):
                if a == b:
                    d += 1
                else:
                    break
            M[i, j] = d
    return M


def hierarchical_infonce(
    z: torch.Tensor,
    lca_M: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Hierarchical InfoNCE: weight positive log-prob by normalized LCA depth.

    Loss for anchor i:
        L_i = - sum_j w_ij * log( exp(s_ij / T) / sum_k exp(s_ik / T) )
        where w_ij = lca(i,j) / 7  (so species-level shares weight=1, kingdom-level w=1/7).
    """
    sim = (z @ z.t()) / temperature
    # numerical stability
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # remove self (set diag to 0 weight)
    w = lca_M / 7.0
    eye = torch.eye(w.shape[0], device=w.device)
    w = w * (1 - eye)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
    w_norm = w / w_sum
    loss = -(w_norm * log_prob).sum(dim=1).mean()
    return loss


def train_C5_adapter(
    image_embeddings: np.ndarray,
    tax_labels_per_sample: List[List[str]],
    epochs: int = 5,
    lr: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
) -> np.ndarray:
    """Train the LinearAdapter on top of frozen image embeddings.

    Returns the adapted (and L2-normalized) embeddings.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    z = torch.from_numpy(image_embeddings).float().to(device)
    z = F.normalize(z, dim=-1)
    M = torch.from_numpy(lca_matrix(tax_labels_per_sample)).float().to(device)

    adapter = LinearAdapter(z.shape[1]).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr)
    for ep in range(epochs):
        adapter.train()
        opt.zero_grad()
        z_out = adapter(z)
        loss = hierarchical_infonce(z_out, M, temperature=0.07)
        loss.backward()
        opt.step()
    adapter.eval()
    with torch.no_grad():
        z_final = adapter(z).cpu().numpy()
    return z_final


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, size=(20, 64)).astype(np.float32)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    tax = [
        ["Animalia", "Chordata", "Aves", "Pa", "F", "G", f"sp{i % 4}"] for i in range(20)
    ]
    out = train_C5_adapter(z, tax, epochs=3)
    print("C5 output shape:", out.shape)

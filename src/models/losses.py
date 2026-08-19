"""
Contrastive Loss implementations.
Provides NT-Xent (Normalized Temperature-scaled Cross Entropy Loss) with logsumexp numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss (SimCLR style).
    Uses logsumexp across negative pairs for numerical stability.

    Given a batch of N pairs (z1, z2) from two augmented views:
      - Normalizes embeddings to unit hypersphere.
      - Computes pairwise cosine similarity matrix of size (2N, 2N) / temperature.
      - Masks out self-similarity elements.
      - Calculates cross-entropy against positive pairs.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss between two sets of representation vectors.

        Args:
            z1: Tensor of shape (N, D)
            z2: Tensor of shape (N, D)

        Returns:
            Scalar loss tensor.
        """
        # Normalize representations to unit norm
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        n = z1.size(0)

        # Concatenate into 2N representations
        z = torch.cat([z1, z2], dim=0)  # (2N, D)

        # Pairwise cosine similarity matrix scaled by temperature
        sim = torch.mm(z, z.T) / self.temperature  # (2N, 2N)

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -9e15)

        # Positive pairs: (i, i+N) and (i+N, i)
        positives = torch.cat([torch.diag(sim, n), torch.diag(sim, -n)], dim=0)

        # Denominator: log-sum-exp over all non-self elements
        log_prob = torch.logsumexp(sim, dim=1)

        # SimCLR loss: -positives + log_prob
        loss = -positives + log_prob
        return loss.mean()

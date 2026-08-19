"""
Attribute-Conditioned Attention Pooling Mechanism.
Aggregates variable-length or patch-level embeddings into a single fixed-dimensional representation
with optional machine attribute bias modulation.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Computes learnable attention weights over patch embeddings:
      1. Computes raw attention score per patch via a 2-layer MLP with Tanh activation.
      2. Injects linear machine attribute bias: scores = scores + attr_bias(attrs).
      3. Applies softmax over patches to produce normalized attention weights.
      4. Aggregates patches into a single pooled representation via weighted sum.

    Shapes:
      patch_embeddings: (B, N, embed_dim)
      attrs           : (B, attr_dim) or None
      Output          : (B, embed_dim)
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128, attr_dim: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.attr_dim = attr_dim

        self.attn_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        if attr_dim > 0:
            self.attr_bias = nn.Linear(attr_dim, 1)
        else:
            self.attr_bias = None

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        attrs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for attention pooling.

        Args:
            patch_embeddings: Tensor of shape (B, N, embed_dim)
            attrs: Optional attribute tensor of shape (B, attr_dim)

        Returns:
            Pooled tensor of shape (B, embed_dim)
        """
        # 1) Compute raw attention scores: (B, N, 1)
        scores = self.attn_net(patch_embeddings)

        # 2) Modulate scores with machine attribute bias if available
        if self.attr_bias is not None and attrs is not None:
            bias = self.attr_bias(attrs)  # (B, 1)
            scores = scores + bias.unsqueeze(1)  # broadcast to (B, N, 1)

        # 3) Normalize scores across patches (dimension 1) via Softmax
        weights = F.softmax(scores, dim=1)  # (B, N, 1)

        # 4) Weighted sum aggregation
        pooled = (weights * patch_embeddings).sum(dim=1)  # (B, embed_dim)
        return pooled

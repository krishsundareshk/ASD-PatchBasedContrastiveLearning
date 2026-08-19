"""
Patch-based Attention Contrastive Learning Model (PatchAttentionCLModel).
Combines patch feature extraction (ResNet backbone), projection MLP,
attribute-conditioned linear attention pooling, and optional early attribute fusion.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .attention import AttentionPooling


class PatchAttentionCLModel(nn.Module):
    """
    End-to-end Contrastive Learning model over patchified spectrograms.

    Architecture pipeline:
      1. Patch Feature Extraction: ResNet backbone extracts features (e.g. 512-dim) per patch.
      2. Non-linear Projector: 2-layer MLP with ReLU projecting backbone features to embed_dim.
      3. Attribute-Conditioned Attention Pooling: Computes patch weights conditioned on attributes
         and pools patches into a single spectrogram-level representation (embed_dim).
      4. Early Fusion Attribute MLP: If attr_dim > 0, embeds attributes to embed_dim and
         concatenates with pooled representation -> final_dim = 2 * embed_dim.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        attr_dim: int = 0,
        attn_hidden_dim: int = 128,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attr_dim = attr_dim

        # 1) Patch Encoder (Backbone)
        self.encoder, feat_dim = build_backbone(architecture=backbone_name, pretrained=pretrained)

        # 2) Projector MLP: feat_dim -> 512 -> embed_dim
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.embed_dim),
        )

        # 3) Attribute-conditioned attention pooling
        self.attn_pool = AttentionPooling(
            embed_dim=self.embed_dim,
            hidden_dim=attn_hidden_dim,
            attr_dim=attr_dim,
        )

        # 4) Early-fusion attribute MLP
        if attr_dim > 0:
            self.attr_mlp = nn.Sequential(
                nn.Linear(attr_dim, 32),
                nn.ReLU(),
                nn.Linear(32, self.embed_dim),
            )
            self.final_dim = 2 * self.embed_dim
        else:
            self.attr_mlp = None
            self.final_dim = self.embed_dim

        self.fusion = nn.Identity()

    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Encode per-patch embeddings.

        - 5D input: (B, N, C, H, W) -> returns (B, N, embed_dim) with L2 normalized patch vectors.
        - 4D input: (B*, C, H, W)   -> returns (B*, embed_dim) unnormalized projected features.
        """
        if patches.dim() == 5:
            b, n, c, h, w = patches.shape
            flat = patches.view(b * n, c, h, w)
            features = self.encoder(flat)                 # (B*N, feat_dim)
            projected = self.projector(features)          # (B*N, embed_dim)
            normalized = F.normalize(projected, dim=1)    # L2 normalize
            return normalized.view(b, n, -1)              # (B, N, embed_dim)

        elif patches.dim() == 4:
            features = self.encoder(patches)              # (B*, feat_dim)
            projected = self.projector(features)          # (B*, embed_dim)
            return projected

        else:
            raise ValueError(
                f"encode_patches expects 4D or 5D input tensor, got shape {tuple(patches.shape)}"
            )

    def forward(
        self,
        patches: torch.Tensor,
        batch_size: Optional[int] = None,
        num_patches: Optional[int] = None,
        attrs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for a batch of spectrogram patches.

        Args:
            patches: Tensor of shape (B, N, 3, H, W)
            batch_size: Optional batch size (inferred if None)
            num_patches: Optional patch count (inferred if None)
            attrs: Optional machine attribute tensor of shape (B, attr_dim)

        Returns:
            Pooled and fused representation tensor of shape (B, final_dim)
        """
        if patches.dim() != 5:
            raise ValueError(f"Expected 5D tensor (B, N, C, H, W), got shape {tuple(patches.shape)}")

        b, n, c, h, w = patches.shape
        flat = patches.view(b * n, c, h, w)
        proj = self.encode_patches(flat)  # (B*N, embed_dim)
        proj = proj.view(b, n, -1)        # (B, N, embed_dim)

        # Attribute-conditioned attention pooling: (B, embed_dim)
        pooled = self.attn_pool(proj, attrs=attrs)

        # Early fusion with attribute embeddings if available
        if self.attr_mlp is not None and attrs is not None:
            attr_embed = self.attr_mlp(attrs)  # (B, embed_dim)
            fused = torch.cat([pooled, attr_embed], dim=1)  # (B, 2*embed_dim)
            return self.fusion(fused)

        return pooled

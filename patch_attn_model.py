# patch_attn_model.py (updated)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from attention_pooling import AttentionPooling


class ResNet34Encoder(nn.Module):
    """ResNet34 backbone encoder without the classification head."""
    def __init__(self):
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        return self.backbone(x)  # (B, 512)


class PatchAttentionCLModel(nn.Module):
    """
    Contrastive-learning over patches with:
      1) attribute-conditioned linear attention pooling
      2) early-fusion concatenation with attributes

    NOTE: embed_dim is the _projected_ embedding dimension (projector output) and must match
    the attention pooling input and attribute-mlp output. If your checkpoint was trained with
    embed_dim=256, set embed_dim=256 here before loading to avoid mismatch errors.
    """
    def __init__(self, embed_dim: int = 256, attr_dim: int = 0, attn_hidden_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim  # final per-patch embedding dim after projector

        # patch encoder (backbone)
        self.encoder = ResNet34Encoder()

        # projector: maps ResNet features (512) -> embed_dim (configurable)
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.embed_dim)
        )

        # attribute-conditioned linear attention pooling
        # AttentionPooling should accept input_dim=self.embed_dim
        # and produce output of shape (B, self.embed_dim).
        self.attn_pool = AttentionPooling(embed_dim=self.embed_dim,
                                  hidden_dim=attn_hidden_dim,
                                  attr_dim=attr_dim)

        # early-fusion attribute MLP: maps attrs -> embed_dim (so concatenation works)
        if attr_dim > 0:
            self.attr_mlp = nn.Sequential(
                nn.Linear(attr_dim, 32),
                nn.ReLU(),
                nn.Linear(32, self.embed_dim)
            )
            self.final_dim = 2 * self.embed_dim
        else:
            self.attr_mlp = None
            self.final_dim = self.embed_dim

        self.fusion = nn.Identity()  # placeholder for possible later fusion


    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Encode per-patch embeddings.

        - 5D input: (B, N, 3, H, W) -> returns (B, N, embed_dim) normalized.
        - 4D input: (B*, 3, H, W) -> returns (B*, embed_dim) unnormalized.
        """
        if patches.dim() == 5:
            B, N, C, H, W = patches.shape
            flat = patches.view(B * N, C, H, W)
            f = self.encoder(flat)          # (B*N, 512)
            proj = self.projector(f)        # (B*N, embed_dim)
            proj = F.normalize(proj, dim=1)
            return proj.view(B, N, -1)      # (B, N, embed_dim)

        elif patches.dim() == 4:
            f = self.encoder(patches)       # (B*, 512)
            proj = self.projector(f)        # (B*, embed_dim)
            return proj

        else:
            raise ValueError(
                f"encode_patches expects 4D or 5D input, got {tuple(patches.shape)}"
            )


    def forward(self, patches: torch.Tensor, batch_size: int, num_patches: int, attrs: torch.Tensor = None) -> torch.Tensor:
        # Encode patches
        B, N, C, H, W = patches.shape
        flat = patches.view(B * N, C, H, W)
        proj = self.encode_patches(flat)         # (B*N, embed_dim)
        proj = proj.view(B, N, -1)               # (B, N, embed_dim)

        # Attribute-conditioned pooling -> (B, embed_dim)
        pooled = self.attn_pool(proj, attrs)

        # Early fusion with attributes if available
        if self.attr_mlp is not None and attrs is not None:
            a = self.attr_mlp(attrs)               # (B, embed_dim)
            fused = torch.cat([pooled, a], dim=1) # (B, 2*embed_dim)
            return self.fusion(fused)

        return pooled  # (B, embed_dim)


class NTXentLoss(nn.Module):
    """Corrected NT-Xent loss (SimCLR-style) with logsumexp for numerical stability."""
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N = z1.size(0)

        # Concatenate
        z = torch.cat([z1, z2], dim=0)  # (2N, D)

        # Similarity matrix
        sim = torch.mm(z, z.T) / self.temperature  # (2N, 2N)

        # Mask out self-similarity
        mask = torch.eye(2*N, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -9e15)

        # Positive pairs: diagonal offset by N
        positives = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], dim=0)

        # Denominator: logsumexp over all other pairs
        log_prob = torch.logsumexp(sim, dim=1)

        # Loss
        loss = -positives + log_prob
        return loss.mean()
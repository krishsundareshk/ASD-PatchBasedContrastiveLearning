"""
Vision backbone encoders for patch representation learning.
Provides ResNet architectures (ResNet-18, ResNet-34, ResNet-50) with ImageNet pretrained weights
and the final classification layer replaced by Identity.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
)


class ResNetEncoder(nn.Module):
    """
    ResNet backbone encoder extracting deep feature vectors from 2D patches.
    Replaces the classification head with Identity.
    """

    def __init__(self, architecture: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.architecture = architecture.lower()

        if self.architecture == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet18(weights=weights)
            self.feature_dim = 512
        elif self.architecture == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet34(weights=weights)
            self.feature_dim = 512
        elif self.architecture == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model = resnet50(weights=weights)
            self.feature_dim = 2048
        else:
            raise ValueError(
                f"Unsupported architecture: {architecture}. Choose 'resnet18', 'resnet34', or 'resnet50'."
            )

        model.fc = nn.Identity()
        self.backbone = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch tensor.
        Input: (B, 3, H, W)
        Output: (B, feature_dim)
        """
        return self.backbone(x)


def build_backbone(architecture: str = "resnet34", pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Helper function to instantiate a backbone and return (model, feature_dim)."""
    encoder = ResNetEncoder(architecture=architecture, pretrained=pretrained)
    return encoder, encoder.feature_dim

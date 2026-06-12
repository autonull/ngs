"""
Pretrained backbones for quick evaluation.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class PretrainedBackbone(nn.Module):
    """Wrapper for pretrained feature extractors."""
    
    BACKBONES = {
        'resnet18': {'model': models.resnet18, 'feat_dim': 512, 'weights': models.ResNet18_Weights.DEFAULT},
        'resnet34': {'model': models.resnet34, 'feat_dim': 512, 'weights': models.ResNet34_Weights.DEFAULT},
        'resnet50': {'model': models.resnet50, 'feat_dim': 2048, 'weights': models.ResNet50_Weights.DEFAULT},
        'vit_b_16': {'model': models.vit_b_16, 'feat_dim': 768, 'weights': models.ViT_B_16_Weights.DEFAULT},
        'vit_b_32': {'model': models.vit_b_32, 'feat_dim': 768, 'weights': models.ViT_B_32_Weights.DEFAULT},
        'mobilenet_v3_small': {'model': models.mobilenet_v3_small, 'feat_dim': 576, 'weights': models.MobileNet_V3_Small_Weights.DEFAULT},
        'efficientnet_b0': {'model': models.efficientnet_b0, 'feat_dim': 1280, 'weights': models.EfficientNet_B0_Weights.DEFAULT},
    }
    
    def __init__(self, name: str = 'resnet18', freeze: bool = True, pretrained: bool = True):
        super().__init__()
        if name not in self.BACKBONES:
            raise ValueError(f"Unknown backbone: {name}. Choose from {list(self.BACKBONES.keys())}")
        
        config = self.BACKBONES[name]
        model_fn = config['model']
        weights = config['weights'] if pretrained else None
        
        self.backbone = model_fn(weights=weights)
        self.feature_dim = config['feat_dim']
        self.name = name
        
        # Remove classifier head
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'heads'):
            self.backbone.heads = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features."""
        with torch.set_grad_enabled(self.training):
            return self.backbone(x)
    
    def train(self, mode: bool = True):
        if hasattr(self, 'freeze') and self.freeze:
            # Keep backbone in eval mode even when training head
            self.backbone.eval()
        else:
            self.backbone.train(mode)
        return super().train(mode)


class BackboneNGS(nn.Module):
    """Combined backbone + LeanNGS head."""
    
    def __init__(self, backbone: PretrainedBackbone, ngs_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.ngs_head = ngs_head
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.ngs_head(feat)
    
    def parameters_head(self):
        """Return only head parameters for optimizer."""
        return self.ngs_head.parameters()


def create_backbone_ngs(
    backbone_name: str,
    num_classes: int,
    d_latent: int = 64,
    k_init: int = 32,
    max_k: int = 256,
    top_k: int = 8,
    lora_rank: int = 4,
    freeze_backbone: bool = True
) -> BackboneNGS:
    """Create backbone + LeanNGS head combo."""
    from experiments.lean_ngs_trainer import create_lean_ngs
    
    backbone = PretrainedBackbone(backbone_name, freeze=freeze_backbone)
    ngs_head = create_lean_ngs(
        backbone.feature_dim, num_classes,
        d_latent=d_latent, k_init=k_init, max_k=max_k, top_k=top_k, lora_rank=lora_rank
    )
    
    return BackboneNGS(backbone, ngs_head)


if __name__ == '__main__':
    # Test backbones
    for name in ['resnet18', 'vit_b_16', 'mobilenet_v3_small']:
        try:
            bb = PretrainedBackbone(name, freeze=True)
            x = torch.randn(2, 3, 224, 224) if 'vit' in name else torch.randn(2, 3, 32, 32)
            out = bb(x)
            print(f"{name}: feat_dim={bb.feature_dim}, out_shape={out.shape}")
        except Exception as e:
            print(f"{name}: ERROR - {e}")
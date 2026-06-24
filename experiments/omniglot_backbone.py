"""
Omniglot CNN backbone for MAML (standard 4-layer conv net).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OmniglotCNN(nn.Module):
    """
    Standard 4-layer CNN for Omniglot MAML (Finn et al., 2017).
    Input: 1x28x28 -> Output: 64-d feature vector
    """
    
    def __init__(self, num_filters: int = 64, out_dim: int = 64):
        super().__init__()
        self.num_filters = num_filters
        self.out_dim = out_dim
        
        self.conv1 = nn.Conv2d(1, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_filters)
        
        # 28 -> 14 -> 7 -> 3 -> 1 (with 2x2 maxpool after each)
        # Actually: 28 -> 14 -> 7 -> 3 -> 1 (4 maxpools)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Final feature size: 64 * 1 * 1 = 64
        self.fc = nn.Linear(num_filters, out_dim)
    
    def forward(self, x):
        # x: [B, 784] or [B, 1, 28, 28]
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28 -> 14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14 -> 7
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 7 -> 3
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 3 -> 1
        
        x = x.view(x.size(0), -1)  # [B, 64]
        x = self.fc(x)  # [B, out_dim]
        return x


class CNNNGS(nn.Module):
    """CNN backbone + NGS head for MAML."""
    
    def __init__(self, cnn_backbone: nn.Module, ngs_head: nn.Module):
        super().__init__()
        self.backbone = cnn_backbone
        self.ngs_head = ngs_head
    
    def forward(self, x):
        feat = self.backbone(x)
        return self.ngs_head(feat)
    
    def parameters_backbone(self):
        return self.backbone.parameters()
    
    def parameters_head(self):
        return self.ngs_head.parameters()
    
    def named_parameters(self, prefix='', recurse=True):
        # Override to allow separate LR for backbone vs head
        for name, param in self.backbone.named_parameters(prefix='backbone.', recurse=recurse):
            yield name, param
        for name, param in self.ngs_head.named_parameters(prefix='ngs_head.', recurse=recurse):
            yield name, param


def create_cnn_ngs_maml(
    input_channels: int = 1,
    num_filters: int = 64,
    latent_dim: int = 64,
    max_k: int = 64,
    k_init: int = 32,
    top_k: int = 8,
    hypernetwork_hidden_dim: int = 64,
    hypernetwork_code_dim: int = 16,
    num_classes: int = 5
) -> CNNNGS:
    """Create CNN + NGS for MAML on Omniglot."""
    from ngs.models.ngs import build_ngs
    from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    
    cnn = OmniglotCNN(num_filters=num_filters, out_dim=latent_dim)
    
    cfg = NGSConfig(
        latent_dim=latent_dim,
        max_k=max_k,
        k_init=k_init,
        top_k=top_k,
        routing='monolithic_mahalanobis',
        parameter_storage='hypernetwork_generated',
        topology_control='discrete_heuristic',
        memory_management='dynamic',
        hypernetwork_hidden_dim=hypernetwork_hidden_dim,
        hypernetwork_code_dim=hypernetwork_code_dim,
        use_lora=False,
    )
    ngs_head = build_ngs(latent_dim, num_classes, cfg)
    
    return CNNNGS(cnn, ngs_head)
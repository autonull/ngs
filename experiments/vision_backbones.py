"""
Simple CNN backbones for Omniglot (28x28) and CIFAR (32x32).
Not pretrained - trained from scratch with NGS head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet4(nn.Module):
    """4-layer ConvNet for Omniglot (28x28 grayscale) -> 64-d features."""
    def __init__(self, in_channels=1, num_filters=64, out_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_filters)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(num_filters, out_dim)
        self.feature_dim = out_dim

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28->14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14->7
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 7->3
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 3->1
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvNet4CIFAR(nn.Module):
    """4-layer ConvNet for CIFAR (32x32 RGB) -> 128-d features."""
    def __init__(self, in_channels=3, num_filters=128, out_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_filters)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(num_filters * 2 * 2, out_dim)  # 32->16->8->4->2
        self.feature_dim = out_dim

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32->16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16->8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8->4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 4->2
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BackboneNGS(nn.Module):
    """Combined backbone + NGS head."""
    def __init__(self, backbone: nn.Module, ngs_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.ngs_head = ngs_head

    def forward(self, x):
        feat = self.backbone(x)
        return self.ngs_head(feat)

    def parameters_backbone(self):
        return self.backbone.parameters()

    def parameters_head(self):
        return self.ngs_head.parameters()

    def named_parameters(self, prefix='', recurse=True):
        for name, param in self.backbone.named_parameters(prefix='backbone.', recurse=recurse):
            yield name, param
        for name, param in self.ngs_head.named_parameters(prefix='ngs_head.', recurse=recurse):
            yield name, param


def create_omniglot_ngs(
    num_classes=5,
    latent_dim=64,
    max_k=64,
    k_init=32,
    top_k=8,
    hypernetwork_hidden_dim=64,
    hypernetwork_code_dim=16,
    parameter_storage='hypernetwork_generated',
    topology_control='discrete_heuristic',
) -> BackboneNGS:
    """Create ConvNet4 + NGS for Omniglot MAML."""
    from ngs.models.ngs import build_ngs
    from ngs.core.interfaces import NGSConfig
    
    backbone = ConvNet4(in_channels=1, num_filters=64, out_dim=latent_dim)
    
    cfg = NGSConfig(
        latent_dim=latent_dim,
        max_k=max_k,
        k_init=k_init,
        top_k=top_k,
        routing='monolithic_mahalanobis',
        parameter_storage=parameter_storage,
        topology_control=topology_control,
        memory_management='dynamic',
        hypernetwork_hidden_dim=hypernetwork_hidden_dim,
        hypernetwork_code_dim=hypernetwork_code_dim,
        use_lora=False,
    )
    ngs_head = build_ngs(latent_dim, num_classes, cfg)
    return BackboneNGS(backbone, ngs_head)


def create_cifar_ngs(
    num_classes=100,
    latent_dim=128,
    max_k=512,
    k_init=64,
    top_k=8,
    parameter_storage='direct_adapter',
    topology_control='autopoietic',
) -> BackboneNGS:
    """Create ConvNet4CIFAR + NGS for CIFAR-100."""
    from ngs.models.ngs import build_ngs
    from ngs.core.interfaces import NGSConfig
    
    backbone = ConvNet4CIFAR(in_channels=3, num_filters=128, out_dim=latent_dim)
    
    cfg = NGSConfig(
        latent_dim=latent_dim,
        max_k=max_k,
        k_init=k_init,
        top_k=top_k,
        routing='monolithic_mahalanobis',
        parameter_storage=parameter_storage,
        topology_control=topology_control,
        memory_management='dynamic',
        ema_decay=0.99,
        split_threshold=0.05,
        prune_threshold=0.01,
    )
    ngs_head = build_ngs(latent_dim, num_classes, cfg)
    return BackboneNGS(backbone, ngs_head)


if __name__ == '__main__':
    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Omniglot
    print("Testing Omniglot backbone...")
    bb = ConvNet4()
    x = torch.randn(2, 1, 28, 28)
    out = bb(x)
    print(f"  ConvNet4: {out.shape}")
    
    # CIFAR
    print("Testing CIFAR backbone...")
    bb = ConvNet4CIFAR()
    x = torch.randn(2, 3, 32, 32)
    out = bb(x)
    print(f"  ConvNet4CIFAR: {out.shape}")
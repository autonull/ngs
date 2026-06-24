"""
Smoke test for Autopoietic CIFAR-100 with ConvNet4CIFAR backbone + NGS head.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.vision_backbones import create_cifar_ngs
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEVICE}")

# Create model with ConvNet4CIFAR backbone + Autopoietic NGS head
model = create_cifar_ngs(
    num_classes=100,
    latent_dim=128,
    max_k=256,  # Smaller for smoke test
    k_init=32,
    top_k=8,
    parameter_storage='direct_adapter',
    topology_control='autopoietic',
).to(DEVICE)

print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
backbone_params = sum(p.numel() for p in model.parameters_backbone())
head_params = sum(p.numel() for p in model.parameters_head())
print(f"  Backbone: {backbone_params:,}")
print(f"  NGS head: {head_params:,}")
print(f"  Topology: {type(model.ngs_head.topology_manager).__name__}")

# Test forward
print("\n--- Forward pass ---")
x = torch.randn(32, 3, 32, 32).to(DEVICE)
out = model(x)
print(f"Output: {out.logits.shape}")
print(f"Initial active K: {model.ngs_head.router.active_mask.sum().item()}")

# Test autopoietic step
print("\n--- Autopoietic step ---")
z = model.ngs_head.p_down(model.backbone(x))
num_merged, num_split, num_spawned = model.ngs_head.topology_manager.step(model.ngs_head, z)
print(f"  Merged: {num_merged}, Split: {num_split}, Spawned: {num_spawned}")
print(f"  Active K after: {model.ngs_head.router.active_mask.sum().item()}")

# Tree stats
tree_stats = model.ngs_head.topology_manager.get_tree_stats()
print(f"  Tree stats: {tree_stats}")

# Quick training loop (2 epochs on subset)
print("\n--- Quick training (2 epochs, 10 batches) ---")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
train_ds = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()

for epoch in range(2):
    epoch_loss = 0
    for i, (x, y) in enumerate(train_loader):
        if i >= 10:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        opt.zero_grad()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out.logits, y)
        loss.backward()
        opt.step()
        
        # Autopoietic adaptation
        z = model.ngs_head.p_down(model.backbone(x))
        model.ngs_head.topology_manager.step(model.ngs_head, z)
        
        epoch_loss += loss.item()
    
    active_k = model.ngs_head.router.active_mask.sum().item()
    tree_stats = model.ngs_head.topology_manager.get_tree_stats()
    print(f"  Epoch {epoch+1}: loss={epoch_loss/10:.4f}, active_K={active_k}, tree={tree_stats}")

print("\n✅ Autopoietic CIFAR smoke test PASSED")
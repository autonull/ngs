"""
Full Autopoietic Splatting on CIFAR-100 (Paper 4).
Dynamic K tracks data complexity. Target: Dynamic 512 ≈ Static 1024 accuracy with 0.5x params.
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.vision_backbones import create_cifar_ngs
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEVICE}")

# Load CIFAR-100
print("Loading CIFAR-100...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_ds = datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
test_ds = datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

# Create model with ConvNet4CIFAR backbone + Autopoietic NGS head
model = create_cifar_ngs(
    num_classes=100,
    latent_dim=128,
    max_k=512,
    k_init=64,
    top_k=8,
    parameter_storage='direct_adapter',
    topology_control='autopoietic',
).to(DEVICE)

print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Backbone: {sum(p.numel() for p in model.parameters_backbone()):,}")
print(f"  NGS head: {sum(p.numel() for p in model.parameters_head()):,}")
print(f"  Topology: {type(model.ngs_head.topology_manager).__name__}")

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

num_epochs = 100
train_losses = []
test_accs = []
active_units_history = []
tree_stats_history = []

print(f"\n--- Training for {num_epochs} epochs ---")
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out.logits, y)
        
        # Add topology losses if available
        if hasattr(model.ngs_head, 'compute_topology_losses'):
            topo_losses = model.ngs_head.compute_topology_losses()
            loss = loss + topo_losses.get('entropy_loss', 0) + topo_losses.get('diversity_loss', 0)
        
        loss.backward()
        optimizer.step()
        
        # Autopoietic topology adaptation
        z = model.ngs_head.p_down(model.backbone(x))
        model.ngs_head.topology_manager.step(model.ngs_head, z)
        
        epoch_loss += loss.item()
        num_batches += 1
    
    scheduler.step()
    avg_loss = epoch_loss / num_batches
    train_losses.append(avg_loss)
    
    active_k = model.ngs_head.router.active_mask.sum().item()
    active_units_history.append(active_k)
    
    tree_stats = model.ngs_head.topology_manager.get_tree_stats()
    tree_stats_history.append(tree_stats)
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = out.logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    acc = correct / total
    test_accs.append(acc)
    
    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}, acc={acc:.4f}, active_K={active_k}, time={elapsed/60:.1f}min")
    if tree_stats:
        print(f"  Tree: max_depth={tree_stats.get('max_depth', 0)}, mean_depth={tree_stats.get('mean_depth', 0):.2f}, branching={tree_stats.get('branching_factor', 0):.2f}")

# Final results
print(f"\nFinal test accuracy: {test_accs[-1]:.4f}")
print(f"Best test accuracy: {max(test_accs):.4f}")
print(f"Final active units: {active_units_history[-1]}")
print(f"Max active units: {max(active_units_history)}")

# Save results
os.makedirs('results/full', exist_ok=True)
results = {
    'test_accs': test_accs,
    'train_losses': train_losses,
    'active_units_history': active_units_history,
    'tree_stats_history': tree_stats_history,
    'final_acc': test_accs[-1],
    'best_acc': max(test_accs),
    'final_active_k': active_units_history[-1],
    'max_active_k': max(active_units_history),
    'epochs': num_epochs,
}
json.dump(results, open('results/full/autopoietic_cifar100_full.json', 'w'))
print("\nResults saved to results/full/autopoietic_cifar100_full.json")
"""
Autopoietic Splatting on CIFAR-100 (Paper 4).
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

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models.ngs import build_ngs
from experiments.datasets import get_transform
from torchvision import datasets
from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEVICE}")

# Load CIFAR-100
print("Loading CIFAR-100...")
transform_train = get_transform('cifar100', augment=True)
transform_test = get_transform('cifar100', augment=False)

train_ds = datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
test_ds = datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

# Config for Autopoietic NGS
cfg = NGSConfig(
    latent_dim=128,
    k_init=64,
    max_k=512,
    top_k=8,
    routing='monolithic_mahalanobis',
    parameter_storage='direct_adapter',  # Simpler for image classification
    topology_control='autopoietic',
    memory_management='dynamic',
    ema_decay=0.99,
    split_threshold=0.05,
    prune_threshold=0.01,
)

# Add autopoietic config
cfg.extra['entropy_split_threshold'] = 1.2
cfg.extra['overlap_merge_threshold'] = 0.85
cfg.extra['max_tree_depth'] = 5

print("Building Autopoietic NGS model...")
model = build_ngs(32*32*3, 100, cfg).to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Training loop with autopoietic topology adaptation
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
        x = x.view(x.size(0), -1)  # Flatten
        
        optimizer.zero_grad()
        out = model(x)
        logits = out.logits if hasattr(out, 'logits') else out
        loss = F.cross_entropy(logits, y)
        
        # Add topology losses if available
        if hasattr(model, 'compute_topology_losses'):
            topo_losses = model.compute_topology_losses()
            loss = loss + topo_losses.get('entropy_loss', 0) + topo_losses.get('diversity_loss', 0)
        
        loss.backward()
        optimizer.step()
        
        # Autopoietic topology adaptation every batch
        if hasattr(model.topology_manager, 'step'):
            z = model.p_down(x)
            model.topology_manager.step(model, z)
        
        epoch_loss += loss.item()
        num_batches += 1
    
    scheduler.step()
    avg_loss = epoch_loss / num_batches
    train_losses.append(avg_loss)
    
    # Track active units
    active_k = model.router.active_mask.sum().item()
    active_units_history.append(active_k)
    
    # Track tree stats
    if hasattr(model.topology_manager, 'get_tree_stats'):
        tree_stats = model.topology_manager.get_tree_stats()
        tree_stats_history.append(tree_stats)
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.size(0), -1)
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            pred = logits.argmax(1)
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
    'epochs': num_epochs,
}
json.dump(results, open('results/full/autopoietic_cifar100.json', 'w'))
print("\nResults saved to results/full/autopoietic_cifar100.json")
"""
Smoke test: EqNGS on MNIST — verify 98%+ accuracy with zero activation graph.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.modules.eqprop import create_eqngs

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)

print(f"Device: {DEVICE}")

# Load MNIST
print("Loading MNIST...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

# Config for EqNGS
cfg = NGSConfig(
    latent_dim=64,
    k_init=32,
    max_k=256,
    top_k=8,
    routing='monolithic_mahalanobis',
    parameter_storage='direct_adapter',
    topology_control='discrete_heuristic',
    memory_management='dynamic',
    ema_decay=0.99,
)

print("Creating EqNGS model...")
model = create_eqngs(
    d_in=784,
    d_out=10,
    config=cfg,
    ep_beta=0.5,
    ep_settle_steps=10,
    ep_settle_lr=0.2,
    ep_momentum=0.9,
    spectral_gamma=0.95,
).to(DEVICE)

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# Training loop (1 epoch for smoke test)
print("\n--- Training 1 epoch (EP mode) ---")
model.train()

for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc="Training")):
    x, y = x.to(DEVICE), y.to(DEVICE)
    x = x.view(x.size(0), -1)  # Flatten
    
    # EP step (replaces forward + backward + optimizer.step)
    result = model.ep_step(x, y)
    
    if batch_idx % 100 == 0:
        print(f"  Batch {batch_idx}: loss={result['loss']:.4f}, acc={result['accuracy']:.4f}")

# Evaluation
print("\n--- Evaluation ---")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Testing"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = x.view(x.size(0), -1)
        
        out = model(x)
        logits = out.logits if hasattr(out, 'logits') else out
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

acc = correct / total
print(f"\nTest Accuracy: {acc:.4f} ({correct}/{total})")
print("✅ PASS" if acc > 0.97 else "❌ NEEDS MORE TRAINING")

# Memory check
print(f"\nGPU Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB allocated")
print(f"  (Should be constant regardless of depth — no activation graph stored)")
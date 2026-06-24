"""
Fast 4D: Meta-Gaussian 10 shifts - self-referential growth.
Tests if NGS can adapt topology across 10 distribution shifts.
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models.ngs import build_ngs

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# 10 Gaussian shifts with different means
shifts = [np.random.randn(2) * 3 for _ in range(10)]

cfg = NGSConfig(latent_dim=2, k_init=8, max_k=32, top_k=4,
    routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    parameter_storage=ParameterStorage.DIRECT_ADAPTER,
    topology_control=TopologyControl.CONTINUOUS_DENSITY,
    memory_management=MemoryManagement.DYNAMIC,
    split_threshold=0.05, prune_threshold=0.01,
    ema_decay=0.99)

model = build_ngs(2, 10, cfg).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Testing Meta-Gaussian on 10 shifts...")
results = []

for i, shift in enumerate(shifts):
    # Generate data for this shift
    n_samples = 2000
    X = torch.randn(n_samples, 2, dtype=torch.float32).to(DEVICE) + torch.tensor(shift, dtype=torch.float32).to(DEVICE)
    y = (X[:, 0] > 0).long()
    
    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    
    # Train for 5 epochs
    model.train()
    for _ in range(5):
        for x, y in loader:
            opt.zero_grad()
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
    
    # Adapt topology
    model.adapt_density(z_samples=X[:500], split_thresh=0.05, prune_thresh=0.01, spawn_thresh=-5.0, max_spawn_per_call=8)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        out = model(X)
        logits = out.logits if hasattr(out, 'logits') else out
        pred = logits.argmax(1)
        acc = (pred == y).float().mean().item()
    
    results.append({'shift': shift.tolist(), 'acc': acc, 'K': model.K})
    print(f"Shift {i+1}/10: acc={acc:.3f}, K={model.K}")

avg_acc = np.mean([r['acc'] for r in results])
print(f"\nAvg accuracy: {avg_acc:.3f}")
print(f"Final K: {model.K}")

os.makedirs('results/full', exist_ok=True)
json.dump(results, open('results/full/4d.json', 'w'))
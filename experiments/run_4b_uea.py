"""
4B: UEA FactorizedRouter on CharacterTrajectories.
1 seed, 20 epochs, validates >5% over monolithic.
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models.ngs import build_ngs
from experiments.datasets_uea import get_uea_loaders

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# Load UEA CharacterTrajectories
train_loader, test_loader, n_classes, seq_len, n_channels = get_uea_loaders('CharacterTrajectories', batch_size=64)
print(f"CharacterTrajectories: {n_classes} classes, seq_len={seq_len}, n_channels={n_channels}")
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# Monolithic baseline
mono_cfg = NGSConfig(latent_dim=64, k_init=32, max_k=128, top_k=8,
    routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    parameter_storage=ParameterStorage.DIRECT_ADAPTER,
    topology_control=TopologyControl.DISCRETE_HEURISTIC,
    memory_management=MemoryManagement.DYNAMIC)

# Factorized config - 3 subspaces (one per channel)
fact_cfg = NGSConfig(latent_dim=64, k_init=32, max_k=128, top_k=8,
    routing=RoutingStrategy.FACTORIZED_SUBSPACE,
    parameter_storage=ParameterStorage.DIRECT_ADAPTER,
    topology_control=TopologyControl.DISCRETE_HEURISTIC,
    memory_management=MemoryManagement.DYNAMIC,
    num_subspaces=n_channels, top_k_factorized=2)

def train_eval(model, train_loader, test_loader, epochs=20, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            # x: [B, seq_len, n_channels] -> flatten to [B, seq_len * n_channels]
            x = x.contiguous().view(x.size(0), -1)
            if epoch == 0 and i == 0:
                print(f"DEBUG: x shape before model: {x.shape}")
            opt.zero_grad()
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
    model.eval(); corr=tot=0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.contiguous().view(x.size(0), -1)
            if i == 0:
                print(f"DEBUG EVAL: x shape before model: {x.shape}")
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            p = logits.argmax(1); corr += (p==y).sum().item(); tot += y.size(0)
    return corr/tot

# Both models get concatenated input [B, seq_len * n_channels]
input_dim = seq_len * n_channels
print(f"Input dim (concatenated): {input_dim}")

# Monolithic
mono = build_ngs(input_dim, n_classes, mono_cfg).to(DEVICE)
mono_acc = train_eval(mono, train_loader, test_loader)
print(f"Monolithic acc: {mono_acc:.3f}")

# Factorized
fact = build_ngs(input_dim, n_classes, fact_cfg).to(DEVICE)
fact_acc = train_eval(fact, train_loader, test_loader)
print(f"Factorized acc: {fact_acc:.3f}")

improvement = fact_acc - mono_acc
print(f"\nImprovement: {improvement:.3f} (target >0.05)")
print("✅ PASS" if improvement > 0.05 else "❌ NEEDS WORK")

os.makedirs('results/full', exist_ok=True)
json.dump({'mono': mono_acc, 'fact': fact_acc, 'improvement': improvement},
          open('results/full/4b_uea.json', 'w'))
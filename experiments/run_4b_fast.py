"""
Fast 4B: UEA FactorizedRouter - CharacterTrajectories.
1 seed, 10 epochs, validates >5% over monolithic on multimodal data.
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models.ngs import build_ngs
from experiments.datasets import get_task_loaders

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# Check if UEA data exists, else use multimodal MNIST as proxy
def get_multimodal_data():
    """Create multimodal MNIST as UEA proxy."""
    from experiments.datasets import MultimodalMNIST
    ds = MultimodalMNIST(modality_types=("original", "permuted", "rotated", "noisy"), seed=SEED)
    return ds.get_loaders(batch_size=128)

# Monolithic baseline
mono_cfg = NGSConfig(latent_dim=64, k_init=32, max_k=128, top_k=8,
    routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    parameter_storage=ParameterStorage.DIRECT_ADAPTER,
    topology_control=TopologyControl.DISCRETE_HEURISTIC,
    memory_management=MemoryManagement.DYNAMIC)

# Factorized config
fact_cfg = NGSConfig(latent_dim=64, k_init=32, max_k=128, top_k=8,
    routing=RoutingStrategy.FACTORIZED_SUBSPACE,
    parameter_storage=ParameterStorage.DIRECT_ADAPTER,
    topology_control=TopologyControl.DISCRETE_HEURISTIC,
    memory_management=MemoryManagement.DYNAMIC,
    num_subspaces=4, top_k_factorized=2)

def train_eval(model, train_loader, test_loader, epochs=10, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # MultimodalMNIST returns [B, M, D] - concatenate for both monolithic and factorized
            if x.dim() == 3:  # [B, M, D]
                B, M, D = x.shape
                x = x.view(B, M * D)
            opt.zero_grad()
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
    model.eval(); corr=tot=0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if x.dim() == 3:
                B, M, D = x.shape
                x = x.view(B, M * D)
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            p = logits.argmax(1); corr += (p==y).sum().item(); tot += y.size(0)
    return corr/tot

# Try UEA first
try:
    from experiments.datasets import get_task_loaders
    tr, te, _ = get_task_loaders('CharacterTrajectories', 0, 20, 128)
    print("Using UEA CharacterTrajectories")
    d_in = tr.dataset[0][0].shape[-1]
except:
    print("UEA not available, using Multimodal MNIST (4 modalities)")
    tr, te = get_multimodal_data()
    d_in = 784

print(f"Input dim per modality: {d_in}")

# Multimodal MNIST has 4 modalities
M = 4
# Both use concatenated input (4 modalities * 784 = 3136)
input_dim = d_in * M
mono = build_ngs(input_dim, 10, mono_cfg).to(DEVICE)
mono_acc = train_eval(mono, tr, te)
print(f"Monolithic acc: {mono_acc:.3f}")

fact = build_ngs(input_dim, 10, fact_cfg).to(DEVICE)
fact_acc = train_eval(fact, tr, te)
print(f"Factorized acc: {fact_acc:.3f}")

improvement = fact_acc - mono_acc
print(f"\nImprovement: {improvement:.3f} (target >0.05)")
print("✅ PASS" if improvement > 0.05 else "❌ NEEDS WORK")

os.makedirs('results/full', exist_ok=True)
json.dump({'mono': mono_acc, 'fact': fact_acc, 'improvement': improvement},
          open('results/full/4b.json', 'w'))
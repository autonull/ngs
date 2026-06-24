"""
4B: UEA FactorizedRouter on multiple datasets.
Tests CharacterTrajectories, EigenWorms, Heartbeat.
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

datasets = ['CharacterTrajectories', 'EigenWorms', 'Heartbeat']
results = {}

for ds_name in datasets:
    try:
        train_loader, test_loader, n_classes, seq_len, n_channels = get_uea_loaders(ds_name, batch_size=64)
        print(f"\n{'='*50}")
        print(f"{ds_name}: {n_classes} classes, seq_len={seq_len}, n_channels={n_channels}")
        print(f"{'='*50}")
    except Exception as e:
        print(f"{ds_name}: FAILED to load - {e}")
        continue
    
    input_dim = seq_len * n_channels
    
    # Configs
    mono_cfg = NGSConfig(latent_dim=64, k_init=32, max_k=128, top_k=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.DYNAMIC)
    
    fact_cfg = NGSConfig(latent_dim=64, k_init=32, max_k=128, top_k=8,
        routing=RoutingStrategy.FACTORIZED_SUBSPACE,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.DYNAMIC,
        num_subspaces=n_channels, top_k_factorized=2)
    
    def train_eval(model, train_loader, test_loader, epochs=20, lr=1e-3):
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()
        for _ in range(epochs):
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = x.contiguous().view(x.size(0), -1)
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
                x = x.contiguous().view(x.size(0), -1)
                out = model(x)
                logits = out.logits if hasattr(out, 'logits') else out
                p = logits.argmax(1); corr += (p==y).sum().item(); tot += y.size(0)
        return corr/tot
    
    # Monolithic
    mono = build_ngs(input_dim, n_classes, mono_cfg).to(DEVICE)
    mono_acc = train_eval(mono, train_loader, test_loader)
    print(f"Monolithic acc: {mono_acc:.3f}")
    
    # Factorized
    fact = build_ngs(input_dim, n_classes, fact_cfg).to(DEVICE)
    fact_acc = train_eval(fact, train_loader, test_loader)
    print(f"Factorized acc: {fact_acc:.3f}")
    
    improvement = fact_acc = fact_acc - mono_acc
    print(f"Improvement: {improvement:.3f}")
    
    results[ds_name] = {'mono': mono_acc, 'fact': fact_acc, 'improvement': improvement}

print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
for ds, r in results.items():
    print(f"{ds}: mono={r['mono']:.3f}, fact={r['fact']:.3f}, imp={r['improvement']:.3f}")

os.makedirs('results/full', exist_ok=True)
json.dump(results, open('results/full/4b_all.json', 'w'))
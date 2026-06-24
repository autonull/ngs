"""
Fast 4H: Omniglot meta-learning + hypernet.
1 seed, 5-way 1-shot, 200 meta-train tasks, validates >80% (target 95% for full).
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models.ngs import build_ngs
from experiments.datasets import get_omniglot_loaders

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

print("Loading Omniglot...")
try:
    tasks = get_omniglot_loaders(n_way=5, k_shot=1, n_query=15, n_tasks=200, image_size=28)
    print(f"Loaded {len(tasks)} few-shot tasks")
except Exception as e:
    print(f"Omniglot failed: {e}")
    sys.exit(1)

# Simple hypernet adapter for few-shot
cfg = NGSConfig(latent_dim=64, k_init=16, max_k=32, top_k=4,
    routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
    topology_control=TopologyControl.DISCRETE_HEURISTIC,
    memory_management=MemoryManagement.DYNAMIC,
    hypernetwork_hidden_dim=32, hypernetwork_code_dim=8)

model = build_ngs(784, 5, cfg).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Meta-training on 200 tasks...")
for i, (supp_loader, query_loader, _) in enumerate(tasks):
    if i >= 200: break
    
    supp_x, supp_y = next(iter(supp_loader))
    supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)
    if supp_x.dim() > 2: supp_x = supp_x.view(supp_x.size(0), -1)
    
    query_x, query_y = next(iter(query_loader))
    query_x, query_y = query_x.to(DEVICE), query_y.to(DEVICE)
    if query_x.dim() > 2: query_x = query_x.view(query_x.size(0), -1)
    
    # Fast adaptation: 5 steps on support
    model.train()
    for _ in range(5):
        opt.zero_grad()
        out = model(supp_x)
        logits = out.logits if hasattr(out, 'logits') else out
        loss = F.cross_entropy(logits, supp_y)
        loss.backward()
        opt.step()
    
    # Evaluate on query
    model.eval()
    with torch.no_grad():
        out = model(query_x)
        logits = out.logits if hasattr(out, 'logits') else out
        pred = logits.argmax(1)
        acc = (pred == query_y).float().mean().item()
    
    if i % 50 == 0:
        print(f"Task {i}: query_acc={acc:.3f}")

# Final eval on 20 held-out tasks
print("Evaluating on 20 held-out tasks...")
eval_accs = []
for i, (supp_loader, query_loader, _) in enumerate(tasks[200:220]):
    if i >= 20: break
    supp_x, supp_y = next(iter(supp_loader))
    supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)
    if supp_x.dim() > 2: supp_x = supp_x.view(supp_x.size(0), -1)
    query_x, query_y = next(iter(query_loader))
    query_x, query_y = query_x.to(DEVICE), query_y.to(DEVICE)
    if query_x.dim() > 2: query_x = query_x.view(query_x.size(0), -1)
    
    # Reset to meta-init - reinitialize model
    model = build_ngs(784, 5, cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for _ in range(5):
        opt.zero_grad()
        out = model(supp_x)
        logits = out.logits if hasattr(out, 'logits') else out
        loss = F.cross_entropy(logits, supp_y)
        loss.backward()
        opt.step()
    
    model.eval()
    with torch.no_grad():
        out = model(query_x)
        logits = out.logits if hasattr(out, 'logits') else out
        pred = logits.argmax(1)
        acc = (pred == query_y).float().mean().item()
        eval_accs.append(acc)

avg_acc = np.mean(eval_accs) if eval_accs else 0
print(f"\n5-way 1-shot acc: {avg_acc:.3f} (target >0.80 for fast, >0.95 for full)")
print("✅ PASS" if avg_acc > 0.80 else "❌ NEEDS SCALE")

os.makedirs('results/full', exist_ok=True)
json.dump({'avg_acc': float(avg_acc), 'eval_accs': eval_accs},
          open('results/full/4h.json', 'w'))
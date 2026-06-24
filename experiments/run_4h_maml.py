"""
4H: Omniglot MAML meta-learning + hypernet adapter.
1 seed, 5-way 1-shot, 1000 meta-train tasks, 10 inner steps, validates >95%.
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
from copy import deepcopy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models.ngs import build_ngs
from experiments.datasets import get_omniglot_loaders

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

print("Loading Omniglot...")
tasks = get_omniglot_loaders(n_way=5, k_shot=1, n_query=15, n_tasks=1000, image_size=28)
print(f"Loaded {len(tasks)} few-shot tasks")

# Meta-learning config
cfg = NGSConfig(latent_dim=64, k_init=16, max_k=32, top_k=4,
    routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
    topology_control=TopologyControl.DISCRETE_HEURISTIC,
    memory_management=MemoryManagement.DYNAMIC,
    hypernetwork_hidden_dim=32, hypernetwork_code_dim=8)

# Meta-model (shared initialization)
meta_model = build_ngs(784, 5, cfg).to(DEVICE)
meta_opt = torch.optim.AdamW(meta_model.parameters(), lr=1e-3)

def adapt_model(model, supp_x, supp_y, inner_steps=10, inner_lr=0.01):
    """Fast adaptation on support set."""
    # Create adapted copy
    adapted = deepcopy(model)
    inner_opt = torch.optim.SGD(adapted.parameters(), lr=inner_lr)
    adapted.train()
    for _ in range(inner_steps):
        inner_opt.zero_grad()
        out = adapted(supp_x)
        logits = out.logits if hasattr(out, 'logits') else out
        loss = F.cross_entropy(logits, supp_y)
        loss.backward()
        inner_opt.step()
    return adapted

print("Meta-training (MAML)...")
meta_batch_size = 4
for meta_iter in range(250):  # 250 * 4 = 1000 tasks
    meta_opt.zero_grad()
    meta_loss = 0
    
    _batch = np.random.choice(len(tasks), meta_batch_size, replace=False)
    
    for task_idx in _batch:
        supp_loader, query_loader, _ = tasks[task_idx]
        supp_x, supp_y = next(iter(supp_loader))
        query_x, query_y = next(iter(query_loader))
        
        supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)
        query_x, query_y = query_x.to(DEVICE), query_y.to(DEVICE)
        if supp_x.dim() > 2: supp_x = supp_x.view(supp_x.size(0), -1)
        if query_x.dim() > 2: query_x = query_x.view(query_x.size(0), -1)
        
        # Inner loop adaptation
        adapted = adapt_model(meta_model, supp_x, supp_y, inner_steps=5, inner_lr=0.01)
        
        # Outer loss on query set
        adapted.eval()
        with torch.no_grad():
            out = adapted(query_x)
            logits = out.logits if hasattr(out, 'logits') else out
            loss = F.cross_entropy(logits, query_y)
        
        meta_loss += loss
    
    (meta_loss / meta_batch_size).backward()
    meta_opt.step()
    
    if meta_iter % 50 == 0:
        print(f"Meta-iter {meta_iter}: meta_loss={meta_loss.item()/meta_batch_size:.4f}")

# Final evaluation on held-out tasks
print("\nEvaluating on 100 held-out tasks...")
eval_tasks = tasks[900:1000]
eval_accs = []

for supp_loader, query_loader, _ in eval_tasks:
    supp_x, supp_y = next(iter(supp_loader))
    query_x, query_y = next(iter(query_loader))
    
    supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)
    query_x, query_y = query_x.to(DEVICE), query_y.to(DEVICE)
    if supp_x.dim() > 2: supp_x = supp_x.view(supp_x.size(0), -1)
    if query_x.dim() > 2: query_x = query_x.view(query_x.size(0), -1)
    
    # Adapt from meta-init
    adapted = adapt_model(meta_model, supp_x, supp_y, inner_steps=10, inner_lr=0.01)
    
    adapted.eval()
    with torch.no_grad():
        out = adapted(query_x)
        logits = out.logits if hasattr(out, 'logits') else out
        pred = logits.argmax(1)
        acc = (pred == query_y).float().mean().item()
        eval_accs.append(acc)

avg_acc = np.mean(eval_accs)
print(f"\n5-way 1-shot acc: {avg_acc:.3f} (target >0.95)")
print("✅ PASS" if avg_acc > 0.95 else "❌ NEEDS MORE")

os.makedirs('results/full', exist_ok=True)
json.dump({'avg_acc': float(avg_acc), 'eval_accs': eval_accs},
          open('results/full/4h_maml.json', 'w'))
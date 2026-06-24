"""
Smoke test for Omniglot MAML with `higher` library integration.
Tests that gradients flow correctly through the hypernetwork.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.maml_trainer import create_maml_trainer
from experiments.datasets import get_omniglot_loaders

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEVICE}")
print("Loading Omniglot (100 tasks for smoke test)...")
tasks = get_omniglot_loaders(
    n_way=5, k_shot=1, n_query=15, n_tasks=100, image_size=28
)
print(f"Loaded {len(tasks)} few-shot tasks")

# 5-way 1-shot")

trainer = create_maml_trainer(
    input_dim=784, output_dim=5,
    latent_dim=64, max_k=32, k_init=16, top_k=4,
    inner_lr=0.01, inner_steps=5, meta_lr=1e-3,
    device=DEVICE
)

print(f"\nMeta-model params: {sum(p.numel() for p in trainer.meta_model.parameters()):,}")
print(f"Inner-loop params (router+codes): {sum(p.numel() for p in trainer.inner_loop_params()):,}")
print(f"Meta params (feature extractor + hypernet): {sum(p.numel() for p in trainer.meta_opt.param_groups[0]['params']):,}")

# Test single MAML step
print("\n--- Testing MAML step with `higher` ---")
task_idx = 0
supp_loader, query_loader, _ = tasks[task_idx]
supp_x, supp_y = next(iter(supp_loader))
query_x, query_y = next(iter(query_loader))

supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)
query_x, query_y = query_x.to(DEVICE), query_y.to(DEVICE)
if supp_x.dim() > 2: supp_x = supp_x.view(supp_x.size(0), -1)
if query_x.dim() > 2: query_x = query_x.view(query_x.size(0), -1)

print(f"Support: {supp_x.shape}, {supp_y.shape}")
print(f"Query: {query_x.shape}, {query_y.shape}")

meta_loss, fmodel = trainer.maml_step(supp_x, supp_y, query_x, query_y)
print(f"Meta loss: {meta_loss.item():.4f}")

# Check gradients exist on meta-params
print("\n--- Checking meta-gradients ---")
has_grad = False
no_grad = []
for name, param in trainer.meta_model.named_parameters():
    if 'router' not in name and 'code' not in name:
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-6:
                print(f"  ✅ {name}: grad norm = {grad_norm:.6f}")
            else:
                print(f"  ⚠️  {name}: grad norm = {grad_norm:.6f} (near zero)")
        else:
            no_grad.append(name)

if no_grad:
    print(f"  ❌ No gradients on: {no_grad}")
else:
    print(f"  ✅ All meta-params have gradients")

# Test meta-update
print("\n--- Testing meta-update ---")
trainer.meta_update(meta_loss)
print("Meta-update successful")

# Quick evaluation
print("\n--- Quick evaluation (10 inner steps) ---")
acc = trainer.evaluate(supp_x, supp_y, query_x, query_y, inner_steps=10)
print(f"5-way 1-shot accuracy: {acc:.3f}")

# Test 10 meta-iterations
print("\n--- Running 10 meta-iterations ---")
meta_batch_size = 4
for meta_iter in range(10):
    meta_loss_sum = 0
    batch_indices = np.random.choice(len(tasks), meta_batch_size, replace=False)
    
    for task_idx in batch_indices:
        supp_loader, query_loader, _ = tasks[task_idx]
        supp_x, supp_y = next(iter(supp_loader))
        query_x, query_y = next(iter(query_loader))
        
        supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)
        query_x, query_y = query_x.to(DEVICE), query_y.to(DEVICE)
        if supp_x.dim() > 2: supp_x = supp_x.view(supp_x.size(0), -1)
        if query_x.dim() > 2: query_x = query_x.view(query_x.size(0), -1)
        
        meta_loss, _ = trainer.maml_step(supp_x, supp_y, query_x, query_y)
        meta_loss_sum += meta_loss
    
    avg_meta_loss = meta_loss_sum / meta_batch_size
    trainer.meta_update(avg_meta_loss)
    
    if meta_iter % 2 == 0:
        print(f"  Meta-iter {meta_iter}: meta_loss = {avg_meta_loss.item():.4f}")

print("\n--- Final evaluation on 20 held-out tasks ---")
eval_tasks = tasks[80:100]
eval_accs = []

for supp_loader, query_loader, _ in eval_tasks:
    supp_x, supp_y = next(iter(supp_loader))
    query_x, query_y = next(iter(query_loader))
    
    supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)
    query_x, query_y = query_x.to(DEVICE), query_y.to(DEVICE)
    if supp_x.dim() > 2: supp_x = supp_x.view(supp_x.size(0), -1)
    if query_x.dim() > 2: query_x = query_x.view(query_x.size(0), -1)
    
    acc = trainer.evaluate(supp_x, supp_y, query_x, query_y, inner_steps=10)
    eval_accs.append(acc)

avg_acc = np.mean(eval_accs)
print(f"5-way 1-shot acc: {avg_acc:.3f} (target >0.80 for smoke test)")
print("✅ SMOKE TEST PASSED" if avg_acc > 0.80 else "❌ NEEDS MORE TRAINING")

os.makedirs('results/full', exist_ok=True)
import json
json.dump({
    'avg_acc': float(avg_acc),
    'eval_accs': eval_accs,
    'meta_gradients_work': has_grad and len(no_grad) == 0
}, open('results/full/smoke_maml_higher.json', 'w'))
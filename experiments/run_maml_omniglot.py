"""
Full MAML training on Omniglot for Paper 2 (Meta-Learned Gaussian Priors).
Target: 5-way 1-shot >95% after 2000 meta-tasks.
"""
import os
import sys
import json
import time
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.maml_trainer import create_maml_trainer
from experiments.datasets import get_omniglot_loaders

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEVICE}")

# Load Omniglot: 2000 meta-train tasks + 100 held-out for eval
print("Loading Omniglot (2000 meta-train + 100 eval tasks)...")
tasks = get_omniglot_loaders(
    n_way=5, k_shot=1, n_query=15, n_tasks=2100, image_size=28
)
print(f"Loaded {len(tasks)} few-shot tasks")

meta_train_tasks = tasks[:2000]
eval_tasks = tasks[2000:]

# Create trainer
trainer = create_maml_trainer(
    input_dim=784, output_dim=5,
    latent_dim=64, max_k=32, k_init=16, top_k=4,
    inner_lr=0.01, inner_steps=10, meta_lr=1e-3,
    device=DEVICE
)

print(f"\nMeta-model params: {sum(p.numel() for p in trainer.meta_model.parameters()):,}")
print(f"Inner-loop params: {sum(p.numel() for p in trainer.inner_loop_params()):,}")
print(f"Meta params: {sum(p.numel() for p in trainer.meta_opt.param_groups[0]['params']):,}")

# Training
meta_batch_size = 4
num_meta_iters = 500  # 500 * 4 = 2000 tasks
meta_losses = []

print(f"\n--- Meta-training for {num_meta_iters} iterations ({num_meta_iters * meta_batch_size} tasks) ---")
start_time = time.time()

for meta_iter in range(num_meta_iters):
    meta_loss_sum = 0
    batch_indices = np.random.choice(len(meta_train_tasks), meta_batch_size, replace=False)
    
    for task_idx in batch_indices:
        supp_loader, query_loader, _ = meta_train_tasks[task_idx]
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
    meta_losses.append(avg_meta_loss.item())
    
    if meta_iter % 50 == 0:
        elapsed = time.time() - start_time
        print(f"  Meta-iter {meta_iter}/{num_meta_iters}: meta_loss = {avg_meta_loss.item():.4f} (elapsed: {elapsed/60:.1f}min)")
        
        # Quick eval every 100 iters
        if meta_iter % 100 == 0 and meta_iter > 0:
            eval_accs = []
            for supp_loader, query_loader, _ in eval_tasks[:20]:
                supp_x, supp_y = next(iter(supp_loader))
                query_x, query_y = next(iter(query_loader))
                supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)
                query_x, query_y = query_x.to(DEVICE), query_y.to(DEVICE)
                if supp_x.dim() > 2: supp_x = supp_x.view(supp_x.size(0), -1)
                if query_x.dim() > 2: query_x = query_x.view(query_x.size(0), -1)
                acc = trainer.evaluate(supp_x, supp_y, query_x, query_y, inner_steps=10)
                eval_accs.append(acc)
            print(f"    Quick eval (20 tasks): {np.mean(eval_accs):.3f}")

# Final evaluation
print("\n--- Final evaluation on 100 held-out tasks ---")
eval_accs = []

for supp_loader, query_loader, _ in tqdm(eval_tasks, desc="Evaluating"):
    supp_x, supp_y = next(iter(supp_loader))
    query_x, query_y = next(iter(query_loader))
    
    supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)
    query_x, query_y = query_x.to(DEVICE), query_y.to(DEVICE)
    if supp_x.dim() > 2: supp_x = supp_x.view(supp_x.size(0), -1)
    if query_x.dim() > 2: query_x = query_x.view(query_x.size(0), -1)
    
    acc = trainer.evaluate(supp_x, supp_y, query_x, query_y, inner_steps=10)
    eval_accs.append(acc)

avg_acc = np.mean(eval_accs)
std_acc = np.std(eval_accs)

print(f"\n5-way 1-shot accuracy: {avg_acc:.3f} ± {std_acc:.3f}")
print(f"Target: >0.95")
print("✅ PASS" if avg_acc > 0.95 else "❌ NEEDS MORE TRAINING")

# Save results
os.makedirs('results/full', exist_ok=True)
results = {
    'avg_acc': float(avg_acc),
    'std_acc': float(std_acc),
    'eval_accs': eval_accs,
    'meta_losses': meta_losses,
    'num_meta_iters': num_meta_iters,
    'meta_batch_size': meta_batch_size,
    'inner_steps': 10,
    'inner_lr': 0.01,
    'meta_lr': 1e-3,
    'target_met': bool(avg_acc > 0.95)
}
json.dump(results, open('results/full/maml_omniglot_full.json', 'w'))
print("\nResults saved to results/full/maml_omniglot_full.json")
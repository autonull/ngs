"""
Quick MAML Omniglot with ConvNet4 backbone (50 meta-tasks, 10 eval).
Verifies end-to-end pipeline works.
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.maml_trainer_cnn import MAMLTrainerCNN
from experiments.datasets import get_omniglot_loaders

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEVICE}")

# Load 50 meta-train + 10 eval
print("Loading Omniglot (50 meta-train + 10 eval)...")
tasks = get_omniglot_loaders(n_way=5, k_shot=1, n_query=15, n_tasks=60, image_size=28)
meta_train = tasks[:50]
eval_tasks = tasks[50:]

trainer = MAMLTrainerCNN(
    num_classes=5, latent_dim=64, max_k=64, k_init=32, top_k=8,
    inner_lr=0.01, inner_steps=5, meta_lr=1e-3, backbone_lr=1e-4, device=DEVICE
)

print(f"Meta params: {sum(p.numel() for p in trainer.meta_model.parameters()):,}")

# 25 meta-iterations (100 tasks)
meta_batch_size = 4
num_iters = 25

print(f"\n--- Meta-training {num_iters} iters ({num_iters * meta_batch_size} tasks) ---")
for meta_iter in range(num_iters):
    meta_loss_sum = 0
    for task_idx in np.random.choice(len(meta_train), meta_batch_size, replace=False):
        supp_x, supp_y = next(iter(meta_train[task_idx][0]))
        query_x, query_y = next(iter(meta_train[task_idx][1]))
        supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)
        query_x, query_y = query_x.to(DEVICE), query_y.to(DEVICE)
        meta_loss, _ = trainer.maml_step(supp_x, supp_y, query_x, query_y)
        meta_loss_sum += meta_loss
    trainer.meta_update(meta_loss_sum / meta_batch_size)
    
    if meta_iter % 5 == 0:
        print(f"  Iter {meta_iter}: meta_loss = {meta_loss_sum.item()/meta_batch_size:.4f}")

# Quick eval
print("\n--- Eval on 10 held-out tasks ---")
accs = []
for supp_loader, query_loader, _ in eval_tasks:
    supp_x, supp_y = next(iter(supp_loader))
    query_x, query_y = next(iter(query_loader))
    supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)
    query_x, query_y = query_x.to(DEVICE), query_y.to(DEVICE)
    acc = trainer.evaluate(supp_x, supp_y, query_x, query_y, inner_steps=10)
    accs.append(acc)

avg = np.mean(accs)
print(f"5-way 1-shot: {avg:.3f} (target >0.95 for full run)")

# This is just a pipeline test - full run needs more tasks/tuning
print("✅ Pipeline test complete" if avg > 0.1 else "⚠️  Needs debugging")
"""
Split-MNIST Continual Learning with EqNGS + EWC (λ=100).
5 tasks, each with 2 classes (0/1, 2/3, 4/5, 6/7, 8/9).
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import Dict, List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.core.interfaces import NGSConfig
from ngs.modules.eqprop import create_eqngs
from ngs.optim.eqprop_wrapper import create_ewc_regularizer

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

# Split MNIST into 5 tasks (2 classes each)
task_classes = [
    [0, 1],  # Task 0
    [2, 3],  # Task 1
    [4, 5],  # Task 2
    [6, 7],  # Task 3
    [8, 9],  # Task 4
]

def get_task_loaders(dataset, classes, batch_size=128, train=True):
    """Get DataLoader for specific classes."""
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=train, num_workers=0, pin_memory=True)

# Create task loaders
train_loaders = [get_task_loaders(train_ds, c, train=True) for c in task_classes]
test_loaders = [get_task_loaders(test_ds, c, train=False) for c in task_classes]

# Base config
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
    d_out=10,  # Full 10 classes
    config=cfg,
    ep_beta=0.5,
    ep_settle_steps=10,
    ep_settle_lr=0.2,
    ep_momentum=0.9,
    spectral_gamma=0.95,
    spectral_mode='post_update',
).to(DEVICE)

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# EWC regularizer
ewc = create_ewc_regularizer(model, ewc_lambda=100.0)

# Training per task
EPOCHS_PER_TASK = 3
results = {'task_accuracies': [], 'forgetting': []}

for task_id, (train_loader, classes) in enumerate(zip(train_loaders, task_classes)):
    print(f"\n{'='*60}")
    print(f"TASK {task_id}: Classes {classes}")
    print(f"{'='*60}")
    
    # Train on current task
    for epoch in range(EPOCHS_PER_TASK):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_batches = 0
        
        for x, y in tqdm(train_loader, desc=f"Task {task_id} Epoch {epoch+1}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.size(0), -1)
            
            # EP step with EWC loss added to energy
            model.ngs.train()
            z = model.ngs.p_down(x)
            
            # Free phase
            router_free, params_free = model._settle_free_phase(z, model.ep_settle_steps)
            
            # Nudged phase
            router_nudged = model._settle_nudged_phase(z, y, model.ep_settle_steps)
            
            # Contrastive update
            with torch.no_grad():
                for p, p_free in zip(model.ep_params, params_free):
                    p_nudged = p.clone()
                    delta = p_nudged - p_free
                    p.add_(delta, alpha=model.ep_settle_lr)
            
            # Enforce spectral constraints
            model.enforce_spectral_constraints()
            
            # Compute EWC loss and add gradient
            ewc_loss = ewc.compute_ewc_loss()
            if ewc_loss > 0:
                ewc_grads = torch.autograd.grad(
                    ewc_loss, model.ep_params,
                    retain_graph=False, allow_unused=True
                )
                with torch.no_grad():
                    for p, g in zip(model.ep_params, ewc_grads):
                        if g is not None:
                            p.sub_(g, alpha=model.ep_settle_lr)
            
            # Metrics
            with torch.no_grad():
                out = model.ngs(x)
                logits = out.logits if hasattr(out, 'logits') else out
                pred = logits.argmax(1)
                acc = (pred == y).float().mean().item()
                loss = F.cross_entropy(logits, y).item()
            
            train_loss += loss
            train_acc += acc
            train_batches += 1
        
        train_loss /= train_batches
        train_acc /= train_batches
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
    
    # Consolidate task (compute Fisher)
    print(f"  Consolidating task {task_id} (computing Fisher)...")
    ewc.update_fisher(train_loader, task_id, device=DEVICE)
    
    # Evaluate on all tasks so far
    task_accs = []
    for eval_task_id, test_loader in enumerate(test_loaders[:task_id+1]):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = x.view(x.size(0), -1)
                
                out = model.ngs(x)
                logits = out.logits if hasattr(out, 'logits') else out
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = correct / total
        task_accs.append(acc)
        print(f"    Task {eval_task_id} (classes {task_classes[eval_task_id]}): {acc:.4f}")
    
    results['task_accuracies'].append(task_accs)

# Compute forgetting
print(f"\n{'='*60}")
print("FORGETTING ANALYSIS")
print(f"{'='*60}")

final_accs = results['task_accuracies'][-1]
for i, acc in enumerate(final_accs):
    # Peak accuracy for this task (when it was trained)
    peak_acc = max(results['task_accuracies'][j][i] for j in range(i, len(results['task_accuracies'])))
    forgetting = peak_acc - acc
    results['forgetting'].append({
        'task': i,
        'classes': task_classes[i],
        'peak_acc': peak_acc,
        'final_acc': acc,
        'forgetting': forgetting,
    })
    print(f"  Task {i} (classes {task_classes[i]}): peak={peak_acc:.4f}, final={acc:.4f}, forgetting={forgetting:.4f}")

avg_forgetting = np.mean([f['forgetting'] for f in results['forgetting']])
print(f"\nAverage Forgetting: {avg_forgetting:.4f}")

# Save results
import json
import os
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'tier1')
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, 'eqprop_continual_results.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")
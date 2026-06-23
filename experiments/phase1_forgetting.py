"""
Phase 1: Reconcile the forgetting claim.
Tests three conditions on domain-incremental learning (PermutedMNIST):
- A: No replay, no KD, fully trainable (control — fails badly)
- B: No replay, no KD, freeze adapters + decay input_proj LR on task boundary
- C: Replay + KD enabled (control — works)
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.modules.ngs_layer import build_stacked_ngs
from experiments.datasets import PermutedMNIST, ReplayBuffer
from experiments.metrics import evaluate_model_on_task, compute_metrics, print_results, CLMetrics

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
N_TASKS = 3
EPOCHS_PER_TASK = 10
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def build_model(input_dim: int, output_dim: int, k_init: int = 128) -> nn.Module:
    """Build the validated 2-layer NGSLayer config."""
    model = build_stacked_ngs(
        d_in=input_dim,
        d_out=output_dim,
        n_layers=2,
        d_latent=128,
        n_experts=256,
        n_heads=1,
        top_k=8,
        use_residual=True,
        use_norm=True,
    ).to(DEVICE)
    # Initialize with fewer active units
    for layer in model.layers:
        layer.router.active_mask[:] = False
        layer.router.active_mask[:k_init] = True
    return model


def train_task(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    old_model: Optional[nn.Module],
    epochs: int,
    lr: float,
    weight_decay: float,
    replay_buffer: Optional[ReplayBuffer],
    replay_ratio: float,
    kd_weight: float,
    kd_temperature: float,
    freeze_router_mu: bool = False,
    input_proj_lr_scale: float = 1.0,
    enable_topology: bool = False,
    old_active_masks: Optional[List[Tuple[int, torch.Tensor]]] = None,
) -> float:
    """Train model on a single task. Returns average loss."""
    
    # Build optimizer with parameter-specific learning rates
    param_groups = []
    for name, param in model.named_parameters():
        base_lr = lr
        
        if 'router.mu' in name and freeze_router_mu and old_active_masks is not None:
            # Freeze ONLY the OLD Gaussians (active before this task)
            layer_idx = int(name.split('.')[1])  # layers.0.router.mu -> layer_idx=0
            old_mask = None
            for idx, mask in old_active_masks:
                if idx == layer_idx:
                    old_mask = mask
                    break
            if old_mask is not None:
                # Create a mask for this parameter tensor
                # router.mu is [max_k, latent_dim], we need to freeze rows where old_mask is True
                # We'll handle this via a hook instead of param_groups
                param_groups.append({'params': param, 'lr': base_lr})
            else:
                param_groups.append({'params': param, 'lr': base_lr})
        elif 'input_proj' in name:
            param_groups.append({'params': param, 'lr': lr * input_proj_lr_scale})
        else:
            param_groups.append({'params': param, 'lr': base_lr})
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    
    # Enable topology adaptation if requested
    if enable_topology:
        for layer in model.layers:
            layer.enable_topology_adaptation(split_thresh=0.05, prune_thresh=0.01)
    
    # Register hook to freeze OLD router.mu gradients
    mu_hooks = []
    if freeze_router_mu and old_active_masks is not None:
        for layer_idx, layer in enumerate(model.layers):
            old_mask = None
            for idx, mask in old_active_masks:
                if idx == layer_idx:
                    old_mask = mask
                    break
            if old_mask is not None:
                def make_hook(mask):
                    def hook(grad):
                        # Zero out gradients for OLD Gaussians
                        grad = grad.clone()
                        grad[mask] = 0.0
                        return grad
                    return hook
                hook = make_hook(old_mask)
                mu_hooks.append(layer.router.mu.register_hook(hook))
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(DEVICE)
            y = y.to(DEVICE)
            
            # Replay
            if replay_buffer is not None and len(replay_buffer) > x.size(0):
                rx, ry = replay_buffer.sample(int(x.size(0) * replay_ratio))
                if rx is not None:
                    rx, ry = rx.to(DEVICE), ry.to(DEVICE)
                    x = torch.cat([x, rx], dim=0)
                    y = torch.cat([y, ry.argmax(dim=1)], dim=0)
            
            optimizer.zero_grad()
            logits = model(x)
            
            # CE loss
            ce_loss = F.cross_entropy(logits, y)
            
            # KD loss
            kd_loss = torch.tensor(0.0, device=DEVICE)
            if old_model is not None and kd_weight > 0:
                with torch.no_grad():
                    old_logits = old_model(x)
                n_new = x.size(0) // (1 + int(replay_ratio)) if replay_buffer else x.size(0)
                if n_new < x.size(0):
                    kd_loss = F.kl_div(
                        F.log_softmax(logits[n_new:] / kd_temperature, dim=-1),
                        F.softmax(old_logits[n_new:] / kd_temperature, dim=-1),
                        reduction='batchmean'
                    ) * (kd_temperature ** 2)
            
            total_loss = ce_loss + kd_weight * kd_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += ce_loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Topology adaptation: spawn new Gaussians for uncovered regions
        if enable_topology:
            # Collect latent samples per layer by running forward pass
            with torch.no_grad():
                for bx, _ in train_loader:
                    bx = bx.view(bx.size(0), -1).to(DEVICE)
                    x = bx
                    for layer_idx, layer in enumerate(model.layers):
                        if not hasattr(layer, '_z_samples'):
                            layer._z_samples = []
                        z = layer.input_proj(x)
                        layer._z_samples.append(z)
                        x = layer(x)  # Continue forward to next layer
                        if len(torch.cat(layer._z_samples)) >= 500:
                            break
            
            for layer in model.layers:
                if hasattr(layer, '_z_samples') and layer._z_samples:
                    z_samples = torch.cat(layer._z_samples)[:500]
                    layer.adapt_density(
                        z_samples=z_samples,
                        split_thresh=0.05,
                        prune_thresh=0.01,
                        spawn_thresh=-5.0,
                        max_spawn_per_call=8,
                    )
                    layer._z_samples = []
    
    # Remove hooks
    for h in mu_hooks:
        h.remove()
    
    return epoch_loss / max(n_batches, 1)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(DEVICE)
            y = y.to(DEVICE)
            
            # Replay
            if replay_buffer is not None and len(replay_buffer) > x.size(0):
                rx, ry = replay_buffer.sample(int(x.size(0) * replay_ratio))
                if rx is not None:
                    rx, ry = rx.to(DEVICE), ry.to(DEVICE)
                    x = torch.cat([x, rx], dim=0)
                    y = torch.cat([y, ry.argmax(dim=1)], dim=0)
            
            optimizer.zero_grad()
            logits = model(x)
            
            # CE loss
            ce_loss = F.cross_entropy(logits, y)
            
            # KD loss
            kd_loss = torch.tensor(0.0, device=DEVICE)
            if old_model is not None and kd_weight > 0:
                with torch.no_grad():
                    old_logits = old_model(x)
                n_new = x.size(0) // (1 + int(replay_ratio)) if replay_buffer else x.size(0)
                if n_new < x.size(0):
                    kd_loss = F.kl_div(
                        F.log_softmax(logits[n_new:] / kd_temperature, dim=-1),
                        F.softmax(old_logits[n_new:] / kd_temperature, dim=-1),
                        reduction='batchmean'
                    ) * (kd_temperature ** 2)
            
            total_loss = ce_loss + kd_weight * kd_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += ce_loss.item()
            n_batches += 1
        
        scheduler.step()
    
    return epoch_loss / max(n_batches, 1)


def evaluate_all_tasks(model: nn.Module, permuted: PermutedMNIST, n_tasks: int) -> np.ndarray:
    """Evaluate model on all tasks seen so far. Returns accuracy matrix row."""
    accs = np.zeros(n_tasks)
    for task_id in range(n_tasks):
        _, test_loader = permuted.get_task_data(task_id, BATCH_SIZE)
        acc = evaluate_model_on_task(model, test_loader, DEVICE)
        accs[task_id] = acc
    return accs


def run_condition(
    name: str,
    use_replay: bool,
    use_kd: bool,
    freeze_router_mu: bool = False,
    input_proj_lr_scale: float = 1.0,
    enable_topology: bool = False,
    k_init: int = 64,
) -> Tuple[np.ndarray, List[int]]:
    """Run a full continual learning experiment for one condition."""
    print(f"\n{'='*60}")
    print(f"Condition {name}: replay={use_replay}, kd={use_kd}, "
          f"freeze_router_mu={freeze_router_mu}, input_proj_lr_scale={input_proj_lr_scale}, "
          f"enable_topology={enable_topology}, k_init={k_init}")
    print(f"{'='*60}")
    
    set_seed(SEED)
    
    permuted = PermutedMNIST(n_tasks=N_TASKS, seed=SEED)
    input_dim, output_dim = 784, 10
    
    model = build_model(input_dim, output_dim, k_init=k_init)
    replay_buffer = ReplayBuffer(max_size=5000, seed=SEED) if use_replay else None
    
    # accuracy_matrix[i, j] = accuracy on task i after learning task j
    accuracy_matrix = np.full((N_TASKS, N_TASKS), np.nan)
    active_units_per_task = []
    old_model = None
    
    # Track which Gaussians were "old" (active before current task)
    old_active_masks = []  # list of (layer_idx, active_mask_before_task) tuples
    
    for task_id in range(N_TASKS):
        print(f"\n--- Task {task_id + 1}/{N_TASKS} ---")
        train_loader, _ = permuted.get_task_data(task_id, BATCH_SIZE)
        
        # Determine settings for this task
        should_freeze_old_mu = freeze_router_mu and task_id > 0
        proj_scale = input_proj_lr_scale if task_id > 0 else 1.0
        should_enable_topology = enable_topology and task_id > 0
        
        # Record old active masks BEFORE training this task (for freezing)
        if should_freeze_old_mu:
            old_active_masks = []
            for layer_idx, layer in enumerate(model.layers):
                old_active_masks.append((layer_idx, layer.router.active_mask.clone()))
        
        train_task(
            model, train_loader, old_model,
            epochs=EPOCHS_PER_TASK,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            replay_buffer=replay_buffer,
            replay_ratio=1.0,
            kd_weight=10.0 if use_kd else 0.0,
            kd_temperature=2.0,
            freeze_router_mu=should_freeze_old_mu,
            input_proj_lr_scale=proj_scale,
            enable_topology=should_enable_topology,
            old_active_masks=old_active_masks if should_freeze_old_mu else None,
        )
        
        # Evaluate on all seen tasks
        for eval_task in range(task_id + 1):
            _, test_loader = permuted.get_task_data(eval_task, BATCH_SIZE)
            acc = evaluate_model_on_task(model, test_loader, DEVICE)
            accuracy_matrix[eval_task, task_id] = acc
        
        active_units_per_task.append(model.K)
        print(f"  Accuracies: {accuracy_matrix[:task_id+1, task_id]}")
        print(f"  Active units (K): {model.K}")
        
        # Add to replay buffer
        if replay_buffer is not None:
            for x, y in train_loader:
                replay_buffer.add(x, F.one_hot(y, output_dim).float())
        
        # Save model for KD
        if use_kd:
            old_model = deepcopy(model).eval()
            for p in old_model.parameters():
                p.requires_grad = False
    
    return accuracy_matrix, active_units_per_task


def main():
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Tasks: {N_TASKS}, Epochs/task: {EPOCHS_PER_TASK}")
    print(f"Architecture: 2-layer NGSLayer (d_latent=128, n_experts=256, top_k=8, residual+norm)")
    
    # Condition A: No replay, no KD, fully trainable (all 512 from start)
    acc_A, K_A = run_condition('A', use_replay=False, use_kd=False, k_init=512)
    
    # Condition B: No replay, no KD, SPLATTING: start with 64, spawn new, freeze old mu + decay input_proj LR
    acc_B, K_B = run_condition('B', use_replay=False, use_kd=False, 
                                freeze_router_mu=True, input_proj_lr_scale=0.01,
                                enable_topology=True, k_init=64)
    
    # Condition C: Replay + KD (all 512 from start)
    acc_C, K_C = run_condition('C', use_replay=True, use_kd=True, k_init=512)
    
    # Compute metrics
    random_baseline = 1.0 / 10.0  # 10 classes
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    for name, acc_matrix, K_list in [('A', acc_A, K_A), ('B', acc_B, K_B), ('C', acc_C, K_C)]:
        metrics = compute_metrics(acc_matrix, random_baseline)
        metrics.active_units = K_list[-1] if K_list else 0
        metrics.max_units = max(K_list) if K_list else 0
        
        print(f"\n--- Condition {name} ---")
        print_results(metrics, f"Condition {name}")
        print(f"  K trajectory: {K_list}")
        print(f"  Final K: {metrics.active_units}, Max K: {metrics.max_units}")
    
    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Condition':<10} {'Avg Final Acc':>14} {'Avg Forgetting':>15} {'BWT':>10} {'Final K':>8}")
    print("-" * 60)
    
    for name, acc_matrix, K_list in [('A', acc_A, K_A), ('B', acc_B, K_B), ('C', acc_C, K_C)]:
        metrics = compute_metrics(acc_matrix, random_baseline)
        print(f"{name:<10} {metrics.avg_final_accuracy:>14.4f} {metrics.avg_forgetting:>15.4f} {metrics.bwt:>10.4f} {K_list[-1]:>8}")
    
    # Save results
    results = {
        'condition_A': {
            'accuracy_matrix': acc_A.tolist(),
            'active_units': K_A,
            'metrics': compute_metrics(acc_A, random_baseline).to_dict(),
        },
        'condition_B': {
            'accuracy_matrix': acc_B.tolist(),
            'active_units': K_B,
            'metrics': compute_metrics(acc_B, random_baseline).to_dict(),
        },
        'condition_C': {
            'accuracy_matrix': acc_C.tolist(),
            'active_units': K_C,
            'metrics': compute_metrics(acc_C, random_baseline).to_dict(),
        },
        'config': {
            'seed': SEED,
            'n_tasks': N_TASKS,
            'epochs_per_task': EPOCHS_PER_TASK,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'device': DEVICE,
        }
    }
    
    os.makedirs('results/phase1', exist_ok=True)
    with open('results/phase1/forgetting_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to results/phase1/forgetting_results.json")
    
    # What this does NOT prove
    print(f"\n{'='*60}")
    print("WHAT THIS DOES NOT PROVE")
    print(f"{'='*60}")
    print("• Only 3 tasks (PermutedMNIST); may not generalize to more tasks or other domain shifts")
    print("• Single seed; variance across seeds unknown")
    print("• Freeze mechanism is heuristic (LR decay factor 0.01 chosen arbitrarily)")
    print("• Does not test class-incremental or task-incremental scenarios")
    print("• Short training (10 epochs/task); longer training may change dynamics")


if __name__ == '__main__':
    main()
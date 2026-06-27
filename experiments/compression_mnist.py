#!/usr/bin/env python
"""
Gaussian Pruning / Compression Experiments for NGS on MNIST.

1. Trains an NGS model on MNIST for 5 epochs
2. Prunes Gaussians by importance (bottom 10%, 25%, 50%, 75%)
3. Measures accuracy before and after pruning
4. Tests if fine-tuning recovers accuracy
5. Detects "Gaussian lottery ticket" (50% prune with <1pp drop)

Saves results to results/compression.json
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
import json
import os
from pathlib import Path
from copy import deepcopy

from ngs.models.ngs import build_ngs, NGSModel
from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_mnist_loaders(batch_size: int = 256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            pred = model(x).logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    losses = []
    for x, y in train_loader:
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out.logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def compute_unit_importance(model: NGSModel, train_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Compute per-unit importance as sum of routing weights * frequency by which each unit is selected.
    Returns a tensor of shape [max_k] with importance scores for all units.
    """
    router = model.router
    importance = torch.zeros(router.max_k, device=device)
    counts = torch.zeros(router.max_k, device=device)

    model.eval()
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.view(x.size(0), -1).to(device)
            out = model(x)
            # active_indices: [B, top_k], weights: [B, top_k]
            active_indices = model._last_active_indices  # [B, K]
            weights = model._last_routing_weights        # [B, K]
            if active_indices is None or weights is None:
                continue
            # Accumulate importance
            B, K = active_indices.shape
            for b in range(B):
                for k in range(K):
                    idx = active_indices[b, k].item()
                    w = weights[b, k].item()
                    importance[idx] += w
                    counts[idx] += 1

    # Normalize by count to avoid bias toward frequently selected units
    importance = importance / (counts.clamp(min=1))
    return importance


def prune_bottom_k(model: NGSModel, importance: torch.Tensor, prune_frac: float):
    """
    Mask out the bottom `prune_frac` of units by importance.
    """
    router = model.router
    active_mask = router.active_mask.clone()
    active_idx = active_mask.nonzero(as_tuple=True)[0]
    if len(active_idx) == 0:
        return

    # Get importance of active units
    active_importance = importance[active_idx].clone()
    k_prune = int(len(active_idx) * prune_frac)
    if k_prune == 0:
        return

    # Find the k_prune units with lowest importance
    sorted_idx = torch.argsort(active_importance)
    prune_indices = active_idx[ sorted_idx[:k_prune] ]

    # Set their active_mask to False
    router.active_mask[prune_indices] = False


def get_active_count(model: NGSModel):
    return model.router.active_mask.sum().item()


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Configuration (lean, fast configuration)
    cfg = NGSConfig(
        latent_dim=32,
        max_k=512,
        k_init=64,
        top_k=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.PRE_ALLOCATED,
        tau=1.0,
        gamma_residual=0.1,
    )

    d_in, d_out = 28 * 28, 10
    model = build_ngs(d_in, d_out, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_loader, test_loader = get_mnist_loaders(batch_size=256)

    # 1. Train baseline for 5 epochs
    print("\n=== Training Baseline for 5 epochs ===")
    for epoch in range(5):
        loss = train_epoch(model, train_loader, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"  Epoch {epoch + 1}: loss={loss:.4f}, acc={acc:.4f}, active_units={get_active_count(model)}")

    baseline_acc = evaluate(model, test_loader, device)
    baseline_active = get_active_count(model)
    print(f"\nBaseline accuracy: {baseline_acc:.4f} with {baseline_active} active units")

    # 2. Compute per-unit importance on a subset of training data
    print("\n=== Computing per-unit importance ===")
    importance = compute_unit_importance(model, train_loader, device)
    # Normalize importance
    importance = importance / (importance.max() + 1e-8)

    results = {
        "baseline_accuracy": baseline_acc,
        "baseline_active_units": baseline_active,
        "pruning_levels": {}
    }

    # Save baseline model state
    baseline_state = deepcopy(model.state_dict())

    prune_levels = [0.10, 0.25, 0.50, 0.75]
    for prune_frac in prune_levels:
        print(f"\n=== Pruning {int(prune_frac * 100)}% ===")
        # Restore baseline
        model.load_state_dict(baseline_state)
        model.to(device)

        # Prune
        prune_bottom_k(model, importance, prune_frac)
        remaining = get_active_count(model)
        pruned_acc = evaluate(model, test_loader, device)
        print(f"  After pruning: {remaining} units, accuracy={pruned_acc:.4f}")

        # Fine-tune for 2 epochs to see recovery
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        for ft_epoch in range(2):
            loss = train_epoch(model, train_loader, optimizer, device)
            ft_acc = evaluate(model, test_loader, device)
            print(f"    Fine-tune epoch {ft_epoch + 1}: loss={loss:.4f}, acc={ft_acc:.4f}")

        ft_acc = evaluate(model, test_loader, device)
        print(f"  After fine-tuning: accuracy={ft_acc:.4f}")

        results["pruning_levels"][f"{int(prune_frac * 100)}%"] = {
            "remaining_units": remaining,
            "accuracy_after_pruning": pruned_acc,
            "accuracy_after_finetuning": ft_acc,
            "drop_pp": (baseline_acc - pruned_acc) * 100,
            "recovered_pp": (ft_acc - pruned_acc) * 100,
        }

    # Detect Gaussian lottery ticket: 50% prune with <1pp drop
    level_50 = results["pruning_levels"].get("50%", {})
    if level_50:
        drop = level_50.get("drop_pp", 999)
        results["gaussian_lottery_ticket"] = bool(drop < 1.0)
        results["gaussian_lottery_ticket_drop_pp"] = drop
    else:
        results["gaussian_lottery_ticket"] = False
        results["gaussian_lottery_ticket_drop_pp"] = None

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/compression.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== RESULTS ===")
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    for level, data in results["pruning_levels"].items():
        print(f"  Prune {level}: {data['accuracy_after_pruning']:.4f} (drop {data['drop_pp']:.2f}pp), "
              f"fine-tuned {data['accuracy_after_finetuning']:.4f}")
    print(f"Gaussian Lottery Ticket (50% prune, <1pp drop): {results['gaussian_lottery_ticket']}")
    print("Saved to results/compression.json")


if __name__ == '__main__':
    main()

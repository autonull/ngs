#!/usr/bin/env python
"""
Gaussian Specialization Analysis — TODO11 Phase C5

Per-Gaussian activation frequency, mutual information with classes,
and lottery ticket pruning analysis.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, '/home/me/ngs')
sys.path.insert(0, '/home/me/ngs/bioplausible/mep')

from ngs.models.ngs import NGSModel
from ngs.core.interfaces import NGSConfig, RoutingStrategy


def main():
    parser = argparse.ArgumentParser(description="Gaussian specialization analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="results/diagnostics/gaussian_specialization.json")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Device: {device}")
    
    # Load or train model
    config = NGSConfig(
        latent_dim=64, max_k=32, top_k=8, k_init=16,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    )
    model = NGSModel(784, 10, config).to(device)
    
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Training model from scratch...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        train_loader = DataLoader(
            datasets.MNIST("/tmp/mnist", train=True, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True,
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(args.epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                logits = out.logits if hasattr(out, 'logits') else out
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}")
    
    # Prepare test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    test_dataset = datasets.MNIST("/tmp/mnist", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Collect routing statistics
    K = model.router.K
    max_k = config.max_k
    
    results = {
        "config": {
            "latent_dim": config.latent_dim,
            "max_k": config.max_k,
            "top_k": config.top_k,
            "K_active": K,
        },
        "activation_frequency": {},
        "mutual_information": {},
        "lottery_ticket": {},
    }
    
    # --- ACTIVATION FREQUENCY ---
    print("\n--- Activation Frequency ---")
    activation_counts = torch.zeros(max_k, device=device)
    total_batches = 0
    class_given_active = torch.zeros(max_k, 10, device=device)
    active_given_class = torch.zeros(max_k, 10, device=device)
    class_counts = torch.zeros(10, device=device)
    
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            router_out = out.routing_output
            
            if router_out is None:
                continue
            
            indices = router_out.indices  # [B, K]
            for b in range(indices.size(0)):
                for k in range(indices.size(1)):
                    idx = indices[b, k].item()
                    activation_counts[idx] += 1
                    class_given_active[idx, y[b]] += 1
                active_given_class[:, y[b]] += 1
            class_counts += y.bincount(minlength=10)[:10]
            total_batches += 1
    
    # Normalize
    active_probs = activation_counts / (total_batches * args.batch_size)
    
    # Also compute per-Gaussian: P(active | class)
    P_active_given_class = torch.zeros(max_k, 10, device=device)
    P_class_given_active = torch.zeros(max_k, 10, device=device)
    
    for c in range(10):
        if class_counts[c] > 0:
            # P(active | class=c) = count(active AND class=c) / count(class=c)
            # For each Gaussian, how often is it active when class=c?
            if class_counts[c] > 0:
                P_active_given_class[:, c] = class_given_active[:, c] / class_counts[c]
        
        # P(class=c | active)
        active_total = activation_counts.sum()
        if active_total > 0:
            P_class_given_active[:, c] = class_given_active[:, c] / (activation_counts + 1e-8)
    
    # Compute mutual information I(Gaussian_active; class)
    P_active = active_probs  # [max_k]
    P_active = P_active / (P_active.sum() + 1e-8)
    
    mutual_info = torch.zeros(max_k, device=device)
    for g in range(max_k):
        for c in range(10):
            p_gc = class_given_active[g, c] / (total_batches * args.batch_size + 1e-8)
            p_g = P_active[g]
            p_c = class_counts[c] / (class_counts.sum() + 1e-8)
            if p_gc > 0 and p_g > 0 and p_c > 0:
                mutual_info[g] += p_gc * torch.log(p_gc / (p_g * p_c + 1e-8) + 1e-8)
    
    for g in range(max_k):
        results["activation_frequency"][f"gaussian_{g}"] = {
            "active_prob": active_probs[g].item(),
            "activation_count": activation_counts[g].item(),
        }
        mi = mutual_info[g].item()
        results["mutual_information"][f"gaussian_{g}"] = {
            "mutual_information": mi,
            "preferred_class": torch.argmax(P_class_given_active[g]).item(),
            "p_class_given_active": P_class_given_active[g].cpu().tolist(),
        }
    
    print(f"  Total activations: {activation_counts.sum().item()}")
    print(f"  Active prob range: [{active_probs.min().item():.6f}, {active_probs.max().item():.6f}]")
    print(f"  MI range: [{mutual_info.min().item():.6f}, {mutual_info.max().item():.6f}]")
    print(f"  Mean MI: {mutual_info.mean().item():.6f}")
    
    # Concentration: what fraction of activations go to top Gaussians?
    sorted_probs = active_probs.sort(descending=True).values
    cumulative = torch.cumsum(sorted_probs, dim=0)
    for pct in [0.1, 0.25, 0.5]:
        n_for_pct = (cumulative < pct).sum().item()
        print(f"  Top {pct*100:.0f}% activations from {n_for_pct}/{max_k} Gaussians")
        results["activation_frequency"][f"top_{pct}_cumulative"] = n_for_pct
    
    # --- LOTTERY TICKET PRUNING ---
    print("\n--- Lottery Ticket Pruning ---")
    
    # Rank Gaussians by importance (mutual information)
    importance_order = torch.argsort(mutual_info, descending=True)
    
    pruning_levels = [10, 25, 50, 75]
    lottery_results = {}
    
    baseline_correct = 0
    baseline_total = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            baseline_correct += (logits.argmax(1) == y).sum().item()
            baseline_total += x.size(0)
    baseline_acc = baseline_correct / baseline_total
    print(f"  Baseline accuracy: {baseline_acc:.4f}")
    
    for pct in pruning_levels:
        n_keep = max(1, max_k - int(max_k * pct / 100))
        keep_indices = importance_order[:n_keep]
        
        # Create a mask that only allows keeping these Gaussians
        keep_mask = torch.zeros(max_k, dtype=torch.bool, device=device)
        keep_mask[keep_indices] = True
        
        # Temporarily modify active_mask and evaluate
        original_active = model.router.active_mask.clone()
        
        # Prune: only the kept indices that are also active
        model.router.active_mask = model.router.active_mask & keep_mask
        
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                logits = out.logits if hasattr(out, 'logits') else out
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)
        pruned_acc = correct / total
        
        print(f"  Prune {pct}% (keep {n_keep}/{max_k}): acc={pruned_acc:.4f}")
        
        lottery_results[f"prune_{pct}pct"] = {
            "prune_pct": pct,
            "n_keep": n_keep,
            "accuracy": pruned_acc,
            "drop_from_baseline": baseline_acc - pruned_acc,
        }
        
        model.router.active_mask = original_active
    
    results["lottery_ticket"]["baseline_accuracy"] = baseline_acc
    results["lottery_ticket"]["pruning_results"] = lottery_results
    
    results["summary"] = {
        "activation_concentration": {
            "gini_coefficient": _compute_gini(active_probs.cpu()),
            "top_1_active_prob": sorted_probs[0].item(),
            "top_5_active_prob": sorted_probs[:5].sum().item(),
            "mean_mutual_information": mutual_info.mean().item(),
        },
        "lottery_ticket": {
            "prune_10_drop": lottery_results.get("prune_10pct", {}).get("drop_from_baseline", 0),
            "prune_50_drop": lottery_results.get("prune_50pct", {}).get("drop_from_baseline", 0),
        },
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def _compute_gini(tensor):
    """Compute Gini coefficient (0=perfect equality, 1=max inequality)."""
    sorted_t = torch.sort(tensor)[0]
    n = len(sorted_t)
    cumsum = torch.cumsum(sorted_t, dim=0)
    return float((2 * cumsum.sum() - (n + 1) * sorted_t.sum()) / (n * sorted_t.sum() + 1e-8))


if __name__ == "__main__":
    main()

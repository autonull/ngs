#!/usr/bin/env python
"""
Entropy Distribution Diagnostic — TODO11 Phase A4.1

Records routing entropy distribution during Autopoietic training.
Checks if tau_merge/tau_split thresholds are in the right range.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/me/ngs')
sys.path.insert(0, '/home/me/ngs/bioplausible/mep')

from ngs.models.ngs import NGSModel
from ngs.core.interfaces import NGSConfig, RoutingStrategy
from ngs.modules.topology_managers import AutopoieticManager


def compute_routing_entropy(router_out) -> torch.Tensor:
    """Compute routing entropy from router output."""
    weights = router_out.weights  # [B, K]
    # Add small epsilon for numerical stability
    weights = weights + 1e-8
    # Renormalize
    weights = weights / weights.sum(dim=-1, keepdim=True)
    entropy = -(weights * weights.log()).sum(dim=-1)  # [B]
    return entropy


def main():
    parser = argparse.ArgumentParser(description="Entropy distribution diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tau-split", type=float, default=2.0)
    parser.add_argument("--tau-merge", type=float, default=1.5)
    parser.add_argument("--output", default="results/diagnostics/entropy_distribution.json")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Batches: {args.num_batches}")
    print(f"Batch size: {args.batch_size}")
    print(f"tau_split: {args.tau_split}, tau_merge: {args.tau_merge}")
    
    # Create model with Autopoietic topology manager
    config = NGSConfig(
        latent_dim=64,
        max_k=32,
        top_k=8,
        k_init=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    )
    model = NGSModel(784, 10, config).to(device)
    
    # Replace topology manager with AutopoieticManager
    config.k_init = 8
    config.extra = {
        'entropy_split_threshold': args.tau_split,
        'overlap_merge_threshold': args.tau_merge,
    }
    autopoietic = AutopoieticManager(config=config)
    model.topology_manager = autopoietic
    
    results = {
        "config": {
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "tau_split": args.tau_split,
            "tau_merge": args.tau_merge,
            "latent_dim": config.latent_dim,
            "max_k": config.max_k,
            "top_k": config.top_k,
        },
        "entropy_stats": [],
        "k_evolution": [],
        "threshold_analysis": {},
    }
    
    all_entropies = []
    
    print("\n--- Running training to collect entropy distribution ---")
    model.train()
    
    for batch_idx in range(args.num_batches):
        x = torch.randn(args.batch_size, 784, device=device)
        target = torch.randint(0, 10, (args.batch_size,), device=device)
        
        # Forward pass
        out = model(x)
        router_out = out.routing_output
        
        # Compute entropy
        entropy = compute_routing_entropy(router_out)  # [B]
        batch_entropy = entropy.mean().item()
        all_entropies.extend(entropy.cpu().tolist())
        
        # Record K
        current_k = model.router.K
        
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}: K={current_k}, entropy={batch_entropy:.4f}")
        
        # Autopoietic step
        # We need to compute loss for the manager
        logits = out.logits if hasattr(out, 'logits') else out
        loss = F.cross_entropy(logits, target)
        
        # Call autopoietic step
        try:
            z = model.p_down(x)
            merged, split, spawned = autopoietic.step(model, z)
        except Exception as e:
            print(f"  Autopoietic step error: {e}")
            merged, split, spawned = 0, 0, 0
        
        results["entropy_stats"].append({
            "batch": batch_idx,
            "mean_entropy": batch_entropy,
            "min_entropy": entropy.min().item(),
            "max_entropy": entropy.max().item(),
            "std_entropy": entropy.std().item(),
            "K": current_k,
            "merged": merged,
            "split": split,
            "spawned": spawned,
        })
        
        results["k_evolution"].append(current_k)
        
        # Simple backward for training
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.sub_(p.grad * 0.01)
                    p.grad.zero_()
    
    # Analyze entropy distribution
    all_entropies_tensor = torch.tensor(all_entropies)
    
    results["threshold_analysis"] = {
        "mean_entropy": all_entropies_tensor.mean().item(),
        "std_entropy": all_entropies_tensor.std().item(),
        "min_entropy": all_entropies_tensor.min().item(),
        "max_entropy": all_entropies_tensor.max().item(),
        "median_entropy": all_entropies_tensor.median().item(),
        "q25_entropy": all_entropies_tensor.quantile(0.25).item(),
        "q75_entropy": all_entropies_tensor.quantile(0.75).item(),
        "tau_split": args.tau_split,
        "tau_merge": args.tau_merge,
        "pct_above_tau_split": (all_entropies_tensor > args.tau_split).float().mean().item(),
        "pct_below_tau_merge": (all_entropies_tensor < args.tau_merge).float().mean().item(),
        "pct_between": ((all_entropies_tensor >= args.tau_merge) & 
                        (all_entropies_tensor <= args.tau_split)).float().mean().item(),
    }
    
    print(f"\nENTROPY DISTRIBUTION:")
    print(f"  Mean: {results['threshold_analysis']['mean_entropy']:.4f}")
    print(f"  Std:  {results['threshold_analysis']['std_entropy']:.4f}")
    print(f"  Min:  {results['threshold_analysis']['min_entropy']:.4f}")
    print(f"  Max:  {results['threshold_analysis']['max_entropy']:.4f}")
    print(f"  Median: {results['threshold_analysis']['median_entropy']:.4f}")
    print(f"  Q25-Q75: {results['threshold_analysis']['q25_entropy']:.4f} - {results['threshold_analysis']['q75_entropy']:.4f}")
    print(f"\nTHRESHOLD ANALYSIS:")
    print(f"  tau_split={args.tau_split}: {results['threshold_analysis']['pct_above_tau_split']*100:.1f}% above")
    print(f"  tau_merge={args.tau_merge}: {results['threshold_analysis']['pct_below_tau_merge']*100:.1f}% below")
    print(f"  Between thresholds: {results['threshold_analysis']['pct_between']*100:.1f}%")
    
    if results['threshold_analysis']['pct_above_tau_split'] < 0.01:
        print(f"\n⚠ WARNING: Almost no samples above tau_split ({args.tau_split})")
        print(f"   Split trigger will NEVER fire!")
        results["threshold_analysis"]["split_never_fires"] = True
    else:
        results["threshold_analysis"]["split_never_fires"] = False
    
    if results['threshold_analysis']['pct_below_tau_merge'] < 0.01:
        print(f"\n⚠ WARNING: Almost no samples below tau_merge ({args.tau_merge})")
        print(f"   Merge trigger will NEVER fire!")
        results["threshold_analysis"]["merge_never_fires"] = True
    else:
        results["threshold_analysis"]["merge_never_fires"] = False
    
    # K evolution
    results["k_evolution_stats"] = {
        "initial_k": results["k_evolution"][0] if results["k_evolution"] else 0,
        "final_k": results["k_evolution"][-1] if results["k_evolution"] else 0,
        "max_k": max(results["k_evolution"]) if results["k_evolution"] else 0,
        "min_k": min(results["k_evolution"]) if results["k_evolution"] else 0,
        "mean_k": sum(results["k_evolution"]) / len(results["k_evolution"]) if results["k_evolution"] else 0,
    }
    
    print(f"\nK EVOLUTION:")
    print(f"  Initial: {results['k_evolution_stats']['initial_k']}")
    print(f"  Final:   {results['k_evolution_stats']['final_k']}")
    print(f"  Range:   {results['k_evolution_stats']['min_k']} - {results['k_evolution_stats']['max_k']}")
    print(f"  Mean:    {results['k_evolution_stats']['mean_k']:.1f}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
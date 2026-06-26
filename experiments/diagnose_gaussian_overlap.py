#!/usr/bin/env python
"""
Gaussian Overlap Diagnostic — TODO11 Phase A4.2

Computes and visualizes the overlap matrix between active Gaussians.
Checks for redundancy and measures overlap distribution.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, '/home/me/ngs')
sys.path.insert(0, '/home/me/ngs/bioplausible/mep')

from ngs.models.ngs import NGSModel
from ngs.core.interfaces import NGSConfig, RoutingStrategy


def compute_overlap_matrix(mu: torch.Tensor, log_s: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Gaussian overlap.
    
    For diagonal Gaussians, overlap ~ exp(-0.5 * Mahalanobis distance)
    """
    K, d = mu.shape
    
    # Compute Mahalanobis distance between all pairs
    # mu: [K, d], log_s: [K, d]
    # Pairwise diff: [K, K, d]
    diff = mu.unsqueeze(1) - mu.unsqueeze(0)  # [K, K, d]
    
    # Average sigma^2 for pairwise
    sigma_sq_i = torch.exp(2 * log_s)  # [K, d]
    sigma_sq_j = torch.exp(2 * log_s)  # [K, d]
    sigma_sq_avg = (sigma_sq_i.unsqueeze(1) + sigma_sq_j.unsqueeze(0)) / 2  # [K, K, d]
    
    # Mahalanobis squared
    mahalanobis_sq = (diff ** 2 / (sigma_sq_avg + 1e-8)).sum(dim=-1)  # [K, K]
    
    # Overlap = exp(-0.5 * mahalanobis_sq)
    overlap = torch.exp(-0.5 * mahalanobis_sq)
    
    return overlap


def main():
    parser = argparse.ArgumentParser(description="Gaussian overlap diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--output", default="results/diagnostics/gaussian_overlap.json")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    
    # Create model
    config = NGSConfig(
        latent_dim=64,
        max_k=32,
        top_k=8,
        k_init=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    )
    model = NGSModel(784, 10, config).to(device)
    
    results = {
        "config": {
            "latent_dim": config.latent_dim,
            "max_k": config.max_k,
            "top_k": config.top_k,
            "num_samples": args.num_samples,
        },
        "static_analysis": {},
        "dynamic_analysis": {},
    }
    
    # --- STATIC ANALYSIS: Current Gaussian parameters ---
    router = model.router
    
    if hasattr(router, 'mu') and router.mu.dim() == 2:
        active_mask = router.active_mask
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        mu = router.mu[active_indices]
        log_s = router.log_s[active_indices]
        log_alpha = router.log_alpha[active_indices]
        
        K_active = len(active_indices)
        print(f"Active Gaussians: {K_active}")
        
        # Compute overlap matrix
        overlap = compute_overlap_matrix(mu, log_s)  # [K, K]
        
        # Diagonal is 1.0 (self-overlap)
        # Off-diagonal: pairwise overlap
        off_diag_mask = ~torch.eye(K_active, dtype=torch.bool, device=device)
        off_diag_overlaps = overlap[off_diag_mask]
        
        results["static_analysis"] = {
            "K_active": K_active,
            "mean_overlap": off_diag_overlaps.mean().item(),
            "std_overlap": off_diag_overlaps.std().item(),
            "min_overlap": off_diag_overlaps.min().item(),
            "max_overlap": off_diag_overlaps.max().item(),
            "median_overlap": off_diag_overlaps.median().item(),
            "pct_high_overlap_05": (off_diag_overlaps > 0.5).float().mean().item(),
            "pct_high_overlap_01": (off_diag_overlaps > 0.1).float().mean().item(),
            "pct_high_overlap_001": (off_diag_overlaps > 0.01).float().mean().item(),
            "mean_mu_norm": mu.norm(dim=-1).mean().item(),
            "mean_log_s": log_s.mean().item(),
            "mean_log_alpha": log_alpha.mean().item(),
        }
        
        print(f"\nSTATIC OVERLAP ANALYSIS:")
        print(f"  Mean overlap: {results['static_analysis']['mean_overlap']:.6f}")
        print(f"  Max overlap:  {results['static_analysis']['max_overlap']:.6f}")
        print(f"  >0.5: {results['static_analysis']['pct_high_overlap_05']*100:.1f}%")
        print(f"  >0.1: {results['static_analysis']['pct_high_overlap_01']*100:.1f}%")
        print(f"  >0.01: {results['static_analysis']['pct_high_overlap_001']*100:.1f}%")
        
        # Full overlap matrix for saving (only if small)
        if K_active <= 32:
            results["static_analysis"]["overlap_matrix"] = overlap.cpu().tolist()
    
    # --- DYNAMIC ANALYSIS: Overlap during routing ---
    print(f"\n--- Dynamic overlap analysis ({args.num_samples} samples) ---")
    
    all_entropies = []
    all_max_weights = []
    all_min_distances = []
    
    model.eval()
    with torch.no_grad():
        for i in range(args.num_samples // 32 + 1):
            batch_size = min(32, args.num_samples - i * 32)
            if batch_size <= 0:
                break
            
            x = torch.randn(batch_size, 784, device=device)
            z = model.p_down(x)
            router_out = model.router(z)
            
            weights = router_out.weights  # [B, K]
            
            # Entropy
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1)
            all_entropies.extend(entropy.cpu().tolist())
            
            # Max weight (confidence)
            max_weight = weights.max(dim=-1).values
            all_max_weights.extend(max_weight.cpu().tolist())
            
            # Min Mahalanobis distance to active Gaussians
            if hasattr(router, 'mu') and router.mu.dim() == 2:
                active_indices = router.active_mask.nonzero(as_tuple=True)[0]
                mu = router.mu[active_indices]
                log_s = router.log_s[active_indices]
                
                diff = z.unsqueeze(1) - mu.unsqueeze(0)
                sigma_sq = torch.exp(2 * log_s) + 1e-5
                mahalanobis_sq = (diff ** 2 / sigma_sq).sum(dim=-1)  # [B, K]
                
                min_dist = mahalanobis_sq.min(dim=-1).values.sqrt()
                all_min_distances.extend(min_dist.cpu().tolist())
    
    all_entropies = torch.tensor(all_entropies[:args.num_samples])
    all_max_weights = torch.tensor(all_max_weights[:args.num_samples])
    all_min_distances = torch.tensor(all_min_distances[:args.num_samples])
    
    results["dynamic_analysis"] = {
        "num_samples": len(all_entropies),
        "entropy": {
            "mean": all_entropies.mean().item(),
            "std": all_entropies.std().item(),
            "min": all_entropies.min().item(),
            "max": all_entropies.max().item(),
            "median": all_entropies.median().item(),
        },
        "max_weight": {
            "mean": all_max_weights.mean().item(),
            "std": all_max_weights.std().item(),
            "min": all_max_weights.min().item(),
            "max": all_max_weights.max().item(),
            "median": all_max_weights.median().item(),
        },
        "min_mahalanobis_distance": {
            "mean": all_min_distances.mean().item() if len(all_min_distances) > 0 else 0,
            "std": all_min_distances.std().item() if len(all_min_distances) > 0 else 0,
            "min": all_min_distances.min().item() if len(all_min_distances) > 0 else 0,
            "max": all_min_distances.max().item() if len(all_min_distances) > 0 else 0,
            "median": all_min_distances.median().item() if len(all_min_distances) > 0 else 0,
        },
    }
    
    print(f"\nDYNAMIC ROUTING ANALYSIS:")
    print(f"  Entropy: mean={results['dynamic_analysis']['entropy']['mean']:.4f}, "
          f"std={results['dynamic_analysis']['entropy']['std']:.4f}")
    print(f"  Max weight: mean={results['dynamic_analysis']['max_weight']['mean']:.4f}")
    print(f"  Min Mahalanobis: mean={results['dynamic_analysis']['min_mahalanobis_distance']['mean']:.4f}")
    
    # Redundancy check: if overlap is very high, Gaussians are redundant
    if results["static_analysis"].get("mean_overlap", 0) > 0.1:
        results["static_analysis"]["redundancy_warning"] = True
        print(f"\n⚠ WARNING: High mean overlap ({results['static_analysis']['mean_overlap']:.4f})")
        print(f"   Gaussians may be redundant!")
    else:
        results["static_analysis"]["redundancy_warning"] = False
    
    # Coverage check: if min Mahalanobis distance is large, coverage is sparse
    if results["dynamic_analysis"]["min_mahalanobis_distance"]["mean"] > 3.0:
        results["dynamic_analysis"]["coverage_warning"] = True
        print(f"\n⚠ WARNING: Sparse coverage (mean min distance = {results['dynamic_analysis']['min_mahalanobis_distance']['mean']:.2f})")
        print(f"   OOD detection via distance may not work")
    else:
        results["dynamic_analysis"]["coverage_warning"] = False
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
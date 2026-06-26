#!/usr/bin/env python
"""
Spectral Norm Diagnostic — TODO11 Phase A1.1

Measures singular values of router.mu projections before/after SpectralConstraint application.
Answers: is SN actually constraining anything?
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
from ngs.optim.eqprop_wrapper import SpectralConstraint, add_spectral_constraint


def compute_singular_values(W: torch.Tensor, niter: int = 10) -> torch.Tensor:
    """Compute singular values via power iteration (more iterations for accuracy)."""
    if W.ndim > 2:
        W = W.view(W.shape[0], -1)
    
    h, w = W.shape
    u = torch.randn(h, device=W.device, dtype=W.dtype)
    u = u / (u.norm() + 1e-8)
    v = torch.randn(w, device=W.device, dtype=W.dtype)
    v = v / (v.norm() + 1e-8)
    
    for _ in range(niter):
        v = W.T @ u
        v = v / (v.norm() + 1e-8)
        u = W @ v
        u = u / (u.norm() + 1e-8)
    
    sigma = (u @ W @ v).abs()
    return sigma


def compute_all_singular_values(W: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Compute top-k singular values via SVD."""
    if W.ndim > 2:
        W = W.view(W.shape[0], -1)
    return torch.linalg.svdvals(W)[:k]


def main():
    parser = argparse.ArgumentParser(description="Spectral norm diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--output", default="results/diagnostics/spectral_norm.json")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Gamma: {args.gamma}")
    
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
            "gamma": args.gamma,
        },
        "measurements": [],
    }
    
    # Find router.mu parameter
    router_mu = None
    for name, param in model.named_parameters():
        if 'router' in name and 'mu' in name and param.ndim >= 2:
            router_mu = param
            print(f"Found router parameter: {name}, shape: {param.shape}")
            break
    
    if router_mu is None:
        print("ERROR: Could not find router.mu parameter")
        results["error"] = "router.mu not found"
        return
    
    # Measure initial singular values
    with torch.no_grad():
        sigma_initial = compute_singular_values(router_mu.data, niter=20)
        sigmas_initial_topk = compute_all_singular_values(router_mu.data, k=5)
    
    print(f"Initial max sigma (power iter): {sigma_initial.item():.6f}")
    print(f"Initial top-5 sigmas (SVD): {sigmas_initial_topk.tolist()}")
    
    results["measurements"].append({
        "stage": "initial",
        "max_sigma_power_iter": sigma_initial.item(),
        "top_5_sigmas_svd": sigmas_initial_topk.tolist(),
    })
    
    # Apply SpectralConstraint
    constraint = SpectralConstraint(gamma=args.gamma, power_iter=10, timing='post_update')
    state = {}
    
    # Check if already constrained
    sigma_before = compute_singular_values(router_mu.data, niter=10)
    print(f"Before constraint: {sigma_before.item():.6f}")
    
    constraint.enforce(router_mu, state, {})
    
    sigma_after = compute_singular_values(router_mu.data, niter=10)
    sigmas_after_topk = compute_all_singular_values(router_mu.data, k=5)
    print(f"After constraint:  {sigma_after.item():.6f}")
    print(f"After top-5 sigmas: {sigmas_after_topk.tolist()}")
    
    results["measurements"].append({
        "stage": "after_constraint",
        "max_sigma_power_iter": sigma_after.item(),
        "top_5_sigmas_svd": sigmas_after_topk.tolist(),
        "was_constrained": sigma_after.item() < sigma_before.item(),
        "gamma": args.gamma,
    })
    
    # Test add_spectral_constraint utility
    print("\n--- Testing add_spectral_constraint utility ---")
    config.k_init = 8
    model2 = NGSModel(784, 10, config).to(device)
    
    constraints = add_spectral_constraint(model2, gamma=args.gamma, timing='post_update')
    print(f"Registered {len(constraints)} constraints")
    
    for name, param, constraint in constraints:
        sigma_before = compute_singular_values(param.data, niter=10)
        constraint.enforce(param, {}, {})
        sigma_after = compute_singular_values(param.data, niter=10)
        print(f"  {name}: {sigma_before.item():.6f} -> {sigma_after.item():.6f} "
              f"({'constrained' if sigma_after < sigma_before else 'no change'})")
        results["measurements"].append({
            "stage": "utility_test",
            "param_name": name,
            "max_sigma_before": sigma_before.item(),
            "max_sigma_after": sigma_after.item(),
            "was_constrained": sigma_after.item() < sigma_before.item(),
        })
    
    # Test: what if we run multiple enforcement steps?
    print("\n--- Multiple enforcement steps ---")
    model3 = NGSModel(784, 10, config).to(device)
    # Ensure initialized
    router_mu3 = dict(model3.named_parameters())['router.mu']
    
    constraint3 = SpectralConstraint(gamma=args.gamma, power_iter=10, timing='post_update')
    state3 = {}
    
    for step in range(5):
        sigma_b = compute_singular_values(router_mu3.data, niter=10)
        constraint3.enforce(router_mu3, state3, {})
        sigma_a = compute_singular_values(router_mu3.data, niter=10)
        print(f"  Step {step}: {sigma_b.item():.6f} -> {sigma_a.item():.6f}")
        results["measurements"].append({
            "stage": f"multi_step_{step}",
            "max_sigma_before": sigma_b.item(),
            "max_sigma_after": sigma_a.item(),
        })
    
    # Summary
    initial_max = results["measurements"][0]["max_sigma_power_iter"]
    after_max = results["measurements"][1]["max_sigma_power_iter"]
    
    results["summary"] = {
        "initial_max_sigma": initial_max,
        "after_constraint_max_sigma": after_max,
        "constraint_effective": after_max < initial_max,
        "constraint_effective_ratio": after_max / initial_max if initial_max > 0 else 1.0,
        "gamma_target": args.gamma,
        "naturally_below_gamma": initial_max <= args.gamma,
    }
    
    print(f"\nSUMMARY:")
    print(f"  Initial max sigma: {initial_max:.6f}")
    print(f"  After constraint:  {after_max:.6f}")
    print(f"  Gamma target:      {args.gamma:.6f}")
    print(f"  Constraint effective: {results['summary']['constraint_effective']}")
    print(f"  Naturally below gamma: {results['summary']['naturally_below_gamma']}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
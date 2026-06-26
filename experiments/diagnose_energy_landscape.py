#!/usr/bin/env python
"""
Energy Landscape Diagnostic — TODO11 Phase A1.2

Visualizes the Mahalanobis energy landscape along random directions in parameter space.
Measures convexity and identifies local minima.
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


def compute_energy(model: nn.Module, z: torch.Tensor, router_out, target=None, beta=0.0):
    """Compute the routing energy for a given state."""
    B, K = router_out.weights.shape
    device = z.device
    
    active_indices = router_out.indices
    router = model.router
    
    if hasattr(router, 'mu') and router.mu.dim() == 2:
        mu = router.mu[active_indices]
        log_s = router.log_s[active_indices]
        log_alpha = router.log_alpha[active_indices]
    else:
        from ngs.modules.topology_managers import _flat_access
        mu, log_s, log_alpha = _flat_access(router)
        if mu is None:
            return torch.tensor(0.0, device=device)
        mu_active = mu[router.active_mask]
        log_s_active = log_s[router.active_mask]
        return torch.tensor(0.0, device=device)
    
    diff = z.unsqueeze(1) - mu
    sigma_sq = torch.exp(2 * log_s) + 1e-5
    mahalanobis_sq = (diff ** 2 / sigma_sq).sum(dim=-1)
    weights = router_out.weights
    
    internal_energy = (weights * mahalanobis_sq).sum(dim=-1).mean()
    
    nudge_energy = torch.tensor(0.0, device=device)
    if target is not None and beta > 0:
        active_indices = router_out.indices
        param_out = model.param_store(active_indices, z)
        weighted_out = (router_out.weights.unsqueeze(-1) * param_out).sum(dim=1)
        logits = model.p_up(weighted_out)
        nudge_energy = beta * F.cross_entropy(logits, target)
    
    return internal_energy + nudge_energy


def main():
    parser = argparse.ArgumentParser(description="Energy landscape diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-directions", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--alpha-range", type=float, default=1.0)
    parser.add_argument("--output", default="results/diagnostics/energy_landscape.json")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Directions: {args.num_directions}")
    print(f"Steps per direction: {args.num_steps}")
    print(f"Alpha range: [-{args.alpha_range}, {args.alpha_range}]")
    
    # Create model
    config = NGSConfig(
        latent_dim=64,
        max_k=32,
        top_k=8,
        k_init=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    )
    model = NGSModel(784, 10, config).to(device)
    
    # Get EP parameters (router + param_store)
    ep_params = []
    ep_param_names = []
    for name, param in model.named_parameters():
        if ('router' in name or 'param_store' in name) and param.requires_grad:
            ep_params.append(param)
            ep_param_names.append(name)
    
    print(f"EP parameters: {len(ep_params)}")
    for name in ep_param_names:
        print(f"  {name}: {dict(model.named_parameters())[name].shape}")
    
    # Flatten all EP parameters into a single vector
    param_shapes = [p.shape for p in ep_params]
    param_numel = [p.numel() for p in ep_params]
    total_params = sum(param_numel)
    print(f"Total EP parameters: {total_params}")
    
    def get_flat_params():
        return torch.cat([p.data.view(-1) for p in ep_params])
    
    def set_flat_params(flat_vec):
        idx = 0
        for i, p in enumerate(ep_params):
            n = param_numel[i]
            p.data.copy_(flat_vec[idx:idx+n].view(param_shapes[i]))
            idx += n
    
    # Generate input data
    x = torch.randn(32, 784, device=device)
    z = model.p_down(x)
    
    # Get baseline router output
    with torch.no_grad():
        router_out = model.router(z)
    
    # Compute energy at current parameters
    energy_0 = compute_energy(model, z, router_out).item()
    print(f"Baseline energy: {energy_0:.6f}")
    
    # Save initial parameters
    theta_0 = get_flat_params().clone()
    
    results = {
        "config": {
            "num_directions": args.num_directions,
            "num_steps": args.num_steps,
            "alpha_range": args.alpha_range,
            "total_ep_params": total_params,
        },
        "baseline_energy": energy_0,
        "directions": [],
        "convexity_scores": [],
    }
    
    # Generate random directions
    directions = []
    for _ in range(args.num_directions):
        d = torch.randn(total_params, device=device)
        d = d / (d.norm() + 1e-8)
        directions.append(d)
    
    alphas = torch.linspace(-args.alpha_range, args.alpha_range, args.num_steps)
    
    for dir_idx, d in enumerate(directions):
        print(f"\nDirection {dir_idx + 1}/{args.num_directions}")
        energies = []
        
        for alpha in alphas:
            # Set parameters: theta = theta_0 + alpha * d
            theta_new = theta_0 + alpha * d
            set_flat_params(theta_new)
            
            # Compute energy
            with torch.no_grad():
                router_out = model.router(z)
                energy = compute_energy(model, z, router_out).item()
            
            energies.append(energy)
        
        # Restore
        set_flat_params(theta_0)
        
        # Compute convexity score: fraction of monotonic segments
        # For a convex function along a line, energy should be unimodal
        monotonic_segments = 0
        for i in range(1, len(energies) - 1):
            # Check if we're in a "valley" (local minimum pattern)
            if energies[i] < energies[i-1] and energies[i] < energies[i+1]:
                monotonic_segments += 1
        
        # Better convexity measure: check if function is convex (second derivative positive)
        # Use finite differences
        second_derivs = []
        for i in range(1, len(energies) - 1):
            d2 = energies[i-1] - 2*energies[i] + energies[i+1]
            second_derivs.append(d2)
        
        convex_frac = sum(1 for d2 in second_derivs if d2 >= 0) / len(second_derivs) if second_derivs else 0
        
        direction_result = {
            "direction_idx": dir_idx,
            "alphas": alphas.tolist(),
            "energies": energies,
            "convex_fraction": convex_frac,
            "min_energy": min(energies),
            "min_alpha": alphas[energies.index(min(energies))].item(),
            "energy_at_zero": energies[len(energies)//2],
        }
        
        results["directions"].append(direction_result)
        results["convexity_scores"].append(convex_frac)
        
        print(f"  Energies: {[f'{e:.4f}' for e in energies]}")
        print(f"  Convex fraction: {convex_frac:.3f}")
        print(f"  Min at alpha={direction_result['min_alpha']:.3f}")
    
    # Overall convexity
    mean_convexity = sum(results["convexity_scores"]) / len(results["convexity_scores"])
    results["summary"] = {
        "mean_convexity_fraction": mean_convexity,
        "baseline_energy": energy_0,
        "is_convex": mean_convexity > 0.7,
        "landscape_type": "convex" if mean_convexity > 0.7 else "non-convex" if mean_convexity < 0.3 else "mixed",
    }
    
    print(f"\nSUMMARY:")
    print(f"  Mean convexity: {mean_convexity:.3f}")
    print(f"  Landscape type: {results['summary']['landscape_type']}")
    print(f"  Is convex: {results['summary']['is_convex']}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    import torch.nn.functional as F
    main()
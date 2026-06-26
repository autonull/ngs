#!/usr/bin/env python
"""
EP vs Backprop Update Comparison — TODO11 Phase A1.5

THE CRITICAL DIAGNOSTIC: Compare EP contrastive update with true backprop gradient
for router parameters. If cosine similarity is low, EP mechanism is fundamentally broken.
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
from ngs.modules.eqprop import EqNGSLayer, create_eqngs


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    a_flat = a.view(-1)
    b_flat = b.view(-1)
    cos_sim = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=1)
    return cos_sim.item()


def main():
    parser = argparse.ArgumentParser(description="EP vs BP update comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--ep-beta", type=float, default=0.5)
    parser.add_argument("--output", default="results/diagnostics/ep_vs_bp_updates.json")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Settle steps: {args.settle_steps}")
    print(f"EP beta: {args.ep_beta}")
    
    # Create model
    config = NGSConfig(
        latent_dim=64,
        max_k=32,
        top_k=8,
        k_init=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    )
    
    eqngs = create_eqngs(
        d_in=784,
        d_out=10,
        config=config,
        ep_beta=args.ep_beta,
        ep_settle_steps=args.settle_steps,
        ep_settle_lr=0.2,
        spectral_mode='none',  # Disable SN for clean comparison
    ).to(device)
    
    # Get EP parameters
    ep_params = eqngs.ep_params
    ep_param_names = eqngs.ep_param_names
    print(f"EP parameters ({len(ep_params)}):")
    for name in ep_param_names:
        print(f"  {name}")
    
    # Generate dummy data
    x = torch.randn(args.batch_size, 784, device=device)
    target = torch.randint(0, 10, (args.batch_size,), device=device)
    
    results = {
        "config": {
            "batch_size": args.batch_size,
            "settle_steps": args.settle_steps,
            "ep_beta": args.ep_beta,
            "latent_dim": config.latent_dim,
            "max_k": config.max_k,
            "top_k": config.top_k,
        },
        "per_parameter": [],
        "aggregated": {},
    }
    
    # --- BACKPROP GRADIENT ---
    print("\n--- Computing backprop gradient ---")
    eqngs.train()
    eqngs.zero_grad()
    
    # Forward pass
    z = eqngs.ngs.p_down(x)
    out = eqngs.ngs(x)
    logits = out.logits if hasattr(out, 'logits') else out
    loss_bp = F.cross_entropy(logits, target)
    loss_bp.backward()
    
    # Collect backprop gradients
    bp_grads = {}
    for name, param in eqngs.ngs.named_parameters():
        if param.grad is not None and ('router' in name or 'param_store' in name):
            bp_grads[name] = param.grad.clone()
    
    print(f"Backprop gradients collected for {len(bp_grads)} parameters")
    
    # --- EP CONTRASTIVE UPDATE ---
    print("\n--- Computing EP contrastive update ---")
    
    # Save initial parameters
    initial_params = {name: param.clone() for name, param in eqngs.ngs.named_parameters()
                      if 'router' in name or 'param_store' in name}
    
    # FREE PHASE
    eqngs._ep_training = True
    eqngs.ngs.train()
    
    param_states_free = [p.clone() for p in eqngs.ep_params]
    
    for step in range(args.settle_steps):
        router_out = eqngs._router_forward(z)
        if router_out is None:
            break
        
        energy = eqngs._compute_routing_energy(z, router_out, target=None, beta=0.0)
        energy = energy.to(z.device)
        
        grads = torch.autograd.grad(
            energy, eqngs.ep_params,
            retain_graph=True, create_graph=False, allow_unused=True
        )
        grads = [g.to(p.device) if g is not None else None for g, p in zip(grads, eqngs.ep_params)]
        
        with torch.no_grad():
            for p, buf, g in zip(eqngs.ep_params, eqngs.ep_buffers, grads):
                if g is not None:
                    buf.mul_(eqngs.ep_momentum).add_(g)
                    p.sub_(buf, alpha=eqngs.ep_settle_lr)
    
    router_free = eqngs._router_forward(z)
    params_free = {name: param.clone() for name, param in eqngs.ngs.named_parameters()
                   if 'router' in name or 'param_store' in name}
    
    # Restore initial parameters
    with torch.no_grad():
        for name, param in eqngs.ngs.named_parameters():
            if name in initial_params:
                param.data.copy_(initial_params[name])
    
    # NUDGED PHASE
    for step in range(args.settle_steps):
        router_out = eqngs._router_forward(z)
        if router_out is None:
            break
        
        energy = eqngs._compute_routing_energy(z, router_out, target=target, beta=args.ep_beta)
        energy = energy.to(z.device)
        
        grads = torch.autograd.grad(
            energy, eqngs.ep_params,
            retain_graph=True, create_graph=False, allow_unused=True
        )
        grads = [g.to(p.device) if g is not None else None for g, p in zip(grads, eqngs.ep_params)]
        
        with torch.no_grad():
            for p, buf, g in zip(eqngs.ep_params, eqngs.ep_buffers, grads):
                if g is not None:
                    buf.mul_(eqngs.ep_momentum).add_(g)
                    p.sub_(buf, alpha=eqngs.ep_settle_lr)
    
    router_nudged = eqngs._router_forward(z)
    params_nudged = {name: param.clone() for name, param in eqngs.ngs.named_parameters()
                     if 'router' in name or 'param_store' in name}
    
    # Compute EP contrastive update: Δθ_EP = θ_nudged - θ_free
    ep_updates = {}
    for name in params_free:
        if name in params_nudged:
            ep_updates[name] = params_nudged[name] - params_free[name]
    
    print(f"EP updates computed for {len(ep_updates)} parameters")
    
    # --- COMPARE ---
    print("\n--- Comparing EP vs BP ---")
    cos_sims = []
    mag_ratios = []
    sign_agreements = []
    
    for name in bp_grads:
        if name not in ep_updates:
            continue
        
        bp_grad = bp_grads[name]
        ep_update = ep_updates[name]
        
        # Ensure same shape
        if bp_grad.shape != ep_update.shape:
            print(f"  {name}: Shape mismatch BP={bp_grad.shape} EP={ep_update.shape}")
            continue
        
        # Cosine similarity
        cos_sim = cosine_similarity(bp_grad, ep_update)
        
        # Magnitude ratio
        bp_norm = bp_grad.norm().item()
        ep_norm = ep_update.norm().item()
        mag_ratio = ep_norm / bp_norm if bp_norm > 0 else float('inf')
        
        # Sign agreement
        sign_agree = (bp_grad * ep_update > 0).float().mean().item()
        
        print(f"  {name}:")
        print(f"    Cosine sim: {cos_sim:.6f}")
        print(f"    |BP grad|: {bp_norm:.6f}, |EP update|: {ep_norm:.6f}, Ratio: {mag_ratio:.6f}")
        print(f"    Sign agreement: {sign_agree:.6f}")
        
        results["per_parameter"].append({
            "name": name,
            "cosine_similarity": cos_sim,
            "magnitude_ratio": mag_ratio,
            "bp_gradient_norm": bp_norm,
            "ep_update_norm": ep_norm,
            "sign_agreement": sign_agree,
            "shape": list(bp_grad.shape),
        })
        
        cos_sims.append(cos_sim)
        mag_ratios.append(mag_ratio)
        sign_agreements.append(sign_agree)
    
    # Aggregated stats
    if cos_sims:
        results["aggregated"] = {
            "mean_cosine_similarity": sum(cos_sims) / len(cos_sims),
            "min_cosine_similarity": min(cos_sims),
            "max_cosine_similarity": max(cos_sims),
            "mean_magnitude_ratio": sum(mag_ratios) / len(mag_ratios),
            "mean_sign_agreement": sum(sign_agreements) / len(sign_agreements),
            "num_parameters": len(cos_sims),
        }
        
        print(f"\nAGGREGATED:")
        print(f"  Mean cosine similarity: {results['aggregated']['mean_cosine_similarity']:.6f}")
        print(f"  Min cosine similarity:  {results['aggregated']['min_cosine_similarity']:.6f}")
        print(f"  Max cosine similarity:  {results['aggregated']['max_cosine_similarity']:.6f}")
        print(f"  Mean magnitude ratio:   {results['aggregated']['mean_magnitude_ratio']:.6f}")
        print(f"  Mean sign agreement:    {results['aggregated']['mean_sign_agreement']:.6f}")
        
        # Showstopper check
        if results['aggregated']['mean_cosine_similarity'] < 0.1:
            results["showstopper"] = True
            print(f"\n⚠ SHOWSTOPPER: Mean cosine similarity < 0.1")
        elif results['aggregated']['mean_cosine_similarity'] < 0.3:
            results["warning"] = True
            print(f"\n⚠ WARNING: Mean cosine similarity < 0.3")
    else:
        results["aggregated"]["error"] = "No matching parameters found"
        print("ERROR: No matching parameters to compare")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
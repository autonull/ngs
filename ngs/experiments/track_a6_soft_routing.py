#!/usr/bin/env python
"""
Track A6: Soft Routing (Ablate top-k)
Hypothesis: Using all K units with entropy regularization eliminates information loss from hard top-k.
Target: 4-layer NGS >= 92% on MNIST (5 epochs)
"""
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/me/ngs')

from ngs.models.ngs import MultiLayerNGS
from ngs.core.interfaces import NGSConfig, RoutingStrategy
from experiments.fast_data import load_mnist_fast


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def train_eval(model, train_loader, test_loader, device, epochs=5, lr=1e-3, entropy_weight=0.01):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).logits
            loss = criterion(logits, y)
            
            # Entropy regularization (if model supports it)
            if hasattr(model, 'entropy_loss'):
                entropy_loss = model.entropy_loss(x)
                loss = loss + entropy_weight * entropy_loss
            
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).logits
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        best_acc = max(best_acc, acc)
    
    return best_acc


def run_a6_experiment(config_name, layer_configs, entropy_weight, seed=42, epochs=5, device='cuda'):
    """Run a single A6 config."""
    set_seed(seed)
    
    train_ds, test_ds, d_in, d_out = load_mnist_fast()
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    configs = []
    for i, lc in enumerate(layer_configs):
        configs.append(NGSConfig(
            latent_dim=lc.get('latent_dim', 64),
            max_k=lc.get('max_k', 32),
            top_k=lc.get('top_k', 8),
            k_init=min(lc.get('top_k', 8), lc.get('max_k', 32)),
            routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
            gamma_residual=lc.get('gamma_residual', 0.1),
            beta_residual=lc.get('beta_residual', 0.1),
            soft_routing=True,  # KEY: enable soft routing (ablate top-k)
        ))
    
    model = MultiLayerNGS(d_in, d_out, len(configs), configs)
    
    print(f"  [{config_name}] Entropy weight: {entropy_weight}, K: {[c.max_k for c in configs]}")
    
    start = time.time()
    acc = train_eval(model, train_loader, test_loader, device, epochs=epochs, entropy_weight=entropy_weight)
    elapsed = time.time() - start
    
    return {
        'config_name': config_name,
        'entropy_weight': entropy_weight,
        'layer_configs': layer_configs,
        'test_accuracy': acc,
        'time_seconds': elapsed,
        'seed': seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Track A6: Soft Routing sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/track_a/a6_soft_routing.json")
    args = parser.parse_args()
    
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    
    # Test different entropy weights
    entropy_weights = [0.0, 0.001, 0.01, 0.05, 0.1]
    
    # Layer configurations to test
    layer_configs_list = [
        # 4-layer progressive top_k (from A1 best)
        {
            'name': 'progressive_4L',
            'layers': [
                {'top_k': 4, 'max_k': 32, 'latent_dim': 64},
                {'top_k': 8, 'max_k': 32, 'latent_dim': 64},
                {'top_k': 16, 'max_k': 32, 'latent_dim': 64},
                {'top_k': 32, 'max_k': 32, 'latent_dim': 64},
            ]
        },
        # 4-layer constant K=32
        {
            'name': 'constant_4L_K32',
            'layers': [
                {'top_k': 8, 'max_k': 32, 'latent_dim': 64},
                {'top_k': 8, 'max_k': 32, 'latent_dim': 64},
                {'top_k': 8, 'max_k': 32, 'latent_dim': 64},
                {'top_k': 8, 'max_k': 32, 'latent_dim': 64},
            ]
        },
        # 4-layer growing K
        {
            'name': 'growing_4L_K16_32_64_128',
            'layers': [
                {'top_k': 8, 'max_k': 16, 'latent_dim': 64},
                {'top_k': 8, 'max_k': 32, 'latent_dim': 64},
                {'top_k': 8, 'max_k': 64, 'latent_dim': 64},
                {'top_k': 8, 'max_k': 128, 'latent_dim': 64},
            ]
        },
        # 3-layer
        {
            'name': 'progressive_3L',
            'layers': [
                {'top_k': 4, 'max_k': 32, 'latent_dim': 64},
                {'top_k': 8, 'max_k': 32, 'latent_dim': 64},
                {'top_k': 16, 'max_k': 32, 'latent_dim': 64},
            ]
        },
    ]
    
    sweep_configs = []
    for lc in layer_configs_list:
        for ew in entropy_weights:
            sweep_configs.append({
                'name': f"{lc['name']}_ew{ew}",
                'layers': lc['layers'],
                'entropy_weight': ew,
            })
    
    # Also test hard top-k baselines (soft_routing=False) for comparison
    baseline_configs = [
        {
            'name': 'hard_topk_progressive_4L',
            'layers': [
                {'top_k': 4, 'max_k': 32, 'latent_dim': 64},
                {'top_k': 8, 'max_k': 32, 'latent_dim': 64},
                {'top_k': 16, 'max_k': 32, 'latent_dim': 64},
                {'top_k': 32, 'max_k': 32, 'latent_dim': 64},
            ],
            'soft_routing': False,
            'entropy_weight': 0.0,
        },
        {
            'name': 'hard_topk_constant_4L_K32',
            'layers': [
                {'top_k': 8, 'max_k': 32, 'latent_dim': 64},
                {'top_k': 8, 'max_k': 32, 'latent_dim': 64},
                {'top_k': 8, 'max_k': 32, 'latent_dim': 64},
                {'top_k': 8, 'max_k': 32, 'latent_dim': 64},
            ],
            'soft_routing': False,
            'entropy_weight': 0.0,
        },
    ]
    
    print(f"Running Track A6: {len(sweep_configs)} soft routing + {len(baseline_configs)} hard top-k baselines, {args.epochs} epochs each")
    print(f"Device: {device}, Seed: {args.seed}")
    
    results = []
    
    # Soft routing experiments
    for config in sweep_configs:
        try:
            result = run_a6_experiment(
                config['name'], config['layers'], 
                config['entropy_weight'],
                args.seed, args.epochs, device
            )
            results.append(result)
            print(f"  Result: {result['test_accuracy']:.4f} ({result['time_seconds']:.1f}s)")
        except Exception as e:
            print(f"  ERROR in {config['name']}: {e}")
            results.append({'config_name': config['name'], 'error': str(e)})
    
    # Hard top-k baselines (using MultiLayerNGS directly with soft_routing=False)
    from ngs.models.ngs import MultiLayerNGS
    
    for config in baseline_configs:
        set_seed(args.seed)
        train_ds, test_ds, d_in, d_out = load_mnist_fast()
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
        
        configs = []
        for i, lc in enumerate(config['layers']):
            configs.append(NGSConfig(
                latent_dim=lc.get('latent_dim', 64),
                max_k=lc.get('max_k', 32),
                top_k=lc.get('top_k', 8),
                k_init=min(lc.get('top_k', 8), lc.get('max_k', 32)),
                routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
                gamma_residual=lc.get('gamma_residual', 0.1),
                beta_residual=lc.get('beta_residual', 0.1),
                soft_routing=False,
            ))
        
        model = MultiLayerNGS(d_in, d_out, len(configs), configs)
        
        print(f"  [{config['name']}] Hard top-k baseline")
        
        start = time.time()
        acc = train_eval(model, train_loader, test_loader, device, epochs=args.epochs, entropy_weight=0.0)
        elapsed = time.time() - start
        
        results.append({
            'config_name': config['name'],
            'entropy_weight': 0.0,
            'layer_configs': config['layers'],
            'test_accuracy': acc,
            'time_seconds': elapsed,
            'seed': args.seed,
        })
        print(f"  Result: {acc:.4f} ({elapsed:.1f}s)")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'track': 'A6',
            'description': 'Soft Routing (ablate top-k) sweep',
            'epochs': args.epochs,
            'seed': args.seed,
            'results': results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "="*60)
    print("TRACK A6 SUMMARY")
    print("="*60)
    for r in results:
        if 'test_accuracy' in r:
            marker = " ***" if r['test_accuracy'] >= 0.92 else ""
            print(f"  {r['config_name']:40s} : {r['test_accuracy']:.4f}{marker}")
        else:
            print(f"  {r['config_name']:40s} : ERROR - {r.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
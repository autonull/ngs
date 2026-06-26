#!/usr/bin/env python
"""
Track A4: Cross-Layer Router Sharing
Hypothesis: Single router shared across layers with per-layer param_stores.
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

from ngs.models.ngs import SharedRouterNGS
from ngs.core.interfaces import NGSConfig, RoutingStrategy
from experiments.fast_data import load_mnist_fast


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def train_eval(model, train_loader, test_loader, device, epochs=5, lr=1e-3):
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


def run_a4_experiment(config_name, num_layers, config_params, seed=42, epochs=5, device='cuda'):
    """Run a single A4 config."""
    set_seed(seed)
    
    train_ds, test_ds, d_in, d_out = load_mnist_fast()
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    config = NGSConfig(
        latent_dim=config_params.get('latent_dim', 64),
        max_k=config_params.get('max_k', 32),
        top_k=config_params.get('top_k', 8),
        k_init=min(config_params.get('top_k', 8), config_params.get('max_k', 32)),
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        gamma_residual=config_params.get('gamma_residual', 0.1),
    )
    
    model = SharedRouterNGS(d_in, d_out, num_layers, config)
    
    print(f"  [{config_name}] Layers: {num_layers}, K: {config.max_k}, top_k: {config.top_k}")
    
    start = time.time()
    acc = train_eval(model, train_loader, test_loader, device, epochs=epochs)
    elapsed = time.time() - start
    
    return {
        'config_name': config_name,
        'num_layers': num_layers,
        'config': config_params,
        'test_accuracy': acc,
        'time_seconds': elapsed,
        'seed': seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Track A4: Shared Router test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/track_a/a4_shared_router.json")
    args = parser.parse_args()
    
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    
    sweep_configs = [
        # Baseline: 4 layers, shared router, K=32
        {
            'name': 'shared_router_4L_K32_tk8',
            'num_layers': 4,
            'params': {'latent_dim': 64, 'max_k': 32, 'top_k': 8, 'gamma_residual': 0.1},
        },
        # 3 layers
        {
            'name': 'shared_router_3L_K32_tk8',
            'num_layers': 3,
            'params': {'latent_dim': 64, 'max_k': 32, 'top_k': 8, 'gamma_residual': 0.1},
        },
        # 2 layers
        {
            'name': 'shared_router_2L_K32_tk8',
            'num_layers': 2,
            'params': {'latent_dim': 64, 'max_k': 32, 'top_k': 8, 'gamma_residual': 0.1},
        },
        # Larger K
        {
            'name': 'shared_router_4L_K64_tk8',
            'num_layers': 4,
            'params': {'latent_dim': 64, 'max_k': 64, 'top_k': 8, 'gamma_residual': 0.1},
        },
        # Larger K=128
        {
            'name': 'shared_router_4L_K128_tk8',
            'num_layers': 4,
            'params': {'latent_dim': 64, 'max_k': 128, 'top_k': 8, 'gamma_residual': 0.1},
        },
        # Stronger residual
        {
            'name': 'shared_router_4L_K32_tk8_gamma02',
            'num_layers': 4,
            'params': {'latent_dim': 64, 'max_k': 32, 'top_k': 8, 'gamma_residual': 0.2},
        },
    ]
    
    # Compare with MultiLayerNGS (non-shared) for reference
    from ngs.models.ngs import MultiLayerNGS
    
    print(f"Running Track A4: {len(sweep_configs)} shared router configs + 4 baselines, {args.epochs} epochs each")
    print(f"Device: {device}, Seed: {args.seed}")
    
    results = []
    
    # Shared router experiments
    for config in sweep_configs:
        try:
            result = run_a4_experiment(
                config['name'], config['num_layers'], config['params'],
                args.seed, args.epochs, device
            )
            results.append(result)
            print(f"  Result: {result['test_accuracy']:.4f} ({result['time_seconds']:.1f}s)")
        except Exception as e:
            print(f"  ERROR in {config['name']}: {e}")
            results.append({'config_name': config['name'], 'error': str(e)})
    
    # Baseline comparisons using MultiLayerNGS (non-shared)
    baseline_configs = [
        {'name': 'multilayer_baseline_4L_K32_tk8', 'num_layers': 4, 'params': {'latent_dim': 64, 'max_k': 32, 'top_k': 8, 'gamma_residual': 0.1}},
        {'name': 'multilayer_baseline_4L_K64_tk8', 'num_layers': 4, 'params': {'latent_dim': 64, 'max_k': 64, 'top_k': 8, 'gamma_residual': 0.1}},
        {'name': 'multilayer_baseline_4L_K128_tk8', 'num_layers': 4, 'params': {'latent_dim': 64, 'max_k': 128, 'top_k': 8, 'gamma_residual': 0.1}},
    ]
    
    for config in baseline_configs:
        set_seed(args.seed)
        train_ds, test_ds, d_in, d_out = load_mnist_fast()
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
        
        configs = []
        for i in range(config['num_layers']):
            configs.append(NGSConfig(
                latent_dim=config['params']['latent_dim'],
                max_k=config['params']['max_k'],
                top_k=config['params']['top_k'],
                k_init=min(config['params']['top_k'], config['params']['max_k']),
                routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
                gamma_residual=config['params']['gamma_residual'],
            ))
        
        model = MultiLayerNGS(d_in, d_out, config['num_layers'], configs)
        
        print(f"  [{config['name']}] Layers: {config['num_layers']}, K: {config['params']['max_k']}, top_k: {config['params']['top_k']} (MultiLayerNGS)")
        
        start = time.time()
        acc = train_eval(model, train_loader, test_loader, device, epochs=args.epochs)
        elapsed = time.time() - start
        
        results.append({
            'config_name': config['name'],
            'num_layers': config['num_layers'],
            'config': config['params'],
            'test_accuracy': acc,
            'time_seconds': elapsed,
            'seed': args.seed,
        })
        print(f"  Result: {acc:.4f} ({elapsed:.1f}s)")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'track': 'A4',
            'description': 'Shared Router test (vs MultiLayerNGS baseline)',
            'epochs': args.epochs,
            'seed': args.seed,
            'results': results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "="*60)
    print("TRACK A4 SUMMARY")
    print("="*60)
    for r in results:
        if 'test_accuracy' in r:
            marker = " ***" if r['test_accuracy'] >= 0.92 else ""
            print(f"  {r['config_name']:40s} : {r['test_accuracy']:.4f}{marker}")
        else:
            print(f"  {r['config_name']:40s} : ERROR - {r.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
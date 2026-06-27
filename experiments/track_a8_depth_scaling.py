#!/usr/bin/env python
"""
Track A8: Depth Scaling (8L, 16L)
Hypothesis: Multi-layer NGS scales to deeper networks without collapse.
Target: 8L and 16L >= 95% on MNIST (5 epochs)
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


def run_a8_experiment(config_name, num_layers, layer_configs, seed=42, epochs=5, device='cuda'):
    set_seed(seed)
    
    train_ds, test_ds, d_in, d_out = load_mnist_fast()
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    configs = []
    for lc in layer_configs:
        configs.append(NGSConfig(
            latent_dim=lc.get('latent_dim', 64),
            max_k=lc.get('max_k', 32),
            top_k=lc.get('top_k', 8),
            k_init=min(lc.get('top_k', 8), lc.get('max_k', 32)),
            routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
            gamma_residual=lc.get('gamma_residual', 0.1),
            beta_residual=lc.get('beta_residual', 0.1),
            use_mlp_projections=lc.get('use_mlp_projections', False),
            mlp_hidden_multiplier=lc.get('mlp_hidden_multiplier', 4),
        ))
    
    model = MultiLayerNGS(d_in, d_out, num_layers, configs)
    
    print(f"  [{config_name}] {num_layers} layers")
    
    start = time.time()
    acc = train_eval(model, train_loader, test_loader, device, epochs=epochs)
    elapsed = time.time() - start
    
    return {
        'config_name': config_name,
        'num_layers': num_layers,
        'test_accuracy': acc,
        'time_seconds': elapsed,
        'seed': seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Track A8: Depth scaling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/track_a/a8_depth_scaling.json")
    args = parser.parse_args()
    
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    
    # Best config from A7 applied to 8L and 16L
    # Use progressive top_k per layer, with MLP projections
    def make_configs(num_layers):
        configs = []
        for i in range(num_layers):
            top_k_val = min(4 * (2**i), 32)
            configs.append({
                'top_k': top_k_val,
                'max_k': 32,
                'latent_dim': 64,
                'gamma_residual': 0.2,
                'beta_residual': 0.2,
                'use_mlp_projections': True,
                'mlp_hidden_multiplier': 4,
            })
        return configs
    
    sweep_configs = [
        {'name': 'depth_8L', 'num_layers': 8, 'configs': make_configs(8)},
        {'name': 'depth_16L', 'num_layers': 16, 'configs': make_configs(16)},
        {'name': 'depth_4L_baseline', 'num_layers': 4, 'configs': make_configs(4)},
    ]
    
    print(f"Running Track A8: depth scaling, {args.epochs} epochs each")
    print(f"Device: {device}, Seed: {args.seed}")
    
    results = []
    for config in sweep_configs:
        try:
            result = run_a8_experiment(
                config['name'], config['num_layers'], config['configs'],
                args.seed, args.epochs, device
            )
            results.append(result)
            print(f"  Result: {result['test_accuracy']:.4f} ({result['time_seconds']:.1f}s)")
        except Exception as e:
            print(f"  ERROR in {config['name']}: {e}")
            results.append({'config_name': config['name'], 'error': str(e)})
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'track': 'A8',
            'description': 'Depth scaling (8L, 16L)',
            'epochs': args.epochs,
            'seed': args.seed,
            'results': results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "="*60)
    print("TRACK A8 SUMMARY")
    print("="*60)
    for r in results:
        if 'test_accuracy' in r:
            marker = " ***" if r['test_accuracy'] >= 0.95 else ""
            print(f"  {r['config_name']:30s} : {r['test_accuracy']:.4f}{marker}")
        else:
            print(f"  {r['config_name']:30s} : ERROR - {r.get('error', 'unknown')}")


if __name__ == "__main__":
    main()

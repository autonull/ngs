#!/usr/bin/env python
"""
Track A2: Growing K Sweep
Hypothesis: Capacity should scale with depth (more units in deeper layers).
Target: 4-layer NGS >= 93% on MNIST (5 epochs)
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
            logits = model(x).logits = model(x).logits
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


def run_a2_experiment(config_name, layer_configs, seed=42, epochs=5, device='cuda'):
    """Run a single A2 config."""
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
        ))
    
    model = MultiLayerNGS(d_in, d_out, len(configs), configs)
    
    print(f"  [{config_name}] Layers: {len(configs)}, K: {[c.max_k for c in configs]}, top_k: {[c.top_k for c in configs]}")
    
    start = time.time()
    acc = train_eval(model, train_loader, test_loader, device, epochs=epochs)
    elapsed = time.time() - start
    
    return {
        'config_name': config_name,
        'layer_configs': layer_configs,
        'test_accuracy': acc,
        'time_seconds': elapsed,
        'seed': seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Track A2: Growing K sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/track_a/a2_growing_k.json")
    args = parser.parse_args()
    
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    
    sweep_configs = [
        # Baseline: 4L, K=32, top_k=8
        {
            'name': 'baseline_4L_K32_tk8',
            'layers': [
                {'max_k': 32, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 32, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 32, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 32, 'top_k': 8, 'latent_dim': 64},
            ]
        },
        # Growing K: [16, 32, 64, 128]
        {
            'name': 'growing_K_16_32_64_128',
            'layers': [
                {'max_k': 16, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 32, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 64, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 128, 'top_k': 8, 'latent_dim': 64},
            ]
        },
        # Growing K: [8, 16, 32, 64]
        {
            'name': 'growing_K_8_16_32_64',
            'layers': [
                {'max_k': 8, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 16, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 32, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 64, 'top_k': 8, 'latent_dim': 64},
            ]
        },
        # 3-layer growing K
        {
            'name': 'growing_3L_K_16_32_64',
            'layers': [
                {'max_k': 16, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 32, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 64, 'top_k': 8, 'latent_dim': 64},
            ]
        },
        # Shrinking K: [128, 64, 32, 16] (reverse)
        {
            'name': 'shrinking_K_128_64_32_16',
            'layers': [
                {'max_k': 128, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 64, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 32, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 16, 'top_k': 8, 'latent_dim': 64},
            ]
        },
        # Constant large K=64
        {
            'name': 'constant_K64_tk8',
            'layers': [
                {'max_k': 64, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 64, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 64, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 64, 'top_k': 8, 'latent_dim': 64},
            ]
        },
        # Constant large K=128
        {
            'name': 'constant_K128_tk8',
            'layers': [
                {'max_k': 128, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 128, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 128, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 128, 'top_k': 8, 'latent_dim': 64},
            ]
        },
        # 2-layer baseline
        {
            'name': 'baseline_2L_K32_tk8',
            'layers': [
                {'max_k': 32, 'top_k': 8, 'latent_dim': 64},
                {'max_k': 32, 'top_k': 8, 'latent_dim': 64},
            ]
        },
    ]
    
    print(f"Running Track A2: {len(sweep_configs)} configs, {args.epochs} epochs each")
    print(f"Device: {device}, Seed: {args.seed}")
    
    results = []
    for config in sweep_configs:
        try:
            result = run_a2_experiment(config['name'], config['layers'], args.seed, args.epochs, device)
            results.append(result)
            print(f"  Result: {result['test_accuracy']:.4f} ({result['time_seconds']:.1f}s)")
        except Exception as e:
            print(f"  ERROR in {config['name']}: {e}")
            results.append({'config_name': config['name'], 'error': str(e)})
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'track': 'A2',
            'description': 'Growing K sweep',
            'epochs': args.epochs,
            'seed': args.seed,
            'results': results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "="*60)
    print("TRACK A2 SUMMARY")
    print("="*60)
    for r in results:
        if 'test_accuracy' in r:
            marker = " ***" if r['test_accuracy'] >= 0.93 else ""
            print(f"  {r['config_name']:35s} : {r['test_accuracy']:.4f}{marker}")
        else:
            print(f"  {r['config_name']:35s} : ERROR - {r.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
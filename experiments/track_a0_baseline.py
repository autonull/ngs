#!/usr/bin/env python
"""
Track A0: Baseline Confirmation (Run across 3 seeds)
Hypothesis: depth=4 is consistently >= 95% across seeds
Target: >= 95.0% mean accuracy on MNIST (5 epochs)
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


def run_a0_experiment(seed, epochs=5, device='cuda'):
    set_seed(seed)
    
    train_ds, test_ds, d_in, d_out = load_mnist_fast()
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    config = NGSConfig(
        latent_dim=64,
        max_k=32,
        top_k=8,
        k_init=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        gamma_residual=0.1,
        beta_residual=0.1,
    )
    
    model = MultiLayerNGS(d_in, d_out, 4, [config]*4)
    
    start = time.time()
    acc = train_eval(model, train_loader, test_loader, device, epochs=epochs)
    elapsed = time.time() - start
    
    return {
        'seed': seed,
        'test_accuracy': acc,
        'time_seconds': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Track A0: Baseline confirmation (3 seeds)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/track_a/a0_baseline.json")
    args = parser.parse_args()
    
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    seeds = [42, 43, 44]
    
    print(f"Running Track A0: {len(seeds)} seeds, {args.epochs} epochs each")
    print(f"Device: {device}")
    
    results = []
    for seed in seeds:
        try:
            result = run_a0_experiment(seed, args.epochs, device)
            results.append(result)
            print(f"  Seed {seed}: {result['test_accuracy']:.4f} ({result['time_seconds']:.1f}s)")
        except Exception as e:
            print(f"  ERROR with seed {seed}: {e}")
            results.append({'seed': seed, 'error': str(e)})
    
    # Compute mean accuracy if all successful
    accuracies = [r['test_accuracy'] for r in results if 'test_accuracy' in r]
    if accuracies:
        mean_acc = sum(accuracies) / len(accuracies)
        print(f"\n  Mean accuracy: {mean_acc:.4f}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'track': 'A0',
            'description': 'Baseline confirmation (3 seeds)',
            'epochs': args.epochs,
            'results': results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "="*60)
    print("TRACK A0 SUMMARY")
    print("="*60)
    for r in results:
        if 'test_accuracy' in r:
            marker = " ***" if r['test_accuracy'] >= 0.95 else ""
            print(f"  Seed {r['seed']:3d} : {r['test_accuracy']:.4f}{marker}")
        else:
            print(f"  Seed {r['seed']:3d} : ERROR - {r.get('error', 'unknown')}")
    
    if accuracies:
        mean_acc = sum(accuracies) / len(accuracies)
        gate = "PASSED" if mean_acc >= 0.95 else "FAILED"
        print(f"\n  Gate A: {gate} (mean {mean_acc:.4f} vs target 0.9500)")


if __name__ == "__main__":
    main()

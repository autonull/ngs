#!/usr/bin/env python
"""
EPOptimizer Direct on NGS — TODO11 Phase C6

Apply bioplausible EPOptimizer directly to NGSModel.
Uses MSE-based internal energy instead of Mahalanobis routing energy.
This is THE critical gate experiment (G6).

If this achieves >85% on MNIST, the defect is in our custom EqNGSLayer,
not in NGS+EP in general.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, '/home/me/ngs')
sys.path.insert(0, '/home/me/ngs/bioplausible/mep')

from ngs.modules.epopt_ngs_wrapper import EPOptimizerNGSWrapper
from ngs.core.interfaces import NGSConfig, RoutingStrategy

# Import EPOptimizer directly from bioplausible
from mep.optimizers.ep_optimizer import EPOptimizer


def get_mnist_loaders(batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    train = datasets.MNIST("/tmp/mnist", train=True, download=True, transform=transform)
    test = datasets.MNIST("/tmp/mnist", train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def main():
    parser = argparse.ArgumentParser(description="EPOptimizer on NGS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit batches per epoch for quick tests")
    parser.add_argument("--mode", choices=["ep", "backprop"], default="ep")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--settle-lr", type=float, default=0.2)
    parser.add_argument("--output", default="results/diagnostics/eqprop_via_epoptimizer.json")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    if args.mode == "ep":
        print(f"  Beta: {args.beta}, Settle steps: {args.settle_steps}, Settle LR: {args.settle_lr}")
    
    # Create model
    config = NGSConfig(
        latent_dim=64,
        max_k=32,
        top_k=8,
        k_init=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        tau=1.0,
    )
    
    model = EPOptimizerNGSWrapper(784, 10, config)
    model = model.to(device)
    
    # For EP mode, also try with backprop to compare
    if args.mode == "ep":
        opt = EPOptimizer(
            model.parameters(),
            model=model,
            mode='ep',
            lr=args.lr,
            beta=args.beta,
            settle_steps=args.settle_steps,
            settle_lr=args.settle_lr,
            loss_type='mse',
        )
    else:
        opt = EPOptimizer(
            model.parameters(),
            model=model,
            mode='backprop',
            lr=args.lr,
        )
    
    train_loader, test_loader = get_mnist_loaders(args.batch_size)
    
    results = {
        "config": {
            "mode": args.mode,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "beta": args.beta,
            "settle_steps": args.settle_steps,
            "settle_lr": args.settle_lr,
            "max_batches": args.max_batches,
            "latent_dim": config.latent_dim,
            "max_k": config.max_k,
            "top_k": config.top_k,
            "tau": config.tau,
        },
        "epochs": [],
        "final_accuracy": 0.0,
        "total_time": 0.0,
    }
    
    total_start = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            
            x, y = x.to(device), y.to(device)
            
            if args.mode == "ep":
                opt.step(x=x, target=y)
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
            
            # Track metrics
            with torch.no_grad():
                logits = model(x)
                acc = (logits.argmax(1) == y).float().mean().item()
                epoch_acc += acc
                epoch_loss += F.cross_entropy(logits, y).item()
                n_batches += 1
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches
        
        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                test_correct += (logits.argmax(1) == y).sum().item()
                test_total += x.size(0)
        test_acc = test_correct / test_total
        
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, train_acc={avg_acc:.4f}, test_acc={test_acc:.4f} ({epoch_time:.1f}s)")
        
        results["epochs"].append({
            "epoch": epoch,
            "loss": avg_loss,
            "train_accuracy": avg_acc,
            "test_accuracy": test_acc,
            "time": epoch_time,
        })
        
        results["final_accuracy"] = test_acc
    
    results["total_time"] = time.time() - total_start
    results["final_accuracy"] = results["epochs"][-1]["test_accuracy"] if results["epochs"] else 0.0
    
    print(f"\nTotal time: {results['total_time']:.1f}s")
    print(f"Final test accuracy: {results['final_accuracy']:.4f}")
    
    if args.mode == "ep":
        print(f"\nDecision rule (G6):")
        if results["final_accuracy"] >= 0.85:
            print(f"  ✓ EPOptimizer achieves >85% on NGS")
            print(f"  → Defect is in EqNGSLayer, not NGS+EP. Revive EqNGS paper.")
        elif results["final_accuracy"] >= 0.70:
            print(f"  ⚠ EPOptimizer >70% but <85%. Mixed outcome.")
        else:
            print(f"  ✗ EPOptimizer <70% on NGS")
            print(f"  → NGS+EP is fundamentally broken. Publish negative results.")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

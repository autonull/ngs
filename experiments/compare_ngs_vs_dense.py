#!/usr/bin/env python
"""
NGS vs Dense MLP Comparison — TODO11 Phase C1a

Fair comparison between NGS sparse routing and dense MLP of equivalent parameter count.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

sys.path.insert(0, '/home/me/ngs')
sys.path.insert(0, '/home/me/ngs/bioplausible/mep')

from ngs.models.ngs import NGSModel
from ngs.core.interfaces import NGSConfig, RoutingStrategy


class DenseMLP(nn.Module):
    """Dense MLP with configurable hidden size."""
    
    def __init__(self, d_in: int, d_out: int, hidden_sizes: list):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NGSWrapper(nn.Module):
    """Wrapper for NGSModel to match DenseMLP interface."""
    
    def __init__(self, d_in: int, d_out: int, config: NGSConfig):
        super().__init__()
        self.ngs = NGSModel(d_in, d_out, config)
    
    def forward(self, x):
        out = self.ngs(x)
        return out.logits if hasattr(out, 'logits') else out
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_mnist_loaders(batch_size: int = 128, data_dir: str = "/tmp/mnist"):
    """Get MNIST data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_model(model: nn.Module, train_loader: DataLoader, device: str, 
                epochs: int = 10, lr: float = 0.001) -> dict:
    """Train model with standard backprop."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        
        train_acc = correct / total
        avg_loss = total_loss / total
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch}: loss={avg_loss:.4f}, acc={train_acc:.4f}")
    
    return {"final_train_loss": avg_loss, "final_train_acc": train_acc}


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    
    return correct / total


def count_flops(model: nn.Module, input_shape: tuple) -> int:
    """Rough FLOP count for forward pass."""
    # This is a rough estimate
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += module.in_features * module.out_features * input_shape[0]
    return total


def main():
    parser = argparse.ArgumentParser(description="NGS vs Dense MLP comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", default="results/diagnostics/ngs_vs_dense.json")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    
    # Get data
    train_loader, test_loader = get_mnist_loaders(args.batch_size)
    
    results = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        },
        "models": {},
    }
    
    # --- Configuration 1: NGS with top_k=8, K=32, latent_dim=64 ---
    print("\n" + "="*60)
    print("CONFIG 1: NGS (top_k=8, K=32, latent_dim=64)")
    print("="*60)
    
    config1 = NGSConfig(
        latent_dim=64,
        max_k=32,
        top_k=8,
        k_init=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    )
    ngs1 = NGSWrapper(784, 10, config1)
    ngs_params = ngs1.count_parameters()
    print(f"NGS parameters: {ngs_params}")
    
    # Train NGS
    print("Training NGS...")
    train_stats_ngs1 = train_model(ngs1, train_loader, device, args.epochs, args.lr)
    test_acc_ngs1 = evaluate_model(ngs1, test_loader, device)
    
    print(f"NGS test accuracy: {test_acc_ngs1:.4f}")
    
    results["models"]["ngs_k32_tk8"] = {
        "type": "NGS",
        "config": {"latent_dim": 64, "max_k": 32, "top_k": 8},
        "parameters": ngs_params,
        "train_loss": train_stats_ngs1["final_train_loss"],
        "train_acc": train_stats_ngs1["final_train_acc"],
        "test_acc": test_acc_ngs1,
    }
    
    # --- Configuration 2: Dense MLP with matched parameters ---
    print("\n" + "="*60)
    print(f"CONFIG 2: Dense MLP (matched params ~{ngs_params})")
    print("="*60)
    
    # Find hidden size that gives similar param count
    # 784 * h + h * h + h * 10 ≈ ngs_params
    # For 2 hidden layers: 784*h + h^2 + 10*h ≈ ngs_params
    # h^2 + 794*h - ngs_params = 0
    import math
    h = int((-794 + math.sqrt(794**2 + 4 * ngs_params)) / 2)
    h = max(h, 32)
    
    dense1 = DenseMLP(784, 10, [h, h])
    dense_params = dense1.count_parameters()
    print(f"Dense MLP hidden size: {h}, parameters: {dense_params}")
    
    print("Training Dense MLP...")
    train_stats_dense1 = train_model(dense1, train_loader, device, args.epochs, args.lr)
    test_acc_dense1 = evaluate_model(dense1, test_loader, device)
    
    print(f"Dense test accuracy: {test_acc_dense1:.4f}")
    
    results["models"]["dense_matched"] = {
        "type": "DenseMLP",
        "config": {"hidden_sizes": [h, h]},
        "parameters": dense_params,
        "train_loss": train_stats_dense1["final_train_loss"],
        "train_acc": train_stats_dense1["final_train_acc"],
        "test_acc": test_acc_dense1,
    }
    
    # --- Configuration 3: Dense MLP with 256 hidden (standard) ---
    print("\n" + "="*60)
    print("CONFIG 3: Dense MLP (256 hidden, standard)")
    print("="*60)
    
    dense2 = DenseMLP(784, 10, [256, 256])
    dense2_params = dense2.count_parameters()
    print(f"Dense MLP parameters: {dense2_params}")
    
    print("Training Dense MLP...")
    train_stats_dense2 = train_model(dense2, train_loader, device, args.epochs, args.lr)
    test_acc_dense2 = evaluate_model(dense2, test_loader, device)
    
    print(f"Dense test accuracy: {test_acc_dense2:.4f}")
    
    results["models"]["dense_256"] = {
        "type": "DenseMLP",
        "config": {"hidden_sizes": [256, 256]},
        "parameters": dense2_params,
        "train_loss": train_stats_dense2["final_train_loss"],
        "train_acc": train_stats_dense2["final_train_acc"],
        "test_acc": test_acc_dense2,
    }
    
    # --- Configuration 4: NGS sweep (K=8, 16, 32, 64, top_k=4, 8, 16) ---
    print("\n" + "="*60)
    print("CONFIG 4: NGS sweep over K and top_k")
    print("="*60)
    
    sweep_results = []
    for K in [8, 16, 32, 64, 128]:
        for top_k in [4, 8, 16]:
            if top_k > K:
                continue
            
            config = NGSConfig(
                latent_dim=64,
                max_k=K,
                top_k=top_k,
                k_init=min(8, K),
                routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
            )
            ngs = NGSWrapper(784, 10, config)
            params = ngs.count_parameters()
            
            print(f"  K={K}, top_k={top_k}, params={params}")
            
            train_stats = train_model(ngs, train_loader, device, args.epochs, args.lr)
            test_acc = evaluate_model(ngs, test_loader, device)
            
            sweep_results.append({
                "K": K,
                "top_k": top_k,
                "parameters": params,
                "train_loss": train_stats["final_train_loss"],
                "train_acc": train_stats["final_train_acc"],
                "test_acc": test_acc,
            })
            
            print(f"    Test acc: {test_acc:.4f}")
    
    results["models"]["ngs_sweep"] = sweep_results
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"NGS (K=32, tk=8):     {test_acc_ngs1:.4f} ({ngs_params:,} params)")
    print(f"Dense (matched):      {test_acc_dense1:.4f} ({dense_params:,} params)")
    print(f"Dense (256x2):        {test_acc_dense2:.4f} ({dense2_params:,} params)")
    print(f"Gap (NGS vs matched): {test_acc_ngs1 - test_acc_dense1:+.4f}")
    print(f"Gap (NGS vs 256x2):   {test_acc_ngs1 - test_acc_dense2:+.4f}")
    
    results["summary"] = {
        "ngs_vs_dense_matched_gap": test_acc_ngs1 - test_acc_dense1,
        "ngs_vs_dense_256_gap": test_acc_ngs1 - test_acc_dense2,
        "ngs_params": ngs_params,
        "dense_matched_params": dense_params,
        "dense_256_params": dense2_params,
    }
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
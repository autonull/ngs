#!/usr/bin/env python
"""
Projection Ablation — TODO11 Phase C4

Compare different p_down/p_up projection strategies for NGS:
1. Learned linear (baseline)
2. Fixed random projection (Gaussian)
3. Random Fourier Features (high-dim)
4. Learned MLP projection (non-linear)
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, '/home/me/ngs')
sys.path.insert(0, '/home/me/ngs/bioplausible/mep')

from ngs.models.ngs import NGSModel
from ngs.core.interfaces import NGSConfig, RoutingStrategy, RoutingOutput, BaseRouter


class RandomProjectionNGS(nn.Module):
    """NGS with fixed random p_down projection."""
    
    def __init__(self, d_in: int, d_out: int, d_latent: int, config: NGSConfig, proj_type: str = "random"):
        super().__init__()
        self.d_latent = d_latent
        
        if proj_type == "random":
            self.p_down = nn.Linear(d_in, d_latent, bias=False)
            nn.init.normal_(self.p_down.weight, std=0.1)
            self.p_down.weight.requires_grad_(False)
        elif proj_type == "rff":
            # Random Fourier Features
            self.register_buffer('rff_W', torch.randn(d_in, d_latent) * 0.1)
            self.register_buffer('rff_b', torch.rand(d_latent) * 2 * 3.14159)
        elif proj_type == "mlp":
            self.p_down = nn.Sequential(
                nn.Linear(d_in, d_latent * 2),
                nn.ReLU(),
                nn.Linear(d_latent * 2, d_latent),
            )
        
        self.ngs = NGSModel(d_latent, d_out, config)
        self.proj_type = proj_type
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        if self.proj_type == "rff":
            z = torch.cos(x @ self.rff_W + self.rff_b)
        else:
            z = self.p_down(x)
        
        out = self.ngs(z)
        return out.logits if hasattr(out, 'logits') else out


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
    parser = argparse.ArgumentParser(description="Projection ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="results/diagnostics/projection_ablation.json")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Device: {device}")
    
    train_loader, test_loader = get_mnist_loaders(args.batch_size)
    
    base_config = NGSConfig(
        latent_dim=64, max_k=32, top_k=8, k_init=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    )
    
    projections = [
        ("learned_linear", "learned"),
        ("random_linear", "random"),
        ("random_fourier", "rff"),
        ("mlp_projection", "mlp"),
    ]
    
    results = {
        "config": {"epochs": args.epochs, "batch_size": args.batch_size},
        "results": [],
    }
    
    for name, proj_type in projections:
        print(f"\n--- {name} ---")
        
        if proj_type == "learned":
            model = NGSModel(784, 10, base_config)
        else:
            model = RandomProjectionNGS(
                784, 10, base_config.latent_dim, base_config,
                proj_type=proj_type,
            )
        
        model = model.to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {params}, Total params: {total_params}")
        
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.001,
        )
        
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                logits = out.logits if hasattr(out, 'logits') else out
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                logits = out.logits if hasattr(out, 'logits') else out
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)
        acc = correct / total
        print(f"Test acc: {acc:.4f}")
        
        results["results"].append({
            "name": name,
            "projection_type": proj_type,
            "trainable_params": params,
            "total_params": total_params,
            "test_accuracy": acc,
        })
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

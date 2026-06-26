#!/usr/bin/env python
"""
3DGS Hardness Diagnostic — TODO11 Phase A3

Systematic hardness scaling for 3DGS classification.
Tests NGS vs MLP across varying Gaussian counts, noise, occlusion, and class counts.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, '/home/me/ngs')
sys.path.insert(0, '/home/me/ngs/bioplausible/mep')

from ngs.models.ngs import NGSModel
from ngs.core.interfaces import NGSConfig, RoutingStrategy
from experiments.load_3dgs import create_3dgs_dataset


class SmallMLP(nn.Module):
    """Simple MLP for comparison."""
    
    def __init__(self, d_in: int, d_out: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out),
        )
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class NGSClassifier(nn.Module):
    """NGS-based classifier."""
    
    def __init__(self, d_in: int, d_out: int, config: NGSConfig):
        super().__init__()
        self.ngs = NGSModel(d_in, d_out, config)
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        out = self.ngs(x)
        return out.logits if hasattr(out, 'logits') else out


def generate_synthetic_3dgs(
    num_samples: int,
    num_gaussians: int,
    seed: int = 42,
    noise: float = 0.01,
    occlusion: float = 0.0,
    num_classes: int = 4,
) -> tuple:
    """Generate synthetic 3DGS data.
    
    Returns:
        (X, y) where X shape [N, num_gaussians, 13] (position 3 + rotation 6 + scale 3 + opacity 1)
    """
    torch.manual_seed(seed)
    
    # Each Gaussian: [x, y, z, qx, qy, qz, qw, sx, sy, sz, alpha]
    # For simplicity: [pos_3, rot_6, scale_3, opacity_1] = 13
    
    X_list = []
    y_list = []
    
    for class_idx in range(num_classes):
        # Each class has a characteristic spatial pattern
        for _ in range(num_samples // num_classes):
            centers = torch.randn(num_gaussians, 3) * 2.0 + torch.tensor([class_idx * 2, 0.0, 0.0])
            
            rotations = torch.randn(num_gaussians, 6) * 0.1
            scales = torch.rand(num_gaussians, 3) * 0.5 + 0.1
            opacities = torch.rand(num_gaussians, 1) * 0.5 + 0.5
            
            gaussians = torch.cat([centers, rotations, scales, opacities], dim=-1)
            
            # Add noise
            if noise > 0:
                gaussians = gaussians + torch.randn_like(gaussians) * noise
            
            # Apply occlusion (mask out some Gaussians)
            if occlusion > 0:
                occlude_mask = torch.rand(num_gaussians) < occlusion
                gaussians[occlude_mask] = 0.0
            
            X_list.append(gaussians.flatten())
            y_list.append(class_idx)
    
    X = torch.stack(X_list)
    y = torch.tensor(y_list)
    
    # Shuffle
    perm = torch.randperm(len(X))
    X = X[perm]
    y = y[perm]
    
    return X, y


def main():
    parser = argparse.ArgumentParser(description="3DGS hardness diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--output", default="results/diagnostics/3dgs_hardness.json")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    
    results = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "seed": args.seed,
        },
        "experiments": {},
    }
    
    # --- Experiment 1: Vary Gaussian count ---
    print("\n" + "="*60)
    print("EXPERIMENT 1: Vary Gaussian Count (K)")
    print("="*60)
    
    num_gaussians_list = [1, 4, 8, 16, 32, 64]
    gaussian_results = []
    
    for ng in num_gaussians_list:
        X, y = generate_synthetic_3dgs(
            args.num_samples, ng, seed=args.seed,
            noise=0.01, occlusion=0.0, num_classes=4
        )
        in_features = X.shape[1]
        
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        # NGS
        config = NGSConfig(
            latent_dim=32,
            max_k=min(16, ng * 2),
            top_k=min(4, ng),
            k_init=min(4, ng),
            routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        )
        ngs_model = NGSClassifier(in_features, 4, config)
        ngs_model = ngs_model.to(device)
        
        ngs_optimizer = torch.optim.Adam(ngs_model.parameters(), lr=0.001)
        for epoch in range(args.epochs):
            ngs_model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = ngs_model(xb)
                loss = F.cross_entropy(logits, yb)
                ngs_optimizer.zero_grad()
                loss.backward()
                ngs_optimizer.step()
        
        ngs_model.eval()
        with torch.no_grad():
            logits = ngs_model(X.to(device))
            ngs_acc = (logits.argmax(1) == y.to(device)).float().mean().item()
        
        # MLP
        mlp = SmallMLP(in_features, 4).to(device)
        mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
        for epoch in range(args.epochs):
            mlp.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = mlp(xb)
                loss = F.cross_entropy(logits, yb)
                mlp_optimizer.zero_grad()
                loss.backward()
                mlp_optimizer.step()
        
        mlp.eval()
        with torch.no_grad():
            logits = mlp(X.to(device))
            mlp_acc = (logits.argmax(1) == y.to(device)).float().mean().item()
        
        print(f"  Gaussians={ng}: NGS={ngs_acc:.4f}, MLP={mlp_acc:.4f}")
        gaussian_results.append({
            "num_gaussians": ng,
            "ngs_accuracy": ngs_acc,
            "mlp_accuracy": mlp_acc,
        })
    
    results["experiments"]["vary_gaussians"] = gaussian_results
    
    # --- Experiment 2: Vary Noise ---
    print("\n" + "="*60)
    print("EXPERIMENT 2: Vary Noise Level")
    print("="*60)
    
    noise_levels = [0.01, 0.05, 0.1, 0.5, 1.0]
    noise_results = []
    
    for noise in noise_levels:
        X, y = generate_synthetic_3dgs(
            args.num_samples, 16, seed=args.seed,
            noise=noise, occlusion=0.0, num_classes=4
        )
        in_features = X.shape[1]
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        config = NGSConfig(
            latent_dim=32, max_k=16, top_k=4, k_init=4,
            routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        )
        ngs_model = NGSClassifier(in_features, 4, config)
        ngs_model = ngs_model.to(device)
        
        ngs_optimizer = torch.optim.Adam(ngs_model.parameters(), lr=0.001)
        for epoch in range(args.epochs):
            ngs_model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = ngs_model(xb)
                loss = F.cross_entropy(logits, yb)
                ngs_optimizer.zero_grad()
                loss.backward()
                ngs_optimizer.step()
        
        ngs_model.eval()
        with torch.no_grad():
            logits = ngs_model(X.to(device))
            ngs_acc = (logits.argmax(1) == y.to(device)).float().mean().item()
        
        mlp = SmallMLP(in_features, 4).to(device)
        mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
        for epoch in range(args.epochs):
            mlp.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = mlp(xb)
                loss = F.cross_entropy(logits, yb)
                mlp_optimizer.zero_grad()
                loss.backward()
                mlp_optimizer.step()
        
        mlp.eval()
        with torch.no_grad():
            logits = mlp(X.to(device))
            mlp_acc = (logits.argmax(1) == y.to(device)).float().mean().item()
        
        print(f"  Noise={noise}: NGS={ngs_acc:.4f}, MLP={mlp_acc:.4f}")
        noise_results.append({
            "noise": noise,
            "ngs_accuracy": ngs_acc,
            "mlp_accuracy": mlp_acc,
        })
    
    results["experiments"]["vary_noise"] = noise_results
    
    # --- Experiment 3: Vary Occlusion ---
    print("\n" + "="*60)
    print("EXPERIMENT 3: Vary Occlusion Level")
    print("="*60)
    
    occlusion_levels = [0.0, 0.25, 0.5, 0.75]
    occlusion_results = []
    
    for occlusion in occlusion_levels:
        X, y = generate_synthetic_3dgs(
            args.num_samples, 16, seed=args.seed,
            noise=0.01, occlusion=occlusion, num_classes=4
        )
        in_features = X.shape[1]
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        config = NGSConfig(
            latent_dim=32, max_k=16, top_k=4, k_init=4,
            routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        )
        ngs_model = NGSClassifier(in_features, 4, config)
        ngs_model = ngs_model.to(device)
        
        ngs_optimizer = torch.optim.Adam(ngs_model.parameters(), lr=0.001)
        for epoch in range(args.epochs):
            ngs_model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = ngs_model(xb)
                loss = F.cross_entropy(logits, yb)
                ngs_optimizer.zero_grad()
                loss.backward()
                ngs_optimizer.step()
        
        ngs_model.eval()
        with torch.no_grad():
            logits = ngs_model(X.to(device))
            ngs_acc = (logits.argmax(1) == y.to(device)).float().mean().item()
        
        mlp = SmallMLP(in_features, 4).to(device)
        mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
        for epoch in range(args.epochs):
            mlp.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = mlp(xb)
                loss = F.cross_entropy(logits, yb)
                mlp_optimizer.zero_grad()
                loss.backward()
                mlp_optimizer.step()
        
        mlp.eval()
        with torch.no_grad():
            logits = mlp(X.to(device))
            mlp_acc = (logits.argmax(1) == y.to(device)).float().mean().item()
        
        print(f"  Occlusion={occlusion}: NGS={ngs_acc:.4f}, MLP={mlp_acc:.4f}")
        occlusion_results.append({
            "occlusion": occlusion,
            "ngs_accuracy": ngs_acc,
            "mlp_accuracy": mlp_acc,
        })
    
    results["experiments"]["vary_occlusion"] = occlusion_results
    
    # --- Experiment 4: Vary Class Count ---
    print("\n" + "="*60)
    print("EXPERIMENT 4: Vary Number of Classes")
    print("="*60)
    
    class_counts = [2, 4, 8]
    class_results = []
    
    for num_classes in class_counts:
        X, y = generate_synthetic_3dgs(
            args.num_samples, 16, seed=args.seed,
            noise=0.01, occlusion=0.0, num_classes=num_classes
        )
        in_features = X.shape[1]
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        config = NGSConfig(
            latent_dim=32, max_k=16, top_k=4, k_init=4,
            routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        )
        ngs_model = NGSClassifier(in_features, 4, config)
        ngs_model = ngs_model.to(device)
        ngs_model = ngs_model.to(device)
        ngs_model = ngs_model.to(device)
        
        ngs_optimizer = torch.optim.Adam(ngs_model.parameters(), lr=0.001)
        for epoch in range(args.epochs):
            ngs_model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = ngs_model(xb)
                loss = F.cross_entropy(logits, yb)
                ngs_optimizer.zero_grad()
                loss.backward()
                ngs_optimizer.step()
        
        ngs_model.eval()
        with torch.no_grad():
            logits = ngs_model(X.to(device))
            ngs_acc = (logits.argmax(1) == y.to(device)).float().mean().item()
        
        mlp = SmallMLP(in_features, num_classes).to(device)
        mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
        for epoch in range(args.epochs):
            mlp.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = mlp(xb)
                loss = F.cross_entropy(logits, yb)
                mlp_optimizer.zero_grad()
                loss.backward()
                mlp_optimizer.step()
        
        mlp.eval()
        with torch.no_grad():
            logits = mlp(X.to(device))
            mlp_acc = (logits.argmax(1) == y.to(device)).float().mean().item()
        
        print(f"  Classes={num_classes}: NGS={ngs_acc:.4f}, MLP={mlp_acc:.4f}")
        class_results.append({
            "num_classes": num_classes,
            "ngs_accuracy": ngs_acc,
            "mlp_accuracy": mlp_acc,
        })
    
    results["experiments"]["vary_classes"] = class_results
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    results["summary"] = {}
    for exp_name, exp_data in results["experiments"].items():
        # Compute average NGS advantage
        ngs_advantage = [e["ngs_accuracy"] - e["mlp_accuracy"] for e in exp_data]
        avg_adv = sum(ngs_advantage) / len(ngs_advantage)
        results["summary"][exp_name] = {
            "avg_ngs_advantage": avg_adv,
            "ngs_wins": sum(1 for a in ngs_advantage if a > 0),
            "total": len(ngs_advantage),
        }
        print(f"  {exp_name}: avg NGS advantage = {avg_adv:.4f} (NGS wins {results['summary'][exp_name]['ngs_wins']}/{len(ngs_advantage)})")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

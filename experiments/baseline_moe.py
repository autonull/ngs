#!/usr/bin/env python
"""
Sparse MoE Baseline — TODO11 Phase C1c

Compare NGS routing against standard TopK MoE (Switch Transformer style).
Same K experts, same top_k routing, same parameter count, backprop training.
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
from ngs.core.interfaces import NGSConfig, RoutingStrategy


class TopKMoE(nn.Module):
    """Standard TopK MoE with dot-product routing."""
    
    def __init__(self, d_in: int, d_out: int, num_experts: int, top_k: int, d_expert: int = 128):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_expert = d_expert
        
        # Router: projects input to expert logits
        self.router = nn.Linear(d_in, num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in, d_expert),
                nn.ReLU(),
                nn.Linear(d_expert, d_out),
            )
            for _ in range(num_experts)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_out, d_out)
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Router logits
        router_logits = self.router(x)  # [B, num_experts]
        
        # Top-k routing
        topk_vals, topk_idx = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)
        
        # Compute expert outputs
        batch_outputs = []
        for b in range(x.size(0)):
            expert_out = torch.zeros(self.d_expert if hasattr(self.experts[0], 'out_features') else 128, 
                                      device=x.device)
            for k in range(self.top_k):
                expert_idx = topk_idx[b, k].item()
                weight = topk_weights[b, k]
                out = self.experts[expert_idx](x[b:b+1])
                batch_outputs.append(weight * out)
        
        # Simplified: use all experts via averaging
        # Actually implement proper sparse computation
        B = x.size(0)
        output = torch.zeros(B, self.experts[0](x[:1]).size(-1), device=x.device)
        
        # For efficiency, compute all then mask (small models)
        for k in range(self.top_k):
            expert_idx = topk_idx[:, k]  # [B]
            weight = topk_weights[:, k].unsqueeze(-1)  # [B, 1]
            
            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    expert_out = self.experts[e_idx](x[mask])
                    output[mask] += weight[mask] * expert_out
        
        return self.output_proj(output)


class NGSEqualParam(nn.Module):
    """NGS model with same parameter budget as MoE."""
    
    def __init__(self, d_in: int, d_out: int, config: NGSConfig):
        super().__init__()
        self.ngs = NGSModel(d_in, d_out, config)
    
    def forward(self, x):
        out = self.ngs(x)
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


def train_and_eval(model, train_loader, test_loader, device, epochs=10, lr=0.001):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
    
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


def main():
    parser = argparse.ArgumentParser(description="NGS vs MoE baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="results/diagnostics/baseline_moe.json")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Device: {device}")
    
    train_loader, test_loader = get_mnist_loaders(args.batch_size)
    
    results = {
        "config": {"epochs": args.epochs, "batch_size": args.batch_size},
        "comparisons": [],
    }
    
    # Configurations: N experts / NGS K variants
    configs = [
        {"num_experts": 8, "top_k": 4, "ngs_K": 8, "ngs_top_k": 4, "latent_dim": 64},
        {"num_experts": 16, "top_k": 4, "ngs_K": 16, "ngs_top_k": 4, "latent_dim": 64},
        {"num_experts": 32, "top_k": 8, "ngs_K": 32, "ngs_top_k": 8, "latent_dim": 64},
    ]
    
    for cfg in configs:
        print(f"\n--- MoE({cfg['num_experts']} experts, tk={cfg['top_k']}) vs NGS(K={cfg['ngs_K']}, tk={cfg['ngs_top_k']}) ---")
        
        # MoE
        moe = TopKMoE(784, 10, cfg['num_experts'], cfg['top_k'])
        moe_params = sum(p.numel() for p in moe.parameters())
        print(f"MoE params: {moe_params}")
        moe_acc = train_and_eval(moe, train_loader, test_loader, device, args.epochs)
        print(f"MoE test acc: {moe_acc:.4f}")
        
        # NGS
        ngs_cfg = NGSConfig(
            latent_dim=cfg['latent_dim'],
            max_k=cfg['ngs_K'],
            top_k=cfg['ngs_top_k'],
            k_init=min(8, cfg['ngs_K']),
            routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        )
        ngs = NGSEqualParam(784, 10, ngs_cfg)
        ngs_params = sum(p.numel() for p in ngs.parameters())
        print(f"NGS params: {ngs_params}")
        ngs_acc = train_and_eval(ngs, train_loader, test_loader, device, args.epochs)
        print(f"NGS test acc: {ngs_acc:.4f}")
        
        # Dense baseline (single MLP with comparable params)
        h = max(32, int((moe_params / 2) ** 0.5))
        dense = nn.Sequential(
            nn.Linear(784, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 10),
        )
        dense_params = sum(p.numel() for p in dense.parameters())
        print(f"Dense params: {dense_params}")
        dense_acc = train_and_eval(dense, train_loader, test_loader, device, args.epochs)
        print(f"Dense test acc: {dense_acc:.4f}")
        
        results["comparisons"].append({
            "num_experts": cfg['num_experts'],
            "top_k": cfg['top_k'],
            "ngs_K": cfg['ngs_K'],
            "ngs_top_k": cfg['ngs_top_k'],
            "moe_params": moe_params,
            "ngs_params": ngs_params,
            "dense_params": dense_params,
            "moe_accuracy": moe_acc,
            "ngs_accuracy": ngs_acc,
            "dense_accuracy": dense_acc,
        })
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

"""Recursive NGS on own weights benchmark (Experiment 3C).
Meta-NGS 'splats' improvements on its own parameters.
Target: Self-compression, autonomous representation discovery."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import json
from copy import deepcopy


class MetaNGS(nn.Module):
    """Meta-NGS that operates on another NGS's parameters."""
    
    def __init__(self, target_ngs_config, meta_config):
        super().__init__()
        from ngs.models import build_ngs
        from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
        
        # The target NGS (to be optimized)
        self.target_ngs = build_ngs(784, 10, target_ngs_config)
        
        # Meta-NGS: takes flattened target parameters as input, outputs parameter updates
        target_params = sum(p.numel() for p in self.target_ngs.parameters())
        meta_input_dim = target_params
        meta_output_dim = target_params
        
        self.meta_ngs = build_ngs(meta_input_dim, meta_output_dim, meta_config)
        
    def forward(self, x):
        """Forward through target NGS."""
        return self.target_ngs(x)
    
    def meta_step(self, loss):
        """Use meta-NGS to generate parameter updates based on loss."""
        # Flatten target parameters
        target_params = torch.cat([p.data.flatten() for p in self.target_ngs.parameters()])
        target_grads = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p.data) for p in self.target_ngs.parameters()])
        
        # Meta input: [params, grads, loss]
        meta_input = torch.cat([
            target_params.unsqueeze(0),
            target_grads.unsqueeze(0),
            torch.tensor([[loss]], device=target_params.device)
        ], dim=-1)
        
        # Meta-NGS predicts parameter updates
        with torch.no_grad():
            meta_out = self.meta_ngs(meta_input)
            updates = meta_out.logits.squeeze(0)
        
        # Apply updates to target
        idx = 0
        for p in self.target_ngs.parameters():
            n = p.numel()
            p.data.add_(updates[idx:idx+n].view_as(p.data))
            idx += n


def run_recursive_ngs_benchmark(
    epochs: int = 20,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./recursive_ngs_results",
    latent_dim: int = 32,
    k_init: int = 16,
    max_k: int = 64,
    lr: float = 1e-3,
    meta_lr: float = 1e-4,
    batch_size: int = 64,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running recursive NGS benchmark")

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    
    # Target NGS config - small for meta-NGS compatibility
    target_config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=4,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,  # Has codes
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.DYNAMIC,
        hypernetwork_code_dim=8,
        hypernetwork_hidden_dim=32,
    )
    
    # Meta NGS config - operates on hypernetwork codes (much smaller)
    meta_config = NGSConfig(
        latent_dim=16,
        k_init=8,
        max_k=32,
        top_k=2,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.PRE_ALLOCATED,
    )
    
    # Simpler approach: target NGS with hypernetwork, meta-NGS operates on codes
    from ngs.models import build_ngs
    
    target_ngs = build_ngs(784, 10, target_config).to(device)
    meta_ngs = build_ngs(target_config.max_k * target_config.hypernetwork_code_dim, 
                         target_config.max_k * target_config.hypernetwork_code_dim, 
                         meta_config).to(device)
    
    # Optimizers
    target_optimizer = torch.optim.Adam(target_ngs.parameters(), lr=lr)
    meta_optimizer = torch.optim.Adam(meta_ngs.parameters(), lr=meta_lr)
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    criterion = nn.CrossEntropyLoss()
    
    results = {
        "train_losses": [],
        "test_accs": [],
        "target_K": [],
        "meta_K": [],
    }
    
    for epoch in range(epochs):
        target_ngs.train()
        epoch_losses = []
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            
            # Target forward
            out_obj = target_ngs(x)
            logits = out_obj.logits
            loss = criterion(logits, y)
            
            # Target backward
            target_optimizer.zero_grad()
            loss.backward()
            target_optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Meta step: meta-NGS operates on hypernetwork codes
        if epoch > 0 and epoch % 5 == 0:
            meta_ngs.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                x = x.view(x.size(0), -1)
                
                out_obj = target_ngs(x)
                logits = out_obj.logits
                loss = criterion(logits, y)
                
                # Get hypernetwork codes
                codes = target_ngs.param_store.codes.data.clone().flatten().unsqueeze(0)  # [1, K*code_dim]
                
                # Meta forward
                meta_optimizer.zero_grad()
                meta_out = meta_ngs(codes)
                meta_loss = F.mse_loss(meta_out.logits, codes)  # Self-reconstruction
                
                meta_loss.backward()
                meta_optimizer.step()
                
                # Apply meta-updates to codes
                with torch.no_grad():
                    updated_codes = meta_out.logits.squeeze(0).view_as(target_ngs.param_store.codes.data)
                    target_ngs.param_store.codes.data.add_((updated_codes - target_ngs.param_store.codes.data) * 0.1)
                
                break
        
        # Evaluate
        target_ngs.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                x = x.view(x.size(0), -1)
                out_obj = target_ngs(x)
                logits = out_obj.logits
                _, pred = logits.max(1)
                total += y.size(0)
                correct += pred.eq(y).sum().item()
        
        test_acc = correct / total
        avg_loss = np.mean(epoch_losses)
        
        results["train_losses"].append(avg_loss)
        results["test_accs"].append(test_acc)
        results["target_K"].append(target_ngs.K)
        results["meta_K"].append(meta_ngs.K)
        
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, acc={test_acc:.4f}, target_K={target_ngs.K}, meta_K={meta_ngs.K}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "recursive_ngs_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    run_recursive_ngs_benchmark(epochs=args.epochs, device=args.device)
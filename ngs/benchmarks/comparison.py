"""Comparison benchmarks: NGS vs ProtoNet vs MAML vs Fine-tuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import copy

from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models import build_ngs
from ngs.training import NGSTrainer, TrainerConfig


class ProtoNet(nn.Module):
    """Prototypical Network baseline."""
    
    def __init__(self, d_in: int, d_latent: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Linear(128, d_latent),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MAMLModel(nn.Module):
    """MAML-compatible model with inner-loop adaptation."""
    
    def __init__(self, d_in: int, d_latent: int = 64, n_way: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Linear(128, d_latent),
        )
        self.classifier = nn.Linear(d_latent, n_way)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))
    
    def clone(self):
        """Create a copy for inner-loop adaptation."""
        clone = copy.deepcopy(self)
        return clone


def generate_tasks(n_way: int, k_shot: int, n_tasks: int, n_query: int = 15, d: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic few-shot tasks."""
    all_x, all_y = [], []
    
    for task in range(n_tasks):
        centers = torch.randn(n_way, d) * 5.0
        for c in range(n_way):
            support = centers[c].unsqueeze(0) + torch.randn(k_shot, d) * 0.5
            all_x.append(support)
            all_y.extend([task * n_way + c] * k_shot)
            
            query = centers[c].unsqueeze(0) + torch.randn(n_query, d) * 0.5
            all_x.append(query)
            all_y.extend([task * n_way + c] * n_query)
    
    return torch.cat(all_x, dim=0), torch.tensor(all_y)


def evaluate_protonet(model: ProtoNet, n_way: int, k_shot: int, n_tasks: int, 
                      n_query: int, device: torch.device) -> float:
    """Evaluate ProtoNet on tasks."""
    model.eval()
    acc = 0.0
    
    with torch.no_grad():
        for _ in range(n_tasks):
            x, y = generate_tasks(n_way, k_shot, 1, n_query)
            x, y = x.to(device), y.to(device)
            
            n_support = n_way * k_shot
            support_x, support_y = x[:n_support], y[:n_support]
            query_x, query_y = x[n_support:], y[n_support:] % n_way
            
            support_z = model(support_x)
            query_z = model(query_x)
            
            prototypes = torch.zeros(n_way, support_z.shape[1], device=device)
            for c in range(n_way):
                mask = (support_y == c)
                if mask.any():
                    prototypes[c] = support_z[mask].mean(0)
            
            logits = -torch.cdist(query_z, prototypes)
            preds = logits.argmax(1)
            acc += (preds == query_y).float().mean().item()
    
    return acc / n_tasks


def evaluate_maml(model: MAMLModel, n_way: int, k_shot: int, n_tasks: int,
                  n_query: int, device: torch.device, inner_lr: float = 0.1,
                  inner_steps: int = 5) -> float:
    """Evaluate MAML with inner-loop adaptation."""
    acc = 0.0
    
    for _ in range(n_tasks):
        x, y = generate_tasks(n_way, k_shot, 1, n_query)
        x, y = x.to(device), y.to(device)
        
        n_support = n_way * k_shot
        support_x, support_y = x[:n_support], y[:n_support] % n_way
        query_x, query_y = x[n_support:], y[n_support:] % n_way
        
        adapted = model.clone().to(device)
        opt = torch.optim.SGD(adapted.parameters(), lr=inner_lr)
        
        for _ in range(inner_steps):
            logits = adapted(support_x)
            loss = F.cross_entropy(logits, support_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        with torch.no_grad():
            query_logits = adapted(query_x)
            preds = query_logits.argmax(1)
            acc += (preds == query_y).float().mean().item()
    
    return acc / n_tasks


def evaluate_finetune(model: nn.Module, n_way: int, k_shot: int, n_tasks: int,
                      n_query: int, device: torch.device, lr: float = 1e-3,
                      steps: int = 50) -> float:
    """Evaluate standard fine-tuning on support set."""
    model.train()
    acc = 0.0
    
    for _ in range(n_tasks):
        x, y = generate_tasks(n_way, k_shot, 1, n_query)
        x, y = x.to(device), y.to(device)
        
        n_support = n_way * k_shot
        support_x, support_y = x[:n_support], y[:n_support] % n_way
        query_x, query_y = x[n_support:], y[n_support:] % n_way
        
        adapted = copy.deepcopy(model).to(device)
        opt = torch.optim.Adam(adapted.parameters(), lr=lr)
        
        for _ in range(steps):
            logits = adapted(support_x)
            loss = F.cross_entropy(logits, support_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        adapted.eval()
        with torch.no_grad():
            query_logits = adapted(query_x)
            preds = query_logits.argmax(1)
            acc += (preds == query_y).float().mean().item()
    
    return acc / n_tasks


def train_ngs_meta(n_way: int, k_shot: int, epochs: int, device: torch.device,
                   latent_dim: int = 64, meta_lr: float = 1e-3) -> nn.Module:
    """Train NGS as meta-feature extractor."""
    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=16,
        max_k=128,
        top_k=8,
        routing=RoutingStrategy.FACTORIZED_SUBSPACE,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.PRE_ALLOCATED_MASKED,
        num_subspaces=4,
        hypernetwork_code_dim=8,
        hypernetwork_hidden_dim=32,
    )
    
    model = build_ngs(2, latent_dim, config).to(device)
    classifier = nn.Linear(latent_dim, n_way).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=meta_lr)
    
    for epoch in range(epochs):
        model.train()
        classifier.train()
        
        x, y = generate_tasks(n_way, k_shot, n_tasks=32, n_query=10)
        x, y = x.to(device), y.to(device)
        
        n_support = n_way * k_shot
        support_x, support_y = x[:n_support], y[:n_support]
        query_x, query_y = x[n_support:n_support+n_way*10], y[n_support:n_support+n_way*10]
        
        support_features = model(support_x)
        query_features = model(query_x)
        
        prototypes = torch.zeros(n_way, latent_dim, device=device)
        for c in range(n_way):
            mask = (support_y == c)
            if mask.any():
                prototypes[c] = support_features[mask].mean(0)
        
        logits = -torch.cdist(query_features, prototypes)
        loss = F.cross_entropy(logits, query_y % n_way)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model


def evaluate_ngs(model: nn.Module, n_way: int, k_shot: int, n_tasks: int,
                 n_query: int, device: torch.device) -> float:
    """Evaluate NGS with prototypical classification."""
    model.eval()
    acc = 0.0
    
    with torch.no_grad():
        for _ in range(n_tasks):
            x, y = generate_tasks(n_way, k_shot, 1, n_query)
            x, y = x.to(device), y.to(device)
            
            n_support = n_way * k_shot
            support_x, support_y = x[:n_support], y[:n_support]
            query_x, query_y = x[n_support:], y[n_support:] % n_way
            
            support_features = model(support_x)
            query_features = model(query_x)
            
            prototypes = torch.zeros(n_way, support_features.shape[1], device=device)
            for c in range(n_way):
                mask = (support_y == c)
                if mask.any():
                    prototypes[c] = support_features[mask].mean(0)
            
            logits = -torch.cdist(query_features, prototypes)
            preds = logits.argmax(1)
            acc += (preds == query_y).float().mean().item()
    
    return acc / n_tasks


def run_comparison_benchmark(
    n_way: int = 5,
    k_shot: int = 1,
    epochs: int = 50,
    n_test_tasks: int = 100,
    n_query: int = 15,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./comparison_results",
) -> Dict[str, Any]:
    """Run comparison: NGS vs ProtoNet vs MAML vs Fine-tuning."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running comparison: {n_way}-way {k_shot}-shot on {device}")
    
    results = {}
    
    # Train NGS
    print("\n=== Training NGS ===")
    ngs_model = train_ngs_meta(n_way, k_shot, epochs, device)
    ngs_acc = evaluate_ngs(ngs_model, n_way, k_shot, n_test_tasks, n_query, device)
    results["ngs"] = {"test_acc": ngs_acc}
    print(f"NGS: {ngs_acc:.4f}")
    
    # Train ProtoNet
    print("\n=== Training ProtoNet ===")
    protonet = ProtoNet(2, 64).to(device)
    opt = torch.optim.Adam(protonet.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        protonet.train()
        x, y = generate_tasks(n_way, k_shot, 32, 10)
        x, y = x.to(device), y.to(device)
        
        n_support = n_way * k_shot
        support_x, support_y = x[:n_support], y[:n_support]
        query_x, query_y = x[n_support:n_support+n_way*10], y[n_support:n_support+n_way*10]
        
        support_z = protonet(support_x)
        query_z = protonet(query_x)
        
        prototypes = torch.zeros(n_way, 64, device=device)
        for c in range(n_way):
            mask = (support_y == c)
            if mask.any():
                prototypes[c] = support_z[mask].mean(0)
        
        logits = -torch.cdist(query_z, prototypes)
        loss = F.cross_entropy(logits, query_y % n_way)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    protonet_acc = evaluate_protonet(protonet, n_way, k_shot, n_test_tasks, n_query, device)
    results["protonet"] = {"test_acc": protonet_acc}
    print(f"ProtoNet: {protonet_acc:.4f}")
    
    # Train MAML
    print("\n=== Training MAML ===")
    maml = MAMLModel(2, 64, n_way).to(device)
    meta_opt = torch.optim.Adam(maml.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        maml.train()
        task_losses = []
        
        for _ in range(16):  # meta-batch
            x, y = generate_tasks(n_way, k_shot, 1, 10)
            x, y = x.to(device), y.to(device)
            
            n_support = n_way * k_shot
            support_x, support_y = x[:n_support], y[:n_support] % n_way
            query_x, query_y = x[n_support:], y[n_support:] % n_way
            
            adapted = maml.clone().to(device)
            inner_opt = torch.optim.SGD(adapted.parameters(), lr=0.1)
            
            for _ in range(5):
                inner_logits = adapted(support_x)
                inner_loss = F.cross_entropy(inner_logits, support_y)
                inner_opt.zero_grad()
                inner_loss.backward()
                inner_opt.step()
            
            query_logits = adapted(query_x)
            query_loss = F.cross_entropy(query_logits, query_y)
            task_losses.append(query_loss)
        
        meta_loss = torch.stack(task_losses).mean()
        meta_opt.zero_grad()
        meta_loss.backward()
        meta_opt.step()
    
    maml_acc = evaluate_maml(maml, n_way, k_shot, n_test_tasks, n_query, device)
    results["maml"] = {"test_acc": maml_acc}
    print(f"MAML: {maml_acc:.4f}")
    
    # Train Fine-tuning baseline
    print("\n=== Training Fine-tuning baseline ===")
    finetune_model = MAMLModel(2, 64, n_way).to(device)
    opt = torch.optim.Adam(finetune_model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        finetune_model.train()
        x, y = generate_tasks(n_way, k_shot, 32, 10)
        x, y = x.to(device), y.to(device)
        
        n_support = n_way * k_shot
        support_x, support_y = x[:n_support], y[:n_support] % n_way
        
        logits = finetune_model(support_x)
        loss = F.cross_entropy(logits, support_y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    ft_acc = evaluate_finetune(finetune_model, n_way, k_shot, n_test_tasks, n_query, device)
    results["finetune"] = {"test_acc": ft_acc}
    print(f"Fine-tuning: {ft_acc:.4f}")
    
    # Summary
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    for name, res in results.items():
        print(f"{name:12s}: {res['test_acc']:.4f}")
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / f"comparison_{n_way}way_{k_shot}shot.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-test-tasks", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./comparison_results")
    args = parser.parse_args()
    
    run_comparison_benchmark(
        n_way=args.n_way,
        k_shot=args.k_shot,
        epochs=args.epochs,
        n_test_tasks=args.n_test_tasks,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir
    )
"""Few-shot learning benchmarks for NGS."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
from pathlib import Path
import json

from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models import build_ngs
from ngs.training import NGSTrainer, TrainerConfig


def generate_omniglot_tasks(n_way: int, k_shot: int, n_tasks: int, n_query: int = 15) -> tuple:
    """Generate synthetic Omniglot-like tasks for testing."""
    # Generate synthetic data: n_way classes, each with k_shot + n_query samples
    # Each class is a Gaussian cluster in 2D
    d = 2
    all_x = []
    all_y = []
    
    for task in range(n_tasks):
        # Random class centers
        centers = torch.randn(n_way, d) * 5.0
        
        for c in range(n_way):
            # Support samples
            support = centers[c].unsqueeze(0) + torch.randn(k_shot, d) * 0.5
            all_x.append(support)
            all_y.extend([task * n_way + c] * k_shot)
            
            # Query samples
            query = centers[c].unsqueeze(0) + torch.randn(n_query, d) * 0.5
            all_x.append(query)
            all_y.extend([task * n_way + c] * n_query)
    
    x = torch.cat(all_x, dim=0)
    y = torch.tensor(all_y)
    return x, y


class PrototypicalNetwork(nn.Module):
    """Simple prototypical network for few-shot classification."""
    
    def __init__(self, d_in: int, d_latent: int, n_way: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.ReLU(),
            nn.Linear(64, d_latent),
        )
        self.n_way = n_way
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def run_fewshot_benchmark(
    dataset: str = "omniglot",
    n_way: int = 5,
    k_shot: int = 1,
    epochs: int = 50,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./fewshot_results",
    latent_dim: int = 64,
    meta_lr: float = 1e-3,
    inner_lr: float = 0.1,
) -> Dict[str, Any]:
    """Run NGS few-shot learning benchmark (simplified with synthetic data)."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running few-shot on {dataset}: {n_way}-way {k_shot}-shot using {device}")
    
    # Generate synthetic tasks
    n_meta_train = 1000
    n_meta_test = 100
    
    # Build NGS model as feature extractor
    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=16,
        max_k=128,
        top_k=8,
        routing=RoutingStrategy.FACTORIZED_SUBSPACE,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.PRE_ALLOCATED,
        num_subspaces=4,
        hypernetwork_code_dim=8,
        hypernetwork_hidden_dim=32,
    )
    
    # NGS as feature extractor
    model = build_ngs(2, latent_dim, config).to(device)
    
    # Simple linear classifier on top
    classifier = nn.Linear(latent_dim, n_way).to(device)
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=meta_lr)
    
    # Training loop
    acc_history = []
    
    for epoch in range(epochs):
        model.train()
        classifier.train()
        
        # Generate meta-training tasks
        x, y = generate_omniglot_tasks(n_way, k_shot, n_tasks=32, n_query=10)
        x, y = x.to(device), y.to(device)
        
        # Split support/query
        n_support = n_way * k_shot
        n_query_total = n_way * 10
        
        support_x = x[:n_support]
        support_y = y[:n_support]
        query_x = x[n_support:n_support+n_query_total]
        query_y = y[n_support:n_support+n_query_total]
        
        # Forward pass
        support_output = model(support_x)
        query_output = model(query_x)
        support_features = support_output.logits if hasattr(support_output, 'logits') else support_output
        query_features = query_output.logits if hasattr(query_output, 'logits') else query_output
        
        # Compute prototypes
        prototypes = torch.zeros(n_way, latent_dim, device=device)
        for c in range(n_way):
            mask = (support_y == c)
            if mask.any():
                prototypes[c] = support_features[mask].mean(0)
        
        # Classify queries
        logits = torch.cdist(query_features, prototypes)
        logits = -logits  # Distance to similarity
        
        # Loss
        loss = F.cross_entropy(logits, query_y % n_way)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate
        if epoch % 10 == 0:
            model.eval()
            classifier.eval()
            with torch.no_grad():
                test_accs = []
                for _ in range(20):
                    x_test, y_test = generate_omniglot_tasks(n_way, k_shot, n_tasks=1, n_query=15)
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    
                    n_support = n_way * k_shot
                    support_x = x_test[:n_support]
                    support_y = y_test[:n_support]
                    query_x = x_test[n_support:]
                    query_y = y_test[n_support:] % n_way
                    
                    support_output = model(support_x)
                    query_output = model(query_x)
                    support_features = support_output.logits if hasattr(support_output, 'logits') else support_output
                    query_features = query_output.logits if hasattr(query_output, 'logits') else query_output
                    
                    prototypes = torch.zeros(n_way, latent_dim, device=device)
                    for c in range(n_way):
                        mask = (support_y == c)
                        if mask.any():
                            prototypes[c] = support_features[mask].mean(0)
                    
                    logits = -torch.cdist(query_features, prototypes)
                    preds = logits.argmax(1)
                    acc = (preds == query_y).float().mean().item()
                    test_accs.append(acc)
                
                mean_acc = np.mean(test_accs)
                acc_history.append(mean_acc)
                print(f"Epoch {epoch}: Test Acc = {mean_acc:.4f}")
    
    # Final evaluation
    model.eval()
    classifier.eval()
    with torch.no_grad():
        test_accs = []
        for _ in range(100):
            x_test, y_test = generate_omniglot_tasks(n_way, k_shot, n_tasks=1, n_query=15)
            x_test, y_test = x_test.to(device), y_test.to(device)
            
            n_support = n_way * k_shot
            support_x = x_test[:n_support]
            support_y = y_test[:n_support]
            query_x = x_test[n_support:]
            query_y = y_test[n_support:] % n_way
            
            support_output = model(support_x)
            query_output = model(query_x)
            support_features = support_output.logits if hasattr(support_output, 'logits') else support_output
            query_features = query_output.logits if hasattr(query_output, 'logits') else query_output
            
            prototypes = torch.zeros(n_way, latent_dim, device=device)
            for c in range(n_way):
                mask = (support_y == c)
                if mask.any():
                    prototypes[c] = support_features[mask].mean(0)
            
            logits = -torch.cdist(query_features, prototypes)
            preds = logits.argmax(1)
            acc = (preds == query_y).float().mean().item()
            test_accs.append(acc)
        
        final_acc = np.mean(test_accs)
        final_std = np.std(test_accs)
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {
        "dataset": dataset,
        "n_way": n_way,
        "k_shot": k_shot,
        "test_acc": final_acc,
        "test_acc_std": final_std,
        "acc_history": acc_history,
        "config": {
            "latent_dim": latent_dim,
            "epochs": epochs,
            "meta_lr": meta_lr,
        }
    }
    
    with open(Path(output_dir) / f"{dataset}_{n_way}way_{k_shot}shot_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFinal Test Accuracy: {final_acc:.4f} ± {final_std:.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="omniglot", choices=["omniglot", "miniimagenet"])
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./fewshot_results")
    args = parser.parse_args()
    
    run_fewshot_benchmark(
        dataset=args.dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir
    )
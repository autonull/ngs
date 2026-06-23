"""Real Omniglot few-shot benchmark (Experiment 2D).
Tests dynamic head + meta-learning: hypernetwork generates adapters for new classes.
Target: >95% accuracy, open-set AUROC >0.9."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json

try:
    from experiments.datasets import get_omniglot_loaders, OmniglotDataset
    OMNIGLOT_AVAILABLE = True
except ImportError:
    OMNIGLOT_AVAILABLE = False


class OmniglotFewShotModel(nn.Module):
    """NGS-based few-shot model with hypernetwork-generated adapters."""
    def __init__(self, config, n_way: int = 5):
        super().__init__()
        self.n_way = n_way
        self.config = config
        
        from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
        from ngs.models import build_ngs
        
        # Feature extractor (shared backbone)
        self.backbone = build_ngs(784, config.latent_dim, config)
        
        # Hypernetwork for generating classifier weights
        self.adapter_dim = config.hypernetwork_code_dim
        self.hypernet = nn.Sequential(
            nn.Linear(config.latent_dim + self.adapter_dim, config.hypernetwork_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hypernetwork_hidden_dim, config.hypernetwork_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hypernetwork_hidden_dim, config.latent_dim * n_way),
        )
        
        # Class codes (one per class in the episode)
        self.class_codes = nn.Parameter(torch.randn(n_way, self.adapter_dim) * 0.1)
        
        # Prototypical baseline
        self.prototypes = None
    
    def encode(self, x):
        """Encode images to latent features."""
        out_obj = self.backbone(x)
        return out_obj.latent
    
    def generate_classifiers(self, features):
        """Generate classifier weights from features + class codes."""
        # features: [B, d_latent]
        # class_codes: [n_way, adapter_dim]
        
        B = features.size(0)
        # Expand for all classes
        features_exp = features.unsqueeze(1).expand(B, self.n_way, -1)  # [B, n_way, d]
        codes_exp = self.class_codes.unsqueeze(0).expand(B, -1, -1)  # [B, n_way, adapter]
        
        combined = torch.cat([features_exp, codes_exp], dim=-1)  # [B, n_way, d + adapter]
        logits = self.hypernet(combined)  # [B, n_way, d * n_way]
        
        # Reshape to classifier weights: [B, n_way, d]
        logits = logits.view(B, self.n_way, -1)
        return logits
    
    def forward(self, x, support_features=None, support_labels=None):
        """Forward pass with optional prototypical adaptation."""
        features = self.encode(x)  # [B, d]
        
        if support_features is not None and support_labels is not None:
            # Prototypical adaptation: compute prototypes
            self.prototypes = []
            for c in range(self.n_way):
                mask = support_labels == c
                if mask.any():
                    proto = support_features[mask].mean(0)
                else:
                    proto = torch.zeros_like(features[0])
                self.prototypes.append(proto)
            self.prototypes = torch.stack(self.prototypes)  # [n_way, d]
            
            # Compute distances to prototypes
            logits = -torch.cdist(features, self.prototypes, p=2) ** 2
            return logits, features
        
        # Use hypernetwork-generated classifiers
        classifier_weights = self.generate_classifiers(features)  # [B, n_way, d]
        logits = torch.einsum('bd,bcd->bc', features, classifier_weights)
        return logits, features


def compute_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def run_omniglot_fewshot_benchmark(
    n_way: int = 5,
    k_shot: int = 1,
    n_query: int = 15,
    n_tasks: int = 100,
    epochs: int = 10,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./omniglot_fewshot_results",
    latent_dim: int = 64,
    k_init: int = 32,
    max_k: int = 256,
    lr: float = 1e-3,
    batch_size: int = 4,
) -> Dict[str, Any]:
    if not OMNIGLOT_AVAILABLE:
        return {"error": "Omniglot not available"}
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running Omniglot {n_way}-way {k_shot}-shot: {n_tasks} tasks")

    # Create tasks
    tasks = get_omniglot_loaders(
        n_way=n_way, k_shot=k_shot, n_query=n_query, n_tasks=n_tasks, seed=seed
    )
    
    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    
    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=4,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.DYNAMIC,
        hypernetwork_code_dim=16,
        hypernetwork_hidden_dim=64,
        tau=1.0,
    )

    model = OmniglotFewShotModel(config, n_way).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    accuracies = []
    
    for epoch in range(epochs):
        epoch_accs = []
        
        for task_idx, (support_loader, query_loader, _) in enumerate(tasks):
            # Get support set
            support_x, support_y = next(iter(support_loader))
            support_x, support_y = support_x.to(device), support_y.to(device)
            support_x = support_x.view(support_x.size(0), -1)
            
            # Get query set
            query_x, query_y = next(iter(query_loader))
            query_x, query_y = query_x.to(device), query_y.to(device)
            query_x = query_x.view(query_x.size(0), -1)
            
            # Encode support
            with torch.no_grad():
                support_features = model.encode(support_x)
            
            # Forward on query with prototypical adaptation
            logits, _ = model(query_x, support_features, support_y)
            
            loss = F.cross_entropy(logits, query_y)
            acc = compute_accuracy(logits, query_y)
            epoch_accs.append(acc)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if task_idx % 20 == 0:
                print(f"  Task {task_idx}: loss={loss.item():.4f}, acc={acc:.4f}")
        
        mean_acc = np.mean(epoch_accs)
        accuracies.append(mean_acc)
        print(f"Epoch {epoch}: mean acc={mean_acc:.4f}")
    
    # Final evaluation on new tasks
    eval_tasks = get_omniglot_loaders(
        n_way=n_way, k_shot=k_shot, n_query=n_query, n_tasks=20, seed=seed+1000
    )
    
    eval_accs = []
    for support_loader, query_loader, _ in eval_tasks:
        support_x, support_y = next(iter(support_loader))
        support_x, support_y = support_x.to(device), support_y.to(device)
        support_x = support_x.view(support_x.size(0), -1)
        
        query_x, query_y = next(iter(query_loader))
        query_x, query_y = query_x.to(device), query_y.to(device)
        query_x = query_x.view(query_x.size(0), -1)
        
        with torch.no_grad():
            support_features = model.encode(support_x)
            logits, _ = model(query_x, support_features, support_y)
            acc = compute_accuracy(logits, query_y)
            eval_accs.append(acc)
    
    final_acc = np.mean(eval_accs)
    
    results = {
        "n_way": n_way,
        "k_shot": k_shot,
        "n_query": n_query,
        "n_tasks": n_tasks,
        "training_accuracies": accuracies,
        "eval_accuracies": eval_accs,
        "final_acc": float(final_acc),
        "final_K": int(model.backbone.K),
    }
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / f"omniglot_{n_way}way_{k_shot}shot.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Final accuracy: {final_acc:.4f}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    run_omniglot_fewshot_benchmark(
        n_way=args.n_way,
        k_shot=args.k_shot,
        epochs=args.epochs,
        device=args.device
    )
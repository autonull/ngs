"""Meta-learning benchmarks for NGS (MAML-style)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
from pathlib import Path
import json
import copy


class MAMLNGS(nn.Module):
    """MAML-style meta-learner with NGS as the task model."""
    
    def __init__(self, input_dim: int, output_dim: int, ngs_config, meta_lr: float = 0.01):
        super().__init__()
        from ngs.models import build_ngs
        self.model = build_ngs(input_dim, output_dim, ngs_config)
        self.meta_lr = meta_lr
    
    def inner_adapt(self, support_x, support_y, n_steps=5):
        """Perform inner-loop adaptation on support set."""
        fast_model = copy.deepcopy(self.model)
        fast_optimizer = torch.optim.SGD(fast_model.parameters(), lr=self.meta_lr)
        
        for _ in range(n_steps):
            logits = fast_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            fast_optimizer.zero_grad()
            loss.backward()
            fast_optimizer.step()
        
        return fast_model
    
    def forward(self, x):
        return self.model(x)


def run_metalearn_benchmark(
    n_way: int = 5,
    k_shot: int = 5,
    n_tasks: int = 100,
    meta_epochs: int = 100,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./metalearn_results",
    latent_dim: int = 32,
    inner_lr: float = 0.01,
    meta_lr: float = 1e-3,
    inner_steps: int = 5,
) -> Dict[str, Any]:
    """Run MAML-style meta-learning with NGS."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running MAML meta-learning: {n_way}-way {k_shot}-shot using {device}")

    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement

    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=16,
        max_k=64,
        top_k=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
    )

    meta_learner = MAMLNGS(2, n_way, config, inner_lr).to(device)
    meta_optimizer = torch.optim.Adam(meta_learner.parameters(), lr=meta_lr)

    acc_history = []
    loss_history = []

    for epoch in range(meta_epochs):
        meta_optimizer.zero_grad()
        meta_loss = 0
        meta_correct = 0
        meta_total = 0

        for _ in range(16):
            centers = torch.randn(n_way, 2, device=device) * 5.0
            support_x, support_y = [], []
            query_x, query_y = [], []

            for c in range(n_way):
                s = centers[c] + torch.randn(k_shot, 2, device=device) * 0.5
                support_x.append(s)
                support_y.extend([c] * k_shot)
                q = centers[c] + torch.randn(15, 2, device=device) * 0.5
                query_x.append(q)
                query_y.extend([c] * 15)

            support_x = torch.cat(support_x, 0)
            support_y = torch.tensor(support_y, device=device)
            query_x = torch.cat(query_x, 0)
            query_y = torch.tensor(query_y, device=device)

            fast_model = meta_learner.inner_adapt(support_x, support_y, inner_steps)
            query_logits = fast_model(query_x)
            task_loss = F.cross_entropy(query_logits, query_y)
            meta_loss += task_loss

            preds = query_logits.argmax(dim=-1)
            meta_correct += (preds == query_y).sum().item()
            meta_total += query_y.size(0)

        meta_loss /= 16
        meta_loss.backward()
        meta_optimizer.step()

        acc = meta_correct / meta_total
        acc_history.append(acc)
        loss_history.append(meta_loss.item())

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss={meta_loss.item():.4f}, Acc={acc:.4f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "n_way": n_way,
        "k_shot": k_shot,
        "final_acc": acc_history[-1],
        "acc_history": acc_history,
        "loss_history": loss_history,
        "final_k": meta_learner.model.K,
    }
    with open(Path(output_dir) / f"{n_way}way_{k_shot}shot_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Final accuracy: {acc_history[-1]:.4f}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_metalearn_benchmark(args.n_way, args.k_shot, meta_epochs=args.epochs, device=args.device, seed=args.seed)

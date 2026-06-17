"""Contextual bandit benchmark for NGS."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
from pathlib import Path
import json


class DriftingBandit:
    """Contextual bandit with drifting reward functions."""
    
    def __init__(self, n_arms: int, context_dim: int, drift_speed: float = 0.01):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.drift_speed = drift_speed
        self.true_weights = torch.randn(n_arms, context_dim)
        self.step = 0
    
    def drift(self):
        self.true_weights += torch.randn_like(self.true_weights) * self.drift_speed
        self.step += 1
    
    def get_reward(self, context: torch.Tensor, arm: int) -> float:
        noise = torch.randn(1).item() * 0.1
        return (self.true_weights[arm] * context).sum().item() + noise
    
    def get_optimal_reward(self, context: torch.Tensor) -> float:
        rewards = (self.true_weights * context.unsqueeze(0)).sum(dim=-1)
        return rewards.max().item()


def run_bandit_benchmark(
    n_arms: int = 10,
    context_dim: int = 8,
    n_steps: int = 5000,
    drift_speed: float = 0.005,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./bandit_results",
    latent_dim: int = 16,
) -> Dict[str, Any]:
    """Run contextual bandit with NGS."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running bandit: {n_arms} arms, {context_dim}D context using {device}")

    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    from ngs.models import build_ngs

    bandit = DriftingBandit(n_arms, context_dim, drift_speed)
    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=n_arms,
        max_k=n_arms * 4,
        top_k=n_arms,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
    )
    model = build_ngs(context_dim, n_arms, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epsilon = 0.1
    regret_history = []
    cumulative_regret = 0
    reward_history = []

    for t in range(n_steps):
        context = torch.randn(1, context_dim).to(device)
        with torch.no_grad():
            q_values = model(context)
        
        if np.random.random() < epsilon:
            arm = np.random.randint(n_arms)
        else:
            arm = q_values.argmax(dim=-1).item()
        
        reward = bandit.get_reward(context.squeeze().cpu(), arm)
        optimal = bandit.get_optimal_reward(context.squeeze().cpu())
        regret = optimal - reward
        cumulative_regret += regret
        
        target = q_values.clone()
        target[0, arm] = reward
        loss = F.mse_loss(q_values, target.detach())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if t % 100 == 0:
            bandit.drift()
        
        regret_history.append(cumulative_regret)
        reward_history.append(reward)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "n_arms": n_arms,
        "context_dim": context_dim,
        "final_regret": cumulative_regret,
        "avg_regret_last_1000": np.mean(np.diff(regret_history[-1000:])) if len(regret_history) > 1000 else 0,
        "regret_history": regret_history[::100],
        "final_k": model.K,
    }
    with open(Path(output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Final cumulative regret: {cumulative_regret:.1f}, K={model.K}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-arms", type=int, default=10)
    parser.add_argument("--context-dim", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_bandit_benchmark(args.n_arms, args.context_dim, args.steps, device=args.device, seed=args.seed)

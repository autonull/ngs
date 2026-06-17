"""Continual RL benchmarks for NGS."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from typing import Dict, Any, List
from pathlib import Path
import json


class ContinualRLEnv:
    """Sequence of environments with changing dynamics."""
    
    def __init__(self, env_name: str, n_tasks: int = 5, seed: int = 42):
        self.env_name = env_name
        self.n_tasks = n_tasks
        self.seed = seed
        self.shifts = [
            {},  # default
            {"gravity_mult": 0.8},
            {"gravity_mult": 1.2},
            {"mass_mult": 0.7},
            {"mass_mult": 1.5},
        ][:n_tasks]
    
    def make_env(self, task_id: int):
        env = gym.make(self.env_name)
        shift = self.shifts[task_id]
        if "gravity_mult" in shift and hasattr(env.unwrapped, 'gravity'):
            env.unwrapped.gravity *= shift["gravity_mult"]
        if "mass_mult" in shift and hasattr(env.unwrapped, 'masscart'):
            env.unwrapped.masscart *= shift["mass_mult"]
        return env


def simple_rollout(env, policy, max_steps=500, device="cpu"):
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(max_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy(obs_t).argmax(dim=-1).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


class SimplePolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
    def forward(self, x):
        return self.net(x)


def run_continual_rl_benchmark(
    env_name: str = "CartPole-v1",
    n_tasks: int = 5,
    epochs_per_task: int = 5,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./continual_rl_results",
) -> Dict[str, Any]:
    """Run continual RL benchmark with NGS."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running continual RL on {env_name} with {n_tasks} tasks using {device}")

    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    from ngs.models import build_ngs

    continual_env = ContinualRLEnv(env_name, n_tasks, seed)
    env0 = continual_env.make_env(0)
    obs_dim = env0.observation_space.shape[0]
    action_dim = env0.action_space.n
    env0.close()

    config = NGSConfig(
        latent_dim=32,
        k_init=16,
        max_k=128,
        top_k=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
    )

    model = build_ngs(obs_dim, action_dim, config).to(device)
    head = nn.Linear(64, action_dim).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

    reward_matrix = np.zeros((n_tasks, n_tasks))
    active_units = []

    for task_id in range(n_tasks):
        env = continual_env.make_env(task_id)
        for epoch in range(epochs_per_task):
            obs, _ = env.reset()
            for _ in range(500):
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                features = model(obs_t)
                logits = head(features)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
                next_obs, reward, terminated, truncated, _ = env.step(action)
                
                obs_t_next = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    next_val = head(model(obs_t_next)).max()
                    target = reward + 0.99 * next_val * (1 - terminated)
                val = head(features).gather(1, torch.tensor([action], device=device))
                loss = F.mse_loss(val.squeeze(), target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                obs = next_obs
                if terminated or truncated:
                    break
        env.close()

        for eval_task in range(task_id + 1):
            eval_env = continual_env.make_env(eval_task)
            rewards = [simple_rollout(eval_env, lambda o: head(model(o)), device=device) for _ in range(5)]
            reward_matrix[eval_task, task_id] = np.mean(rewards)
            eval_env.close()

        active_units.append(model.K)
        print(f"Task {task_id}: K={model.K}, avg_reward={reward_matrix[task_id, task_id]:.1f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "env_name": env_name,
        "n_tasks": n_tasks,
        "reward_matrix": reward_matrix.tolist(),
        "active_units": active_units,
    }
    with open(Path(output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Reward matrix diagonal: {np.diag(reward_matrix)}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_continual_rl_benchmark(args.env, args.n_tasks, device=args.device, seed=args.seed)

"""Reinforcement learning benchmarks for NGS."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional
from pathlib import Path
import json

from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models import build_ngs
from ngs.training import NGSTrainer, TrainerConfig


class NGSRLPolicy(nn.Module):
    """NGS-based policy network for RL."""
    
    def __init__(self, obs_dim: int, action_dim: int, config: NGSConfig):
        super().__init__()
        self.ngs = build_ngs(obs_dim, 64, config)
        self.action_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)
    
    def forward(self, obs: torch.Tensor):
        output = self.ngs(obs)
        features = output.logits if hasattr(output, 'logits') else output
        logits = self.action_head(features)
        value = self.value_head(features)
        return logits, value.squeeze(-1)
    
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        logits, _ = self.forward(obs)
        if deterministic:
            return logits.argmax(dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


class RolloutBuffer:
    """Simple rollout buffer for PPO."""
    
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, obs, action, reward, value, log_prob, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def __len__(self):
        return len(self.obs)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[i + 1]
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return torch.tensor(advantages), torch.tensor(returns)


def make_env(env_name: str, domain_shift: str = "none"):
    """Create environment with optional domain shift."""
    env = gym.make(env_name)
    
    if domain_shift != "none":
        # Apply domain shift by modifying environment parameters
        if hasattr(env.unwrapped, 'gravity') and domain_shift == "gravity":
            env.unwrapped.gravity *= 1.5
        elif hasattr(env.unwrapped, 'masscart') and domain_shift == "mass":
            env.unwrapped.masscart *= 2.0
        elif hasattr(env.unwrapped, 'length') and domain_shift == "length":
            env.unwrapped.length *= 1.5
        elif domain_shift == "noise":
            # Add observation noise wrapper
            class NoisyWrapper(gym.ObservationWrapper):
                def observation(self, obs):
                    return obs + np.random.randn(*obs.shape) * 0.1
            env = NoisyWrapper(env)
    
    return env


def run_rl_benchmark(
    env_name: str = "CartPole-v1",
    domain_shift: str = "none",
    total_timesteps: int = 100000,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./rl_results",
    latent_dim: int = 32,
    k_init: int = 32,
    max_k: int = 128,
    lr: float = 3e-4,
) -> Dict[str, Any]:
    """Run NGS RL benchmark with domain shift."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running RL on {env_name} with {domain_shift} shift using {device}")
    
    # Create environment
    env = make_env(env_name, domain_shift)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Build NGS policy
    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.PRE_ALLOCATED,
    )
    
    policy = NGSRLPolicy(obs_dim, action_dim, config).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    buffer = RolloutBuffer()
    
    # PPO hyperparameters
    clip_eps = 0.2
    ppo_epochs = 4
    batch_size = 64
    gamma = 0.99
    lam = 0.95
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    
    obs, _ = env.reset(seed=seed)
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    
    for timestep in range(total_timesteps):
        # Collect rollout
        policy.eval()
        with torch.no_grad():
            action = policy.act(obs.unsqueeze(0)).item()
            logits, value = policy(obs.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(torch.tensor([action], device=device))
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        buffer.add(
            obs.cpu().numpy(), action, reward, 
            value.item(), log_prob.item(), done
        )
        
        obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
        
        if done:
            episode_rewards.append(sum(buffer.rewards))
            episode_lengths.append(len(buffer))
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
        
        # Update policy when buffer is full
        if len(buffer) >= 2048:
            # Compute advantages
            advantages, returns = compute_gae(
                buffer.rewards, buffer.values, buffer.dones, gamma, lam
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Prepare batch data
            obs_batch = torch.tensor(np.array(buffer.obs), dtype=torch.float32).to(device)
            actions_batch = torch.tensor(buffer.actions, dtype=torch.long).to(device)
            old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32).to(device)
            returns_batch = returns.to(device)
            advantages_batch = advantages.to(device)
            
            # PPO update
            policy.train()
            for _ in range(ppo_epochs):
                # Shuffle
                idx = torch.randperm(len(buffer))
                for i in range(0, len(buffer), batch_size):
                    batch_idx = idx[i:i+batch_size]
                    
                    b_obs = obs_batch[batch_idx]
                    b_actions = actions_batch[batch_idx]
                    b_old_log_probs = old_log_probs[batch_idx]
                    b_returns = returns_batch[batch_idx]
                    b_advantages = advantages_batch[batch_idx]
                    
                    log_probs, values, entropy = policy.evaluate_actions(b_obs, b_actions)
                    
                    ratio = (log_probs - b_old_log_probs).exp()
                    surr1 = ratio * b_advantages
                    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(values, b_returns)
                    entropy_loss = -entropy.mean()
                    
                    loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                    optimizer.step()
            
            buffer.clear()
            
            # Evaluate
            if len(episode_rewards) % 10 == 0:
                policy.eval()
                eval_env = make_env(env_name, domain_shift)
                eval_total = 0
                for _ in range(5):
                    e_obs, _ = eval_env.reset()
                    e_obs = torch.tensor(e_obs, dtype=torch.float32).to(device)
                    e_done = False
                    ep_reward = 0
                    while not e_done:
                        with torch.no_grad():
                            e_action = policy.act(e_obs.unsqueeze(0), deterministic=True).item()
                        e_obs, e_reward, e_terminated, e_truncated, _ = eval_env.step(e_action)
                        e_done = e_terminated or e_truncated
                        e_obs = torch.tensor(e_obs, dtype=torch.float32).to(device)
                        ep_reward += e_reward
                    eval_total += ep_reward
                eval_rewards.append(eval_total / 5)
                
                print(f"Timestep {timestep}: Eval Reward = {eval_rewards[-1]:.1f}, K = {policy.ngs.K}")
    
    # Final evaluation
    policy.eval()
    eval_env = make_env(env_name, domain_shift)
    final_rewards = []
    for _ in range(20):
        e_obs, _ = eval_env.reset()
        e_obs = torch.tensor(e_obs, dtype=torch.float32).to(device)
        e_done = False
        ep_reward = 0
        while not e_done:
            with torch.no_grad():
                e_action = policy.act(e_obs.unsqueeze(0), deterministic=True).item()
            e_obs, e_reward, e_terminated, e_truncated, _ = eval_env.step(e_action)
            e_done = e_terminated or e_truncated
            e_obs = torch.tensor(e_obs, dtype=torch.float32).to(device)
            ep_reward += e_reward
        final_rewards.append(ep_reward)
    
    final_mean = np.mean(final_rewards)
    final_std = np.std(final_rewards)
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {
        "env_name": env_name,
        "domain_shift": domain_shift,
        "final_eval_reward": final_mean,
        "final_eval_std": final_std,
        "episode_rewards": episode_rewards[-100:],  # Last 100
        "eval_rewards": eval_rewards,
        "final_k": policy.ngs.K,
        "config": {
            "latent_dim": latent_dim,
            "k_init": k_init,
            "max_k": max_k,
            "lr": lr,
            "total_timesteps": total_timesteps,
        }
    }
    
    with open(Path(output_dir) / f"{env_name}_{domain_shift}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFinal Eval Reward: {final_mean:.2f} ± {final_std:.2f}")
    print(f"Final K: {policy.ngs.K}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--domain-shift", default="none", choices=["none", "gravity", "mass", "length", "noise"])
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./rl_results")
    args = parser.parse_args()
    
    run_rl_benchmark(
        env_name=args.env,
        domain_shift=args.domain_shift,
        total_timesteps=args.timesteps,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir
    )
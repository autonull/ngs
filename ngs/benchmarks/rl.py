"""Reinforcement Learning Benchmark for NGS.

Tests NGS on Gym/MinAtar environments with domain shift.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import random


@dataclass
class RLConfig:
    """Configuration for RL benchmark."""
    # Environment
    env_name: str = "CartPole-v1"  # "CartPole-v1", "Acrobot-v1", "MountainCar-v0"
    domain_shift: str = "none"  # "none", "gravity", "mass", "length", "noise"
    shift_magnitude: float = 0.5
    
    # Model
    latent_dim: int = 64
    k_init: int = 32
    max_k: int = 256
    top_k: int = 8
    routing: str = "factorized"
    parameter_storage: str = "hypernetwork"
    topology_control: str = "continuous_density"
    memory_management: str = "pre_allocated"
    
    # PPO hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training
    n_envs: int = 8
    n_steps: int = 256
    n_epochs: int = 4
    batch_size: int = 64
    total_timesteps: int = 1_000_000
    
    # Domain adaptation
    adapt_every: int = 10000  # timesteps
    n_adaptation_steps: int = 5
    
    # Evaluation
    eval_episodes: int = 10
    eval_freq: int = 50000


def make_env(env_name: str, domain_shift: str = "none", shift_magnitude: float = 0.5, seed: int = 42):
    """Create environment with optional domain shift."""
    try:
        import gymnasium as gym
    except ImportError:
        import gym
    
    env = gym.make(env_name)
    
    # Apply domain shift if specified
    if domain_shift != "none" and hasattr(env, 'unwrapped'):
        unwrapped = env.unwrapped
        if domain_shift == "gravity" and hasattr(unwrapped, 'gravity'):
            unwrapped.gravity *= (1 + shift_magnitude)
        elif domain_shift == "mass" and hasattr(unwrapped, 'masscart'):
            unwrapped.masscart *= (1 + shift_magnitude)
        elif domain_shift == "length" and hasattr(unwrapped, 'length'):
            unwrapped.length *= (1 + shift_magnitude)
        # Add more shifts as needed
    
    env.reset(seed=seed)
    return env


class RLModel(nn.Module):
    """NGS-based actor-critic for RL."""
    
    def __init__(self, obs_dim: int, act_dim: int, config: RLConfig, discrete: bool = True):
        super().__init__()
        self.config = config
        self.discrete = discrete
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.latent_dim),
        )
        
        # NGS for policy
        from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
        from ngs.models.ngs import build_ngs
        
        ngs_config = NGSConfig(
            latent_dim=config.latent_dim,
            k_init=config.k_init,
            max_k=config.max_k,
            top_k=config.top_k,
            routing=RoutingStrategy(config.routing),
            parameter_storage=ParameterStorage(config.parameter_storage),
            topology_control=TopologyControl(config.topology_control),
            memory_management=MemoryManagement(config.memory_management),
        )
        
        # Policy head
        self.ngs_policy = build_ngs(config.latent_dim, act_dim if discrete else act_dim * 2, ngs_config)
        
        # Value head (simpler)
        self.value_head = nn.Sequential(
            nn.Linear(config.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits/mean, value)."""
        z = self.encoder(obs)
        
        # Policy
        policy_out = self.ngs_policy(z)
        action_out = policy_out.logits
        
        # Value
        value = self.value_head(z).squeeze(-1)
        
        return action_out, value
    
    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        """Get action, log_prob, entropy, value for PPO."""
        action_out, value = self.forward(obs)
        
        if self.discrete:
            logits = action_out
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            
            if action is None:
                action = torch.multinomial(probs, 1).squeeze(-1)
            log_prob = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            
        else:
            # Continuous: action_out is [mean, log_std]
            mean, log_std = action_out.chunk(2, dim=-1)
            log_std = log_std.clamp(-20, 2)
            std = log_std.exp()
            
            probs = torch.distributions.Normal(mean, std)
            if action is None:
                action = probs.sample()
            log_prob = probs.log_prob(action).sum(dim=-1)
            entropy = probs.entropy().sum(dim=-1)
            
        return action, log_prob, entropy, value
    
    def adapt_topology(self, z_samples: torch.Tensor):
        """Adapt topology based on recent observations."""
        self.ngs_policy.adapt_density(z_samples)


class RolloutBuffer:
    """PPO rollout buffer."""
    
    def __init__(self, n_steps: int, n_envs: int, obs_dim: int, act_dim: int, device: str):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        
        self.obs = torch.zeros((n_steps, n_envs, obs_dim), device=device)
        self.actions = torch.zeros((n_steps, n_envs), device=device, dtype=torch.long)
        self.log_probs = torch.zeros((n_steps, n_envs), device=device)
        self.rewards = torch.zeros((n_steps, n_envs), device=device)
        self.dones = torch.zeros((n_steps, n_envs), device=device)
        self.values = torch.zeros((n_steps, n_envs), device=device)
        
        self.step = 0
        
    def add(self, obs, action, log_prob, reward, done, value):
        if self.step >= self.n_steps:
            return
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.step += 1
        
    def compute_returns_and_advantage(self, last_values: torch.Tensor, gamma: float, gae_lambda: float):
        """Compute GAE advantages and returns."""
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0
        
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]
                
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            
        returns = advantages + self.values
        return advantages, returns
    
    def get_batches(self, batch_size: int):
        """Yield batches for PPO update."""
        indices = torch.randperm(self.n_steps * self.n_envs, device=self.device)
        
        flat_obs = self.obs.view(-1, *self.obs.shape[2:])
        flat_actions = self.actions.view(-1)
        flat_log_probs = self.log_probs.view(-1)
        flat_advantages = self.advantages.view(-1)
        flat_returns = self.returns.view(-1)
        
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_idx = indices[start:end]
            
            yield (
                flat_obs[batch_idx],
                flat_actions[batch_idx],
                flat_log_probs[batch_idx],
                flat_advantages[batch_idx],
                flat_returns[batch_idx],
            )
    
    def reset(self):
        self.step = 0


class RLBenchmark:
    """RL benchmark runner."""
    
    def __init__(self, config: RLConfig, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results = {}
        
    def run(self, seed: int = 42) -> Dict:
        """Run RL benchmark."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Create environments
        envs = []
        for i in range(self.config.n_envs):
            env = make_env(self.config.env_name, self.config.domain_shift, 
                          self.config.shift_magnitude, seed + i)
            envs.append(env)
            
        # Get dimensions
        obs_dim = envs[0].observation_space.shape[0]
        if hasattr(envs[0].action_space, 'n'):
            act_dim = envs[0].action_space.n
            discrete = True
        else:
            act_dim = envs[0].action_space.shape[0]
            discrete = False
            
        # Create model
        model = RLModel(obs_dim, act_dim, self.config, discrete).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        
        # Buffer
        buffer = RolloutBuffer(self.config.n_steps, self.config.n_envs, obs_dim, act_dim, self.device)
        
        # Training loop
        obs_list = [torch.tensor(env.reset()[0], dtype=torch.float32, device=self.device) for env in envs]
        obs = torch.stack(obs_list)
        
        episode_rewards = []
        episode_lengths = []
        eval_rewards = []
        
        total_steps = 0
        
        while total_steps < self.config.total_timesteps:
            # Collect rollouts
            for step in range(self.config.n_steps):
                with torch.no_grad():
                    action, log_prob, _, value = model.get_action_and_value(obs)
                    
                # Step environments
                next_obs_list = []
                reward_list = []
                done_list = []
                
                for i, env in enumerate(envs):
                    next_obs, reward, terminated, truncated, _ = env.step(action[i].item())
                    done = terminated or truncated
                    
                    next_obs_list.append(torch.tensor(next_obs, dtype=torch.float32, device=self.device))
                    reward_list.append(reward)
                    done_list.append(done)
                    
                    if done:
                        episode_rewards.append(sum(reward_list))  # Simplified
                        obs_list[i] = torch.tensor(env.reset()[0], dtype=torch.float32, device=self.device)
                    else:
                        obs_list[i] = next_obs_list[i]
                        
                obs = torch.stack(obs_list)
                rewards = torch.tensor(reward_list, device=self.device)
                dones = torch.tensor(done_list, device=self.device, dtype=torch.float32)
                
                buffer.add(obs, action, log_prob, rewards, dones, value)
                total_steps += self.config.n_envs
                
            # Compute advantages
            with torch.no_grad():
                _, last_values = model.forward(obs)
            buffer.compute_returns_and_advantage(last_values, self.config.gamma, self.config.gae_lambda)
            
            # PPO update
            for _ in range(self.config.n_epochs):
                for batch_obs, batch_actions, batch_log_probs, batch_advantages, batch_returns in buffer.get_batches(self.config.batch_size):
                    _, new_log_probs, entropy, new_values = model.get_action_and_value(batch_obs, batch_actions)
                    
                    # Ratio
                    ratio = (new_log_probs - batch_log_probs).exp()
                    
                    # Surrogate loss
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_loss = F.mse_loss(new_values, batch_returns)
                    
                    # Entropy loss
                    entropy_loss = -entropy.mean()
                    
                    loss = policy_loss + self.config.value_coef * value_loss + self.config.entropy_coef * entropy_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    
            buffer.reset()
            
            # Topology adaptation
            if total_steps % self.config.adapt_every == 0 and total_steps > 0:
                with torch.no_grad():
                    z_samples = model.encoder(obs)
                    model.adapt_topology(z_samples)
                    
            # Evaluation
            if total_steps % self.config.eval_freq == 0:
                eval_reward = self._evaluate(model, self.config.env_name, self.config.eval_episodes, seed)
                eval_rewards.append((total_steps, eval_reward))
                print(f"Steps: {total_steps}, Eval Reward: {eval_reward:.2f}, K: {model.ngs_policy.K}")
                
        # Final evaluation
        final_eval = self._evaluate(model, self.config.env_name, self.config.eval_episodes, seed)
        
        self.results = {
            "config": self.config.__dict__,
            "seed": seed,
            "episode_rewards": episode_rewards[-100:],  # Last 100
            "eval_rewards": eval_rewards,
            "final_eval_reward": final_eval,
            "final_k": model.ngs_policy.K,
        }
        
        # Close envs
        for env in envs:
            env.close()
            
        return self.results
    
    def _evaluate(self, model: RLModel, env_name: str, n_episodes: int, seed: int) -> float:
        """Evaluate policy."""
        model.eval()
        rewards = []
        
        for ep in range(n_episodes):
            env = make_env(env_name, self.config.domain_shift, self.config.shift_magnitude, seed + 1000 + ep)
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            ep_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    action, _, _, _ = model.get_action_and_value(obs)
                obs_np, reward, terminated, truncated, _ = env.step(action.item())
                obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                ep_reward += reward
                done = terminated or truncated
                
            rewards.append(ep_reward)
            env.close()
            
        return np.mean(rewards)


def run_rl_benchmark(
    env_name: str = "CartPole-v1",
    domain_shift: str = "none",
    total_timesteps: int = 1_000_000,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./rl_results"
) -> Dict:
    """Run RL benchmark with default config."""
    config = RLConfig(
        env_name=env_name,
        domain_shift=domain_shift,
        total_timesteps=total_timesteps,
    )
    
    benchmark = RLBenchmark(config, device)
    results = benchmark.run(seed)
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{env_name}_{domain_shift}_seed{seed}_results.json"
    with open(Path(output_dir) / fname, "w") as f:
        serializable = {k: v for k, v in results.items() if k != "config"}
        serializable["config"] = config.__dict__
        json.dump(serializable, f, indent=2)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RL benchmark")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--domain-shift", default="none", choices=["none", "gravity", "mass", "length", "noise"])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./rl_results")
    args = parser.parse_args()
    
    run_rl_benchmark(args.env, args.domain_shift, args.timesteps, args.device, args.seed, args.output_dir)
"""CartPole domain shift environments for Experiment 2C."""
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class CartPoleShiftWrapper(gym.Wrapper):
    """Wrapper to modify CartPole physics parameters."""
    def __init__(self, env, gravity: float = 9.8, length: float = 0.5, mass: float = 1.0):
        super().__init__(env)
        self.gravity = gravity
        self.length = length
        self.mass = mass
        
        # Unwrap TimeLimit to get the actual CartPole env
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        self.base_env = base_env
        # Modify the underlying environment
        base_env.gravity = gravity
        base_env.length = length
        base_env.masspole = mass
        base_env.total_mass = base_env.masscart + base_env.masspole
        base_env.polemass_length = base_env.masspole * base_env.length
    
    def step(self, action):
        return self.env.step(action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def make_cartpole_shift(gravity: float = 9.8, length: float = 0.5, mass: float = 1.0, seed: int = 42):
    """Create CartPole environment with modified physics."""
    env = gym.make('CartPole-v1')
    env = CartPoleShiftWrapper(env, gravity, length, mass)
    env.reset(seed=seed)
    return env


def get_cartpole_shifts() -> list:
    """Get list of domain shifts to test."""
    return [
        {"name": "normal", "gravity": 9.8, "length": 0.5, "mass": 1.0},
        {"name": "high_gravity", "gravity": 15.0, "length": 0.5, "mass": 1.0},
        {"name": "low_gravity", "gravity": 5.0, "length": 0.5, "mass": 1.0},
        {"name": "long_pole", "gravity": 9.8, "length": 1.0, "mass": 1.0},
        {"name": "short_pole", "gravity": 9.8, "length": 0.25, "mass": 1.0},
        {"name": "heavy_pole", "gravity": 9.8, "length": 0.5, "mass": 2.0},
        {"name": "light_pole", "gravity": 9.8, "length": 0.5, "mass": 0.5},
    ]


class NGSDiscretePolicy(nn.Module):
    """NGS-based discrete control policy (for CartPole)."""
    def __init__(self, obs_dim: int, action_dim: int, config):
        super().__init__()
        from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
        from ngs.models import build_ngs
        
        self.ngs = build_ngs(obs_dim, config.latent_dim, config)
        self.config = config
        
        # Actor head (logits)
        self.actor = nn.Linear(config.latent_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(config.latent_dim, 1)
    
    def forward(self, x):
        out_obj = self.ngs(x)
        latent = out_obj.latent
        
        logits = self.actor(latent)
        value = self.critic(latent)
        
        return logits, value, out_obj.routing_output
    
    def act(self, x):
        logits, value, _ = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


def run_cartpole_shift_benchmark(
    use_ngs: bool = True,
    n_shifts: int = 5,
    episodes_per_shift: int = 50,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./cartpole_shift_results",
    latent_dim: int = 64,
    k_init: int = 32,
    max_k: int = 256,
    lr: float = 3e-4,
) -> Dict[str, Any]:
    """Run CartPole domain shift adaptation benchmark."""
    import gymnasium as gym
    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running CartPole shifts: NGS={use_ngs}")

    shifts = get_cartpole_shifts()[:n_shifts]
    
    # Use first shift for dimension info
    shift_params = {k: v for k, v in shifts[0].items() if k != 'name'}
    env = make_cartpole_shift(**shift_params, seed=seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")

    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=4,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.DYNAMIC,
        split_threshold=0.05,
        prune_threshold=0.01,
        hypernetwork_code_dim=16,
        hypernetwork_hidden_dim=64,
        tau=1.0,
    )

    model = NGSDiscretePolicy(obs_dim, action_dim, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    shift_results = {}
    old_model = None
    
    for shift_idx, shift in enumerate(shifts):
        print(f"\n=== Shift {shift_idx}: {shift['name']} ===")
        shift_params = {k: v for k, v in shift.items() if k != 'name'}
        env = make_cartpole_shift(**shift_params, seed=seed + shift_idx)
        
        # Collect trajectories
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        
        for ep in range(episodes_per_shift):
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    action, log_prob, value = model.act(obs_tensor)
                
                next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
                done = terminated or truncated
                
                states.append(obs_tensor)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(done)
                
                obs = next_obs
                ep_reward += reward
            
            if ep % 10 == 0:
                print(f"  Episode {ep}: reward={ep_reward}")
        
        env.close()
        
        # Compute returns
        returns = []
        gae = 0
        gamma, lam = 0.99, 0.95
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            if d:
                gae = 0
            delta = r + gamma * v.item() - gae
            gae = delta + gamma * lam * gae
            returns.insert(0, gae + v.item())
        
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(5):
            logits, value, _ = model(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            ratio = (new_log_probs - log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(value.squeeze(), returns)
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        
        # Evaluate
        eval_shift_params = {k: v for k, v in shift.items() if k != 'name'}
        eval_env = make_cartpole_shift(**eval_shift_params, seed=seed + shift_idx + 100)
        eval_returns = []
        for _ in range(10):
            obs, _ = eval_env.reset()
            done = False
            ep_reward = 0
            while not done:
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _, _ = model.act(obs_tensor)
                obs, reward, terminated, truncated, _ = eval_env.step(action.cpu().numpy()[0])
                done = terminated or truncated
                ep_reward += reward
            eval_returns.append(ep_reward)
        eval_env.close()
        
        shift_results[shift['name']] = {
            'train_returns': [sum(rewards[i:i+500]) for i in range(0, len(rewards), 500)],
            'eval_return': np.mean(eval_returns),
        }
        print(f"  Eval return: {np.mean(eval_returns):.1f}")

    results = {
        "use_ngs": use_ngs,
        "n_shifts": n_shifts,
        "shift_results": shift_results,
    }
    
    import json
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    suffix = "ngs" if use_ngs else "mlp"
    with open(Path(output_dir) / f"cartpole_shifts_{suffix}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngs", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    run_cartpole_shift_benchmark(use_ngs=args.ngs, device=args.device)
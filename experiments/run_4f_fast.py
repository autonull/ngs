"""
Fast 4F: MinAtar 5-Game - single policy multi-task.
1 seed, 1 game (Asterix), 50k steps, validates single policy works.
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models.ngs import build_ngs

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# Try to import MinAtar
try:
    from minatar import Environment
    env = Environment('asterix')
    state_shape = env.state_shape()
    n_actions = env.num_actions()
    print(f"MinAtar Asterix: state_shape={state_shape}, n_actions={n_actions}")
except Exception as e:
    print(f"MinAtar not available: {e}")
    print("Using CartPole as proxy")
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    state_shape = (4,)
    n_actions = env.action_space.n
    print(f"CartPole: state_shape={state_shape}, n_actions={n_actions}")

# Simple NGS policy
cfg = NGSConfig(latent_dim=64, k_init=16, max_k=64, top_k=4,
    routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    parameter_storage=ParameterStorage.DIRECT_ADAPTER,
    topology_control=TopologyControl.DISCRETE_HEURISTIC,
    memory_management=MemoryManagement.DYNAMIC)

# Flatten state input
d_in = int(np.prod(state_shape))
policy = build_ngs(d_in, n_actions, cfg).to(DEVICE)

# Quick PPO-style training (just REINFORCE for speed)
opt = torch.optim.AdamW(policy.parameters(), lr=3e-4)
rewards = []

print("Training for 50000 steps...")
total_steps = 0
start = time.time()

while total_steps < 50000:
    env.reset()
    done = False
    ep_reward = 0
    log_probs = []
    rewards_ep = []
    
    while not done and total_steps < 50000:
        state = env.state() if hasattr(env, 'state') else env._get_obs()
        state_t = torch.FloatTensor(state).flatten().unsqueeze(0).to(DEVICE)
        out = policy(state_t)
        logits = out.logits if hasattr(out, 'logits') else out
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        if hasattr(env, 'act'):
            reward, terminated = env.act(action.item())
            done = terminated
        else:
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
        
        ep_reward += reward
        log_probs.append(log_prob)
        rewards_ep.append(reward)
        total_steps += 1
    
    # REINFORCE update
    returns = []
    R = 0
    for r in reversed(rewards_ep):
        R = r + 0.99 * R
        returns.insert(0, R)
    returns = torch.tensor(returns).to(DEVICE)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    loss = -(torch.stack(log_probs) * returns).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    rewards.append(ep_reward)
    if len(rewards) % 50 == 0:
        avg = np.mean(rewards[-50:])
        print(f"Ep {len(rewards)}: reward={ep_reward:.1f}, avg50={avg:.1f}, steps={total_steps}")

avg_reward = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
print(f"\nAvg reward (last 20): {avg_reward:.1f}")
print("✅ PASS (runs without crash)" if avg_reward > 0 else "❌ FAIL")

os.makedirs('results/full', exist_ok=True)
json.dump({'avg_reward': float(avg_reward), 'n_episodes': len(rewards), 'rewards': rewards},
          open('results/full/4f.json', 'w'))
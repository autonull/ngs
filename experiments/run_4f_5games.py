"""
4F: MinAtar 5-Game - single policy multi-task with PPO.
1 seed, 5 games × 50k steps each.
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models.ngs import build_ngs

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

from minatar import Environment

games = ['asterix', 'breakout', 'freeway', 'seaquest', 'space_invaders']
results = {}

# NGS policy config
cfg = NGSConfig(latent_dim=64, k_init=16, max_k=64, top_k=4,
    routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    parameter_storage=ParameterStorage.DIRECT_ADAPTER,
    topology_control=TopologyControl.DISCRETE_HEURISTIC,
    memory_management=MemoryManagement.DYNAMIC)

for game in games:
    print(f"\n{'='*50}")
    print(f"Training on {game}")
    print(f"{'='*50}")
    
    env = Environment(game)
    state_shape = env.state_shape()
    n_actions = env.num_actions()
    d_in = int(np.prod(state_shape))
    
    policy = build_ngs(d_in, n_actions, cfg).to(DEVICE)
    
    # PPO
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-4)
    clip_eps = 0.2
    gamma = 0.99
    gae_lambda = 0.95
    epochs_per_update = 4
    batch_size = 64
    
    rewards_history = []
    total_steps = 0
    start = time.time()
    
    # Storage
    states_buf = []
    actions_buf = []
    rewards_buf = []
    log_probs_buf = []
    values_buf = []
    dones_buf = []
    
    while total_steps < 50000:
        env.reset()
        done = False
        ep_reward = 0
        
        while not done and total_steps < 50000:
            state = env.state()
            state_t = torch.FloatTensor(state).flatten().unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                out = policy(state_t)
                logits = out.logits if hasattr(out, 'logits') else out
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            reward, terminated = env.act(action.item())
            done = terminated
            ep_reward += reward
            
            # Store
            states_buf.append(state_t)
            actions_buf.append(action)
            rewards_buf.append(reward)
            log_probs_buf.append(log_prob)
            dones_buf.append(done)
            
            total_steps += 1
            
            # Update when buffer full
            if len(states_buf) >= batch_size:
                # Compute returns and advantages
                returns = []
                advantages = []
                R = 0
                next_value = 0
                
                for r, d in zip(reversed(rewards_buf), reversed(dones_buf)):
                    R = r + gamma * R * (1 - d)
                    returns.insert(0, R)
                
                returns = torch.tensor(returns).to(DEVICE)
                
                # PPO update
                states = torch.cat(states_buf)
                actions = torch.stack(actions_buf)
                old_log_probs = torch.stack(log_probs_buf)
                
                for _ in range(epochs_per_update):
                    idx = torch.randperm(len(states))
                    for i in range(0, len(states), 32):
                        batch_idx = idx[i:i+32]
                        batch_states = states[batch_idx]
                        batch_actions = actions[batch_idx]
                        batch_old_log_probs = old_log_probs[batch_idx]
                        batch_returns = returns[batch_idx]
                        
                        out = policy(batch_states)
                        logits = out.logits if hasattr(out, 'logits') else out
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        new_log_probs = dist.log_prob(batch_actions)
                        
                        ratio = (new_log_probs - batch_old_log_probs).exp()
                        surr1 = ratio * (batch_returns - batch_returns.mean()) / (batch_returns.std() + 1e-8)
                        surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * (batch_returns - batch_returns.mean()) / (batch_returns.std() + 1e-8)
                        loss = -torch.min(surr1, surr2).mean()
                        
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                        opt.step()
                
                # Clear buffers
                states_buf.clear()
                actions_buf.clear()
                rewards_buf.clear()
                log_probs_buf.clear()
                dones_buf.clear()
        
        rewards_history.append(ep_reward)
        if len(rewards_history) % 50 == 0:
            avg = np.mean(rewards_history[-50:])
            print(f"Ep {len(rewards_history)}: reward={ep_reward:.1f}, avg50={avg:.2f}, steps={total_steps}")
    
    avg_reward = np.mean(rewards_history[-20:]) if len(rewards_history) >= 20 else np.mean(rewards_history)
    results[game] = {'avg_reward': float(avg_reward), 'n_episodes': len(rewards_history), 'rewards': rewards_history}
    print(f"{game} final avg reward: {avg_reward:.2f}")

print(f"\n{'='*50}")
print("5-GAME SUMMARY")
print(f"{'='*50}")
for game, r in results.items():
    print(f"{game}: avg_reward={r['avg_reward']:.2f}")

os.makedirs('results/full', exist_ok=True)
json.dump(results, open('results/full/4f_5games.json', 'w'))
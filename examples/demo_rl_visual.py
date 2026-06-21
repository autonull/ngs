#!/usr/bin/env python
"""
NGS + PPO Demo: Visual Reinforcement Learning Training

Enhanced version with proper NGS feature utilization:
- Dynamic topology adaptation via MonolithicMahalanobis routing
- Sparse top-k activation for efficient computation
- Gaussian mixture model representation of policy features
- Uncertainty tracking through routing weights
- NGS topology losses (entropy, diversity, split_gate)
- Comprehensive metrics and visualizations
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

def parse_args():
    parser = argparse.ArgumentParser(description="NGS + PPO Visual RL Demo")
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium environment name")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    parser.add_argument("--device", default="cpu", help="Device to train on (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="./demo_results", help="Output directory for plots")
    parser.add_argument("--latent-dim", type=int, default=32, help="NGS latent dimension")
    parser.add_argument("--k-init", type=int, default=32, help="Initial number of units")
    parser.add_argument("--max-k", type=int, default=128, help="Maximum units for NGS")
    parser.add_argument("--top-k", type=int, default=8, help="Top-k routing")
    parser.add_argument("--update-every", type=int, default=2048, help="PPO update frequency")
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluation frequency (episodes)")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    parser.add_argument("--routing", default="monolithic_mahalanobis", choices=["monolithic_mahalanobis", "factorized_subspace", "gaussian_attention", "uncertainty_aware"], help="Routing strategy")
    parser.add_argument("--topology", default="discrete_heuristic", choices=["discrete_heuristic", "continuous_density", "merge_aware"], help="Topology control strategy")
    parser.add_argument("--entropy-weight", type=float, default=0.01, help="Entropy regularization weight")
    parser.add_argument("--diversity-weight", type=float, default=0.01, help="Diversity loss weight")
    parser.add_argument("--adapt-every", type=int, default=10, help="Topology adaptation frequency (episodes)")
    return parser.parse_args()

try:
    from ngs.core.interfaces import (
        NGSConfig, RoutingStrategy, ParameterStorage,
        TopologyControl, MemoryManagement
    )
    from ngs.models import build_ngs
    NGS_AVAILABLE = True
except ImportError:
    NGS_AVAILABLE = False
    print("Warning: NGS library not available. Using fallback MLP policy.")


class NGSPolicy(nn.Module):
    """NGS-based policy for PPO with value head."""
    
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        self.ngs = build_ngs(obs_dim, 64, config)
        self.action_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)
    
    def forward(self, obs):
        out = self.ngs(obs)
        features = out.logits if hasattr(out, 'logits') else out
        logits = self.action_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value
    
    def act(self, obs, deterministic=False):
        logits, _ = self.forward(obs)
        if deterministic:
            return logits.argmax(dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()
    
    def evaluate_actions(self, obs, actions):
        logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy
    
    def get_ngs_losses(self):
        """Get NGS-specific topology losses."""
        if hasattr(self.ngs, 'compute_topology_losses'):
            return self.ngs.compute_topology_losses()
        return {}
    
    @property
    def K(self):
        return self.ngs.K


class MLPPolicy(nn.Module):
    """Simple MLP fallback policy when NGS is not available."""
    
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.action_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)
    
    def forward(self, obs):
        features = self.net(obs)
        return self.action_head(features), self.value_head(features).squeeze(-1)
    
    def act(self, obs, deterministic=False):
        logits, _ = self.forward(obs)
        if deterministic:
            return logits.argmax(dim=-1)
        return torch.distributions.Categorical(logits=logits).sample()
    
    def evaluate_actions(self, obs, actions):
        logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), values, dist.entropy()
    
    def get_ngs_losses(self):
        return {}
    
    @property
    def K(self):
        return 0


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation for PPO."""
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        next_value = 0 if i == len(rewards) - 1 else values[i + 1]
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def make_env(env_name, render_mode=None):
    """Create Gymnasium environment with optional rendering."""
    return gym.make(env_name, render_mode=render_mode)


def create_training_plots(args, history, save_dir):
    """Create training progress and NGS dynamics visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    episode_rewards = [h['episode_reward'] for h in history if h['episode_reward'] is not None]
    eval_rewards = [h['eval_reward'] for h in history if h['eval_reward'] is not None]
    episode_lengths = [h['episode_length'] for h in history if h['episode_length'] is not None]
    eval_x = [h['timestep'] for h in history if h['eval_reward'] is not None]
    lengths_x = [h['timestep'] for h in history if h['episode_length'] is not None]
    k_values = [h['k'] for h in history]
    timesteps = [h['timestep'] for h in history]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Episode Rewards
    ax = axes[0, 0]
    ax.plot(range(len(episode_rewards)), episode_rewards, 'b-', lw=1, alpha=0.7, label='Episode Reward')
    if len(episode_rewards) >= 10:
        running_avg = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
        ax.plot(range(9, len(episode_rewards)), running_avg, 'r-', lw=2, label='Running Avg (10)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(f'Training Progress: {args.env}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Evaluation Rewards
    ax = axes[0, 1]
    ax.plot(eval_x, eval_rewards, 'g-o', lw=2, markersize=4)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Eval Reward')
    ax.set_title('Evaluation Rewards')
    ax.grid(True, alpha=0.3)
    
    # NGS Topology Dynamics
    ax = axes[0, 2]
    ax.plot(timesteps, k_values, 'purple', lw=2, label='Active Units (K)')
    ax.axhline(y=args.max_k, color='gray', linestyle='--', alpha=0.5, label='Max Capacity')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Number of Active Units')
    ax.set_title('NGS Topology Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Episode Lengths
    ax = axes[1, 0]
    ax.plot(lengths_x, episode_lengths, 'orange', lw=1, alpha=0.7)
    if len(episode_lengths) >= 10:
        running = np.convolve(episode_lengths, np.ones(10)/10, mode='valid')
        ax.plot(lengths_x[9:], running, 'red', lw=2, label='Running Avg (10)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Lengths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Routing entropy over time
    ax = axes[1, 1]
    if 'routing_entropy' in history[0]:
        entropy_values = [h.get('routing_entropy', 0) for h in history if h.get('routing_entropy') is not None]
        entropy_timesteps = [h['timestep'] for h in history if h.get('routing_entropy') is not None]
        if entropy_values:
            ax.plot(entropy_timesteps, entropy_values, 'brown', lw=2, label='Routing Entropy')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Entropy')
            ax.set_title('Routing Entropy (Exploration)')
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Entropy data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Routing Entropy')
    
    # Loss components
    ax = axes[1, 2]
    if 'policy_loss' in history[0]:
        policy_losses = [h.get('policy_loss', 0) for h in history if h.get('policy_loss') is not None]
        value_losses = [h.get('value_loss', 0) for h in history if h.get('value_loss') is not None]
        entropy_losses = [h.get('entropy_loss', 0) for h in history if h.get('entropy_loss') is not None]
        ngs_losses = [h.get('ngs_loss', 0) for h in history if h.get('ngs_loss') is not None]
        loss_timesteps = [h['timestep'] for h in history if h.get('policy_loss') is not None]
        if policy_losses:
            ax.plot(loss_timesteps, policy_losses, 'r-', lw=1, label='Policy Loss')
            ax.plot(loss_timesteps, value_losses, 'b-', lw=1, label='Value Loss')
            ax.plot(loss_timesteps, entropy_losses, 'g-', lw=1, label='Entropy Loss')
            if any(l != 0 for l in ngs_losses):
                ax.plot(loss_timesteps, ngs_losses, 'purple', lw=1, label='NGS Topology Loss')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Loss')
            ax.set_title('Training Losses')
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Loss data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Training Losses')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_visualization.png', dpi=150)
    plt.close()


def create_routing_heatmap(policy, save_dir, n_samples=100):
    """Create routing weights heatmap showing unit activation patterns."""
    if not NGS_AVAILABLE:
        return
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    save_dir = Path(save_dir)
    policy.eval()
    
    with torch.no_grad():
        z_samples = torch.randn(n_samples, policy.ngs.p_down.in_features)
        out = policy.ngs(z_samples)
        weights = out.routing_output.weights if hasattr(out.routing_output, 'weights') else None
    
    if weights is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(weights.detach().cpu().numpy(), aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_xlabel('Top-K Unit Index')
        ax.set_ylabel('Sample Index')
        ax.set_title('NGS Routing Weights Heatmap')
        plt.colorbar(ax.images[0], ax=ax, label='Weight')
        plt.tight_layout()
        plt.savefig(save_dir / 'routing_heatmap.png', dpi=150)
        plt.close()


def create_gaussian_plot(policy, save_dir):
    """Create 3D scatter plot of Gaussian means in latent space."""
    if not NGS_AVAILABLE:
        return
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    save_dir = Path(save_dir)
    router = policy.ngs.router
    
    if not hasattr(router, 'mu') or not hasattr(router, 'active_mask'):
        return
    
    mu = router.mu.detach().cpu().numpy()
    active_mask = router.active_mask.detach().cpu().numpy()
    active_idx = np.where(active_mask)[0]
    mu_active = mu[active_idx]
    
    if mu_active.shape[0] == 0:
        return
    
    if mu_active.shape[1] > 3:
        try:
            from sklearn.decomposition import PCA
            mu_active = PCA(n_components=3).fit_transform(mu_active)
        except:
            mu_active = mu_active[:, :3]
    elif mu_active.shape[1] < 3:
        mu_active = np.pad(mu_active, ((0, 0), (0, 3 - mu_active.shape[1])))
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(mu_active[:, 0], mu_active[:, 1], mu_active[:, 2],
                         c=active_idx, cmap='viridis', s=80, alpha=0.8, edgecolors='k')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    ax.set_title(f'3D Gaussian Means ({len(active_idx)} active units)')
    plt.colorbar(scatter, ax=ax, shrink=0.6, label='Unit Index')
    plt.tight_layout()
    plt.savefig(save_dir / 'gaussian_means_3d.png', dpi=150)
    plt.close()


def run_demo(args):
    """Run the NGS + PPO demo."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    render_mode = 'human' if args.render else None
    env = make_env(args.env, render_mode=render_mode)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if NGS_AVAILABLE:
        # Map string args to enums
        routing_map = {
            'monolithic_mahalanobis': RoutingStrategy.MONOLITHIC_MAHALANOBIS,
            'factorized_subspace': RoutingStrategy.FACTORIZED_SUBSPACE,
            'gaussian_attention': RoutingStrategy.GAUSSIAN_ATTENTION,
            'uncertainty_aware': RoutingStrategy.UNCERTAINTY_AWARE,
        }
        topology_map = {
            'discrete_heuristic': TopologyControl.DISCRETE_HEURISTIC,
            'continuous_density': TopologyControl.CONTINUOUS_DENSITY,
            'merge_aware': TopologyControl.MERGE_AWARE,
        }
        
        config = NGSConfig(
            latent_dim=args.latent_dim, k_init=args.k_init, max_k=args.max_k, top_k=args.top_k,
            routing=routing_map.get(args.routing, RoutingStrategy.MONOLITHIC_MAHALANOBIS),
            parameter_storage=ParameterStorage.DIRECT_ADAPTER,
            topology_control=topology_map.get(args.topology, TopologyControl.DISCRETE_HEURISTIC),
            memory_management=MemoryManagement.PRE_ALLOCATED,
            entropy_weight=args.entropy_weight,
            diversity_weight=args.diversity_weight,
        )
        policy = NGSPolicy(obs_dim, action_dim, config).to(device)
        print(f"Using NGS policy: latent_dim={args.latent_dim}, k_init={args.k_init}, max_k={args.max_k}")
        print(f"  Routing: {args.routing}, Topology: {args.topology}")
    else:
        policy = MLPPolicy(obs_dim, action_dim).to(device)
        print("Using fallback MLP policy")
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    clip_eps, ppo_epochs, batch_size = 0.2, 4, 64
    gamma, lam = 0.99, 0.95
    ent_coef, vf_coef = 0.01, 0.5
    
    history, episode_rewards, episode_lengths, eval_rewards = [], [], [], []
    
    obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
    
    obs, _ = env.reset(seed=args.seed)
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    
    timestep = episode_reward = episode_length = episode_count = 0
    last_eval_episode = -1
    topology_warned = False
    
    print(f"\nStarting training on {args.env} for {args.timesteps} timesteps...")
    
    while timestep < args.timesteps:
        policy.eval()
        with torch.no_grad():
            logits, value = policy(obs.unsqueeze(0))
            action = policy.act(obs.unsqueeze(0)).item()
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(torch.tensor([action], device=device)).item()
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        obs_buf.append(obs.cpu().numpy())
        act_buf.append(action)
        rew_buf.append(reward)
        val_buf.append(value.item())
        logp_buf.append(log_prob)
        done_buf.append(done)
        
        obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
        episode_reward += reward
        episode_length += 1
        timestep += 1
        
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            history.append({
                'timestep': timestep,
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'eval_reward': None,
                'k': policy.K,
            })
            
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            episode_reward = episode_length = 0
            episode_count += 1
            
            if not args.quiet and episode_count % 20 == 0:
                avg = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
                print(f"Episode {episode_count}: Reward = {avg:.1f}, K = {policy.K}")
            
            # Topology adaptation every N episodes using real observations
            if NGS_AVAILABLE and episode_count % args.adapt_every == 0 and len(obs_buf) > 0:
                # Use recent observations for density adaptation
                recent_obs = torch.tensor(np.array(obs_buf[-min(200, len(obs_buf)):]), dtype=torch.float32).to(device)
                with torch.no_grad():
                    z_samples = policy.ngs.p_down(recent_obs)
                try:
                    num_pruned, num_split, num_spawned = policy.ngs.adapt_density(
                        z_samples=z_samples, 
                        split_thresh=0.05, 
                        prune_thresh=0.01
                    )
                    if (num_spawned > 0 or num_split > 0 or num_pruned > 0) and not args.quiet:
                        print(f"  -> Topology: spawned={num_spawned}, split={num_split}, pruned={num_pruned}, K={policy.K}")
                except AttributeError as e:
                    if not topology_warned:
                        topology_warned = True
                        if not args.quiet:
                            print(f"  -> Topology adaptation not supported for this router: {e}")
        
        if len(obs_buf) >= args.update_every:
            policy.train()
            advantages, returns = compute_gae(rew_buf, val_buf, done_buf, gamma, lam)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            obs_batch = torch.tensor(np.array(obs_buf), dtype=torch.float32).to(device)
            act_batch = torch.tensor(act_buf, dtype=torch.long).to(device)
            old_lp = torch.tensor(logp_buf, dtype=torch.float32).to(device)
            ret_batch = returns.to(device)
            adv_batch = advantages.to(device)
            
            epoch_policy_loss = epoch_value_loss = epoch_entropy_loss = epoch_ngs_loss = 0.0
            
            for _ in range(ppo_epochs):
                idx = torch.randperm(len(obs_buf))
                for i in range(0, len(obs_buf), batch_size):
                    bi = idx[i:i+batch_size]
                    lp, val, ent = policy.evaluate_actions(obs_batch[bi], act_batch[bi])
                    ratio = (lp - old_lp[bi]).exp()
                    surr1, surr2 = ratio * adv_batch[bi], torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv_batch[bi]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(val, ret_batch[bi])
                    entropy_loss = -ent.mean()
                    
                    # Add NGS topology losses
                    ngs_losses = policy.get_ngs_losses()
                    ngs_loss = 0.0
                    if ngs_losses:
                        ngs_loss = (args.entropy_weight * ngs_losses.get('entropy', 0) + 
                                   args.diversity_weight * ngs_losses.get('diversity', 0))
                        if 'split_gate' in ngs_losses:
                            ngs_loss += ngs_losses['split_gate']
                    
                    loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss + ngs_loss
                    
                    epoch_policy_loss += policy_loss.item()
                    epoch_value_loss += value_loss.item()
                    epoch_entropy_loss += entropy_loss.item()
                    epoch_ngs_loss += ngs_loss.item() if isinstance(ngs_loss, torch.Tensor) else ngs_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()
            
            # Record losses for visualization
            num_updates = ppo_epochs * (len(obs_buf) // batch_size + 1)
            history.append({
                'timestep': timestep,
                'episode_reward': None,
                'episode_length': None,
                'eval_reward': None,
                'k': policy.K,
                'policy_loss': epoch_policy_loss / num_updates,
                'value_loss': epoch_value_loss / num_updates,
                'entropy_loss': epoch_entropy_loss / num_updates,
                'ngs_loss': epoch_ngs_loss / num_updates,
            })
            
            obs_buf.clear(); act_buf.clear(); rew_buf.clear(); val_buf.clear(); logp_buf.clear(); done_buf.clear()
        
        if episode_count > 0 and episode_count % args.eval_every == 0 and episode_count != last_eval_episode:
            last_eval_episode = episode_count
            policy.eval()
            eval_env = make_env(args.env)
            eval_scores = []
            for _ in range(5):
                e_obs, _ = eval_env.reset()
                e_obs = torch.tensor(e_obs, dtype=torch.float32).to(device)
                e_done = e_ep = False
                while not e_done:
                    with torch.no_grad():
                        e_logits, _ = policy(e_obs.unsqueeze(0))
                        e_action = e_logits.argmax(dim=-1).item()
                    e_obs, e_r, e_t, e_tr, _ = eval_env.step(e_action)
                    e_done, e_ep = e_t or e_tr, e_ep + e_r
                    e_obs = torch.tensor(e_obs, dtype=torch.float32).to(device)
                eval_scores.append(e_ep)
            eval_env.close()
            eval_rewards.append(np.mean(eval_scores))
            
            # Record routing entropy if available
            routing_entropy = 0.0
            if NGS_AVAILABLE and hasattr(policy.ngs, 'entropy_loss'):
                with torch.no_grad():
                    sample_obs = torch.randn(10, obs_dim, device=device)
                    routing_entropy = policy.ngs.entropy_loss(sample_obs).item()
            
            history.append({
                'timestep': timestep, 
                'episode_reward': None, 
                'episode_length': None, 
                'eval_reward': np.mean(eval_scores), 
                'k': policy.K,
                'routing_entropy': routing_entropy,
            })
            
            if not args.quiet:
                print(f"  Eval at timestep {timestep}: Reward = {np.mean(eval_scores):.1f}, K = {policy.K}")
    
    # Final evaluation
    policy.eval()
    eval_env = make_env(args.env)
    final_scores = []
    for _ in range(10):
        e_obs, _ = eval_env.reset()
        e_obs = torch.tensor(e_obs, dtype=torch.float32).to(device)
        e_done = e_ep = False
        while not e_done:
            with torch.no_grad():
                e_logits, _ = policy(e_obs.unsqueeze(0))
                e_action = e_logits.argmax(dim=-1).item()
            e_obs, e_r, e_t, e_tr, _ = eval_env.step(e_action)
            e_done, e_ep = e_t or e_tr, e_ep + e_r
            e_obs = torch.tensor(e_obs, dtype=torch.float32).to(device)
        final_scores.append(e_ep)
    eval_env.close()
    
    final_mean, final_std = np.mean(final_scores), np.std(final_scores)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Final Eval Reward: {final_mean:.2f} ± {final_std:.2f}")
    print(f"Final Active Units (K): {policy.K}")
    print(f"Total Episodes: {len(episode_rewards)}")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / 'episode_rewards.npy', episode_rewards)
    np.save(out_dir / 'episode_lengths.npy', episode_lengths)
    np.save(out_dir / 'eval_rewards.npy', eval_rewards)
    np.save(out_dir / 'k_history.npy', [h['k'] for h in history])
    
    print("\nGenerating visualizations...")
    create_training_plots(args, history, out_dir)
    create_routing_heatmap(policy, out_dir)
    create_gaussian_plot(policy, out_dir)
    
    print(f"\nResults saved to {out_dir}/")
    print("  - training_visualization.png: Learning curves + NGS dynamics + losses")
    print("  - routing_heatmap.png: Routing weights visualization")
    print("  - gaussian_means_3d.png: 3D Gaussian means scatter")
    
    return history


if __name__ == "__main__":
    args = parse_args()
    try:
        run_demo(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()

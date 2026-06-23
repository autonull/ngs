"""MinAtar 5-game multi-task benchmark (Experiment 2B).
Tests single policy with factorized subspaces per game.
Target: Single policy >5 independent PPO baselines."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import json

try:
    import minatar
    MINATAR_AVAILABLE = True
except ImportError:
    MINATAR_AVAILABLE = False


class MinAtarWrapper:
    """Wrapper for MinAtar environments to standardize interface."""
    def __init__(self, game_name: str):
        self.env = minatar.Environment(game_name)
        self.game_name = game_name
        self.action_space_n = self.env.num_actions()
        self.state_shape = self.env.state_shape()
        
    def reset(self):
        self.env.reset()
        return self.env.state()
    
    def step(self, action):
        reward, terminal = self.env.act(action)
        state = self.env.state()
        return state, reward, terminal, {}
    
    def state(self):
        return self.env.state()


class MultiTaskMinAtar:
    """Multi-task MinAtar with 5 games."""
    GAMES = ['breakout', 'asterix', 'seaquest', 'space_invaders', 'freeway']
    
    def __init__(self, seed: int = 42):
        self.games = self.GAMES
        self.envs = {game: MinAtarWrapper(game) for game in self.games}
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def get_env(self, game: str):
        return self.envs[game]


class MultiTaskActorCritic(nn.Module):
    """Actor-Critic with NGS backbone for multi-task RL."""
    def __init__(self, state_dim: int, action_dim: int, config, num_tasks: int = 5):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_tasks = num_tasks
        
        # NGS backbone
        from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
        from ngs.models import build_ngs
        
        # Project state to latent_dim first
        self.state_proj = nn.Linear(state_dim, config.latent_dim)
        self.ngs = build_ngs(config.latent_dim, config.latent_dim, config)
        self.config = config
        
        # Task-specific heads
        self.actor_heads = nn.ModuleList([
            nn.Linear(config.latent_dim, action_dim) for _ in range(num_tasks)
        ])
        self.critic_heads = nn.ModuleList([
            nn.Linear(config.latent_dim, 1) for _ in range(num_tasks)
        ])
        
        # Task embedding for routing
        self.task_embeddings = nn.Embedding(num_tasks, config.latent_dim)
    
    def forward(self, x, task_id: int):
        # Project state to latent dim
        x = self.state_proj(x)
        
        # Add task embedding
        task_emb = self.task_embeddings(torch.tensor(task_id, device=x.device))
        x = x + task_emb
        
        out_obj = self.ngs(x)
        latent = out_obj.latent  # [B, d_latent]
        
        logits = self.actor_heads[task_id](latent)
        value = self.critic_heads[task_id](latent)
        
        return logits, value, out_obj.routing_output


def compute_returns(rewards, values, gamma=0.99, lam=0.95):
    """Compute GAE returns."""
    returns = []
    gae = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        delta = r + gamma * v - gae
        gae = delta + gamma * lam * gae
        returns.insert(0, gae + v)
    return returns


def run_minatar_multitask_benchmark(
    use_factorized: bool = True,
    n_games: int = 5,
    epochs: int = 10,
    steps_per_epoch: int = 2000,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./minatar_multitask_results",
    latent_dim: int = 64,
    k_init: int = 32,
    max_k: int = 256,
    lr: float = 3e-4,
    batch_size: int = 64,
) -> Dict[str, Any]:
    if not MINATAR_AVAILABLE:
        print("MinAtar not available, returning dummy results")
        return {"error": "MinAtar not installed"}
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running MinAtar multi-task: factorized={use_factorized}")

    mt_env = MultiTaskMinAtar(seed=seed)
    games = mt_env.games[:n_games]
    num_tasks = len(games)
    
    # State dim: MinAtar state is [C, H, W] - flatten
    sample_state = mt_env.get_env(games[0]).reset()
    state_dim = np.prod(sample_state.shape)
    action_dim = mt_env.get_env(games[0]).action_space_n
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    
    if use_factorized:
        routing = RoutingStrategy.FACTORIZED_SUBSPACE
        num_subspaces = num_tasks
    else:
        routing = RoutingStrategy.MONOLITHIC_MAHALANOBIS
        num_subspaces = 1
    
    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=4,
        top_k_factorized=2,
        routing=routing,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.DYNAMIC,
        num_subspaces=num_subspaces,
        hypernetwork_code_dim=16,
        hypernetwork_hidden_dim=64,
        tau=1.0,
    )

    model = MultiTaskActorCritic(state_dim, action_dim, config, num_tasks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    results_per_game = {game: [] for game in games}
    
    for epoch in range(epochs):
        epoch_returns = {game: [] for game in games}
        
        for game_idx, game in enumerate(games):
            env = mt_env.get_env(game)
            state = env.reset()
            done = False
            
            states, actions, rewards, log_probs, values, task_ids = [], [], [], [], [], []
            
            for step in range(steps_per_epoch // num_tasks):
                state_tensor = torch.from_numpy(state.flatten()).float().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits, value, _ = model(state_tensor, game_idx)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                
                next_state, reward, done, _ = env.step(action.item())
                
                states.append(state_tensor)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value.item())
                task_ids.append(game_idx)
                
                state = next_state
                if done:
                    state = env.reset()
            
            # Compute returns
            returns = compute_returns(rewards, values)
            
            # Update
            states = torch.cat(states)
            actions = torch.stack(actions)
            log_probs = torch.stack(log_probs)
            returns = torch.tensor(returns, dtype=torch.float32, device=device)
            values = torch.tensor(values, dtype=torch.float32, device=device)
            task_ids = torch.tensor(task_ids, dtype=torch.long, device=device)
            
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO-style update (simplified)
            for _ in range(2):
                total_loss = 0
                for task_id in range(num_tasks):
                    mask = task_ids == task_id
                    if not mask.any():
                        continue
                    
                    logits, value, _ = model(states[mask], task_id)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions[mask])
                    entropy = dist.entropy().mean()
                    
                    ratio = (new_log_probs - log_probs[mask]).exp()
                    surr1 = ratio * advantages[mask]
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages[mask]
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = F.mse_loss(value.squeeze(), returns[mask])
                    
                    total_loss += actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            
            epoch_return = sum(rewards)
            epoch_returns[game].append(epoch_return)
        
        for game in games:
            results_per_game[game].append(np.mean(epoch_returns[game]))
        
        print(f"Epoch {epoch}: " + ", ".join(f"{g}={np.mean(results_per_game[g][-1]):.1f}" for g in games))

    # Final evaluation
    eval_returns = {}
    for game_idx, game in enumerate(games):
        env = mt_env.get_env(game)
        returns = []
        for _ in range(10):
            state = env.reset()
            done = False
            ep_return = 0
            while not done:
                state_tensor = torch.from_numpy(state.flatten()).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _, _ = model(state_tensor, game_idx)
                    action = logits.argmax(dim=-1).item()
                state, reward, done, _ = env.step(action)
                ep_return += reward
            returns.append(ep_return)
        eval_returns[game] = np.mean(returns)
        print(f"Eval {game}: {eval_returns[game]:.1f}")

    results = {
        "use_factorized": use_factorized,
        "n_games": n_games,
        "training_returns": results_per_game,
        "eval_returns": eval_returns,
        "final_K": int(model.ngs.K),
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    suffix = "factorized" if use_factorized else "monolithic"
    with open(Path(output_dir) / f"minatar_{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--factorized", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    run_minatar_multitask_benchmark(
        use_factorized=args.factorized,
        epochs=args.epochs,
        device=args.device
    )
"""
Phase 3: Two figures visualizing the Phase 1 mechanism.
1. Gaussian-means-drift animation contrasting condition A (drifting) vs B (stable)
2. Accuracy-vs-task line chart overlaying A / B / C
"""
import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.modules.ngs_layer import build_stacked_ngs
from experiments.datasets import PermutedMNIST, ReplayBuffer
from experiments.metrics import evaluate_model_on_task

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
N_TASKS = 3
EPOCHS_PER_TASK = 10
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def build_model(input_dim: int, output_dim: int) -> torch.nn.Module:
    return build_stacked_ngs(
        d_in=input_dim, d_out=output_dim,
        n_layers=2, d_latent=128,
        n_experts=256, n_heads=1,
        top_k=8, use_residual=True, use_norm=True
    ).to(DEVICE)


def train_task(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    old_model,
    epochs: int,
    lr: float,
    weight_decay: float,
    replay_buffer,
    replay_ratio: float,
    kd_weight: float,
    kd_temperature: float,
    freeze_adapters: bool = False,
    input_proj_lr_scale: float = 1.0,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if freeze_adapters or input_proj_lr_scale != 1.0:
        param_groups = []
        for name, param in model.named_parameters():
            if 'expert_weights' in name:
                param_groups.append({'params': param, 'lr': 0.0 if freeze_adapters else lr})
            elif 'input_proj' in name:
                param_groups.append({'params': param, 'lr': lr * input_proj_lr_scale})
            else:
                param_groups.append({'params': param, 'lr': lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(DEVICE)
            y = y.to(DEVICE)
            
            if replay_buffer is not None and len(replay_buffer) > x.size(0):
                rx, ry = replay_buffer.sample(int(x.size(0) * replay_ratio))
                if rx is not None:
                    rx, ry = rx.to(DEVICE), ry.to(DEVICE)
                    x = torch.cat([x, rx], dim=0)
                    y = torch.cat([y, ry.argmax(dim=1)], dim=0)
            
            optimizer.zero_grad()
            logits = model(x)
            ce_loss = torch.nn.functional.cross_entropy(logits, y)
            
            kd_loss = torch.tensor(0.0, device=DEVICE)
            if old_model is not None and kd_weight > 0:
                with torch.no_grad():
                    old_logits = old_model(x)
                n_new = x.size(0) // (1 + int(replay_ratio)) if replay_buffer else x.size(0)
                if n_new < x.size(0):
                    kd_loss = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(logits[n_new:] / kd_temperature, dim=-1),
                        torch.nn.functional.softmax(old_logits[n_new:] / kd_temperature, dim=-1),
                        reduction='batchmean'
                    ) * (kd_temperature ** 2)
            
            total_loss = ce_loss + kd_weight * kd_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()


def get_router_means(model) -> np.ndarray:
    """Extract Gaussian means (mu) from the first layer's router."""
    # Use first layer only for consistent dimensions
    layer = model.layers[0]
    if hasattr(layer, 'router') and hasattr(layer.router, 'mu'):
        mu = layer.router.mu.detach().cpu().numpy()
        active_mask = layer.router.active_mask.detach().cpu().numpy()
        return mu[active_mask]
    return np.array([])


def run_condition_with_tracking(
    name: str,
    use_replay: bool,
    use_kd: bool,
    freeze_adapters: bool = False,
    input_proj_lr_scale: float = 1.0,
) -> Dict:
    """Run experiment and track Gaussian means at each task boundary."""
    set_seed(SEED)
    
    permuted = PermutedMNIST(n_tasks=N_TASKS, seed=SEED)
    input_dim, output_dim = 784, 10
    
    model = build_model(input_dim, output_dim)
    replay_buffer = ReplayBuffer(max_size=5000, seed=SEED) if use_replay else None
    
    accuracy_matrix = np.zeros((N_TASKS, N_TASKS))
    active_units_per_task = []
    means_per_task = []  # Track means at each task boundary
    old_model = None
    
    for task_id in range(N_TASKS):
        train_loader, _ = permuted.get_task_data(task_id, BATCH_SIZE)
        
        should_freeze = freeze_adapters and task_id > 0
        proj_scale = input_proj_lr_scale if task_id > 0 else 1.0
        
        train_task(
            model, train_loader, old_model,
            epochs=EPOCHS_PER_TASK, lr=LR, weight_decay=WEIGHT_DECAY,
            replay_buffer=replay_buffer, replay_ratio=1.0,
            kd_weight=10.0 if use_kd else 0.0, kd_temperature=2.0,
            freeze_adapters=should_freeze, input_proj_lr_scale=proj_scale,
        )
        
        # Evaluate on all seen tasks
        for eval_task in range(task_id + 1):
            _, test_loader = permuted.get_task_data(eval_task, BATCH_SIZE)
            acc = evaluate_model_on_task(model, test_loader, DEVICE)
            accuracy_matrix[eval_task, task_id] = acc
        
        # Track means
        means_per_task.append(get_router_means(model))
        active_units_per_task.append(model.K)
        
        if replay_buffer is not None:
            for x, y in train_loader:
                replay_buffer.add(x, torch.nn.functional.one_hot(y, output_dim).float())
        
        if use_kd:
            old_model = deepcopy(model).eval()
            for p in old_model.parameters():
                p.requires_grad = False
    
    # Fill upper triangle with NaN
    for i in range(N_TASKS):
        for j in range(i + 1, N_TASKS):
            accuracy_matrix[i, j] = np.nan
    
    return {
        'name': name,
        'accuracy_matrix': accuracy_matrix,
        'active_units': active_units_per_task,
        'means_per_task': means_per_task,
    }


def compute_final_accuracies(acc_matrix: np.ndarray) -> np.ndarray:
    """Compute final accuracy per task (last column)."""
    n_tasks = acc_matrix.shape[0]
    final_acc = np.zeros(n_tasks)
    for i in range(n_tasks):
        final_acc[i] = acc_matrix[i, -1]
    return final_acc


def plot_accuracy_vs_task():
    """Figure 2: Accuracy-vs-task line chart overlaying A / B / C."""
    plt = _require_matplotlib()
    
    # Load Phase 1 results
    with open('results/phase1/forgetting_results.json', 'r') as f:
        results = json.load(f)
    
    conditions = ['A', 'B', 'C']
    labels = ['A: No replay, no KD', 'B: Freeze adapters + decay LR', 'C: Replay + KD']
    colors = ['#DC3545', '#FD7E14', '#28A745']
    markers = ['o', 's', '^']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for cond, label, color, marker in zip(conditions, labels, colors, markers):
        acc_matrix = np.array(results[f'condition_{cond}']['accuracy_matrix'])
        final_acc = compute_final_accuracies(acc_matrix)
        tasks = np.arange(1, len(final_acc) + 1)
        
        ax.plot(tasks, final_acc, marker=marker, markersize=8, linewidth=2,
                color=color, label=label, alpha=0.9)
    
    ax.set_xlabel('Task', fontsize=14)
    ax.set_ylabel('Final Accuracy', fontsize=14)
    ax.set_title('Domain-Incremental Forgetting: Final Accuracy per Task', fontsize=16, fontweight='bold')
    ax.set_xticks(tasks)
    ax.set_xticklabels([f'Task {i}' for i in tasks], fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Random (10 classes)')
    
    plt.tight_layout()
    save_path = 'results/phase3/accuracy_vs_task.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def create_gaussian_drift_animation():
    """Figure 1: Gaussian-means-drift animation A vs B."""
    # Run conditions A and B with mean tracking
    print("Running Condition A (drifting)...")
    cond_A = run_condition_with_tracking('A', use_replay=False, use_kd=False)
    
    print("Running Condition B (stable)...")
    cond_B = run_condition_with_tracking('B', use_replay=False, use_kd=False,
                                          freeze_adapters=True, input_proj_lr_scale=0.01)
    
    # For animation, we need PCA projection of means across tasks
    from sklearn.decomposition import PCA
    
    # Collect all means from both conditions for consistent PCA
    all_means = []
    for cond in [cond_A, cond_B]:
        for task_means in cond['means_per_task']:
            if len(task_means) > 0:
                all_means.append(task_means)
    
    if len(all_means) == 0:
        print("No means collected!")
        return
    
    all_means = np.vstack(all_means)
    pca = PCA(n_components=2)
    pca.fit(all_means)
    
    # Create animation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    conditions_data = [
        (cond_A, 'A: No Replay, No KD (Drifting)', '#DC3545'),
        (cond_B, 'B: Freeze Adapters + Decay LR (Stable)', '#28A745'),
    ]
    
    scatters = []
    for idx, (cond, title, color) in enumerate(conditions_data):
        ax = axes[idx]
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Initial scatter (task 0)
        if cond['means_per_task'] and len(cond['means_per_task'][0]) > 0:
            task0_means = cond['means_per_task'][0]
            projected = pca.transform(task0_means)
            scatter = ax.scatter(projected[:, 0], projected[:, 1], c=color, s=30, alpha=0.7)
            scatters.append(scatter)
            ax.set_xlim(projected[:, 0].min() - 1, projected[:, 0].max() + 1)
            ax.set_ylim(projected[:, 1].min() - 1, projected[:, 1].max() + 1)
        else:
            scatters.append(None)
    
    def update(frame):
        """Update animation for each task boundary."""
        for idx, (cond, title, color) in enumerate(conditions_data):
            if scatters[idx] is None:
                continue
            if frame < len(cond['means_per_task']):
                task_means = cond['means_per_task'][frame]
                if len(task_means) > 0:
                    projected = pca.transform(task_means)
                    scatters[idx].set_offsets(projected)
                    axes[idx].set_title(f'{title}\nTask {frame + 1}', fontsize=14, fontweight='bold')
        
        return scatters
    
    n_frames = max(len(cond_A['means_per_task']), len(cond_B['means_per_task']))
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1500, blit=False)
    
    save_path = 'results/phase3/gaussian_drift_animation.gif'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        import imageio
        tmp_dir = "/tmp/opencode/ngs_frames"
        os.makedirs(tmp_dir, exist_ok=True)
        for i in range(n_frames):
            update(i)
            fig.canvas.draw_idle()
            fig.savefig(os.path.join(tmp_dir, f"frame_{i:04d}.png"), dpi=100)
        with imageio.get_writer(save_path, mode="I", duration=1.5) as writer:
            for i in range(n_frames):
                writer.append_data(imageio.imread(os.path.join(tmp_dir, f"frame_{i:04d}.png")))
    except Exception:
        anim.save(save_path, writer="pillow", fps=1)
    
    print(f"Saved: {save_path}")
    plt.close(fig)


def _require_matplotlib():
    import matplotlib.pyplot as plt
    return plt


from copy import deepcopy

if __name__ == '__main__':
    print("="*60)
    print("PHASE 3: VISUALIZATIONS")
    print("="*60)
    
    # Figure 1: Gaussian drift animation
    print("\nCreating Figure 1: Gaussian-means-drift animation...")
    create_gaussian_drift_animation()
    
    # Figure 2: Accuracy vs task line chart
    print("\nCreating Figure 2: Accuracy-vs-task line chart...")
    plot_accuracy_vs_task()
    
    print("\nPhase 3 complete!")
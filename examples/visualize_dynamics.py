#!/usr/bin/env python
"""
Generate visualization plots for paper figures.

Plots topology dynamics, routing heatmaps, merge/split events, etc.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mngs.core.config import MNGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from mngs import build_mngs
from mngs.training.trainer import NGSTrainer, TrainConfig as TrainerConfig
from experiments.datasets import get_task_loaders


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def generate_topology_dynamics_plot(args):
    """Generate topology dynamics over continual learning."""
    config = MNGSConfig(
        routing=RoutingStrategy(args.routing),
        parameter_storage=ParameterStorage(args.param_storage),
        topology_control=TopologyControl(args.topology),
        memory_management=MemoryManagement(args.memory),
        max_k=args.max_k,
        k_init=args.k_init,
        top_k=args.top_k,
        latent_dim=args.latent_dim,
    )

    model = build_mngs(784, 10, config)

    trainer_config = TrainerConfig(
        lr=args.lr,
        epochs=args.epochs_per_task,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        replay_ratio=args.replay_ratio,
        kd_weight=args.kd_weight,
        device=args.device,
    )

    trainer = NGSTrainer(model, trainer_config)

    from experiments.datasets import ReplayBuffer
    replay_buffer = ReplayBuffer(max_size=args.replay_size)

    n_tasks = 5
    k_history = []
    split_history = []
    prune_history = []
    merge_history = []
    accuracy_matrix = np.zeros((n_tasks, n_tasks))

    for task_id in range(n_tasks):
        train_loader, test_loader, _ = get_task_loaders(
            'split_mnist', task_id, 2, args.batch_size
        )

        trainer.train_epoch(train_loader, replay_buffer=replay_buffer)

        # Track topology
        k_history.append(model.K)

        # Adapt topology and track events
        z_samples = torch.randn(200, args.latent_dim)
        action = model.adapt_density(z_samples, split_thresh=0.05, prune_thresh=0.01)

        split_history.append(action[1])
        prune_history.append(action[0])
        merge_history.append(0)

        # Evaluate
        from experiments.metrics import evaluate_model_on_task
        for eval_task in range(task_id + 1):
            _, eval_test_loader, _ = get_task_loaders('split_mnist', eval_task, 2, args.batch_size)
            acc = evaluate_model_on_task(model, eval_test_loader, args.device)
            accuracy_matrix[eval_task, task_id] = acc

        # Update replay
        import torch.nn.functional as F
        for x, y in train_loader:
            x_flat = x.view(x.size(0), -1)
            y_onehot = F.one_hot(y, num_classes=10).float()
            replay_buffer.add(x_flat, y_onehot)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # K over tasks
    axes[0, 0].plot(range(1, n_tasks + 1), k_history, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Task')
    axes[0, 0].set_ylabel('Active Units (K)')
    axes[0, 0].set_title('Topology Growth Over Tasks')
    axes[0, 0].grid(True, alpha=0.3)

    # Split/Prune/Merge events
    x = range(1, n_tasks + 1)
    width = 0.25
    axes[0, 1].bar([xi - width for xi in x], split_history, width, label='Split', alpha=0.7)
    axes[0, 1].bar(x, prune_history, width, label='Prune', alpha=0.7)
    axes[0, 1].bar([xi + width for xi in x], merge_history, width, label='Merge', alpha=0.7)
    axes[0, 1].set_xlabel('Task')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Topology Events Per Task')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Accuracy matrix
    im = axes[1, 0].imshow(accuracy_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 0].set_xlabel('Task Learned')
    axes[1, 0].set_ylabel('Task Evaluated')
    axes[1, 0].set_title('Accuracy Matrix')
    plt.colorbar(im, ax=axes[1, 0])

    # Final accuracy per task
    final_accs = np.diag(accuracy_matrix)
    axes[1, 1].bar(range(1, n_tasks + 1), final_accs, alpha=0.7, color='steelblue')
    axes[1, 1].set_xlabel('Task')
    axes[1, 1].set_ylabel('Final Accuracy')
    axes[1, 1].set_title('Final Accuracy Per Task')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "topology_dynamics.png", dpi=150)
    plt.close()

    print(f"Topology dynamics plot saved to {args.output_dir}/topology_dynamics.png")


def generate_routing_heatmap(args):
    """Generate routing heatmap for a trained model."""
    config = MNGSConfig(
        routing=RoutingStrategy.FACTORIZED_SUBSPACE,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.PRE_ALLOCATED_MASKED,
        max_k=128,
        k_init=32,
        top_k=8,
        latent_dim=32,
        num_subspaces=4,
    )

    model = build_mngs(784, 10, config)

    # Quick train on one task
    trainer_config = TrainerConfig(lr=1e-3, epochs=1, batch_size=256, device=args.device)
    trainer = NGSTrainer(model, trainer_config)

    train_loader, _, _ = get_task_loaders('split_mnist', 0, 2, 256)
    trainer.train_epoch(train_loader)

    # Get routing for test samples
    model.eval()
    with torch.no_grad():
        test_loader, _, _ = get_task_loaders('split_mnist', 0, 2, 100)
        x, y = next(iter(test_loader))
        x = x.view(x.size(0), -1)

        out = model(x)

        # Routing indices
        if hasattr(out, 'routing') and out.routing is not None:
            if isinstance(out.routing.indices, list):
                # Factorized - combine
                all_indices = torch.cat(out.routing.indices, dim=1).cpu().numpy()
                all_weights = torch.cat(out.routing.weights, dim=1).cpu().numpy()
            else:
                all_indices = out.routing.indices.cpu().numpy()
                all_weights = out.routing.weights.cpu().numpy()
        else:
            # Fallback: get routing from router directly
            z = model.p_down(x)
            routing_output = model.router(z)
            if isinstance(routing_output[0], list):
                all_indices = torch.cat(routing_output[0], dim=1).cpu().numpy()
                all_weights = torch.cat(routing_output[1], dim=1).cpu().numpy()
            else:
                all_indices = routing_output[0].cpu().numpy()
                all_weights = routing_output[1].cpu().numpy()

    # Plot heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Routing indices heatmap
    im1 = axes[0].imshow(all_indices[:50].T, aspect='auto', cmap='viridis')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Top-K Position')
    axes[0].set_title('Routing Indices (first 50 samples)')
    plt.colorbar(im1, ax=axes[0])

    # Routing weights heatmap
    im2 = axes[1].imshow(all_weights[:50].T, aspect='auto', cmap='hot')
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Top-K Position')
    axes[1].set_title('Routing Weights (first 50 samples)')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "routing_heatmap.png", dpi=150)
    plt.close()

    print(f"Routing heatmap saved to {args.output_dir}/routing_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--output-dir", default="./plots")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--routing", default="factorized_subspace")
    parser.add_argument("--param-storage", default="hypernetwork_generated")
    parser.add_argument("--topology", default="continuous_density")
    parser.add_argument("--memory", default="pre_allocated_masked")
    parser.add_argument("--max-k", type=int, default=512)
    parser.add_argument("--k-init", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs-per-task", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=10.0)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Generating topology dynamics plot...")
    generate_topology_dynamics_plot(args)

    print("Generating routing heatmap...")
    generate_routing_heatmap(args)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
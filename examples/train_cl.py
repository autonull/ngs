#!/usr/bin/env python
"""
Train NGS on Continual Learning benchmarks.

Reproduces Split-MNIST, Permuted-MNIST, CIFAR-100 results.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json

from mngs.core.config import MNGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from mngs import build_mngs
from mngs.training.trainer import NGSTrainer, TrainConfig as TrainerConfig
from experiments.datasets import get_task_loaders, PermutedMNIST, ReplayBuffer
from experiments.metrics import compute_metrics, evaluate_model_on_task
import copy


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config(args) -> MNGSConfig:
    """Create MNGSConfig from args."""
    return MNGSConfig(
        latent_dim=args.latent_dim,
        k_init=args.k_init,
        max_k=args.max_k,
        top_k=args.top_k,
        routing=RoutingStrategy(args.routing),
        parameter_storage=ParameterStorage(args.param_storage),
        topology_control=TopologyControl(args.topology),
        memory_management=MemoryManagement(args.memory),
        hypernetwork_code_dim=args.hypernet_code_dim,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        num_subspaces=args.num_subspaces,
        split_threshold=args.split_thresh,
        prune_threshold=args.prune_thresh,
        entropy_weight=args.entropy_weight,
        diversity_weight=args.diversity_weight,
    )


def run_split_mnist(args, config, device):
    """Run Split-MNIST continual learning."""
    n_tasks = 5
    classes_per_task = 2
    
    model = build_mngs(784, 10, config).to(device)
    
    trainer_config = TrainerConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs_per_task,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        replay_ratio=args.replay_ratio,
        kd_weight=args.kd_weight,
        kd_temperature=args.kd_temperature,
        split_thresh=args.split_thresh,
        prune_thresh=args.prune_thresh,
        max_spawn_per_call=args.max_spawn,
        adapt_every_epoch=args.adapt_every_epoch,
        device=device,
    )
    
    trainer = NGSTrainer(model, trainer_config, device=device)
    replay_buffer = ReplayBuffer(max_size=args.replay_size) if args.replay_size > 0 else None
    
    accuracy_matrix = np.zeros((n_tasks, n_tasks))
    active_units = []
    old_model = None
    
    for task_id in range(n_tasks):
        train_loader, test_loader, _ = get_task_loaders(
            'split_mnist', task_id, classes_per_task, args.batch_size
        )
        
        print(f"\nTask {task_id + 1}/{n_tasks}")
        trainer.train_epoch(train_loader, replay_buffer=replay_buffer, old_model=old_model)
        
        # Evaluate on all seen tasks
        for eval_task in range(task_id + 1):
            _, eval_test_loader, _ = get_task_loaders(
                'split_mnist', eval_task, classes_per_task, args.batch_size
            )
            acc = evaluate_model_on_task(model, eval_test_loader, device)
            accuracy_matrix[eval_task, task_id] = acc
            print(f"  Task {eval_task} accuracy: {acc:.4f}")
            
        # Update replay buffer
        if replay_buffer:
            import torch.nn.functional as F
            for x, y in train_loader:
                x_flat = x.view(x.size(0), -1).to(device)
                y_onehot = F.one_hot(y, num_classes=10).float().to(device)
                replay_buffer.add(x_flat, y_onehot)
        
        # Save old model for KD
        old_model = copy.deepcopy(model)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False
            
        active_units.append(model.K)
        print(f"  Active units: {model.K}")
        
    metrics = compute_metrics(accuracy_matrix, random_baseline=0.1)
    
    return {
        "accuracy_matrix": accuracy_matrix.tolist(),
        "active_units": active_units,
        "metrics": metrics.to_dict(),
    }


def run_permuted_mnist(args, config, device):
    """Run Permuted-MNIST continual learning."""
    n_tasks = 10
    
    model = build_mngs(784, 10, config).to(device)
    
    trainer_config = TrainerConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs_per_task,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        replay_ratio=args.replay_ratio,
        kd_weight=args.kd_weight,
        kd_temperature=args.kd_temperature,
        split_thresh=args.split_thresh,
        prune_thresh=args.prune_thresh,
        max_spawn_per_call=args.max_spawn,
        adapt_every_epoch=args.adapt_every_epoch,
        device=device,
    )
    
    trainer = NGSTrainer(model, trainer_config, device=device)
    replay_buffer = ReplayBuffer(max_size=args.replay_size) if args.replay_size > 0 else None
    permuted = PermutedMNIST(n_tasks=n_tasks, seed=args.seed)
    
    accuracy_matrix = np.zeros((n_tasks, n_tasks))
    active_units = []
    old_model = None
    
    for task_id in range(n_tasks):
        train_loader, test_loader = permuted.get_task_data(task_id, args.batch_size)
        
        print(f"\nTask {task_id + 1}/{n_tasks}")
        trainer.train_epoch(train_loader, replay_buffer=replay_buffer, old_model=old_model)
        
        for eval_task in range(task_id + 1):
            _, eval_test_loader = permuted.get_task_data(eval_task, args.batch_size)
            acc = evaluate_model_on_task(model, eval_test_loader, device)
            accuracy_matrix[eval_task, task_id] = acc
            print(f"  Task {eval_task} accuracy: {acc:.4f}")
            
        if replay_buffer:
            import torch.nn.functional as F
            for x, y in train_loader:
                x_flat = x.view(x.size(0), -1).to(device)
                y_onehot = F.one_hot(y, num_classes=10).float().to(device)
                replay_buffer.add(x_flat, y_onehot)
                
        old_model = copy.deepcopy(model)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False
            
        active_units.append(model.K)
        print(f"  Active units: {model.K}")
        
    metrics = compute_metrics(accuracy_matrix, random_baseline=0.1)
    
    return {
        "accuracy_matrix": accuracy_matrix.tolist(),
        "active_units": active_units,
        "metrics": metrics.to_dict(),
    }


def run_split_cifar100(args, config, device):
    """Run Split-CIFAR100 continual learning."""
    # This requires a backbone - use the experiments config
    from experiments.config import EXPERIMENTS
    exp_config = EXPERIMENTS['split_cifar100']
    
    n_tasks = exp_config.n_tasks
    classes_per_task = exp_config.classes_per_task
    d_in = exp_config.input_dim
    d_out = exp_config.output_dim
    
    config.latent_dim = args.latent_dim
    config.k_init = args.k_init
    config.max_k = args.max_k
    config.top_k = args.top_k
    
    model = build_mngs(d_in, d_out, config).to(device)
    
    trainer_config = TrainerConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs_per_task,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        replay_ratio=args.replay_ratio,
        kd_weight=args.kd_weight,
        kd_temperature=args.kd_temperature,
        device=device,
    )
    
    trainer = NGSTrainer(model, trainer_config, device=device)
    replay_buffer = ReplayBuffer(max_size=args.replay_size) if args.replay_size > 0 else None
    
    accuracy_matrix = np.zeros((n_tasks, n_tasks))
    active_units = []
    old_model = None
    
    for task_id in range(n_tasks):
        train_loader, test_loader, _ = get_task_loaders(
            'split_cifar100', task_id, classes_per_task, args.batch_size
        )
        
        print(f"\nTask {task_id + 1}/{n_tasks}")
        trainer.train_epoch(train_loader, replay_buffer=replay_buffer, old_model=old_model)
        
        for eval_task in range(task_id + 1):
            _, eval_test_loader, _ = get_task_loaders(
                'split_cifar100', eval_task, classes_per_task, args.batch_size
            )
            acc = evaluate_model_on_task(model, eval_test_loader, device)
            accuracy_matrix[eval_task, task_id] = acc
            print(f"  Task {eval_task} accuracy: {acc:.4f}")
            
        if replay_buffer:
            import torch.nn.functional as F
            for x, y in train_loader:
                x_flat = x.view(x.size(0), -1).to(device)
                y_onehot = F.one_hot(y, num_classes=d_out).float().to(device)
                replay_buffer.add(x_flat, y_onehot)
                
        old_model = copy.deepcopy(model)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False
            
        active_units.append(model.K)
        print(f"  Active units: {model.K}")
        
    metrics = compute_metrics(accuracy_matrix, random_baseline=1.0/d_out)
    
    return {
        "accuracy_matrix": accuracy_matrix.tolist(),
        "active_units": active_units,
        "metrics": metrics.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train NGS on Continual Learning benchmarks")
    
    # Experiment
    parser.add_argument("--experiment", default="split_mnist", 
                        choices=["split_mnist", "permuted_mnist", "split_cifar100"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    
    # Model
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--k-init", type=int, default=128)
    parser.add_argument("--max-k", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--routing", default="factorized_subspace", 
                        choices=["monolithic_mahalanobis", "factorized_subspace", "lsh_approximate"])
    parser.add_argument("--param-storage", default="hypernetwork_generated",
                        choices=["direct_adapter", "hypernetwork_generated"])
    parser.add_argument("--topology", default="continuous_density",
                        choices=["discrete_heuristic", "continuous_density"])
    parser.add_argument("--memory", default="pre_allocated_masked",
                        choices=["dynamic_growth", "pre_allocated_masked", "strict_capacity"])
    parser.add_argument("--hypernet-code-dim", type=int, default=8)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--num-subspaces", type=int, default=4)
    
    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs-per-task", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=10.0)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    
    # Topology
    parser.add_argument("--split-thresh", type=float, default=0.005)
    parser.add_argument("--prune-thresh", type=float, default=0.01)
    parser.add_argument("--merge-thresh", type=float, default=0.1)
    parser.add_argument("--max-spawn", type=int, default=5)
    parser.add_argument("--adapt-every-epoch", action="store_true", default=True)
    
    # Loss weights
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    
    # Output
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run for each seed
    all_results = []
    
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Running seed {seed}")
        print(f"{'='*60}")
        
        set_seed(seed)
        config = get_config(args)
        
        if args.experiment == "split_mnist":
            result = run_split_mnist(args, config, device)
        elif args.experiment == "permuted_mnist":
            result = run_permuted_mnist(args, config, device)
        elif args.experiment == "split_cifar100":
            result = run_split_cifar100(args, config, device)
        else:
            raise ValueError(f"Unknown experiment: {args.experiment}")
            
        result["seed"] = seed
        result["config"] = vars(args)
        all_results.append(result)
        
        # Save individual result
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / f"{args.experiment}_seed{seed}.json", "w") as f:
            json.dump(result, f, indent=2)
            
    # Aggregate
    metric_keys = all_results[0]["metrics"].keys()
    aggregated = {}
    for key in metric_keys:
        vals = [r["metrics"][key] for r in all_results]
        aggregated[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": vals,
        }
        
    print("\n" + "="*60)
    print(f"AGGREGATED RESULTS ({args.experiment})")
    print("="*60)
    for metric, vals in aggregated.items():
        print(f"  {metric}: {vals['mean']:.4f} ± {vals['std']:.4f}")
        
    # Save aggregated
    with open(Path(args.output_dir) / f"{args.experiment}_aggregated.json", "w") as f:
        json.dump({
            "experiment": args.experiment,
            "seeds": args.seeds,
            "individual": all_results,
            "aggregated": aggregated,
        }, f, indent=2)
        

if __name__ == "__main__":
    main()
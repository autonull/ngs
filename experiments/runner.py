"""
Main experiment runner for continual learning evaluation.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import Dict, List, Callable, Optional
from dataclasses import asdict
from copy import deepcopy
from tqdm import tqdm

from experiments.config import ExperimentConfig, EXPERIMENTS, ModelConfig, TrainConfig
from experiments.datasets import get_task_loaders, PermutedMNIST, RotatedMNIST, BlurryMNIST, NoisyMNIST, ReplayBuffer
from experiments.baselines import create_baseline
from experiments.metrics import (
    compute_metrics, run_continual_evaluation, evaluate_model_on_task, print_results
)
from experiments.trainers import get_trainer
from experiments.lean_ngs_trainer import train_lean_ngs, create_lean_ngs


def set_seed(seed: int):
    """Set all random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(
    config: ExperimentConfig,
    model_name: str,
    seed: int = 42,
    output_dir: str = './results',
    verbose: bool = True
) -> Dict:
    """Run a single experiment with given model and config."""
    set_seed(seed)
    device = config.device

    # Create model
    if model_name == 'lean_ngs':
        model = create_lean_ngs(config.input_dim, config.output_dim, **asdict(config.model))
        train_fn = train_lean_ngs
        train_kwargs = asdict(config.train)
    else:
        model = create_baseline(model_name, config.input_dim, config.output_dim)
        train_fn = get_trainer(model_name)
        train_kwargs = asdict(config.train)

    # Setup replay buffer for ER/LeanNGS
    replay_buffer = None
    if model_name in ['er', 'lean_ngs']:
        replay_buffer = ReplayBuffer(max_size=config.train.replay_size)

    # Task loader function
    def get_task_data(task_id):
        if config.dataset == 'permuted_mnist':
            permuted = PermutedMNIST(n_tasks=config.n_tasks, seed=seed)
            train_loader, test_loader = permuted.get_task_data(task_id, config.train.batch_size)
            classes = list(range(10))
        elif config.dataset == 'rotated_mnist':
            rotated = RotatedMNIST(n_tasks=config.n_tasks, seed=seed)
            train_loader, test_loader = rotated.get_task_data(task_id, config.train.batch_size)
            classes = list(range(10))
        elif config.dataset == 'blurry_mnist':
            blurry = BlurryMNIST(n_tasks=config.n_tasks, seed=seed)
            train_loader, test_loader = blurry.get_task_data(task_id, config.train.batch_size)
            classes = list(range(10))
        elif config.dataset == 'noisy_mnist':
            noisy = NoisyMNIST(n_tasks=config.n_tasks, seed=seed)
            train_loader, test_loader = noisy.get_task_data(task_id, config.train.batch_size)
            classes = list(range(10))
        else:
            train_loader, test_loader, classes = get_task_loaders(
                config.dataset, task_id, config.classes_per_task, config.train.batch_size
            )
        return train_loader, test_loader, classes

    # Run continual evaluation
    accuracy_matrix = np.zeros((config.n_tasks, config.n_tasks))
    active_units_list = []
    old_model = None

    for task_id in range(config.n_tasks):
        train_loader, test_loader, classes = get_task_data(task_id)

        # Train
        train_kwargs['replay_buffer'] = replay_buffer
        if model_name == 'lean_ngs':
            train_kwargs['old_model'] = old_model
        if model_name == 'lwf':
            train_kwargs['old_model'] = old_model
        train_fn(model, train_loader, task_id, device=device, **train_kwargs)

        # EWC consolidation after each task
        if model_name == 'ewc' and hasattr(model, 'consolidate'):
            model.consolidate(train_loader, device)

        # SI update after each task
        if model_name == 'si' and hasattr(model, 'update_omega'):
            model.update_omega(train_loader, device)

        # Evaluate on all seen tasks
        for eval_task in range(task_id + 1):
            _, eval_test_loader, _ = get_task_data(eval_task)
            acc = evaluate_model_on_task(model, eval_test_loader, device)
            accuracy_matrix[eval_task, task_id] = acc

        # Add current task data to replay buffer AFTER training
        if replay_buffer:
            for x, y in train_loader:
                x_flat = x.view(x.size(0), -1)
                replay_buffer.add(x_flat, torch.nn.functional.one_hot(y, num_classes=config.output_dim).float())

        # Save old model for KD/LwF
        if model_name in ['lean_ngs', 'lwf']:
            old_model = deepcopy(model)
            old_model.eval()
            for p in old_model.parameters():
                p.requires_grad = False

        # Track capacity
        if hasattr(model, 'K'):
            active_units_list.append(model.K)
        elif hasattr(model, 'active_mask'):
            active_units_list.append(model.active_mask.sum().item())
        else:
            active_units_list.append(0)

        if verbose:
            print(f"  Task {task_id} done. Acc on task {task_id}: {accuracy_matrix[task_id, task_id]:.4f}")

    # Compute metrics
    metrics = compute_metrics(accuracy_matrix)
    metrics.active_units = active_units_list[-1] if active_units_list else 0
    metrics.max_units = config.model.max_k if model_name == 'lean_ngs' else 0

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"{config.name}_{model_name}_seed{seed}.json")
    full_result = {
        'metrics': metrics.to_dict(),
        'accuracy_matrix': accuracy_matrix.tolist(),
        'active_units': active_units_list,
        'config': config.name,
        'model': model_name,
        'seed': seed,
    }
    with open(result_file, 'w') as f:
        json.dump(full_result, f, indent=2)

    if verbose:
        print_results(metrics, f"{config.name} - {model_name.upper()} (seed={seed})")

    return full_result


def run_all_experiments(
    experiment_names: List[str],
    model_names: List[str],
    seeds: List[int] = [42, 123, 456],
    output_dir: str = './results',
    verbose: bool = True
) -> Dict:
    """Run multiple experiments across models and seeds."""
    all_results = {}

    for exp_name in experiment_names:
        if exp_name not in EXPERIMENTS:
            print(f"Unknown experiment: {exp_name}")
            continue
        config = EXPERIMENTS[exp_name]

        for model_name in model_names:
            for seed in seeds:
                key = f"{exp_name}_{model_name}_seed{seed}"
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Running: {key}")
                    print(f"{'='*60}")

                try:
                    result = run_experiment(config, model_name, seed, output_dir, verbose)
                    all_results[key] = result
                except Exception as e:
                    print(f"Error in {key}: {e}")
                    all_results[key] = {'error': str(e)}

    # Save summary
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    return all_results


def aggregate_results(results: Dict, group_by: str = 'model') -> Dict:
    """Aggregate results across seeds."""
    grouped = {}

    for key, result in results.items():
        if 'error' in result:
            continue

        # Parse from key (filename) if not in result
        if 'config' in result:
            exp_name = result['config']
            model = result['model']
            seed = result['seed']
        else:
            parts = key.split('_')
            if len(parts) >= 3:
                exp_name = '_'.join(parts[:-2])
                model = parts[-2]
                seed = parts[-1].replace('seed', '')
            else:
                continue

        if group_by == 'model':
            group_key = model
        elif group_by == 'experiment':
            group_key = exp_name
        else:
            group_key = f"{exp_name}_{model}"

        if group_key not in grouped:
            grouped[group_key] = {
                'avg_final_accuracy': [],
                'avg_forgetting': [],
                'bwt': [],
                'fwt': [],
                'la': [],
            }

        # Handle both formats
        if 'metrics' in result:
            m = result['metrics']
        else:
            # Compute from accuracy_matrix
            acc_matrix = np.array(result['accuracy_matrix'])
            m = compute_metrics(acc_matrix).to_dict()

        grouped[group_key]['avg_final_accuracy'].append(m['avg_final_accuracy'])
        grouped[group_key]['avg_forgetting'].append(m['avg_forgetting'])
        grouped[group_key]['bwt'].append(m['bwt'])
        grouped[group_key]['fwt'].append(m['fwt'])
        grouped[group_key]['la'].append(m['la'])

    # Compute mean/std
    aggregated = {}
    for key, vals in grouped.items():
        aggregated[key] = {
            k: {'mean': np.mean(v), 'std': np.std(v)}
            for k, v in vals.items()
        }

    return aggregated
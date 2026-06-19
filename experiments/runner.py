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

from experiments.config import ExperimentConfig, EXPERIMENTS, ModelConfig, TrainConfig, as_train_kwargs
from experiments.datasets import get_task_loaders, PermutedMNIST, RotatedMNIST, BlurryMNIST, NoisyMNIST, ReplayBuffer
from experiments.datasets_tinyshakespeare import get_tinyshakespeare_loaders
from experiments.baselines import create_baseline
from experiments.metrics import (
    compute_metrics, run_continual_evaluation, evaluate_model_on_task, print_results
)
from experiments.trainers import get_trainer
from experiments.ngs_trainer import train_ngs, create_ngs, create_ngs_from_profile, PROFILE_TRAIN_CONFIGS


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
    if model_name.startswith('ngs_'):
        # Parse profile name (e.g., 'ngs_baseline')
        profile = model_name[4:]
        try:
            model = create_ngs_from_profile(profile, config.input_dim, config.output_dim)
        except ValueError as e:
            print(f"Warning: {e}. Using baseline profile.")
            model = create_ngs_from_profile('baseline', config.input_dim, config.output_dim)
        train_fn = train_ngs
        train_kwargs = as_train_kwargs(config.train)
        # Apply profile-specific train overrides
        if profile in PROFILE_TRAIN_CONFIGS:
            train_kwargs.update(PROFILE_TRAIN_CONFIGS[profile])
    else:
        model = create_baseline(model_name, config.input_dim, config.output_dim)
        train_fn = get_trainer(model_name)
        train_kwargs = as_train_kwargs(config.train)

    # Setup replay buffer for ER/NGS
    replay_buffer = None
    if model_name.startswith('ngs_') or model_name == 'er':
        replay_buffer = ReplayBuffer(max_size=config.train.replay_size)

    # Pre-create all task loaders once (efficient for split datasets)
    if config.dataset in ['permuted_mnist', 'rotated_mnist', 'blurry_mnist', 'noisy_mnist']:
        # These create fresh permutations/transforms per call - keep lazy
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
            return train_loader, test_loader, classes
    elif config.dataset == 'tinyshakespeare':
        def get_task_data(task_id):
            return get_tinyshakespeare_loaders(
                config.dataset, task_id, config.classes_per_task, config.train.batch_size,
                n_tasks=config.n_tasks, seq_len=config.input_dim
            )
    else:
        # Split datasets: pre-create all loaders once
        task_loaders = []
        for tid in range(config.n_tasks):
            task_loaders.append(get_task_loaders(
                config.dataset, tid, config.classes_per_task, config.train.batch_size,
                scenario=config.scenario
            ))
        def get_task_data(task_id):
            return task_loaders[task_id]

    # Run continual evaluation
    accuracy_matrix = np.zeros((config.n_tasks, config.n_tasks))
    active_units_list = []
    old_model = None

    for task_id in range(config.n_tasks):
        train_loader, test_loader, classes = get_task_data(task_id)

        # Store training data for replay buffer update (train_loader gets exhausted)
        train_data = []
        for x, y in train_loader:
            train_data.append((x, y))

        # Train
        train_kwargs['replay_buffer'] = replay_buffer
        ngs_models = ['ngs_baseline', 'ngs_cfg_net', 'ngs_ultra_edge', 'ngs_abl_hyper',
                      'ngs_baseline_lora', 'ngs_cfg_net_lora', 'ngs_abl_hyper_lora']
        if model_name in ['lwf'] + ngs_models or model_name.startswith('ngs_'):
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
        if replay_buffer is not None:
            for x, y in train_data:
                x_flat = x.view(x.size(0), -1)
                replay_buffer.add(x_flat, torch.nn.functional.one_hot(y, num_classes=config.output_dim).float())

        # Save old model for KD/LwF
        if model_name in ['lwf'] or model_name.startswith('ngs_'):
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
    metrics = compute_metrics(accuracy_matrix, random_baseline=1.0 / config.output_dim)
    metrics.active_units = active_units_list[-1] if active_units_list else 0
    metrics.max_units = model.config.max_k if hasattr(model, 'config') and hasattr(model.config, 'max_k') else 0

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
    """Aggregate results across seeds with confidence intervals."""
    from experiments.metrics import compute_confidence_interval
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

    # Compute mean/std + confidence intervals + raw values
    aggregated = {}
    for key, vals in grouped.items():
        aggregated[key] = {}
        for k, v in vals.items():
            arr = np.array(v)
            ci = compute_confidence_interval(arr)
            aggregated[key][k] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'ci_95': ci,
                'values': v,  # keep raw values for significance testing
                'n': len(v)
            }
    return aggregated


def compare_models_statistical(
    results: Dict,
    model_a: str,
    model_b: str,
    metrics: List[str] = None,
    paired: bool = True
) -> Dict:
    """Compare two models with statistical significance testing."""
    from experiments.metrics import compare_models_significance, effect_size_cohens_d
    
    agg = aggregate_results(results, group_by='model')
    
    if model_a not in agg or model_b not in agg:
        return {'error': f'Models not found in results: {model_a}, {model_b}'}
    
    # Extract raw values for each metric
    data_a = {}
    data_b = {}
    if metrics is None:
        metrics = ['avg_final_accuracy', 'avg_forgetting', 'bwt', 'fwt', 'la']
    
    for metric in metrics:
        if metric in agg[model_a] and metric in agg[model_b]:
            data_a[metric] = agg[model_a][metric]['values']
            data_b[metric] = agg[model_b][metric]['values']
    
    comparison = compare_models_significance(data_a, data_b, metrics, paired)
    
    # Add effect sizes
    for metric in metrics:
        if metric in data_a and metric in data_b:
            comparison[metric]['cohens_d'] = effect_size_cohens_d(
                np.array(data_a[metric]), np.array(data_b[metric])
            )
    
    return comparison
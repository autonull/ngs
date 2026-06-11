"""
Ablation study framework for LeanNGS.
Quick hyperparameter exploration with reduced compute.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from typing import Dict, List, Any, Callable
from copy import deepcopy
from itertools import product
from dataclasses import asdict

from experiments.config import ExperimentConfig, EXPERIMENTS, ModelConfig, TrainConfig
from experiments.datasets import get_task_loaders, PermutedMNIST, ReplayBuffer
from experiments.baselines import create_baseline
from experiments.metrics import compute_metrics, evaluate_model_on_task
from experiments.trainers import get_trainer
from experiments.lean_ngs_trainer import train_lean_ngs, create_lean_ngs


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_experiment(
    config: ExperimentConfig,
    model_name: str,
    model_kwargs: Dict = None,
    train_kwargs: Dict = None,
    seed: int = 42,
    device: str = 'cuda'
) -> Dict:
    """Run a single experiment with custom model/train kwargs."""
    set_seed(seed)
    device = config.device if hasattr(config, 'device') else device
    
    # Create model
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {}
    
    if model_name == 'lean_ngs':
        model = create_lean_ngs(config.input_dim, config.output_dim, **model_kwargs)
        train_fn = train_lean_ngs
        default_train_kwargs = asdict(config.train)
    else:
        model = create_baseline(model_name, config.input_dim, config.output_dim, **model_kwargs)
        train_fn = get_trainer(model_name)
        default_train_kwargs = asdict(config.train)
    
    # Merge train kwargs
    for k, v in default_train_kwargs.items():
        if k not in train_kwargs:
            train_kwargs[k] = v
    
    # Setup replay buffer
    replay_buffer = None
    if model_name in ['er', 'lean_ngs']:
        replay_buffer = ReplayBuffer(max_size=config.train.replay_size)
    
    # Task loader function
    def get_task_data(task_id):
        if config.dataset == 'permuted_mnist':
            permuted = PermutedMNIST(n_tasks=config.n_tasks, seed=seed)
            train_loader, test_loader = permuted.get_task_data(task_id, config.train.batch_size)
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
        
        # EWC consolidation
        if model_name == 'ewc' and hasattr(model, 'consolidate'):
            model.consolidate(train_loader, device)
        
        # SI update
        if model_name == 'si' and hasattr(model, 'update_omega'):
            model.update_omega(train_loader, device)
        
        # Evaluate on all seen tasks
        for eval_task in range(task_id + 1):
            _, eval_test_loader, _ = get_task_data(eval_task)
            acc = evaluate_model_on_task(model, eval_test_loader, device)
            accuracy_matrix[eval_task, task_id] = acc
        
        # Update replay buffer
        if replay_buffer:
            for x, y in train_loader:
                x_flat = x.view(x.size(0), -1)
                replay_buffer.add(x_flat, F.one_hot(y, num_classes=config.output_dim).float())
        
        # Save old model
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
    
    # Compute metrics
    metrics = compute_metrics(accuracy_matrix)
    metrics.active_units = active_units_list[-1] if active_units_list else 0
    metrics.max_units = config.model.max_k if model_name == 'lean_ngs' else 0
    
    return {
        'metrics': metrics.to_dict(),
        'accuracy_matrix': accuracy_matrix.tolist(),
        'active_units': active_units_list,
        'config': config.name,
        'model': model_name,
        'seed': seed,
        'model_kwargs': model_kwargs,
        'train_kwargs': train_kwargs,
    }


def run_ablation(
    experiment_name: str,
    model_name: str,
    param_grid: Dict[str, List],
    base_model_kwargs: Dict = None,
    base_train_kwargs: Dict = None,
    seed: int = 42,
    output_dir: str = './ablation_results'
) -> List[Dict]:
    """Run ablation study over parameter grid."""
    config = EXPERIMENTS[experiment_name]
    base_model_kwargs = base_model_kwargs or {}
    base_train_kwargs = base_train_kwargs or {}
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    results = []
    os.makedirs(output_dir, exist_ok=True)
    
    for i, combo in enumerate(combinations):
        param_dict = dict(zip(param_names, combo))
        
        # Split into model_kwargs and train_kwargs
        model_params = {}
        train_params = {}
        for k, v in param_dict.items():
            if k in ['d_latent', 'k_init', 'max_k', 'top_k', 'gamma_init', 'tau_init', 'mu_init_std', 'w_init_std']:
                model_params[k] = v
            else:
                train_params[k] = v
        
        model_kwargs = {**base_model_kwargs, **model_params}
        train_kwargs = {**base_train_kwargs, **train_params}
        
        print(f"\n[{i+1}/{len(combinations)}] Testing: {param_dict}")
        
        try:
            result = run_single_experiment(
                config, model_name, model_kwargs, train_kwargs, seed
            )
            result['params'] = param_dict
            # Remove non-serializable objects
            if 'replay_buffer' in result.get('train_kwargs', {}):
                del result['train_kwargs']['replay_buffer']
            if 'old_model' in result.get('train_kwargs', {}):
                del result['train_kwargs']['old_model']
            results.append(result)
            
            # Save intermediate
            fname = f"{experiment_name}_{model_name}_ablation_{i}.json"
            with open(os.path.join(output_dir, fname), 'w') as f:
                json.dump(result, f, indent=2)
                
            print(f"  Final Acc: {result['metrics']['avg_final_accuracy']:.4f}, "
                  f"Forgetting: {result['metrics']['avg_forgetting']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            results.append({'params': param_dict, 'error': str(e)})
    
    # Save all results
    with open(os.path.join(output_dir, f"{experiment_name}_{model_name}_ablation_all.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def analyze_ablation_results(results: List[Dict], metric: str = 'avg_final_accuracy') -> Dict:
    """Analyze ablation results and find best parameters."""
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        return {}
    
    # Group by each parameter
    param_importance = {}
    all_params = set()
    for r in valid_results:
        all_params.update(r['params'].keys())
    
    for param in all_params:
        values = {}
        for r in valid_results:
            if param in r['params']:
                val = r['params'][param]
                if val not in values:
                    values[val] = []
                values[val].append(r['metrics'][metric])
        
        # Compute mean for each value
        param_importance[param] = {v: np.mean(vals) for v, vals in values.items()}
    
    # Find best combination
    best_result = max(valid_results, key=lambda r: r['metrics'][metric])
    
    return {
        'best_params': best_result['params'],
        'best_metric': best_result['metrics'][metric],
        'param_importance': param_importance,
        'all_results': [(r['params'], r['metrics'][metric]) for r in valid_results]
    }


# Quick ablation configurations for rapid exploration (minimal grid)
QUICK_ABLATIONS = {
    'split_mnist': {
        'kd_weight': [0.0, 2.0],
        'split_thresh': [0.005, 0.01],
        'top_k': [8],
        'd_latent': [32],
        'replay_ratio': [1.0],
        'epochs_per_task': [1],
    },
    'split_fashion': {
        'kd_weight': [0.0, 2.0],
        'split_thresh': [0.005, 0.01],
        'top_k': [8],
        'd_latent': [32],
    },
    'split_cifar10': {
        'kd_weight': [0.0, 2.0],
        'split_thresh': [0.005, 0.01],
        'top_k': [8],
        'd_latent': [32, 64],
        'epochs_per_task': [1],
    },
}


def run_quick_ablation(experiment_name: str, output_dir: str = './ablation_results'):
    """Run predefined quick ablation for an experiment."""
    if experiment_name not in QUICK_ABLATIONS:
        print(f"No predefined ablation for {experiment_name}")
        return
    
    param_grid = QUICK_ABLATIONS[experiment_name]
    
    # For quick runs, use 1 epoch and 1 seed
    config = EXPERIMENTS[experiment_name]
    original_epochs = config.train.epochs_per_task
    config.train.epochs_per_task = 1  # Quick mode
    
    results = run_ablation(
        experiment_name, 'lean_ngs', param_grid,
        base_train_kwargs={'epochs_per_task': 1},
        seed=42,
        output_dir=output_dir
    )
    
    config.train.epochs_per_task = original_epochs
    
    # Analyze
    analysis = analyze_ablation_results(results)
    print("\n" + "="*60)
    print(f"ABLATION ANALYSIS: {experiment_name}")
    print("="*60)
    print(f"Best params: {analysis['best_params']}")
    print(f"Best metric: {analysis['best_metric']:.4f}")
    print("\nParameter importance:")
    for param, importance in analysis['param_importance'].items():
        print(f"  {param}: {importance}")
    
    return results, analysis


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--experiment', default='split_mnist', choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--output-dir', default='./ablation_results')
    args = parser.parse_args()
    
    run_quick_ablation(args.experiment, args.output_dir)
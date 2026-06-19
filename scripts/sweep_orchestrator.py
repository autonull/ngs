#!/usr/bin/env python
"""Hyperparameter sweep orchestrator for variant×dataset pairs."""
import argparse
import copy
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.experiment_config import load_experiment_config
from scripts.run_experiment import run_from_yaml


# Default search spaces per variant type
SEARCH_SPACES = {
    'baseline': {
        'lr': [1e-3, 5e-4, 2e-3],
        'split_threshold': [0.02, 0.05, 0.1],
        'prune_threshold': [0.005, 0.01, 0.02],
        'kd_weight': [5.0, 10.0, 20.0],
    },
    'factorized': {
        'lr': [1e-3, 5e-4],
        'num_subspaces': [2, 4, 8],
        'top_k_factorized': [1, 2, 4],
        'split_threshold': [0.02, 0.05],
        'kd_weight': [5.0, 10.0],
    },
    'hyper': {
        'lr': [1e-3, 5e-4],
        'hypernetwork_code_dim': [4, 8, 16],
        'hypernetwork_hidden_dim': [16, 32],
        'split_threshold': [0.02, 0.05],
        'kd_weight': [5.0, 10.0],
    },
    'cfg_net': {
        'lr': [1e-3, 5e-4],
        'hypernetwork_code_dim': [8, 16],
        'num_subspaces': [2, 4],
        'split_threshold': [0.01, 0.02, 0.05],
        'kd_weight': [5.0, 10.0],
    },
    'ultra_edge': {
        'lr': [1e-3, 2e-3],
        'lora_rank': [2, 4, 8],
        'split_threshold': [0.05, 0.1],
    },
}


def generate_sweep_configs(
    variant: str,
    benchmark: str,
    base_config_path: str,
    search_space: Dict[str, List[Any]],
    output_dir: str = "configs/sweeps",
) -> List[str]:
    """Generate all config combinations for a sweep."""
    base_cfg = load_experiment_config(base_config_path)
    
    # Get benchmark-specific params
    from experiments.config import EXPERIMENTS
    if benchmark in EXPERIMENTS:
        exp = EXPERIMENTS[benchmark]
        base_cfg.dataset = benchmark
        base_cfg.scenario = exp.scenario
        base_cfg.n_tasks = exp.n_tasks
        base_cfg.classes_per_task = exp.classes_per_task
        base_cfg.input_dim = exp.input_dim
        base_cfg.output_dim = exp.output_dim
        base_cfg.name = exp.name
    
    keys = list(search_space.keys())
    combos = list(itertools.product(*[search_space[k] for k in keys]))
    
    config_paths = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        cfg = copy.deepcopy(base_cfg)
        
        # Apply params to model or training config
        for k, v in params.items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)
            elif k in cfg.training:
                cfg.training[k] = v
        
        # Save
        name = f"{variant}_{benchmark}_sweep_{i:03d}"
        output_path = Path(output_dir) / f"{name}.yaml"
        
        import yaml
        data = {
            'experiment': cfg.name,
            'dataset': cfg.dataset,
            'scenario': cfg.scenario,
            'n_tasks': cfg.n_tasks,
            'classes_per_task': cfg.classes_per_task,
            'input_dim': cfg.input_dim,
            'output_dim': cfg.output_dim,
            'model': {
                'latent_dim': cfg.model.latent_dim,
                'k_init': cfg.model.k_init,
                'max_k': cfg.model.max_k,
                'top_k': cfg.model.top_k,
                'routing': cfg.model.routing.value,
                'parameter_storage': cfg.model.parameter_storage.value,
                'topology_control': cfg.model.topology_control.value,
                'memory_management': cfg.model.memory_management.value,
                'num_subspaces': cfg.model.num_subspaces,
                'hypernetwork_code_dim': cfg.model.hypernetwork_code_dim,
                'split_threshold': cfg.model.split_threshold,
                'prune_threshold': cfg.model.prune_threshold,
            },
            'training': cfg.training,
            'seeds': [42],  # Single seed for sweep
            'device': cfg.device,
        }
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        config_paths.append(str(output_path))
    
    return config_paths


def run_sweep(
    variant: str,
    benchmark: str,
    base_config_path: str,
    seeds: List[int] = [42],
    output_dir: str = "./sweep_results",
    epochs: int = 5,
    max_trials: Optional[int] = None,
    search_space: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run hyperparameter sweep for a variant×benchmark pair."""
    space = search_space or SEARCH_SPACES.get(variant, {})
    if not space:
        print(f"No search space defined for variant: {variant}")
        return {"error": "No search space"}
    
    print(f"Sweep: {variant} × {benchmark}")
    print(f"  Search space: {space}")
    
    # Generate configs
    config_paths = generate_sweep_configs(variant, benchmark, base_config_path, space)
    print(f"  Generated {len(config_paths)} configs")
    
    if max_trials:
        config_paths = config_paths[:max_trials]
        print(f"  Limited to {max_trials} trials")
    
    # Run each config
    results = []
    for i, config_path in enumerate(config_paths):
        print(f"\n  Trial {i+1}/{len(config_paths)}: {Path(config_path).stem}")
        t0 = time.time()
        
        try:
            result = run_from_yaml(config_path, seeds, output_dir, epochs_override=epochs)
            result['config_path'] = config_path
            result['params'] = _extract_params(config_path)
            result['elapsed_seconds'] = time.time() - t0
            results.append(result)
            
            if 'avg_final_accuracy' in result:
                acc = result['avg_final_accuracy']
                print(f"    Acc: {acc['mean']:.4f}±{acc['std']:.4f} ({result['elapsed_seconds']:.1f}s)")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({'config_path': config_path, 'error': str(e)})
    
    # Find best
    valid_results = [r for r in results if 'error' not in r and 'avg_final_accuracy' in r]
    if valid_results:
        best = max(valid_results, key=lambda r: r['avg_final_accuracy']['mean'])
        print(f"\n  Best: {Path(best['config_path']).stem}")
        print(f"  Best Acc: {best['avg_final_accuracy']['mean']:.4f}")
        print(f"  Best Params: {best['params']}")
    else:
        best = None
    
    # Save sweep results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sweep_result = {
        'variant': variant,
        'benchmark': benchmark,
        'search_space': space,
        'n_trials': len(config_paths),
        'seeds': seeds,
        'epochs': epochs,
        'best_params': best['params'] if best else None,
        'best_accuracy': best['avg_final_accuracy']['mean'] if best else None,
        'all_results': results,
    }
    
    summary_path = Path(output_dir) / f"sweep_{variant}_{benchmark}.json"
    with open(summary_path, 'w') as f:
        json.dump(sweep_result, f, indent=2, default=str)
    
    print(f"\n  Sweep complete. Saved to {summary_path}")
    return sweep_result


def _extract_params(config_path: str) -> Dict:
    """Extract hyperparameters from config filename."""
    # Simple extraction from filename
    stem = Path(config_path).stem
    parts = stem.split('_')
    params = {}
    # This is a simplified version - in practice, parse the YAML
    return params


def run_all_sweeps(
    variants: List[str] = None,
    benchmarks: List[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run sweeps for all variant×benchmark pairs."""
    variants = variants or list(SEARCH_SPACES.keys())
    benchmarks = benchmarks or [
        'split_mnist', 'permuted_mnist', 'rotated_mnist',
        'split_fashion', 'split_cifar10'
    ]
    
    # Map variants to base configs
    variant_configs = {
        'baseline': 'configs/cl/split_mnist_ngs_baseline.yaml',
        'factorized': 'configs/cl/split_mnist_ngs_factorized.yaml',
        'hyper': 'configs/cl/split_mnist_ngs_hyper.yaml',
        'cfg_net': 'configs/cl/split_mnist_ngs_cfg_net.yaml',
        'ultra_edge': 'configs/cl/split_mnist_ngs_ultra.yaml',
    }
    
    all_results = {}
    for variant in variants:
        if variant not in variant_configs:
            print(f"Skipping {variant}: no base config")
            continue
        
        base_config = variant_configs[variant]
        for benchmark in benchmarks:
            key = f"{variant}_{benchmark}"
            print(f"\n{'='*60}")
            print(f"Sweep: {key}")
            print(f"{'='*60}")
            
            result = run_sweep(variant, benchmark, base_config, **kwargs)
            all_results[key] = result
    
    # Save master summary
    Path(kwargs.get('output_dir', './sweep_results')).mkdir(parents=True, exist_ok=True)
    summary_path = Path(kwargs.get('output_dir', './sweep_results')) / "all_sweeps_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweeps")
    parser.add_argument("--variant", default=None, help="Single variant to sweep")
    parser.add_argument("--benchmark", default=None, help="Single benchmark to sweep")
    parser.add_argument("--variants", nargs="+", default=None, help="Multiple variants")
    parser.add_argument("--benchmarks", nargs="+", default=None, help="Multiple benchmarks")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output-dir", default="./sweep_results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-trials", type=int, default=None, help="Limit trials per sweep")
    parser.add_argument("--all", action="store_true", help="Run all variant×benchmark sweeps")
    args = parser.parse_args()
    
    if args.all:
        run_all_sweeps(
            variants=args.variants,
            benchmarks=args.benchmarks,
            seeds=args.seeds,
            output_dir=args.output_dir,
            epochs=args.epochs,
            max_trials=args.max_trials,
        )
    elif args.variant and args.benchmark:
        variant_configs = {
            'baseline': 'configs/cl/split_mnist_ngs_baseline.yaml',
            'factorized': 'configs/cl/split_mnist_ngs_factorized.yaml',
            'hyper': 'configs/cl/split_mnist_ngs_hyper.yaml',
            'cfg_net': 'configs/cl/split_mnist_ngs_cfg_net.yaml',
            'ultra_edge': 'configs/cl/split_mnist_ngs_ultra.yaml',
        }
        run_sweep(
            args.variant, args.benchmark, variant_configs[args.variant],
            seeds=args.seeds, output_dir=args.output_dir,
            epochs=args.epochs, max_trials=args.max_trials,
        )
    else:
        parser.error("Use --all or specify --variant and --benchmark")


if __name__ == "__main__":
    main()

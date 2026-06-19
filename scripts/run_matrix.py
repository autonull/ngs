#!/usr/bin/env python
"""Matrix experiment runner: all variants × all benchmarks × seeds."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.experiment_config import discover_configs, load_experiment_config
from scripts.run_experiment import run_from_yaml


# Canonical variant configs (9 variants)
VARIANT_CONFIGS = {
    'baseline': 'configs/cl/split_mnist_ngs_baseline.yaml',
    'factorized': 'configs/cl/split_mnist_ngs_factorized.yaml',
    'hyper': 'configs/cl/split_mnist_ngs_hyper.yaml',
    'cfg_net': 'configs/cl/split_mnist_ngs_cfg_net.yaml',
    'merge_aware': 'configs/cl/split_mnist_ngs_merge.yaml',
    'meta_learned': 'configs/cl/split_mnist_ngs_meta.yaml',
    'ultra_edge': 'configs/cl/split_mnist_ngs_ultra.yaml',
    'attention': 'configs/cl/split_mnist_ngs_attention.yaml',
    'hierarchical': 'configs/cl/split_mnist_ngs_hierarchical.yaml',
}

# CL Benchmarks (8 datasets)
CL_BENCHMARKS = [
    'split_mnist',
    'permuted_mnist', 
    'rotated_mnist',
    'blurry_mnist',
    'noisy_mnist',
    'split_fashion',
    'split_cifar10',
    'digits',
]


def create_variant_configs_for_benchmark(variant_name: str, benchmark: str, 
                                           base_config_path: str) -> str:
    """Create a variant config adapted for a specific benchmark."""
    cfg = load_experiment_config(base_config_path)
    
    # Update benchmark-specific params
    from experiments.config import EXPERIMENTS
    if benchmark in EXPERIMENTS:
        exp = EXPERIMENTS[benchmark]
        cfg.dataset = benchmark
        cfg.scenario = exp.scenario
        cfg.n_tasks = exp.n_tasks
        cfg.classes_per_task = exp.classes_per_task
        cfg.input_dim = exp.input_dim
        cfg.output_dim = exp.output_dim
        cfg.name = exp.name
    
    # Save adapted config
    output_path = Path(f"configs/generated/{variant_name}_{benchmark}.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
        'seeds': cfg.seeds,
        'device': cfg.device,
    }
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    return str(output_path)


def run_matrix(
    variants: List[str] = None,
    benchmarks: List[str] = None,
    seeds: List[int] = [42],
    output_dir: str = "./matrix_results",
    epochs: int = 1,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run full variant × benchmark matrix."""
    variants = variants or list(VARIANT_CONFIGS.keys())
    benchmarks = benchmarks or CL_BENCHMARKS
    
    print(f"Matrix: {len(variants)} variants × {len(benchmarks)} benchmarks × {len(seeds)} seeds")
    print(f"Variants: {variants}")
    print(f"Benchmarks: {benchmarks}")
    
    if dry_run:
        print("\nDry run - would execute:")
        for v in variants:
            for b in benchmarks:
                print(f"  {v} × {b}")
        return {"status": "dry_run", "matrix_size": len(variants) * len(benchmarks) * len(seeds)}
    
    results = {}
    for variant in variants:
        base_config = VARIANT_CONFIGS.get(variant)
        if not base_config:
            print(f"Unknown variant: {variant}")
            continue
        
        for benchmark in benchmarks:
            key = f"{variant}_{benchmark}"
            print(f"\n{'='*60}")
            print(f"Running: {key}")
            print(f"{'='*60}")
            
            try:
                # Create adapted config
                config_path = create_variant_configs_for_benchmark(variant, benchmark, base_config)
                
                # Run experiment
                t0 = time.time()
                result = run_from_yaml(config_path, seeds, output_dir, epochs_override=epochs)
                elapsed = time.time() - t0
                
                result['elapsed_seconds'] = elapsed
                results[key] = result
                print(f"  Completed in {elapsed:.1f}s")
                
                if 'avg_final_accuracy' in result:
                    acc = result['avg_final_accuracy']
                    print(f"  Accuracy: {acc['mean']:.4f}±{acc['std']:.4f}")
                    
            except Exception as e:
                print(f"  ERROR: {e}")
                results[key] = {'error': str(e)}
    
    # Save matrix summary
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(output_dir) / "matrix_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nMatrix complete. Summary: {summary_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run variant × benchmark matrix")
    parser.add_argument("--variants", nargs="+", default=None, help="Variant names (default: all)")
    parser.add_argument("--benchmarks", nargs="+", default=None, help="Benchmark names (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output-dir", default="./matrix_results")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per task (smoke test)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    run_matrix(args.variants, args.benchmarks, args.seeds, args.output_dir, args.epochs, args.dry_run)


if __name__ == "__main__":
    main()

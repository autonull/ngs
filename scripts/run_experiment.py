#!/usr/bin/env python
"""Unified experiment runner using YAML configs - smoke test version."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.experiment_config import load_experiment_config, discover_configs, config_to_train_kwargs
from experiments.runner import run_experiment as run_cl_experiment
from experiments.config import EXPERIMENTS


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_from_yaml(
    config_path: str,
    seeds: List[int] = [42],
    output_dir: str = "./results",
    device: str = "cuda",
    epochs_override: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run experiment from YAML config."""
    cfg = load_experiment_config(config_path)
    device = device if torch.cuda.is_available() else "cpu"
    
    if epochs_override:
        cfg.training['epochs_per_task'] = epochs_override
    
    print(f"Experiment: {cfg.name}")
    print(f"  Dataset: {cfg.dataset} ({cfg.scenario})")
    print(f"  Tasks: {cfg.n_tasks}, Classes/task: {cfg.classes_per_task}")
    print(f"  Model: routing={cfg.model.routing.value}, "
          f"storage={cfg.model.parameter_storage.value}, "
          f"topology={cfg.model.topology_control.value}")
    print(f"  Seeds: {seeds}, Device: {device}")
    
    if dry_run:
        return {"status": "dry_run", "config": str(config_path)}
    
    # Map YAML config to EXPERIMENTS format
    exp_name = cfg.name.lower().replace('-', '_')
    if exp_name not in EXPERIMENTS:
        # Create temporary experiment config
        from experiments.config import ExperimentConfig, ModelConfig, TrainConfig
        exp_cfg = ExperimentConfig(
            name=cfg.name,
            dataset=cfg.dataset,
            scenario=cfg.scenario,
            n_tasks=cfg.n_tasks,
            classes_per_task=cfg.classes_per_task,
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
        )
        # Apply training overrides
        t = cfg.training
        exp_cfg.train = TrainConfig(
            lr=t.get('lr', 1e-3),
            weight_decay=t.get('weight_decay', 1e-4),
            epochs_per_task=t.get('epochs_per_task', 5),
            batch_size=t.get('batch_size', 256),
            replay_size=t.get('replay_size', 50000),
            replay_ratio=t.get('replay_ratio', 1.0),
            kd_weight=t.get('kd_weight', 10.0),
            kd_temperature=t.get('kd_temperature', 2.0),
            split_thresh=cfg.model.split_threshold,
            prune_thresh=cfg.model.prune_threshold,
        )
    else:
        exp_cfg = EXPERIMENTS[exp_name]
        # Override with YAML values
        t = cfg.training
        exp_cfg.train.lr = t.get('lr', exp_cfg.train.lr)
        exp_cfg.train.epochs_per_task = t.get('epochs_per_task', exp_cfg.train.epochs_per_task)
        exp_cfg.train.batch_size = t.get('batch_size', exp_cfg.train.batch_size)
        exp_cfg.train.replay_ratio = t.get('replay_ratio', exp_cfg.train.replay_ratio)
        exp_cfg.train.kd_weight = t.get('kd_weight', exp_cfg.train.kd_weight)
        exp_cfg.train.split_thresh = cfg.model.split_threshold
        exp_cfg.train.prune_thresh = cfg.model.prune_threshold
    
    # Determine model name from config
    model_name = _config_to_model_name(cfg.model)
    
    all_results = []
    for seed in seeds:
        set_seed(seed)
        print(f"\n  Seed {seed}...")
        result = run_cl_experiment(exp_cfg, model_name, seed, output_dir, verbose=True)
        result['seed'] = seed
        all_results.append(result)
    
    return _aggregate_results(all_results, cfg.name, output_dir)


def _config_to_model_name(model_cfg: 'NGSConfig') -> str:
    """Map NGSConfig to experiment model name."""
    routing = model_cfg.routing.value
    storage = model_cfg.parameter_storage.value
    topology = model_cfg.topology_control.value
    
    if routing == 'monolithic_mahalanobis' and storage == 'direct_adapter':
        return 'ngs_baseline'
    elif routing == 'factorized_subspace' and storage == 'hypernetwork_generated' and topology == 'continuous_density':
        return 'ngs_cfg_net'
    elif routing == 'factorized_subspace' and storage == 'hypernetwork_generated' and topology == 'discrete_heuristic':
        return 'ngs_abl_hyper'
    elif routing == 'lsh_approximate' and storage == 'lora':
        return 'ngs_ultra_edge'
    elif 'lora' in storage:
        base = _config_to_model_name(type('cfg', (), {'routing': model_cfg.routing, 
                                                        'parameter_storage': type('s', (), {'value': 'direct_adapter'})(),
                                                        'topology_control': model_cfg.topology_control})())
        return base + '_lora'
    return 'ngs_baseline'


def _aggregate_results(results: List[Dict], name: str, output_dir: str) -> Dict[str, Any]:
    """Aggregate results across seeds."""
    metric_keys = set()
    for r in results:
        if "metrics" in r and isinstance(r["metrics"], dict):
            metric_keys.update(r["metrics"].keys())
    
    aggregated = {"seeds": len(results), "individual": results}
    for key in sorted(metric_keys):
        vals = [r["metrics"].get(key, float("nan")) for r in results]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if vals:
            aggregated[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "values": vals,
            }
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{name}_aggregated.json"
    with open(path, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    
    print(f"\n  Aggregated -> {path}")
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Run NGS experiment from YAML config")
    parser.add_argument("config", nargs="?", help="Path to YAML config file")
    parser.add_argument("--list", action="store_true", help="List available configs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs_per_task")
    parser.add_argument("--dry-run", action="store_true", help="Show config without running")
    args = parser.parse_args()
    
    if args.list:
        configs = discover_configs("configs")
        print("Available configs:")
        for name, path in sorted(configs.items()):
            print(f"  {name}")
        return
    
    if not args.config:
        parser.error("Config file required (or use --list)")
    
    run_from_yaml(args.config, args.seeds, args.output_dir, args.device, args.epochs, args.dry_run)


if __name__ == "__main__":
    main()

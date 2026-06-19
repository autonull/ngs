#!/usr/bin/env python
"""Checkpoint-enabled experiment runner with resume capability."""
import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.experiment_config import load_experiment_config
from experiments.runner import run_experiment as run_cl_experiment
from experiments.config import EXPERIMENTS
from scripts.callbacks import CheckpointCallback, CheckpointConfig, ProgressTracker, create_callbacks


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config_hash(config: Dict) -> str:
    """Generate deterministic hash for config."""
    def serialize(obj):
        if hasattr(obj, 'value'):  # Enum
            return obj.value
        if hasattr(obj, '__dict__'):  # Dataclass
            return {k: serialize(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [serialize(v) for v in obj]
        return obj
    
    serializable = serialize(config)
    s = json.dumps(serializable, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def run_with_checkpoints(
    config_path: str,
    seeds: List[int] = [42],
    output_dir: str = "./results",
    device: str = "cuda",
    epochs_override: Optional[int] = None,
    checkpoint_dir: str = "./checkpoints",
    resume: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run experiment with checkpointing and resume support."""
    from scripts.experiment_config import load_experiment_config
    cfg = load_experiment_config(config_path)
    device = device if torch.cuda.is_available() else "cpu"
    
    if epochs_override:
        cfg.training['epochs_per_task'] = epochs_override
    
    # Generate experiment ID from config
    exp_id = f"{cfg.name}_{config_hash(cfg.model.__dict__)}"
    exp_checkpoint_dir = Path(checkpoint_dir) / exp_id
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment: {cfg.name} (ID: {exp_id})")
    print(f"  Dataset: {cfg.dataset} ({cfg.scenario})")
    print(f"  Seeds: {seeds}, Device: {device}")
    print(f"  Checkpoint dir: {exp_checkpoint_dir}")
    
    if dry_run:
        return {"status": "dry_run", "config": str(config_path), "exp_id": exp_id}
    
    # Check for existing progress
    progress_tracker = ProgressTracker(str(exp_checkpoint_dir / "training_state.json"))
    existing_state = progress_tracker.load_state() if resume else None
    
    all_results = []
    start_seed_idx = 0
    
    if existing_state:
        print(f"  Resuming from epoch {existing_state.get('epoch', 0)}")
        completed_seeds = existing_state.get('completed_seeds', [])
        start_seed_idx = len(completed_seeds)
        print(f"  Completed seeds: {completed_seeds}, starting from index {start_seed_idx}")
    
    for seed_idx, seed in enumerate(seeds[start_seed_idx:], start=start_seed_idx):
        set_seed(seed)
        print(f"\n  Seed {seed} ({seed_idx+1}/{len(seeds)})")
        
        seed_checkpoint_dir = exp_checkpoint_dir / f"seed_{seed}"
        seed_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = create_callbacks(
            early_stopping=True,
            checkpointing=True,
            logging=True,
            progress_tracking=True,
            early_stopping_config={'patience': 10, 'min_epochs': 3},
            checkpointing_config={'save_every': 5, 'dir': str(seed_checkpoint_dir)},
            log_path=str(seed_checkpoint_dir / "metrics.jsonl"),
            state_path=str(seed_checkpoint_dir / "training_state.json"),
        )
        
        exp_name = cfg.name.lower().replace('-', '_')
        if exp_name not in EXPERIMENTS:
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
            t = cfg.training
            exp_cfg.train.lr = t.get('lr', exp_cfg.train.lr)
            exp_cfg.train.epochs_per_task = t.get('epochs_per_task', exp_cfg.train.epochs_per_task)
            exp_cfg.train.batch_size = t.get('batch_size', exp_cfg.train.batch_size)
            exp_cfg.train.replay_ratio = t.get('replay_ratio', exp_cfg.train.replay_ratio)
            exp_cfg.train.kd_weight = t.get('kd_weight', exp_cfg.train.kd_weight)
            exp_cfg.train.split_thresh = cfg.model.split_threshold
            exp_cfg.train.prune_thresh = cfg.model.prune_threshold
        
        model_name = _config_to_model_name(cfg.model)
        
        result = _run_with_callbacks(exp_cfg, model_name, seed, output_dir, callbacks, verbose=True)
        result['seed'] = seed
        all_results.append(result)
        
        progress_state = progress_tracker.load_state() or {}
        progress_state['completed_seeds'] = progress_state.get('completed_seeds', []) + [seed]
        progress_state['last_completed'] = seed
        with open(exp_checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(progress_state, f, default=str)
    
    return _aggregate_results(all_results, cfg.name, output_dir, exp_id)


def _config_to_model_name(model_cfg) -> str:
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
    return 'ngs_baseline'


def _run_with_callbacks(exp_cfg, model_name, seed, output_dir, callbacks, verbose=True):
    from experiments.runner import run_experiment
    result = run_experiment(exp_cfg, model_name, seed, output_dir, verbose)
    return result


def _aggregate_results(results: List[Dict], name: str, output_dir: str, exp_id: str) -> Dict[str, Any]:
    import numpy as np
    metric_keys = set()
    for r in results:
        if "metrics" in r and isinstance(r["metrics"], dict):
            metric_keys.update(r["metrics"].keys())
    
    aggregated = {"seeds": len(results), "individual": results, "exp_id": exp_id}
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
    path = Path(output_dir) / f"{name}_{exp_id}_aggregated.json"
    with open(path, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    
    print(f"\n  Aggregated -> {path}")
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Run experiment with checkpointing")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoints")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    run_with_checkpoints(
        args.config,
        seeds=args.seeds,
        output_dir=args.output_dir,
        device=args.device,
        epochs_override=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

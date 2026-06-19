#!/usr/bin/env python
"""Hyperparameter sweep runner (Phase 1.2). Supports grid search, Optuna, and Ray Tune."""

import argparse
import copy
import json
import os
import sys
import itertools
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_grid_sweep(
    experiment_name: str,
    param_grid: Dict[str, List[Any]],
    seeds: List[int] = [42],
    output_dir: str = "./sweep_results",
    device: str = "cuda",
) -> Dict[str, Any]:
    """Grid search over hyperparameter combinations."""
    from experiments.runner import run_experiment
    from experiments.config import EXPERIMENTS

    base_config = copy.deepcopy(EXPERIMENTS[experiment_name])
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    print(f"Grid sweep: {len(combos)} combinations x {len(seeds)} seeds")

    all_results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        config = copy.deepcopy(base_config)

        # Apply parameters to config
        for k, v in params.items():
            if hasattr(config.train, k):
                setattr(config.train, k, v)
            elif hasattr(config.model, k):
                setattr(config.model, k, v)
            else:
                # Try as top-level config attribute
                setattr(config, k, v)

        for seed in seeds:
            result = run_experiment(config, "ngs_baseline", seed=seed, output_dir=output_dir, verbose=False)
            result["params"] = params
            result["seed"] = seed
            all_results.append(result)

            acc = result.get("metrics", {}).get("avg_final_accuracy", 0)
            forget = result.get("metrics", {}).get("avg_forgetting", 1)
            print(f"  {params}: acc={acc:.4f}, forget={forget:.4f} (seed={seed})")

    # Find best
    best = max(all_results, key=lambda r: r.get("metrics", {}).get("avg_final_accuracy", 0))

    return {
        "experiment": experiment_name,
        "param_grid": param_grid,
        "n_trials": len(combos) * len(seeds),
        "best_params": best["params"],
        "best_metrics": best.get("metrics", {}),
        "all_results": all_results,
    }


def run_optuna_sweep(
    experiment_name: str,
    n_trials: int = 50,
    seeds: List[int] = [42],
    output_dir: str = "./sweep_results",
    device: str = "cuda",
) -> Dict[str, Any]:
    """Optuna-based hyperparameter optimization."""
    try:
        import optuna
    except ImportError:
        print("optuna not installed. Install with: pip install optuna")
        return {"error": "optuna not installed"}

    from experiments.hpo import run_hpo
    study = run_hpo(experiment_name, n_trials=n_trials, seed=seeds[0])
    return {
        "experiment": experiment_name,
        "n_trials": n_trials,
        "best_params": study.best_params,
        "best_value": study.best_value,
    }


def main():
    parser = argparse.ArgumentParser(description="NGS Hyperparameter Sweep Runner")
    parser.add_argument("--experiment", default="split_mnist",
                        help="Experiment name (from experiments.config.EXPERIMENTS)")
    parser.add_argument("--method", default="grid", choices=["grid", "optuna"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output-dir", default="./sweep_results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-trials", type=int, default=20, help="Trials for Optuna")

    # Grid sweep arguments
    parser.add_argument("--grid-epochs", nargs="+", type=int, default=None)
    parser.add_argument("--grid-lr", nargs="+", type=float, default=None)
    parser.add_argument("--grid-kd-weight", nargs="+", type=float, default=None)
    parser.add_argument("--grid-top-k", nargs="+", type=int, default=None)
    parser.add_argument("--grid-split-thresh", nargs="+", type=float, default=None)
    parser.add_argument("--grid-prune-thresh", nargs="+", type=float, default=None)
    parser.add_argument("--grid-replay-ratio", nargs="+", type=float, default=None)
    parser.add_argument("--grid-d-latent", nargs="+", type=int, default=None)
    parser.add_argument("--grid-max-k", nargs="+", type=int, default=None)
    args = parser.parse_args()

    device = args.device
    print(f"Experiment: {args.experiment}, Method: {args.method}, Seeds: {args.seeds}")

    if args.method == "grid":
        param_grid = {}
        if args.grid_epochs: param_grid["epochs_per_task"] = args.grid_epochs
        if args.grid_lr: param_grid["lr"] = args.grid_lr
        if args.grid_kd_weight: param_grid["kd_weight"] = args.grid_kd_weight
        if args.grid_top_k: param_grid["top_k"] = args.grid_top_k
        if args.grid_split_thresh: param_grid["split_thresh"] = args.grid_split_thresh
        if args.grid_prune_thresh: param_grid["prune_thresh"] = args.grid_prune_thresh
        if args.grid_replay_ratio: param_grid["replay_ratio"] = args.grid_replay_ratio
        if args.grid_d_latent: param_grid["d_latent"] = args.grid_d_latent
        if args.grid_max_k: param_grid["max_k"] = args.grid_max_k

        if not param_grid:
            print("No grid parameters specified. Use --grid-* flags.")
            return

        result = run_grid_sweep(
            experiment_name=args.experiment,
            param_grid=param_grid,
            seeds=args.seeds,
            output_dir=args.output_dir,
            device=device,
        )
    elif args.method == "optuna":
        result = run_optuna_sweep(
            experiment_name=args.experiment,
            n_trials=args.n_trials,
            seeds=args.seeds,
            output_dir=args.output_dir,
            device=device,
        )
    else:
        print(f"Unknown method: {args.method}")
        return

    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(args.output_dir) / f"{args.experiment}_sweep.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nBest params ({args.experiment}):")
    print(f"  {result.get('best_params', {})}")
    if "best_metrics" in result:
        print(f"  Metrics: {result['best_metrics'].get('avg_final_accuracy', 'N/A')}")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()

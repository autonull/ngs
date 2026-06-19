#!/usr/bin/env python
"""Unified experiment runner for all NGS benchmarks (Phase 1.1)."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


BENCHMARK_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register(name: str, category: str, fn: Callable, default_params: Optional[Dict] = None):
    BENCHMARK_REGISTRY[name] = {
        "fn": fn,
        "category": category,
        "default_params": default_params or {},
    }


# --- Continual Learning (via experiments.runner) ---
register("split_mnist", "cl", None, {
    "experiment": "split_mnist", "epochs_per_task": 5, "batch_size": 256, "lr": 1e-3,
})
register("permuted_mnist", "cl", None, {
    "experiment": "permuted_mnist", "epochs_per_task": 5, "batch_size": 256, "lr": 1e-3,
})
register("rotated_mnist", "cl", None, {
    "experiment": "rotated_mnist", "epochs_per_task": 5, "batch_size": 256, "lr": 1e-3,
})
register("blurry_mnist", "cl", None, {
    "experiment": "blurry_mnist", "epochs_per_task": 5, "batch_size": 256, "lr": 1e-3,
})
register("noisy_mnist", "cl", None, {
    "experiment": "noisy_mnist", "epochs_per_task": 5, "batch_size": 256, "lr": 1e-3,
})
register("split_fashion", "cl", None, {
    "experiment": "split_fashion", "epochs_per_task": 5, "batch_size": 256, "lr": 1e-3,
})
register("split_cifar10", "cl", None, {
    "experiment": "split_cifar10", "epochs_per_task": 10, "batch_size": 128, "lr": 1e-3,
})
register("split_cifar100", "cl", None, {
    "experiment": "split_cifar100", "epochs_per_task": 10, "batch_size": 128, "lr": 1e-3,
})
register("digits", "cl", None, {
    "experiment": "digits", "epochs_per_task": 5, "batch_size": 256, "lr": 1e-3,
})

# --- Vision benchmarks ---
from ngs.benchmarks.extended import run_vision_benchmark
register("cifar10", "vision", run_vision_benchmark, {
    "dataset": "cifar10", "epochs": 20, "batch_size": 128, "lr": 1e-3,
})
register("cifar100", "vision", run_vision_benchmark, {
    "dataset": "cifar100", "epochs": 20, "batch_size": 128, "lr": 1e-3,
})
register("fashion_mnist", "vision", run_vision_benchmark, {
    "dataset": "fashion_mnist", "epochs": 10, "batch_size": 128, "lr": 1e-3,
})

# --- NLP benchmarks ---
from ngs.benchmarks.extended import run_nlp_benchmark
register("ag_news", "nlp", run_nlp_benchmark, {
    "dataset": "ag_news", "epochs": 20, "batch_size": 64, "lr": 5e-4,
})
register("imdb", "nlp", run_nlp_benchmark, {
    "dataset": "imdb", "epochs": 20, "batch_size": 64, "lr": 5e-4,
})

# --- Density estimation ---
from ngs.benchmarks.density import run_density_benchmark
register("moons", "density", run_density_benchmark, {
    "dataset": "moons", "epochs": 200, "batch_size": 256, "lr": 1e-3,
})
register("circles", "density", run_density_benchmark, {
    "dataset": "circles", "epochs": 200, "batch_size": 256, "lr": 1e-3,
})
register("pinwheel", "density", run_density_benchmark, {
    "dataset": "pinwheel", "epochs": 200, "batch_size": 256, "lr": 1e-3,
})
register("swissroll", "density", run_density_benchmark, {
    "dataset": "swissroll", "epochs": 200, "batch_size": 256, "lr": 1e-3,
})

# --- Few-shot ---
from ngs.benchmarks.fewshot import run_fewshot_benchmark
register("omniglot", "fewshot", run_fewshot_benchmark, {
    "dataset": "omniglot", "n_way": 5, "k_shot": 1, "epochs": 50,
})
register("miniimagenet", "fewshot", run_fewshot_benchmark, {
    "dataset": "miniimagenet", "n_way": 5, "k_shot": 1, "epochs": 50,
})

# --- Robotics ---
from ngs.benchmarks.extended import run_robotics_benchmark
register("synthetic_control", "robotics", run_robotics_benchmark, {
    "env": "synthetic_control", "epochs": 30, "batch_size": 128, "lr": 1e-3,
})

# --- RL ---
from ngs.benchmarks.rl import run_rl_benchmark
register("cartpole", "rl", run_rl_benchmark, {
    "env_name": "CartPole-v1", "domain_shift": "none", "total_timesteps": 50000,
})

# --- Federated ---
from ngs.benchmarks.federated import run_federated_benchmark
register("federated_mnist", "federated", run_federated_benchmark, {
    "n_clients": 10, "n_rounds": 20, "local_epochs": 2,
})


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_benchmark(
    name: str,
    seeds: List[int],
    output_dir: str,
    device: str,
    overrides: dict,
) -> Dict[str, Any]:
    entry = BENCHMARK_REGISTRY.get(name)
    if entry is None:
        return {"error": f"Unknown benchmark: {name}"}

    fn = entry["fn"]
    params = dict(entry["default_params"])
    params.update(overrides)

    # CL benchmarks use experiments.runner
    if fn is None:
        return _run_cl_benchmark(name, seeds, output_dir, device, params)

    all_results = []
    for seed in seeds:
        set_seed(seed)
        print(f"\n  [{name}] seed={seed}")
        # Handle different parameter names for different benchmark types
        call_params = {k: v for k, v in params.items() if k not in ("experiment",)}
        category = entry["category"]
        
        # Map epochs_per_task to epochs for non-CL benchmarks
        if "epochs_per_task" in call_params and "epochs" not in call_params:
            call_params["epochs"] = call_params.pop("epochs_per_task")
        
        # Category-specific parameter filtering
        if category == "rl":
            # RL benchmarks use total_timesteps, not epochs or batch_size
            call_params.pop("epochs", None)
            call_params.pop("batch_size", None)
        elif category == "federated":
            # Federated benchmarks use n_rounds, local_epochs
            call_params.pop("epochs", None)
            call_params.pop("epochs_per_task", None)
            call_params.pop("batch_size", None)
        elif category == "fewshot":
            # Few-shot uses epochs (meta-epochs)
            call_params.pop("batch_size", None)
        elif category in ("vision", "nlp", "density", "robotics"):
            # These use epochs and batch_size
            pass
        
        result = fn(
            device=device,
            seed=seed,
            output_dir=output_dir,
            **call_params,
        )
        result["seed"] = seed
        all_results.append(result)

    return _aggregate_seeds(all_results, name, output_dir)


def _run_cl_benchmark(
    name: str, seeds: List[int], output_dir: str, device: str, params: dict
) -> Dict[str, Any]:
    """Run CL benchmarks via experiments.runner."""
    from experiments.runner import run_experiment
    from experiments.config import EXPERIMENTS, ModelConfig, TrainConfig

    exp_config = EXPERIMENTS.get(params.get("experiment", name))
    if exp_config is None:
        return {"error": f"Unknown CL experiment: {name}"}

    # Apply overrides
    epochs_per_task = params.get("epochs_per_task", params.get("epochs"))
    if epochs_per_task is not None:
        exp_config.train.epochs_per_task = epochs_per_task
    if "batch_size" in params:
        exp_config.train.batch_size = params["batch_size"]
    if "lr" in params:
        exp_config.train.lr = params["lr"]
    if "replay_ratio" in params:
        exp_config.train.replay_ratio = params["replay_ratio"]

    all_results = []
    for seed in seeds:
        set_seed(seed)
        print(f"\n  [{name}] seed={seed}")
        result = run_experiment(
            config=exp_config,
            model_name="ngs_baseline",
            seed=seed,
            output_dir=output_dir,
            verbose=False,
        )
        result["seed"] = seed
        all_results.append(result)

    return _aggregate_seeds(all_results, name, output_dir)


def _aggregate_seeds(results: List[Dict], name: str, output_dir: str) -> Dict[str, Any]:
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

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{name}_aggregated.json"
    with open(path, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    print(f"  [{name}] Aggregated -> {path}")
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="NGS Unified Benchmark Runner")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Benchmark names to run (default: all)")
    parser.add_argument("--categories", nargs="+", default=None,
                        choices=["cl", "vision", "nlp", "density", "fewshot", "robotics", "rl", "federated"],
                        help="Filter benchmarks by category")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="List benchmarks without running")
    # Common overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Filter benchmarks
    benchmarks = []
    for name, entry in BENCHMARK_REGISTRY.items():
        if args.benchmarks and name not in args.benchmarks:
            continue
        if args.categories and entry["category"] not in args.categories:
            continue
        benchmarks.append(name)

    if not benchmarks:
        print("No matching benchmarks found. Available:")
        for name, entry in sorted(BENCHMARK_REGISTRY.items()):
            print(f"  {name} ({entry['category']})")
        return

    # Build common overrides
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["lr"] = args.lr

    if args.dry_run:
        print(f"Would run {len(benchmarks)} benchmarks:")
        for name in sorted(benchmarks):
            entry = BENCHMARK_REGISTRY[name]
            print(f"  {name} ({entry['category']}) x {len(args.seeds)} seeds")
        return

    # Run benchmarks
    all_results = {}
    for name in sorted(benchmarks):
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        t0 = time.time()
        result = run_benchmark(name, args.seeds, args.output_dir, device, overrides)
        elapsed = time.time() - t0
        result["elapsed_seconds"] = elapsed
        all_results[name] = result
        print(f"  Completed in {elapsed:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, result in sorted(all_results.items()):
        if "error" in result:
            print(f"  {name}: ERROR - {result['error']}")
        elif "avg_final_accuracy" in result:
            acc = result["avg_final_accuracy"]
            print(f"  {name}: avg_acc={acc['mean']:.4f}+-{acc['std']:.4f}")
        elif "final_test_acc" in result:
            print(f"  {name}: final_acc={result['final_test_acc']:.4f}")
        elif "final_test_mse" in result:
            print(f"  {name}: final_mse={result['final_test_mse']:.6f}")

    # Save master summary
    summary_path = Path(args.output_dir) / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "benchmarks": sorted(benchmarks),
            "seeds": args.seeds,
            "results": {k: v for k, v in all_results.items() if "error" not in v},
            "errors": {k: v["error"] for k, v in all_results.items() if "error" in v},
        }, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

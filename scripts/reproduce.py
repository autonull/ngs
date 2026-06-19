#!/usr/bin/env python
"""Turn-key reproduction script for paper results."""
import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.results_db import ResultsDatabase


PAPER_EXPERIMENTS = {
    # Domain-incremental CL (main claim)
    "domain_incremental": {
        "variants": ["baseline", "factorized", "hyper", "cfg_net", "ultra_edge"],
        "benchmarks": ["permuted_mnist", "rotated_mnist", "blurry_mnist", "noisy_mnist"],
        "seeds": [42, 123, 456],
        "epochs_per_task": 5,
    },
    # Class-incremental CL (competitive)
    "class_incremental": {
        "variants": ["baseline", "factorized", "hyper", "cfg_net"],
        "benchmarks": ["split_mnist", "split_fashion", "split_cifar10", "digits"],
        "seeds": [42, 123, 456],
        "epochs_per_task": 5,
    },
    # Ablation study
    "ablation": {
        "variants": ["baseline", "factorized", "hyper", "cfg_net"],
        "benchmarks": ["split_mnist"],
        "seeds": [42, 123, 456],
        "epochs_per_task": 5,
    },
    # Scaling
    "scaling": {
        "variants": ["cfg_net", "ultra_edge"],
        "benchmarks": ["split_cifar10"],
        "seeds": [42],
        "epochs_per_task": 10,
    },
    # Extended domains
    "vision": {
        "variants": ["cfg_net"],
        "benchmarks": ["cifar10", "cifar100", "fashion_mnist"],
        "seeds": [42],
        "epochs": 20,
    },
    "nlp": {
        "variants": ["cfg_net"],
        "benchmarks": ["ag_news", "imdb"],
        "seeds": [42],
        "epochs": 20,
    },
    "fewshot": {
        "variants": ["cfg_net"],
        "benchmarks": ["omniglot", "miniimagenet"],
        "seeds": [42],
        "epochs": 50,
    },
    "rl": {
        "variants": ["cfg_net"],
        "benchmarks": ["cartpole"],
        "seeds": [42, 123],
        "timesteps": 50000,
    },
}


def run_command(cmd: List[str], cwd: str = None) -> subprocess.CompletedProcess:
    """Run command and return result."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def run_experiment_suite(
    suite_name: str,
    output_dir: str = "./paper_results",
    dry_run: bool = False,
    quick: bool = False,
) -> Dict[str, Any]:
    """Run a complete experiment suite."""
    if suite_name not in PAPER_EXPERIMENTS:
        raise ValueError(f"Unknown suite: {suite_name}. Choose from {list(PAPER_EXPERIMENTS.keys())}")
    
    suite = PAPER_EXPERIMENTS[suite_name]
    if quick:
        suite = suite.copy()
        suite["seeds"] = [42]
        if "epochs_per_task" in suite:
            suite["epochs_per_task"] = 1
        if "epochs" in suite:
            suite["epochs"] = 1
        if "timesteps" in suite:
            suite["timesteps"] = 1000
    
    results = {}
    
    for variant in suite["variants"]:
        for benchmark in suite["benchmarks"]:
            key = f"{variant}_{benchmark}"
            print(f"\n{'='*60}")
            print(f"Running: {key} (suite: {suite_name})")
            print(f"{'='*60}")
            
            if dry_run:
                results[key] = {"status": "dry_run"}
                continue
            
            # Build command
            if benchmark in ["cifar10", "cifar100", "fashion_mnist", "ag_news", "imdb", 
                            "omniglot", "miniimagenet", "cartpole"]:
                # Extended benchmarks
                cmd = [
                    sys.executable, "-m", "ngs.benchmarks.extended",
                    "--domain", _get_domain(benchmark),
                    "--dataset", benchmark,
                    "--seed", str(suite["seeds"][0]),
                    "--output-dir", output_dir,
                ]
                if "epochs" in suite:
                    cmd.extend(["--epochs", str(suite["epochs"])])
                elif "epochs_per_task" in suite:
                    cmd.extend(["--epochs", str(suite["epochs_per_task"])])
                if "timesteps" in suite:
                    cmd.extend(["--timesteps", str(suite["timesteps"])])
            else:
                # CL benchmarks via matrix runner
                cmd = [
                    sys.executable, "scripts/run_matrix.py",
                    "--variants", variant,
                    "--benchmarks", benchmark,
                    "--seeds", *[str(s) for s in suite["seeds"]],
                    "--output-dir", output_dir,
                    "--epochs", str(suite.get("epochs_per_task", 5)),
                ]
            
            result = run_command(cmd)
            results[key] = {
                "returncode": result.returncode,
                "stdout": result.stdout[-5000:] if result.stdout else "",
                "stderr": result.stderr[-2000:] if result.stderr else "",
            }
            
            if result.returncode != 0:
                print(f"  FAILED: {result.stderr[:500]}")
            else:
                print(f"  SUCCESS")
    
    return results


def _get_domain(benchmark: str) -> str:
    """Map benchmark to domain."""
    vision = ["cifar10", "cifar100", "fashion_mnist"]
    nlp = ["ag_news", "imdb"]
    fewshot = ["omniglot", "miniimagenet"]
    rl = ["cartpole"]
    
    if benchmark in vision:
        return "vision"
    elif benchmark in nlp:
        return "nlp"
    elif benchmark in fewshot:
        return "fewshot"
    elif benchmark in rl:
        return "rl"
    return "vision"


def generate_paper_report(output_dir: str = "./paper_results", db_dir: str = "./paper_db"):
    """Generate paper-ready report from results."""
    db = ResultsDatabase(db_dir)
    
    # Ingest all results
    ingest_results_dir(output_dir, db)
    
    report = {
        "domain_incremental": {},
        "class_incremental": {},
        "ablation": {},
        "scaling": {},
        "extended_domains": {},
    }
    
    # Domain-incremental comparison table
    for bench in ["permuted_mnist", "rotated_mnist", "blurry_mnist", "noisy_mnist"]:
        cmp = db.compare_variants(bench, "baseline", ["factorized", "hyper", "cfg_net", "ultra_edge"])
        report["domain_incremental"][bench] = cmp
    
    # Class-incremental
    for bench in ["split_mnist", "split_fashion", "split_cifar10", "digits"]:
        cmp = db.compare_variants(bench, "baseline", ["factorized", "hyper", "cfg_net"])
        report["class_incremental"][bench] = cmp
    
    # Ablation on split_mnist
    cmp = db.compare_variants("split_mnist", "baseline", ["factorized", "hyper", "cfg_net"])
    report["ablation"]["split_mnist"] = cmp
    
    # Pareto frontiers
    for bench in ["split_mnist", "permuted_mnist", "rotated_mnist"]:
        pareto = db.get_pareto_frontier(bench, ["avg_final_accuracy", "avg_forgetting"], [True, False])
        report["extended_domains"][f"pareto_{bench}"] = pareto
    
    # Save report
    report_path = Path(output_dir) / "paper_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nPaper report saved to {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Reproduce paper results")
    parser.add_argument("suite", nargs="?", choices=list(PAPER_EXPERIMENTS.keys()) + ["all"],
                        help="Experiment suite to run")
    parser.add_argument("--output-dir", default="./paper_results")
    parser.add_argument("--db-dir", default="./paper_db")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (1 seed, 1 epoch)")
    parser.add_argument("--generate-report", action="store_true", help="Generate paper report from existing results")
    args = parser.parse_args()
    
    if args.generate_report:
        generate_paper_report(args.output_dir, args.db_dir)
        return
    
    if not args.suite:
        parser.error("Suite required (or use --generate-report)")
    
    if args.suite == "all":
        for suite_name in PAPER_EXPERIMENTS:
            run_experiment_suite(suite_name, args.output_dir, args.dry_run, args.quick)
    else:
        run_experiment_suite(args.suite, args.output_dir, args.dry_run, args.quick)


if __name__ == "__main__":
    main()

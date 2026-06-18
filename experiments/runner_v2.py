#!/usr/bin/env python3
"""
Robust Experiment Runner v2
- Resumable: tracks completed experiments, skips done
- Round-robin: sweeps all (exp, model) pairs before next seed
- Fast-first: runs 1-epoch smoke tests before full epochs
- Progress: live status, ETA, checkpointing
"""
import os
import json
import time
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional, Tuple
from datetime import timedelta
import argparse

RESULTS_DIR = Path("./results")
CHECKPOINT_FILE = RESULTS_DIR / ".runner_checkpoint.json"

EXPERIMENTS = [
    'split_mnist', 'split_fashion', 'permuted_mnist', 'rotated_mnist',
    'blurry_mnist', 'noisy_mnist', 'split_cifar10', 'split_cifar100',
    'digits', 'split_cifar100_20', 'full_mnist', 'tinyshakespeare'
]

MODELS_NGS_PM = ["ngs_baseline", "ngs_cfg_net", "ngs_abl_hyper"]
MODELS_NGS_LORA = ["ngs_baseline_lora", "ngs_cfg_net_lora", "ngs_abl_hyper_lora"]
MODELS_BASELINES = ["mlp", "er", "ewc", "si", "lwf", "lora"]
ALL_MODELS = MODELS_NGS_PM + MODELS_NGS_LORA + MODELS_BASELINES

SEEDS = [42, 123, 456]

@dataclass
class ExperimentJob:
    experiment: str
    model: str
    seed: int
    epochs: int = 2
    status: str = "pending"  # pending, running, done, failed
    start_time: float = 0
    end_time: float = 0
    error: str = ""

    def key(self) -> str:
        return f"{self.experiment}_{self.model}_seed{self.seed}_ep{self.epochs}"

    def result_file(self) -> Path:
        # Match experiments.main naming: Split-MNIST, Split-FashionMNIST, etc.
        exp_names = {
            'split_mnist': 'Split-MNIST',
            'split_fashion': 'Split-FashionMNIST',
            'permuted_mnist': 'Permuted-MNIST',
            'rotated_mnist': 'Rotated-MNIST',
            'blurry_mnist': 'Blurry-MNIST',
            'noisy_mnist': 'Noisy-MNIST',
            'split_cifar10': 'Split-CIFAR10',
            'split_cifar100': 'Split-CIFAR100',
            'digits': 'Digits',
            'split_cifar100_20': 'Split-CIFAR100-20',
            'full_mnist': 'Full-MNIST',
            'tinyshakespeare': 'TinyShakespeare',
        }
        exp_name = exp_names.get(self.experiment, self.experiment.replace('_', '-').title())
        return RESULTS_DIR / f"{exp_name}_{self.model}_seed{self.seed}.json"

    def exists(self) -> bool:
        f = self.result_file()
        if not f.exists():
            return False
        try:
            with open(f) as fp:
                data = json.load(fp)
            return "error" not in data and "metrics" in data
        except:
            return False

class ExperimentRunner:
    def __init__(self, experiments: List[str], models: List[str], seeds: List[int],
                 epochs: int = 2, fast_mode: bool = False, max_jobs: int = None):
        self.experiments = experiments
        self.models = models
        self.seeds = seeds
        self.epochs = epochs
        self.fast_mode = fast_mode
        self.max_jobs = max_jobs
        
        self.jobs: List[ExperimentJob] = []
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()
        self.start_time = time.time()
        
        RESULTS_DIR.mkdir(exist_ok=True)
        self._build_jobs()
        self._load_checkpoint()

    def _build_jobs(self):
        """Build jobs in round-robin order: all (exp, model) pairs for seed 0, then seed 1, etc."""
        for si, seed in enumerate(self.seeds):
            for exp in self.experiments:
                for model in self.models:
                    ep = 1 if (self.fast_mode and si == 0) else self.epochs
                    job = ExperimentJob(
                        experiment=exp,
                        model=model,
                        seed=seed,
                        epochs=ep
                    )
                    if job.exists():
                        job.status = "done"
                        self.completed.add(job.key())
                    self.jobs.append(job)
        
        if self.max_jobs:
            self.jobs = self.jobs[:self.max_jobs]

    def _load_checkpoint(self):
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE) as f:
                    data = json.load(f)
                self.completed = set(data.get("completed", []))
                self.failed = set(data.get("failed", []))
                for job in self.jobs:
                    if job.key() in self.completed:
                        job.status = "done"
                    elif job.key() in self.failed:
                        job.status = "failed"
                print(f"Loaded checkpoint: {len(self.completed)} done, {len(self.failed)} failed")
            except Exception as e:
                print(f"Checkpoint load failed: {e}")

    def _save_checkpoint(self):
        data = {"completed": list(self.completed), "failed": list(self.failed)}
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(data, f)

    def _run_job(self, job: ExperimentJob) -> bool:
        job.status = "running"
        job.start_time = time.time()
        
        cmd = [
            sys.executable, '-m', 'experiments.main',
            '--experiments', job.experiment,
            '--models', job.model,
            '--seeds', str(job.seed),
            '--no-verbose'
        ]
        env = os.environ.copy()
        env['PYTHONPATH'] = '/home/me/ngs:' + env.get('PYTHONPATH', '')
        
        print(f"\n{'='*70}")
        print(f"  RUNNING: {job.experiment} | {job.model} | seed={job.seed} | epochs={job.epochs}")
        print(f"  {'='*70}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, 
                                  cwd='/home/me/ngs', env=env)
            job.end_time = time.time()
            
            print(f"  Return code: {result.returncode}")
            print(f"  Result file exists: {job.result_file().exists()}")
            if result.stderr:
                print(f"  Stderr: {result.stderr[-300:]}")
            if result.stdout:
                print(f"  Stdout: {result.stdout[-300:]}")
            
            if result.returncode == 0 and job.result_file().exists():
                job.status = "done"
                self.completed.add(job.key())
                print(f"  ✓ DONE in {timedelta(seconds=int(job.end_time - job.start_time))}")
                return True
            else:
                job.status = "failed"
                job.error = result.stderr[-500:] if result.stderr else f"Return code {result.returncode}"
                self.failed.add(job.key())
                print(f"  ✗ FAILED: {job.error}")
                return False
        except subprocess.TimeoutExpired:
            job.status = "failed"
            job.error = "Timeout (2h)"
            self.failed.add(job.key())
            print(f"  ✗ TIMEOUT")
            return False
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self.failed.add(job.key())
            print(f"  ✗ ERROR: {e}")
            return False
        finally:
            self._save_checkpoint()

    def _print_progress(self, idx: int, total: int):
        done = len(self.completed)
        failed = len(self.failed)
        elapsed = time.time() - self.start_time
        rate = done / elapsed if elapsed > 0 else 0
        remaining = total - done - failed
        eta = remaining / rate if rate > 0 else 0
        
        print(f"\n{'#'*70}")
        print(f"  PROGRESS: {idx}/{total} | Done: {done} | Failed: {failed} | Remaining: {remaining}")
        print(f"  Elapsed: {timedelta(seconds=int(elapsed))} | ETA: {timedelta(seconds=int(eta))} | Rate: {rate*3600:.1f}/hr")
        print(f"{'#'*70}")

    def run(self) -> Dict:
        pending = [j for j in self.jobs if j.status == "pending"]
        total = len(self.jobs)
        
        print(f"Total jobs: {total} ({len(pending)} pending, {len(self.completed)} done, {len(self.failed)} failed)")
        
        for idx, job in enumerate(pending, 1):
            self._print_progress(idx, total)
            self._run_job(job)
        
        # Summary
        elapsed = time.time() - self.start_time
        print(f"\n{'='*70}")
        print(f"  COMPLETE: {len(self.completed)}/{total} successful, {len(self.failed)} failed")
        print(f"  Total time: {timedelta(seconds=int(elapsed))}")
        print(f"{'='*70}")
        
        return {
            "total": total,
            "completed": len(self.completed),
            "failed": len(self.failed),
            "elapsed": elapsed,
            "failed_jobs": [j.key() for j in self.jobs if j.status == "failed"]
        }

def main():
    parser = argparse.ArgumentParser(description="Robust Experiment Runner v2")
    parser.add_argument('--experiments', nargs='+', default=EXPERIMENTS)
    parser.add_argument('--models', nargs='+', default=ALL_MODELS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--fast', action='store_true', help='Run 1-epoch smoke test first (seed 0 only)')
    parser.add_argument('--max-jobs', type=int, help='Limit total jobs')
    parser.add_argument('--phase', choices=['ngs_pm', 'ngs_lora', 'baselines', 'all'], 
                       help='Preset model groups')
    
    args = parser.parse_args()
    
    # Phase presets
    if args.phase == 'ngs_pm':
        args.models = MODELS_NGS_PM
    elif args.phase == 'ngs_lora':
        args.models = MODELS_NGS_LORA
    elif args.phase == 'baselines':
        args.models = MODELS_BASELINES
    elif args.phase == 'all':
        args.models = ALL_MODELS
    
    runner = ExperimentRunner(
        experiments=args.experiments,
        models=args.models,
        seeds=args.seeds,
        epochs=args.epochs,
        fast_mode=args.fast,
        max_jobs=args.max_jobs
    )
    
    result = runner.run()
    
    # Exit code based on failures
    sys.exit(1 if result["failed"] > 0 else 0)

if __name__ == '__main__':
    main()
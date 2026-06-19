"""
Hyperparameter optimization for LeanNGS using Optuna.
"""
import optuna
import torch
import numpy as np
from typing import Dict, Any, Callable
from dataclasses import asdict
import json
import os

from experiments.config import EXPERIMENTS, ExperimentConfig
from experiments.runner import run_experiment


def create_objective(
    experiment_name: str,
    base_config: ExperimentConfig,
    device: str = 'cuda',
    seed: int = 42
) -> Callable:
    """Create Optuna objective function for an experiment."""
    
    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        config = base_config
        config.train.epochs_per_task = trial.suggest_categorical('epochs_per_task', [2, 3, 5])
        config.train.lr = trial.suggest_float('lr', 5e-4, 5e-3, log=True)
        config.train.kd_weight = trial.suggest_float('kd_weight', 0.5, 4.0)
        config.train.kd_temperature = trial.suggest_float('kd_temperature', 1.5, 3.0)
        config.train.split_thresh = trial.suggest_float('split_thresh', 0.001, 0.05, log=True)
        config.train.prune_thresh = trial.suggest_float('prune_thresh', 0.005, 0.05)
        config.train.replay_ratio = trial.suggest_float('replay_ratio', 0.5, 2.0)
        config.model.top_k = trial.suggest_categorical('top_k', [4, 8, 16])
        config.model.d_latent = trial.suggest_categorical('d_latent', [16, 32, 64, 128])
        
        try:
            result = run_experiment(
                config, 'ngs_baseline', seed=seed, 
                output_dir=f'./hpo_results/{experiment_name}',
                verbose=False
            )
            # Return negative forgetting (we want to minimize forgetting)
            return -result['metrics']['avg_forgetting']
        except Exception as e:
            print(f"Trial failed: {e}")
            return 1.0  # High forgetting = bad
    
    return objective


def run_hpo(
    experiment_name: str,
    n_trials: int = 50,
    seed: int = 42,
    storage: str = None
) -> optuna.Study:
    """Run hyperparameter optimization."""
    config = EXPERIMENTS[experiment_name]
    
    study = optuna.create_study(
        direction='maximize',
        study_name=f'{experiment_name}_ngs_baseline',
        storage=storage,
        load_if_exists=True
    )
    
    objective = create_objective(experiment_name, config, seed=seed)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Save best params
    os.makedirs(f'./hpo_results/{experiment_name}', exist_ok=True)
    with open(f'./hpo_results/{experiment_name}/best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    print(f"\nBest params for {experiment_name}:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Best value (negative forgetting): {study.best_value:.4f}")
    
    return study


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='split_mnist', choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_hpo(args.experiment, n_trials=args.trials, seed=args.seed)
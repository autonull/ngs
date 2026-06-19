#!/usr/bin/env python
"""Bayesian optimizer with meta-learned priors for hyperparameter optimization."""
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    from optuna.distributions import (
        FloatDistribution, IntDistribution, CategoricalDistribution
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


@dataclass
class ParameterSpace:
    """Defines search space for a variant."""
    name: str
    params: Dict[str, Any]
    
    def sample(self, rng: np.random.RandomState) -> Dict:
        config = {}
        for name, spec in self.params.items():
            if isinstance(spec, list):
                config[name] = rng.choice(spec)
            elif spec[0] == "log_uniform":
                config[name] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
            elif spec[0] == "uniform":
                config[name] = float(rng.uniform(spec[1], spec[2]))
            elif spec[0] == "int":
                config[name] = int(rng.randint(spec[1], spec[2] + 1))
        return config
    
    def to_optuna_distributions(self) -> Dict:
        dists = {}
        for name, spec in self.params.items():
            if isinstance(spec, list):
                dists[name] = CategoricalDistribution(spec)
            elif spec[0] == "log_uniform":
                dists[name] = FloatDistribution(spec[1], spec[2], log=True)
            elif spec[0] == "uniform":
                dists[name] = FloatDistribution(spec[1], spec[2], log=False)
            elif spec[0] == "int":
                dists[name] = IntDistribution(spec[1], spec[2])
        return dists


VARIANT_SPACES = {
    "baseline": ParameterSpace("baseline", {
        "lr": ("log_uniform", 1e-4, 1e-2),
        "split_threshold": ("log_uniform", 0.005, 0.2),
        "prune_threshold": ("log_uniform", 0.001, 0.05),
        "kd_weight": ("log_uniform", 1.0, 50.0),
        "entropy_weight": ("log_uniform", 1e-4, 0.1),
    }),
    "factorized": ParameterSpace("factorized", {
        "lr": ("log_uniform", 1e-4, 1e-2),
        "num_subspaces": ("int", 2, 8),
        "top_k_factorized": ("int", 1, 4),
        "split_threshold": ("log_uniform", 0.01, 0.1),
        "kd_weight": ("log_uniform", 1.0, 30.0),
    }),
    "hyper": ParameterSpace("hyper", {
        "lr": ("log_uniform", 1e-4, 1e-2),
        "hypernetwork_code_dim": ("int", 4, 32),
        "hypernetwork_hidden_dim": ("int", 8, 64),
        "split_threshold": ("log_uniform", 0.01, 0.1),
        "kd_weight": ("log_uniform", 1.0, 30.0),
    }),
    "cfg_net": ParameterSpace("cfg_net", {
        "lr": ("log_uniform", 1e-4, 5e-3),
        "hypernetwork_code_dim": ("int", 8, 32),
        "num_subspaces": ("int", 2, 8),
        "split_threshold": ("log_uniform", 0.005, 0.05),
        "kd_weight": ("log_uniform", 5.0, 50.0),
    }),
    "ultra_edge": ParameterSpace("ultra_edge", {
        "lr": ("log_uniform", 5e-4, 5e-3),
        "lora_rank": ("int", 2, 16),
        "split_threshold": ("log_uniform", 0.05, 0.3),
        "prune_threshold": ("log_uniform", 0.01, 0.1),
    }),
}


@dataclass
class Trial:
    config: Dict
    metrics: Dict[str, float]
    fidelity: float = 1.0
    cost: float = 1.0


class BayesianOptimizer:
    """Bayesian optimizer using Optuna study.optimize pattern."""
    
    def __init__(
        self,
        variant: str,
        benchmark: str,
        space: ParameterSpace = None,
        metric: str = "avg_final_accuracy",
        direction: str = "maximize",
        seed: int = 42,
    ):
        self.variant = variant
        self.benchmark = benchmark
        self.space = space or VARIANT_SPACES.get(variant)
        self.metric = metric
        self.direction = direction
        self.seed = seed
        
        self.trials: List[Trial] = []
        self.study = None
        self.rng = np.random.RandomState(seed)
        self.meta_prior: Optional[Dict] = None
        self._pending_suggestions: List[Dict] = []
        
        if OPTUNA_AVAILABLE and self.space:
            self._init_study()
    
    def _init_study(self):
        sampler = TPESampler(
            seed=self.seed,
            multivariate=True,
            group=True,
            warn_independent_sampling=False,
        )
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
        )
        
        if self.meta_prior:
            self._inject_prior()
    
    def _inject_prior(self):
        if not self.meta_prior:
            return
        for config, metrics in self.meta_prior.get("best_configs", []):
            trial_config = {k: v for k, v in config.items() if k in self.space.params}
            if trial_config:
                self.study.enqueue_trial(trial_config)
    
    def set_prior(self, meta_prior: Dict):
        self.meta_prior = meta_prior
        if self.study and meta_prior:
            self._inject_prior()
    
    def observe(self, config: Dict, metrics: Dict[str, float], 
                fidelity: float = 1.0, cost: float = 1.0):
        """Record observation from completed experiment."""
        trial = Trial(config=config, metrics=metrics, fidelity=fidelity, cost=cost)
        self.trials.append(trial)
        
        if self.study:
            value = metrics.get(self.metric, 0.0)
            trial_params = {k: v for k, v in config.items() if k in self.space.params}
            if trial_params:
                distributions = self.space.to_optuna_distributions()
                filtered_dists = {k: v for k, v in distributions.items() if k in trial_params}
                self.study.add_trial(
                    optuna.trial.create_trial(
                        params=trial_params,
                        distributions=filtered_dists,
                        value=value,
                        state=optuna.trial.TrialState.COMPLETE,
                    )
                )
    
    def ask(self, n: int = 1) -> List[Dict]:
        """Generate n new suggestions using study.optimize with dummy objective."""
        if not self.space:
            return []
        
        if not self.study or len(self.trials) < 3:
            return [self.space.sample(self.rng) for _ in range(n)]
        
        # Use study.optimize to generate new trials
        def objective(trial):
            # This will be called with suggested params
            params = {}
            for name, spec in self.space.params.items():
                if isinstance(spec, list):
                    params[name] = trial.suggest_categorical(name, spec)
                elif spec[0] == "log_uniform":
                    params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
                elif spec[0] == "uniform":
                    params[name] = trial.suggest_float(name, spec[1], spec[2], log=False)
                elif spec[0] == "int":
                    params[name] = trial.suggest_int(name, spec[1], spec[2])
            
            # Store suggestion for later retrieval
            self._pending_suggestions.append(params)
            
            # Return dummy value - actual value will come from observe()
            return 0.0
        
        # Run optimization to generate suggestions
        n_existing = len(self.study.trials)
        self.study.optimize(objective, n_trials=n)
        
        # Get the newly generated trials
        new_trials = self.study.trials[n_existing:]
        suggestions = [t.params for t in new_trials]
        self._pending_suggestions = []
        return suggestions
    
    def parameter_importance(self) -> List[Tuple[str, float]]:
        complete = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(complete) < 3:
            return []
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return sorted(importance.items(), key=lambda x: -x[1])
        except:
            return []
    
    def best_config(self) -> Optional[Dict]:
        if not self.study or not self.study.best_trial:
            return None
        return self.study.best_trial.params
    
    def best_value(self) -> float:
        if not self.study or not self.study.best_trial:
            return -float("inf") if self.direction == "maximize" else float("inf")
        return self.study.best_value
    
    def get_pareto_frontier(self, metrics: List[str]) -> List[Dict]:
        complete = [t for t in self.trials if t.metrics]
        if len(complete) < 2:
            return []
        
        pareto = []
        for t in complete:
            dominated = False
            for other in complete:
                if t == other:
                    continue
                better_all = True
                strictly_better = False
                for m in metrics:
                    if m not in t.metrics or m not in other.metrics:
                        better_all = False
                        break
                    if other.metrics[m] < t.metrics[m]:
                        better_all = False
                        break
                    if other.metrics[m] > t.metrics[m]:
                        strictly_better = True
                if better_all and strictly_better:
                    dominated = True
                    break
            if not dominated:
                pareto.append({"config": t.config, "metrics": t.metrics})
        return pareto


class MetaLearner:
    """Learns priors across variants and benchmarks."""
    
    def __init__(self, results_db):
        self.db = results_db
        self.variant_similarity = self._compute_variant_similarity()
        self.benchmark_families = {
            "domain_incremental": ["permuted_mnist", "rotated_mnist", "blurry_mnist", "noisy_mnist"],
            "class_incremental": ["split_mnist", "split_fashion", "split_cifar10", "digits"],
            "vision": ["cifar10", "cifar100", "fashion_mnist"],
            "nlp": ["ag_news", "imdb"],
        }
    
    def _compute_variant_similarity(self) -> Dict[str, Dict[str, float]]:
        variants = list(VARIANT_SPACES.keys())
        sim = {}
        for v1 in variants:
            sim[v1] = {}
            params1 = set(VARIANT_SPACES[v1].params.keys())
            for v2 in variants:
                params2 = set(VARIANT_SPACES[v2].params.keys())
                if params1 or params2:
                    sim[v1][v2] = len(params1 & params2) / len(params1 | params2)
                else:
                    sim[v1][v2] = 0.0
        return sim
    
    def get_meta_prior(self, variant: str, benchmark: str) -> Dict:
        similar_variants = sorted(
            self.variant_similarity[variant].items(), 
            key=lambda x: -x[1]
        )[:3]
        
        family = None
        for fam, benches in self.benchmark_families.items():
            if benchmark in benches:
                family = fam
                break
        
        best_configs = []
        for sim_variant, sim_score in similar_variants:
            if sim_score < 0.3:
                continue
            
            agg = self.db.aggregate(variant=sim_variant, benchmark=benchmark)
            key = f"{sim_variant}_{benchmark}"
            if key in agg:
                trials = self.db.query(variant=sim_variant, benchmark=benchmark)
                if trials:
                    best_trial = max(trials, key=lambda r: r.get("metrics", {}).get("avg_final_accuracy", 0))
                    best_configs.append((best_trial.get("config", {}), best_trial.get("metrics", {})))
            
            if family:
                for bench in self.benchmark_families[family]:
                    if bench == benchmark:
                        continue
                    trials = self.db.query(variant=sim_variant, benchmark=bench)
                    if trials:
                        best_trial = max(trials, key=lambda r: r.get("metrics", {}).get("avg_final_accuracy", 0))
                        best_configs.append((best_trial.get("config", {}), best_trial.get("metrics", {})))
        
        return {
            "similar_variants": similar_variants,
            "benchmark_family": family,
            "best_configs": best_configs[:10],
        }


if __name__ == "__main__":
    space = VARIANT_SPACES["baseline"]
    opt = BayesianOptimizer("baseline", "split_mnist", space)
    
    for i in range(8):
        config = space.sample(np.random.RandomState(i))
        acc = 0.4 + np.random.rand() * 0.3
        opt.observe(config, {"avg_final_accuracy": acc})
    
    print("Best config:", opt.best_config())
    print("Best value:", opt.best_value())
    print("Importance:", opt.parameter_importance())
    print("Next suggestions:", opt.ask(3))
    print("Pareto:", opt.get_pareto_frontier(["avg_final_accuracy"]))

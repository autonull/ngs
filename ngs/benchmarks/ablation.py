"""Ablation Framework for NGS.

Systematic grid sweep over all 4 strategy dimensions (3×3×3×3 = 81 configs).
Component isolation, scaling laws, hyperparameter sensitivity, automated reporting.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
import json
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    # Task
    task: str = "split_mnist"  # "split_mnist", "permuted_mnist", "density"
    d_in: int = 784
    d_out: int = 10
    
    # Grid parameters
    routing_strategies: List[str] = None
    parameter_storages: List[str] = None
    topology_controls: List[str] = None
    memory_managements: List[str] = None
    
    # Scaling law parameters
    max_k_values: List[int] = None
    
    # Training
    epochs_per_task: int = 2
    batch_size: int = 256
    lr: float = 1e-3
    seeds: List[int] = None
    
    # Output
    output_dir: str = "./ablation_results"
    n_workers: int = 1
    
    def __post_init__(self):
        if self.routing_strategies is None:
            self.routing_strategies = ["monolithic", "factorized", "hierarchical"]
        if self.parameter_storages is None:
            self.parameter_storages = ["direct", "hypernetwork", "lora"]
        if self.topology_controls is None:
            self.topology_controls = ["heuristic", "continuous_density", "merge_aware"]
        if self.memory_managements is None:
            self.memory_managements = ["pre_allocated", "dynamic", "strict_capacity"]
        if self.max_k_values is None:
            self.max_k_values = [128, 256, 512, 1024, 2048]
        if self.seeds is None:
            self.seeds = [42, 123, 456]


class AblationFramework:
    """Systematic ablation framework for NGS."""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.results = []
        
    def _create_model_config(self, routing: str, param_storage: str, 
                            topology: str, memory: str, max_k: int) -> 'NGSConfig':
        """Create NGSConfig from strategy choices."""
        from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
        
        return NGSConfig(
            latent_dim=32,
            k_init=min(128, max_k // 4),
            max_k=max_k,
            top_k=min(8, max_k // 16),
            routing=RoutingStrategy(routing),
            parameter_storage=ParameterStorage(param_storage),
            topology_control=TopologyControl(topology),
            memory_management=MemoryManagement(memory),
        )
    
    def _run_single_config(self, config_dict: Dict, seed: int) -> Dict:
        """Run a single configuration."""
        from ngs.models.ngs import build_ngs
        from ngs.training.trainer import NGSTrainer, TrainerConfig
        
        # Reconstruct config
        config = self._create_model_config(**config_dict)
        
        # Build model
        model = build_ngs(self.config.d_in, self.config.d_out, config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Setup trainer
        trainer_config = TrainerConfig(
            lr=self.config.lr,
            epochs=self.config.epochs_per_task,
            batch_size=self.config.batch_size,
            device=device,
        )
        trainer = NGSTrainer(model, trainer_config)
        
        # Run experiment based on task
        if self.config.task == "split_mnist":
            metrics = self._run_split_mnist(trainer, seed)
        elif self.config.task == "permuted_mnist":
            metrics = self._run_permuted_mnist(trainer, seed)
        elif self.config.task == "density":
            metrics = self._run_density(trainer, seed)
        else:
            raise ValueError(f"Unknown task: {self.config.task}")
            
        return {
            **config_dict,
            "seed": seed,
            "metrics": metrics,
        }
    
    def _run_split_mnist(self, trainer: 'NGSTrainer', seed: int) -> Dict:
        """Run Split-MNIST continual learning."""
        from experiments.datasets import get_task_loaders
        from experiments.metrics import compute_metrics, evaluate_model_on_task
        import copy
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        n_tasks = 5
        accuracy_matrix = np.zeros((n_tasks, n_tasks))
        active_units = []
        
        for task_id in range(n_tasks):
            train_loader, test_loader, _ = get_task_loaders(
                'split_mnist', task_id, 2, self.config.batch_size
            )
            
            trainer.train_epoch(train_loader)
            
            # Evaluate on all seen tasks
            for eval_task in range(task_id + 1):
                _, eval_test_loader, _ = get_task_loaders(
                    'split_mnist', eval_task, 2, self.config.batch_size
                )
                acc = evaluate_model_on_task(trainer.model, eval_test_loader, trainer.config.device)
                accuracy_matrix[eval_task, task_id] = acc
                
            active_units.append(trainer.model.K)
            
        metrics = compute_metrics(accuracy_matrix, random_baseline=0.1)
        return {
            "avg_final_accuracy": metrics.avg_final_accuracy,
            "avg_forgetting": metrics.avg_forgetting,
            "forward_transfer": metrics.forward_transfer,
            "final_k": active_units[-1] if active_units else 0,
            "k_history": active_units,
        }
    
    def _run_permuted_mnist(self, trainer: 'NGSTrainer', seed: int) -> Dict:
        """Run Permuted-MNIST continual learning."""
        from experiments.datasets import PermutedMNIST
        from experiments.metrics import compute_metrics, evaluate_model_on_task
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        n_tasks = 10
        permuted = PermutedMNIST(n_tasks=n_tasks, seed=seed)
        accuracy_matrix = np.zeros((n_tasks, n_tasks))
        active_units = []
        
        for task_id in range(n_tasks):
            train_loader, test_loader = permuted.get_task_data(task_id, self.config.batch_size)
            
            trainer.train_epoch(train_loader)
            
            for eval_task in range(task_id + 1):
                _, eval_test_loader = permuted.get_task_data(eval_task, self.config.batch_size)
                acc = evaluate_model_on_task(trainer.model, eval_test_loader, trainer.config.device)
                accuracy_matrix[eval_task, task_id] = acc
                
            active_units.append(trainer.model.K)
            
        metrics = compute_metrics(accuracy_matrix, random_baseline=0.1)
        return {
            "avg_final_accuracy": metrics.avg_final_accuracy,
            "avg_forgetting": metrics.avg_forgetting,
            "forward_transfer": metrics.forward_transfer,
            "final_k": active_units[-1] if active_units else 0,
            "k_history": active_units,
        }
    
    def _run_density(self, trainer: 'NGSTrainer', seed: int) -> Dict:
        """Run density estimation (simplified)."""
        # Quick density test
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate synthetic 2D data
        from sklearn.datasets import make_moons
        X, _ = make_moons(n_samples=2000, noise=0.05, random_state=seed)
        X = torch.tensor(X, dtype=torch.float32).to(trainer.config.device)
        
        # Quick forward pass test
        trainer.model.eval()
        with torch.no_grad():
            out = trainer.model(X[:100])
            
        return {
            "final_k": trainer.model.K,
            "output_shape": list(out.logits.shape),
        }
    
    def run_full_grid(self) -> List[Dict]:
        """Run full 3×3×3×3 grid sweep."""
        # Generate all combinations
        param_grid = {
            "routing": self.config.routing_strategies,
            "param_storage": self.config.parameter_storages,
            "topology": self.config.topology_controls,
            "memory": self.config.memory_managements,
            "max_k": self.config.max_k_values,
        }
        
        # Filter valid combinations
        valid_combos = []
        for combo in product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), combo))
            
            # Validate combination constraints
            if param_dict["routing"] == "hierarchical":
                if param_dict["max_k"] < 64:  # Need enough for coarse + fine
                    continue
            if param_dict["routing"] == "factorized":
                if param_dict["max_k"] % 4 != 0:  # Default num_subspaces=4
                    continue
                    
            valid_combos.append(param_dict)
            
        print(f"Running {len(valid_combos)} valid configs × {len(self.config.seeds)} seeds = {len(valid_combos) * len(self.config.seeds)} total runs")
        
        # Run experiments
        all_results = []
        
        if self.config.n_workers > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
                futures = []
                for param_dict in valid_combos:
                    for seed in self.config.seeds:
                        future = executor.submit(self._run_single_config, param_dict, seed)
                        futures.append((future, param_dict, seed))
                        
                for future, param_dict, seed in futures:
                    try:
                        result = future.result(timeout=3600)  # 1 hour timeout
                        all_results.append(result)
                        print(f"Completed: {param_dict} seed={seed}, Acc={result['metrics'].get('avg_final_accuracy', 'N/A'):.4f}")
                    except Exception as e:
                        print(f"Failed: {param_dict} seed={seed}, Error: {e}")
                        all_results.append({
                            **param_dict,
                            "seed": seed,
                            "error": str(e),
                        })
        else:
            # Sequential execution
            for param_dict in valid_combos:
                for seed in self.config.seeds:
                    try:
                        result = self._run_single_config(param_dict, seed)
                        all_results.append(result)
                        print(f"Completed: {param_dict} seed={seed}, Acc={result['metrics'].get('avg_final_accuracy', 'N/A'):.4f}")
                    except Exception as e:
                        print(f"Failed: {param_dict} seed={seed}, Error: {e}")
                        all_results.append({
                            **param_dict,
                            "seed": seed,
                            "error": str(e),
                        })
                        
        self.results = all_results
        return all_results
    
    def run_component_isolation(self) -> Dict[str, List[Dict]]:
        """Run single-dimension ablations with fixed others."""
        # Base config (best known)
        base = {
            "routing": "factorized",
            "param_storage": "hypernetwork",
            "topology": "continuous_density",
            "memory": "pre_allocated",
            "max_k": 512,
        }
        
        isolation_results = {}
        
        for dim, values in [
            ("routing", self.config.routing_strategies),
            ("param_storage", self.config.parameter_storages),
            ("topology", self.config.topology_controls),
            ("memory", self.config.memory_managements),
        ]:
            dim_results = []
            for val in values:
                config = base.copy()
                config[dim] = val
                
                # Run with single seed for speed
                result = self._run_single_config(config, self.config.seeds[0])
                dim_results.append(result)
                
            isolation_results[dim] = dim_results
            
        return isolation_results
    
    def run_scaling_laws(self) -> List[Dict]:
        """Vary max_k vs performance."""
        base = {
            "routing": "factorized",
            "param_storage": "hypernetwork",
            "topology": "continuous_density",
            "memory": "pre_allocated",
        }
        
        results = []
        for max_k in self.config.max_k_values:
            config = base.copy()
            config["max_k"] = max_k
            
            # Average over seeds
            seed_results = []
            for seed in self.config.seeds:
                result = self._run_single_config(config, seed)
                seed_results.append(result["metrics"])
                
            # Aggregate
            avg_metrics = {}
            for key in seed_results[0].keys():
                vals = [r[key] for r in seed_results]
                avg_metrics[key] = {
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                }
                
            results.append({
                **config,
                "metrics": avg_metrics,
            })
            
        return results
    
    def analyze_results(self) -> Dict:
        """Analyze ablation results."""
        valid_results = [r for r in self.results if "error" not in r]
        
        if not valid_results:
            return {}
            
        # Group by each parameter
        param_importance = {}
        all_params = set()
        for r in valid_results:
            all_params.update([k for k in r.keys() if k not in ["seed", "metrics", "error"]])
            
        for param in all_params:
            values = {}
            for r in valid_results:
                if param in r:
                    val = r[param]
                    if val not in values:
                        values[val] = []
                    values[val].append(r["metrics"].get("avg_final_accuracy", 0))
                    
            param_importance[param] = {v: np.mean(vals) for v, vals in values.items()}
            
        # Find best
        best = max(valid_results, key=lambda r: r["metrics"].get("avg_final_accuracy", 0))
        
        return {
            "best_config": {k: v for k, v in best.items() if k not in ["seed", "metrics", "error"]},
            "best_metric": best["metrics"].get("avg_final_accuracy", 0),
            "param_importance": param_importance,
            "n_valid": len(valid_results),
            "n_failed": len(self.results) - len(valid_results),
        }
    
    def generate_latex_table(self, analysis: Dict) -> str:
        """Generate LaTeX table for paper."""
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{NGS Ablation Study Results}")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append("Component & Best Choice & Mean Acc & Std & Range \\\\")
        lines.append("\\midrule")
        
        for param, importance in analysis.get("param_importance", {}).items():
            sorted_vals = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            best_val, best_acc = sorted_vals[0]
            worst_val, worst_acc = sorted_vals[-1]
            lines.append(f"{param} & {best_val} & {best_acc:.4f} & -- & [{worst_acc:.4f}, {best_acc:.4f}] \\\\")
            
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        return "\n".join(lines)
    
    def save_results(self, filepath: str):
        """Save all results to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
            
    def load_results(self, filepath: str):
        """Load results from JSON."""
        with open(filepath, "r") as f:
            self.results = json.load(f)


def run_ablation_grid(
    task: str = "split_mnist",
    output_dir: str = "./ablation_results",
    n_workers: int = 1,
    quick: bool = False
) -> Dict:
    """Run ablation grid with default config."""
    config = AblationConfig(
        task=task,
        output_dir=output_dir,
        n_workers=n_workers,
    )
    
    if quick:
        # Reduced grid for quick testing
        config.routing_strategies = ["factorized", "monolithic"]
        config.parameter_storages = ["hypernetwork", "direct"]
        config.topology_controls = ["continuous_density", "heuristic"]
        config.memory_managements = ["pre_allocated"]
        config.max_k_values = [256, 512]
        config.seeds = [42]
        config.epochs_per_task = 1
        
    framework = AblationFramework(config)
    results = framework.run_full_grid()
    
    # Analyze
    analysis = framework.analyze_results()
    
    # Save
    framework.save_results(str(Path(output_dir) / f"{task}_full_grid.json"))
    
    with open(Path(output_dir) / f"{task}_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
        
    # Generate LaTeX
    latex = framework.generate_latex_table(analysis)
    with open(Path(output_dir) / f"{task}_table.tex", "w") as f:
        f.write(latex)
        
    print("\n" + "="*60)
    print("ABLATION ANALYSIS")
    print("="*60)
    print(f"Best config: {analysis.get('best_config')}")
    print(f"Best metric: {analysis.get('best_metric'):.4f}")
    print(f"Valid runs: {analysis.get('n_valid')}, Failed: {analysis.get('n_failed')}")
    print("\nParameter importance:")
    for param, imp in analysis.get("param_importance", {}).items():
        print(f"  {param}: {imp}")
        
    return {
        "results": results,
        "analysis": analysis,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ablation framework")
    parser.add_argument("--task", default="split_mnist", choices=["split_mnist", "permuted_mnist", "density"])
    parser.add_argument("--output-dir", default="./ablation_results")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="Run reduced grid for testing")
    args = parser.parse_args()
    
    run_ablation_grid(args.task, args.output_dir, args.n_workers, args.quick)
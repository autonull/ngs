"""Continual Learning Benchmark for NGS.

Reproduces all 11 dataset results, online CL, class-incremental on large scale.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class ContinualConfig:
    """Configuration for continual learning benchmark."""
    # Dataset
    dataset: str = "split_mnist"  # Full list in DATASETS
    
    # Model
    latent_dim: int = 32
    k_init: int = 128
    max_k: int = 512
    top_k: int = 8
    routing: str = "factorized"
    parameter_storage: str = "hypernetwork"
    topology_control: str = "continuous_density"
    memory_management: str = "pre_allocated"
    
    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs_per_task: int = 2
    batch_size: int = 256
    replay_size: int = 50000
    replay_ratio: float = 1.0
    kd_weight: float = 10.0
    kd_temperature: float = 2.0
    
    # Topology
    split_thresh: float = 0.005
    prune_thresh: float = 0.01
    max_spawn_per_call: int = 5
    adapt_every_epoch: bool = True
    
    # Evaluation
    seeds: List[int] = None
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456]


DATASETS = [
    "split_mnist", "split_fashion", "permuted_mnist", 
    "split_cifar10", "split_cifar100", "digits",
    "rotated_mnist", "blurry_mnist", "noisy_mnist",
    "split_cifar100_20", "full_mnist"
]


class ContinualBenchmark:
    """Continual learning benchmark runner."""
    
    def __init__(self, config: ContinualConfig, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results = {}
        
    def run(self, seed: int = 42) -> Dict:
        """Run continual learning benchmark."""
        from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
        from ngs.models.ngs import build_ngs
        from ngs.training.trainer import NGSTrainer, TrainerConfig
        from experiments.datasets import get_task_loaders, PermutedMNIST, ReplayBuffer
        from experiments.metrics import compute_metrics, evaluate_model_on_task
        import copy
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Get dataset config
        from experiments.config import EXPERIMENTS
        exp_config = EXPERIMENTS.get(self.config.dataset)
        if exp_config is None:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
            
        n_tasks = exp_config.n_tasks
        classes_per_task = exp_config.classes_per_task
        d_in = exp_config.input_dim
        d_out = exp_config.output_dim
        
        # Create NGS config
        ngs_config = NGSConfig(
            latent_dim=self.config.latent_dim,
            k_init=self.config.k_init,
            max_k=self.config.max_k,
            top_k=self.config.top_k,
            routing=RoutingStrategy(self.config.routing),
            parameter_storage=ParameterStorage(self.config.parameter_storage),
            topology_control=TopologyControl(self.config.topology_control),
            memory_management=MemoryManagement(self.config.memory_management),
        )
        
        # Build model
        model = build_ngs(d_in, d_out, ngs_config).to(self.device)
        
        # Trainer config
        trainer_config = TrainerConfig(
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            epochs=self.config.epochs_per_task,
            batch_size=self.config.batch_size,
            replay_size=self.config.replay_size,
            replay_ratio=self.config.replay_ratio,
            kd_weight=self.config.kd_weight,
            kd_temperature=self.config.kd_temperature,
            split_thresh=self.config.split_thresh,
            prune_thresh=self.config.prune_thresh,
            max_spawn_per_call=self.config.max_spawn_per_call,
            adapt_every_epoch=self.config.adapt_every_epoch,
            device=self.device,
        )
        
        trainer = NGSTrainer(model, trainer_config)
        
        # Replay buffer
        replay_buffer = ReplayBuffer(max_size=self.config.replay_size) if self.config.replay_size > 0 else None
        
        # Task loader function
        def get_task_data(task_id):
            if self.config.dataset == 'permuted_mnist':
                permuted = PermutedMNIST(n_tasks=n_tasks, seed=seed)
                train_loader, test_loader = permuted.get_task_data(task_id, self.config.batch_size)
                classes = list(range(10))
            else:
                train_loader, test_loader, classes = get_task_loaders(
                    self.config.dataset, task_id, classes_per_task, self.config.batch_size
                )
            return train_loader, test_loader, classes
        
        # Continual evaluation
        accuracy_matrix = np.zeros((n_tasks, n_tasks))
        active_units_list = []
        old_model = None
        
        for task_id in range(n_tasks):
            train_loader, test_loader, classes = get_task_data(task_id)
            
            # Train
            trainer.train_epoch(train_loader, replay_buffer=replay_buffer, old_model=old_model)
            
            # Evaluate on all seen tasks
            for eval_task in range(task_id + 1):
                _, eval_test_loader, _ = get_task_data(eval_task)
                acc = evaluate_model_on_task(model, eval_test_loader, self.device)
                accuracy_matrix[eval_task, task_id] = acc
            
            # Update replay buffer
            if replay_buffer:
                import torch.nn.functional as F
                for x, y in train_loader:
                    x_flat = x.view(x.size(0), -1).to(self.device)
                    y_onehot = F.one_hot(y, num_classes=d_out).float().to(self.device)
                    replay_buffer.add(x_flat, y_onehot)
            
            # Save old model for KD
            old_model = copy.deepcopy(model)
            old_model.eval()
            for p in old_model.parameters():
                p.requires_grad = False
            
            # Track capacity
            active_units_list.append(model.K)
            
            print(f"Task {task_id+1}/{n_tasks}: K={model.K}, Acc={accuracy_matrix[task_id, task_id]:.4f}")
        
        # Compute metrics
        random_baseline = 1.0 / d_out
        metrics = compute_metrics(accuracy_matrix, random_baseline=random_baseline)
        
        self.results = {
            "config": self.config.__dict__,
            "seed": seed,
            "dataset": self.config.dataset,
            "n_tasks": n_tasks,
            "accuracy_matrix": accuracy_matrix.tolist(),
            "active_units": active_units_list,
            "metrics": metrics.to_dict(),
        }
        
        return self.results
    
    def run_all_seeds(self) -> Dict:
        """Run benchmark across all seeds and aggregate."""
        all_results = []
        
        for seed in self.config.seeds:
            print(f"\n{'='*60}")
            print(f"Running seed {seed}")
            print(f"{'='*60}")
            result = self.run(seed)
            all_results.append(result)
            
        # Aggregate
        metric_keys = all_results[0]["metrics"].keys()
        aggregated = {}
        
        for key in metric_keys:
            vals = [r["metrics"][key] for r in all_results]
            aggregated[key] = {
                "mean": np.mean(vals),
                "std": np.std(vals),
                "values": vals,
            }
            
        self.results = {
            "config": self.config.__dict__,
            "dataset": self.config.dataset,
            "seeds": self.config.seeds,
            "individual": all_results,
            "aggregated": aggregated,
        }
        
        return self.results
    
    def save_results(self, filepath: str):
        """Save results to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)


def run_continual_benchmark(
    dataset: str = "split_mnist",
    device: str = "cuda",
    seeds: List[int] = None,
    output_dir: str = "./continual_results"
) -> Dict:
    """Run continual learning benchmark."""
    config = ContinualConfig(dataset=dataset, seeds=seeds or [42, 123, 456])
    
    benchmark = ContinualBenchmark(config, device)
    results = benchmark.run_all_seeds()
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    benchmark.save_results(str(Path(output_dir) / f"{dataset}_results.json"))
    
    # Print summary
    print("\n" + "="*60)
    print(f"CONTINUAL LEARNING RESULTS: {dataset.upper()}")
    print("="*60)
    for metric, vals in results["aggregated"].items():
        print(f"  {metric}: {vals['mean']:.4f} ± {vals['std']:.4f}")
        
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run continual learning benchmark")
    parser.add_argument("--dataset", default="split_mnist", choices=DATASETS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--output-dir", default="./continual_results")
    args = parser.parse_args()
    
    run_continual_benchmark(args.dataset, args.device, args.seeds, args.output_dir)
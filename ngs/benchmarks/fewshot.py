"""Few-Shot Learning Benchmark for NGS.

Tests NGS as meta-learner on Omniglot and miniImageNet.
Uses hypernetwork to generate task-specific adapters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import random


@dataclass
class FewShotConfig:
    """Configuration for few-shot benchmark."""
    # Dataset
    dataset: str = "omniglot"  # "omniglot", "miniimagenet"
    n_way: int = 5
    k_shot: int = 1
    n_query: int = 15
    n_tasks_train: int = 1000
    n_tasks_val: int = 200
    n_tasks_test: int = 600
    
    # Model
    latent_dim: int = 64
    k_init: int = 32
    max_k: int = 256
    top_k: int = 8
    routing: str = "factorized"
    parameter_storage: str = "hypernetwork"
    topology_control: str = "continuous_density"
    memory_management: str = "pre_allocated"
    hypernetwork_code_dim: int = 16
    lora_rank: int = 4
    
    # Training
    meta_lr: float = 1e-3
    task_lr: float = 0.01
    meta_batch_size: int = 4
    adaptation_steps: int = 5
    epochs: int = 50
    
    # Meta-learning
    first_order: bool = True  # FO-MAML approximation
    
    # Evaluation
    n_test_tasks: int = 600


class ConvEncoder(nn.Module):
    """Convolutional encoder for few-shot (Omniglot/MiniImageNet)."""
    
    def __init__(self, latent_dim: int, in_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.fc = nn.Linear(64, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class FewShotNGS(nn.Module):
    """NGS adapted for few-shot classification with task-specific adaptation."""
    
    def __init__(self, config: FewShotConfig, n_classes: int):
        super().__init__()
        self.config = config
        self.n_classes = n_classes
        
        # Determine input channels
        in_channels = 1 if config.dataset == "omniglot" else 3
        
        # Feature encoder
        self.encoder = ConvEncoder(config.latent_dim, in_channels)
        
        # NGS for classification
        from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
        from ngs.models.ngs import build_ngs
        
        ngs_config = NGSConfig(
            latent_dim=config.latent_dim,
            k_init=config.k_init,
            max_k=config.max_k,
            top_k=config.top_k,
            routing=RoutingStrategy(config.routing),
            parameter_storage=ParameterStorage(config.parameter_storage),
            topology_control=TopologyControl(config.topology_control),
            memory_management=MemoryManagement(config.memory_management),
            hypernetwork_code_dim=config.hypernetwork_code_dim,
            use_lora=True,
            lora_rank=config.lora_rank,
        )
        
        self.ngs = build_ngs(config.latent_dim, n_classes, ngs_config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.ngs(z).logits
    
    def adapt_to_task(self, support_x: torch.Tensor, support_y: torch.Tensor, 
                      steps: int = None, lr: float = None) -> 'FewShotNGS':
        """Fast adaptation to a new task (inner loop)."""
        steps = steps or self.config.adaptation_steps
        lr = lr or self.config.task_lr
        
        # Create a copy for adaptation
        adapted_model = FewShotNGS(self.config, self.n_classes)
        adapted_model.load_state_dict(self.state_dict())
        adapted_model.to(support_x.device)
        adapted_model.train()
        
        # Only adapt the hypernetwork/LoRA parameters for efficiency
        # In practice, we'd only adapt the task-specific parameters
        adapter_params = []
        for name, param in adapted_model.named_parameters():
            if 'hypernet' in name.lower() or 'lora' in name.lower() or 'adapter' in name.lower():
                adapter_params.append(param)
        
        if not adapter_params:
            # Fallback: adapt last layer
            adapter_params = list(adapted_model.ngs.p_up.parameters())
            
        optimizer = torch.optim.SGD(adapter_params, lr=lr)
        
        for step in range(steps):
            logits = adapted_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return adapted_model


def load_omniglot(data_dir: str = "./data/omniglot", download: bool = True):
    """Load Omniglot dataset."""
    try:
        from torchvision.datasets import Omniglot
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x),  # Invert: white background -> black
        ])
        
        train_dataset = Omniglot(root=data_dir, background=True, download=download, transform=transform)
        test_dataset = Omniglot(root=data_dir, background=False, download=download, transform=transform)
        
        return train_dataset, test_dataset
    except ImportError:
        print("torchvision not available, using synthetic data")
        return None, None


def load_miniimagenet(data_dir: str = "./data/miniimagenet", download: bool = True):
    """Load miniImageNet dataset (requires manual download)."""
    # miniImageNet typically requires manual download from https://github.com/yaoyao-liu/mini-imagenet-tools
    # For now, return None to indicate not available
    print("miniImageNet requires manual download. Using synthetic data.")
    return None, None


class TaskSampler:
    """Sample few-shot tasks from dataset."""
    
    def __init__(self, dataset, n_way: int, k_shot: int, n_query: int):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        
        # Organize by class
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
            
        self.classes = list(self.class_to_indices.keys())
        
    def sample_task(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a single task: returns (support_x, support_y, query_x, query_y)."""
        selected_classes = random.sample(self.classes, self.n_way)
        
        support_x, support_y = [], []
        query_x, query_y = [], []
        
        for cls_idx, cls in enumerate(selected_classes):
            indices = self.class_to_indices[cls]
            chosen = random.sample(indices, self.k_shot + self.n_query)
            
            for i, idx in enumerate(chosen):
                x, _ = self.dataset[idx]
                if i < self.k_shot:
                    support_x.append(x)
                    support_y.append(cls_idx)
                else:
                    query_x.append(x)
                    query_y.append(cls_idx)
                    
        return (torch.stack(support_x), torch.tensor(support_y, dtype=torch.long),
                torch.stack(query_x), torch.tensor(query_y, dtype=torch.long))
    
    def sample_batch(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of tasks."""
        return [self.sample_task() for _ in range(batch_size)]


class SyntheticTaskSampler:
    """Synthetic few-shot tasks for testing without real datasets."""
    
    def __init__(self, n_way: int, k_shot: int, n_query: int, latent_dim: int = 64):
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.latent_dim = latent_dim
        
    def sample_task(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate synthetic task with Gaussian clusters."""
        # Random class means in latent space
        class_means = torch.randn(self.n_way, self.latent_dim) * 2
        
        support_x, support_y = [], []
        query_x, query_y = [], []
        
        for cls_idx in range(self.n_way):
            mean = class_means[cls_idx]
            
            # Support samples
            support_samples = mean + torch.randn(self.k_shot, self.latent_dim) * 0.5
            support_x.append(support_samples)
            support_y.extend([cls_idx] * self.k_shot)
            
            # Query samples
            query_samples = mean + torch.randn(self.n_query, self.latent_dim) * 0.5
            query_x.append(query_samples)
            query_y.extend([cls_idx] * self.n_query)
            
        return (torch.cat(support_x), torch.tensor(support_y, dtype=torch.long),
                torch.cat(query_x), torch.tensor(query_y, dtype=torch.long))
    
    def sample_batch(self, batch_size: int) -> List[Tuple]:
        return [self.sample_task() for _ in range(batch_size)]


class FewShotBenchmark:
    """Few-shot learning benchmark runner."""
    
    def __init__(self, config: FewShotConfig, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results = {}
        
    def _get_task_sampler(self, split: str = "train") -> TaskSampler:
        """Get task sampler for dataset split."""
        # For now, use synthetic data for testing
        # TODO: Replace with real Omniglot/miniImageNet loaders
        return SyntheticTaskSampler(
            self.config.n_way, self.config.k_shot, self.config.n_query,
            latent_dim=self.config.latent_dim
        )
    
    def run(self, seed: int = 42) -> Dict:
        """Run few-shot benchmark."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Create model
        model = FewShotNGS(self.config, self.config.n_way).to(self.device)
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=self.config.meta_lr)
        
        # Task samplers
        train_sampler = self._get_task_sampler("train")
        val_sampler = self._get_task_sampler("val")
        test_sampler = self._get_task_sampler("test")
        
        # Training loop
        train_accs = []
        val_accs = []
        
        for epoch in range(self.config.epochs):
            model.train()
            epoch_accs = []
            
            for _ in range(self.config.n_tasks_train // self.config.meta_batch_size):
                meta_optimizer.zero_grad()
                meta_loss = 0.0
                
                tasks = train_sampler.sample_batch(self.config.meta_batch_size)
                
                for support_x, support_y, query_x, query_y in tasks:
                    support_x = support_x.to(self.device)
                    support_y = support_y.to(self.device)
                    query_x = query_x.to(self.device)
                    query_y = query_y.to(self.device)
                    
                    # Adapt to task
                    adapted = model.adapt_to_task(support_x, support_y)
                    
                    # Evaluate on query set
                    query_logits = adapted(query_x)
                    loss = F.cross_entropy(query_logits, query_y)
                    meta_loss += loss
                    
                    # Accuracy
                    pred = query_logits.argmax(dim=1)
                    acc = (pred == query_y).float().mean().item()
                    epoch_accs.append(acc)
                    
                meta_loss /= self.config.meta_batch_size
                meta_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                meta_optimizer.step()
                
            train_acc = np.mean(epoch_accs) if epoch_accs else 0
            train_accs.append(train_acc)
            
            # Validation
            if epoch % 5 == 0:
                val_acc = self._evaluate(model, val_sampler, self.config.n_tasks_val)
                val_accs.append(val_acc)
                print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
            else:
                print(f"Epoch {epoch}: Train Acc={train_acc:.4f}")
                
        # Final test evaluation
        test_acc = self._evaluate(model, test_sampler, self.config.n_test_tasks)
        
        self.results = {
            "config": self.config.__dict__,
            "seed": seed,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "test_acc": test_acc,
        }
        
        return self.results
    
    def _evaluate(self, model: FewShotNGS, sampler: TaskSampler, n_tasks: int) -> float:
        """Evaluate model on n_tasks."""
        model.eval()
        accs = []
        
        with torch.no_grad():
            for _ in range(n_tasks):
                support_x, support_y, query_x, query_y = sampler.sample_task()
                support_x = support_x.to(self.device)
                support_y = support_y.to(self.device)
                query_x = query_x.to(self.device)
                query_y = query_y.to(self.device)
                
                # Adapt
                adapted = model.adapt_to_task(support_x, support_y)
                
                # Evaluate
                query_logits = adapted(query_x)
                pred = query_logits.argmax(dim=1)
                acc = (pred == query_y).float().mean().item()
                accs.append(acc)
                
        return np.mean(accs)


def run_fewshot_benchmark(
    dataset: str = "omniglot",
    n_way: int = 5,
    k_shot: int = 1,
    epochs: int = 50,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./fewshot_results"
) -> Dict:
    """Run few-shot benchmark with default config."""
    config = FewShotConfig(
        dataset=dataset,
        n_way=n_way,
        k_shot=k_shot,
        epochs=epochs,
    )
    
    benchmark = FewShotBenchmark(config, device)
    results = benchmark.run(seed)
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / f"{dataset}_{n_way}way_{k_shot}shot_seed{seed}_results.json", "w") as f:
        serializable = {k: v for k, v in results.items() if k != "config"}
        serializable["config"] = config.__dict__
        json.dump(serializable, f, indent=2)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run few-shot benchmark")
    parser.add_argument("--dataset", default="omniglot", choices=["omniglot", "miniimagenet"])
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./fewshot_results")
    args = parser.parse_args()
    
    run_fewshot_benchmark(args.dataset, args.n_way, args.k_shot, args.epochs, args.device, args.seed, args.output_dir)
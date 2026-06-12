"""
Online/incremental continual learning evaluation.
Evaluates models in streaming setting where data arrives one sample at a time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
from copy import deepcopy
import time


@dataclass
class OnlineResult:
    """Results from online evaluation."""
    accuracy_over_time: List[float]
    forgetting_over_time: List[float]
    final_accuracy: float
    final_forgetting: float
    update_time_per_sample: float
    memory_mb: float


def evaluate_online(
    model: nn.Module,
    get_stream: Callable,
    n_tasks: int,
    samples_per_task: int,
    device: str = 'cuda',
    update_fn: Callable = None,
    eval_every: int = 100
) -> OnlineResult:
    """
    Evaluate model in online streaming setting.
    
    Args:
        model: Model to evaluate
        get_stream: Function(task_id) -> iterator of (x, y) samples
        n_tasks: Number of tasks
        samples_per_task: Samples per task
        device: Device to run on
        update_fn: Function(model, x, y) -> loss (for online update)
        eval_every: Evaluate every N samples
    
    Returns:
        OnlineResult with accuracy/forgetting over time
    """
    model.to(device)
    model.train()
    
    if update_fn is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        def default_update(model, x, y):
            optimizer.zero_grad()
            # Add batch dimension
            x_b = x.unsqueeze(0)
            y_b = y.unsqueeze(0)
            logits = model(x_b)
            loss = F.cross_entropy(logits, y_b)
            loss.backward()
            optimizer.step()
            return loss.item()
        update_fn = default_update
    
    # Track accuracy on each task
    task_accuracies = {i: [] for i in range(n_tasks)}
    task_max_acc = {i: 0.0 for i in range(n_tasks)}
    
    accuracy_over_time = []
    forgetting_over_time = []
    
    sample_count = 0
    total_time = 0
    
    for task_id in range(n_tasks):
        stream = get_stream(task_id)
        
        for i, (x, y) in enumerate(stream):
            if i >= samples_per_task:
                break
            
            x = x.to(device)
            y = y.to(device)
            
            # Online update
            start = time.perf_counter()
            loss = update_fn(model, x, y)
            total_time += time.perf_counter() - start
            sample_count += 1
            
            # Periodic evaluation
            if sample_count % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    for eval_task in range(task_id + 1):
                        eval_stream = get_stream(eval_task)
                        correct = 0
                        total = 0
                        for j, (ex, ey) in enumerate(eval_stream):
                            if j >= 200:  # Evaluate on 200 samples
                                break
                            ex, ey = ex.to(device), ey.to(device)
                            ex_b = ex.unsqueeze(0)
                            pred = model(ex_b).argmax(dim=1)
                            correct += (pred == ey).sum().item()
                            total += 1
                        acc = correct / total if total > 0 else 0
                        task_accuracies[eval_task].append(acc)
                        task_max_acc[eval_task] = max(task_max_acc[eval_task], acc)
                
                # Compute average accuracy and forgetting
                current_accs = [task_accuracies[t][-1] for t in range(task_id + 1)]
                avg_acc = np.mean(current_accs)
                avg_forgetting = np.mean([task_max_acc[t] - task_accuracies[t][-1] for t in range(task_id + 1)])
                
                accuracy_over_time.append(avg_acc)
                forgetting_over_time.append(avg_forgetting)
                
                model.train()
    
    # Final evaluation
    model.eval()
    final_accs = []
    final_forgetting = []
    with torch.no_grad():
        for eval_task in range(n_tasks):
            eval_stream = get_stream(eval_task)
            correct = 0
            total = 0
            for j, (ex, ey) in enumerate(eval_stream):
                if j >= 500:
                    break
                ex, ey = ex.to(device), ey.to(device)
                ex_b = ex.unsqueeze(0)
                pred = model(ex_b).argmax(dim=1)
                correct += (pred == ey).sum().item()
                total += 1
            acc = correct / total if total > 0 else 0
            final_accs.append(acc)
            final_forgetting.append(task_max_acc[eval_task] - acc)
    
    return OnlineResult(
        accuracy_over_time=accuracy_over_time,
        forgetting_over_time=forgetting_over_time,
        final_accuracy=np.mean(final_accs),
        final_forgetting=np.mean(final_forgetting),
        update_time_per_sample=total_time / sample_count * 1000 if sample_count > 0 else 0,
        memory_mb=0  # Could add memory tracking
    )


def create_split_mnist_stream(task_id: int, batch_size: int = 1, seed: int = 42):
    """Create online stream for Split-MNIST task."""
    from experiments.datasets import get_task_loaders
    train_loader, _, _ = get_task_loaders('split_mnist', task_id, 2, batch_size)
    
    def stream():
        while True:
            for x, y in train_loader:
                # Flatten the input
                x_flat = x.view(x.size(0), -1)
                for xi, yi in zip(x_flat, y):
                    yield xi, yi
    
    return stream()


def run_online_comparison(
    models: Dict[str, nn.Module],
    n_tasks: int = 5,
    samples_per_task: int = 5000,
    device: str = 'cuda'
) -> Dict[str, OnlineResult]:
    """Run online evaluation on multiple models."""
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name} online...")
        
        # Fresh model for each run
        model_copy = deepcopy(model)
        
        try:
            result = evaluate_online(
                model_copy,
                lambda t: create_split_mnist_stream(t, batch_size=1),
                n_tasks=n_tasks,
                samples_per_task=samples_per_task,
                device=device,
                eval_every=500
            )
            results[name] = result
            print(f"  Final Acc: {result.final_accuracy:.4f}")
            print(f"  Final Forgetting: {result.final_forgetting:.4f}")
            print(f"  Update time: {result.update_time_per_sample:.2f} ms/sample")
        except Exception as e:
            print(f"  Error: {e}")
            results[name] = None
    
    return results


def print_online_results(results: Dict[str, OnlineResult]):
    """Print online evaluation results."""
    print("\n" + "="*80)
    print(f"{'Model':<15} {'Final Acc':>10} {'Final Forget':>12} {'Update (ms)':>12}")
    print("="*80)
    for name, r in results.items():
        if r is None:
            print(f"{name:<15} {'ERROR':>10}")
            continue
        print(f"{name:<15} {r.final_accuracy:>10.4f} {r.final_forgetting:>12.4f} {r.update_time_per_sample:>12.2f}")


if __name__ == '__main__':
    from experiments.config import EXPERIMENTS
    from experiments.baselines import create_baseline
    from experiments.lean_ngs_trainer import create_lean_ngs
    
    config = EXPERIMENTS['split_mnist']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    models = {
        'MLP': create_baseline('mlp', config.input_dim, config.output_dim),
        'ER': create_baseline('er', config.input_dim, config.output_dim),
        'EWC': create_baseline('ewc', config.input_dim, config.output_dim),
        'LwF': create_baseline('lwf', config.input_dim, config.output_dim),
        'LeanNGS': create_lean_ngs(config.input_dim, config.output_dim, **{
            'd_latent': 32, 'k_init': 128, 'max_k': 1024, 'top_k': 8
        }),
    }
    
    # Use smaller sample count for quick test
    results = run_online_comparison(models, n_tasks=3, samples_per_task=1000, device=device)
    print_online_results(results)
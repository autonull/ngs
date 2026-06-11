"""
Evaluation metrics for continual learning.
Includes: Accuracy matrix, Forgetting, BWT, FWT, LA, etc.
"""
import torch
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json


@dataclass
class CLMetrics:
    """Continual Learning metrics."""
    # Accuracy matrix: acc[i][j] = accuracy on task i after training task j
    accuracy_matrix: np.ndarray  # shape (n_tasks, n_tasks)

    # Final accuracies per task
    final_accuracies: np.ndarray  # shape (n_tasks,)

    # Average final accuracy
    avg_final_accuracy: float

    # Forgetting per task: max_acc - final_acc
    forgetting: np.ndarray  # shape (n_tasks,)
    avg_forgetting: float

    # Backward Transfer (BWT): avg_{i<j} (acc[i][j] - acc[i][i])
    bwt: float

    # Forward Transfer (FWT): avg_{i<j} (acc[i][j] - random_baseline)
    fwt: float

    # Learning Accuracy (LA): avg_i acc[i][i]
    la: float

    # Memory stability: ratio of params changed
    param_stability: float = 0.0

    # Model capacity used
    active_units: int = 0
    max_units: int = 0

    def to_dict(self) -> Dict:
        return {
            'accuracy_matrix': self.accuracy_matrix.tolist(),
            'final_accuracies': self.final_accuracies.tolist(),
            'avg_final_accuracy': float(self.avg_final_accuracy),
            'forgetting': self.forgetting.tolist(),
            'avg_forgetting': float(self.avg_forgetting),
            'bwt': float(self.bwt),
            'fwt': float(self.fwt),
            'la': float(self.la),
            'param_stability': float(self.param_stability),
            'active_units': int(self.active_units),
            'max_units': int(self.max_units),
        }

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_metrics(accuracy_matrix: np.ndarray, random_baseline: float = 0.1) -> CLMetrics:
    """Compute all CL metrics from accuracy matrix."""
    n_tasks = accuracy_matrix.shape[0]

    # Final accuracies (diagonal of last column)
    final_accuracies = accuracy_matrix[:, -1]
    avg_final = final_accuracies.mean()

    # Forgetting: max over training - final
    max_accuracies = accuracy_matrix.max(axis=1)
    forgetting = max_accuracies - final_accuracies
    avg_forgetting = forgetting.mean()

    # BWT: how much past tasks improve/forget after learning new tasks
    bwt_sum = 0
    bwt_count = 0
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            bwt_sum += accuracy_matrix[i, j] - accuracy_matrix[i, i]
            bwt_count += 1
    bwt = bwt_sum / bwt_count if bwt_count > 0 else 0

    # FWT: how well new tasks perform before explicit training
    fwt_sum = 0
    fwt_count = 0
    for i in range(n_tasks):
        for j in range(i):
            fwt_sum += accuracy_matrix[i, j] - random_baseline
            fwt_count += 1
    fwt = fwt_sum / fwt_count if fwt_count > 0 else 0

    # LA: average accuracy on each task right after learning it
    la = accuracy_matrix.diagonal().mean()

    return CLMetrics(
        accuracy_matrix=accuracy_matrix,
        final_accuracies=final_accuracies,
        avg_final_accuracy=avg_final,
        forgetting=forgetting,
        avg_forgetting=avg_forgetting,
        bwt=bwt,
        fwt=fwt,
        la=la,
    )


def evaluate_model_on_task(model, test_loader, device) -> float:
    """Evaluate model on a single task."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def run_continual_evaluation(
    model,
    get_task_loaders,
    n_tasks: int,
    device: str,
    train_fn,
    train_kwargs: dict = None
) -> Tuple[np.ndarray, List[float]]:
    """
    Run full continual learning evaluation.
    Returns accuracy matrix and list of active units per task.
    """
    train_kwargs = train_kwargs or {}
    accuracy_matrix = np.zeros((n_tasks, n_tasks))
    active_units = []
    old_model = None

    for task_id in range(n_tasks):
        train_loader, test_loader, _ = get_task_loaders(task_id)

        # Train on current task
        train_fn(model, train_loader, task_id, old_model=old_model, **train_kwargs)

        # Evaluate on all seen tasks
        for eval_task in range(task_id + 1):
            _, eval_test_loader, _ = get_task_loaders(eval_task)
            acc = evaluate_model_on_task(model, eval_test_loader, device)
            accuracy_matrix[eval_task, task_id] = acc

        # Save model for next task (for LwF, KD, etc.)
        if hasattr(model, 'set_old_model') or train_kwargs.get('use_kd', False):
            old_model = deepcopy(model)
            old_model.eval()
            for p in old_model.parameters():
                p.requires_grad = False

        # Track active units if applicable
        if hasattr(model, 'K'):
            active_units.append(model.K)
        elif hasattr(model, 'active_mask'):
            active_units.append(model.active_mask.sum().item())
        else:
            active_units.append(0)

    return accuracy_matrix, active_units


def print_results(metrics: CLMetrics, experiment_name: str = "Experiment"):
    """Pretty print CL metrics."""
    print(f"\n{'='*60}")
    print(f"{experiment_name} Results")
    print(f"{'='*60}")
    print(f"Average Final Accuracy: {metrics.avg_final_accuracy:.4f}")
    print(f"Average Forgetting:     {metrics.avg_forgetting:.4f}")
    print(f"Backward Transfer (BWT): {metrics.bwt:.4f}")
    print(f"Forward Transfer (FWT):  {metrics.fwt:.4f}")
    print(f"Learning Accuracy (LA):  {metrics.la:.4f}")
    print(f"\nPer-task Final Accuracies:")
    for i, acc in enumerate(metrics.final_accuracies):
        print(f"  Task {i}: {acc:.4f} (forgetting: {metrics.forgetting[i]:.4f})")
    print(f"\nAccuracy Matrix (rows=eval task, cols=after task):")
    for i in range(metrics.accuracy_matrix.shape[0]):
        row = []
        for j in range(metrics.accuracy_matrix.shape[1]):
            if j <= i:
                row.append(f"{metrics.accuracy_matrix[i, j]:.4f}")
            else:
                row.append("----")
        print(f"  Task {i}: {row}")


from copy import deepcopy
"""
Evaluation metrics for continual learning.
Includes: Accuracy matrix, Forgetting, BWT, FWT, LA, etc.
Statistical significance testing for rigorous comparison.
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from scipy import stats


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

    # Random baseline used for FWT
    random_baseline: float

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
            'random_baseline': float(self.random_baseline),
            'param_stability': float(self.param_stability),
            'active_units': int(self.active_units),
            'max_units': int(self.max_units),
        }

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_metrics(accuracy_matrix: np.ndarray, random_baseline: float) -> CLMetrics:
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
        random_baseline=random_baseline,
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
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            pred = logits.argmax(dim=1)
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


def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for a set of values using t-distribution."""
    n = len(values)
    if n < 2:
        return (float(values[0]), float(values[0])) if n == 1 else (0.0, 0.0)
    mean = np.mean(values)
    se = stats.sem(values)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return (mean - h, mean + h)


def paired_ttest(values_a: np.ndarray, values_b: np.ndarray) -> Tuple[float, float]:
    """Perform paired t-test between two sets of values. Returns (t-statistic, p-value)."""
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return (0.0, 1.0)
    t_stat, p_val = stats.ttest_rel(values_a, values_b)
    return (float(t_stat), float(p_val))


def welch_ttest(values_a: np.ndarray, values_b: np.ndarray) -> Tuple[float, float]:
    """Perform Welch's t-test (unequal variance) between two sets of values."""
    if len(values_a) < 2 or len(values_b) < 2:
        return (0.0, 1.0)
    t_stat, p_val = stats.ttest_ind(values_a, values_b, equal_var=False)
    return (float(t_stat), float(p_val))


def compare_models_significance(
    results_a: Dict[str, List[float]],
    results_b: Dict[str, List[float]],
    metrics: List[str] = None,
    paired: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compare two models across multiple metrics with statistical significance.
    
    Args:
        results_a: {metric_name: [values across seeds]} for model A
        results_b: {metric_name: [values across seeds]} for model B
        metrics: List of metrics to compare (default: all common metrics)
        paired: Whether to use paired t-test (same seeds) or Welch's t-test
        
    Returns:
        {metric: {'t_stat': t, 'p_value': p, 'significant': bool, 'ci_a': (lo, hi), 'ci_b': (lo, hi)}}
    """
    if metrics is None:
        metrics = list(set(results_a.keys()) & set(results_b.keys()))
    
    comparison = {}
    for metric in metrics:
        if metric not in results_a or metric not in results_b:
            continue
        vals_a = np.array(results_a[metric])
        vals_b = np.array(results_b[metric])
        
        if paired:
            t_stat, p_val = paired_ttest(vals_a, vals_b)
        else:
            t_stat, p_val = welch_ttest(vals_a, vals_b)
        
        ci_a = compute_confidence_interval(vals_a)
        ci_b = compute_confidence_interval(vals_b)
        
        comparison[metric] = {
            't_stat': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'ci_a': ci_a,
            'ci_b': ci_b,
            'mean_a': float(np.mean(vals_a)),
            'mean_b': float(np.mean(vals_b)),
            'mean_diff': float(np.mean(vals_a) - np.mean(vals_b)),
        }
    
    return comparison


def bootstrap_confidence_interval(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval (useful for non-normal distributions)."""
    if len(values) < 2:
        return (float(values[0]), float(values[0])) if len(values) == 1 else (0.0, 0.0)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, 100 * alpha)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha))
    return (float(lower), float(upper))


def effect_size_cohens_d(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n_a, n_b = len(values_a), len(values_b)
    if n_a < 2 or n_b < 2:
        return 0.0
    var_a, var_b = np.var(values_a, ddof=1), np.var(values_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(values_a) - np.mean(values_b)) / pooled_std)
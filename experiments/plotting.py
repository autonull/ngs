"""
Plotting and visualization for continual learning experiments.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from typing import Dict, List, Optional
import pandas as pd

from experiments.metrics import compute_metrics

# Style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12


def plot_accuracy_matrix(accuracy_matrix: np.ndarray, title: str, save_path: str,
                         vmin: float = 0.0, vmax: float = 1.0):
    """Plot accuracy matrix heatmap."""
    n_tasks = accuracy_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))

    # Mask future tasks (upper triangle)
    mask = np.triu(np.ones_like(accuracy_matrix, dtype=bool), k=1)

    sns.heatmap(
        accuracy_matrix,
        mask=mask,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=vmin,
        vmax=vmax,
        center=0.5,
        square=True,
        cbar_kws={'label': 'Accuracy'},
        ax=ax
    )

    ax.set_xlabel('Task Learned (Time)')
    ax.set_ylabel('Task Evaluated')
    ax.set_title(title)
    ax.set_xticks(np.arange(n_tasks) + 0.5)
    ax.set_yticks(np.arange(n_tasks) + 0.5)
    ax.set_xticklabels([f'Task {i}' for i in range(n_tasks)])
    ax.set_yticklabels([f'Task {i}' for i in range(n_tasks)])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_forgetting(forgetting: np.ndarray, task_names: List[str], title: str, save_path: str):
    """Plot per-task forgetting."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['red' if f > 0.1 else 'orange' if f > 0.01 else 'green' for f in forgetting]
    bars = ax.bar(range(len(forgetting)), forgetting, color=colors, alpha=0.7, edgecolor='black')

    ax.set_xlabel('Task')
    ax.set_ylabel('Forgetting (Max Acc - Final Acc)')
    ax.set_title(title)
    ax.set_xticks(range(len(forgetting)))
    ax.set_xticklabels(task_names)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5)

    # Add value labels
    for bar, val in zip(bars, forgetting):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_curves(accuracies: Dict[str, List[List[float]]], title: str, save_path: str):
    """Plot accuracy over tasks for multiple models/seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, seed_runs in accuracies.items():
        seed_runs = np.array(seed_runs)  # (n_seeds, n_tasks)
        mean_acc = seed_runs.mean(axis=0)
        std_acc = seed_runs.std(axis=0)
        tasks = range(1, len(mean_acc) + 1)

        ax.plot(tasks, mean_acc, 'o-', label=model_name, linewidth=2, markersize=8)
        ax.fill_between(tasks, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2)

    ax.set_xlabel('Number of Tasks Learned')
    ax.set_ylabel('Average Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_bar(aggregated: Dict, metric: str, title: str, save_path: str):
    """Bar chart comparing models on a metric."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(aggregated.keys())
    means = [aggregated[m][metric]['mean'] for m in models]
    stds = [aggregated[m][metric]['std'] for m in models]

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')

    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_radar_chart(aggregated: Dict, metrics: List[str], title: str, save_path: str):
    """Radar chart for multi-metric comparison."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    models = list(aggregated.keys())
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        values = [aggregated[model][m]['mean'] for m in metrics]
        # Normalize to [0, 1] for radar (some metrics like forgetting should be inverted)
        norm_values = []
        for m, v in zip(metrics, values):
            if m == 'avg_forgetting':
                norm_values.append(1 - v)  # Lower forgetting is better
            else:
                norm_values.append(v)
        norm_values += norm_values[:1]

        ax.plot(angles, norm_values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, norm_values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_capacity_growth(active_units: Dict[str, List[int]], title: str, save_path: str):
    """Plot model capacity growth over tasks."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, units in active_units.items():
        tasks = range(1, len(units) + 1)
        ax.plot(tasks, units, 'o-', label=model_name, linewidth=2, markersize=8)

    ax.set_xlabel('Task')
    ax.set_ylabel('Active Units / Parameters')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(results: Dict, output_dir: str = './plots'):
    """Generate full report with all plots from results."""
    os.makedirs(output_dir, exist_ok=True)

    # Group by experiment
    for key, result in results.items():
        if 'error' in result:
            continue

        # Parse filename: ExpName_Model_seedN.json
        # key is filename without .json
        parts = key.split('_')
        if len(parts) >= 3:
            exp_name = '_'.join(parts[:-2])  # Everything before model_seed
            model = parts[-2]
            seed = parts[-1].replace('seed', '')
        else:
            exp_name = result.get('config', 'Unknown')
            model = result.get('model', 'Unknown')
            seed = result.get('seed', 0)

        # Handle both old format (direct accuracy_matrix) and new format (wrapped in metrics)
        if 'accuracy_matrix' in result:
            acc_matrix = np.array(result['accuracy_matrix'])
            # Compute metrics if not present
            if 'metrics' in result:
                metrics = result['metrics']
            else:
                metrics = compute_metrics(acc_matrix).to_dict()
        elif 'metrics' in result and 'accuracy_matrix' in result['metrics']:
            acc_matrix = np.array(result['metrics']['accuracy_matrix'])
            metrics = result['metrics']
        else:
            print(f"Skipping {key}: unknown format")
            continue

        # Accuracy matrix heatmap
        plot_accuracy_matrix(
            acc_matrix,
            f"{exp_name} - {model.upper()} (seed={seed})",
            os.path.join(output_dir, f"{exp_name}_{model}_seed{seed}_matrix.png")
        )

        # Forgetting
        task_names = [f"Task {i}" for i in range(len(metrics['forgetting']))]
        plot_forgetting(
            np.array(metrics['forgetting']),
            task_names,
            f"{exp_name} - {model.upper()} Forgetting (seed={seed})",
            os.path.join(output_dir, f"{exp_name}_{model}_seed{seed}_forgetting.png")
        )

    # Aggregate and compare
    from experiments.runner import aggregate_results
    aggregated = aggregate_results(results, group_by='model')

    if aggregated:
        # Comparison bars
        for metric in ['avg_final_accuracy', 'avg_forgetting', 'bwt', 'fwt', 'la']:
            plot_comparison_bar(
                aggregated, metric,
                f"Model Comparison: {metric.replace('_', ' ').title()}",
                os.path.join(output_dir, f"comparison_{metric}.png")
            )

        # Radar chart
        plot_radar_chart(
            aggregated,
            ['avg_final_accuracy', 'avg_forgetting', 'bwt', 'fwt', 'la'],
            "Multi-Metric Comparison",
            os.path.join(output_dir, "radar_comparison.png")
        )

    print(f"Report generated in {output_dir}/")


def load_results(results_dir: str) -> Dict:
    """Load all results from directory."""
    results = {}
    for fname in os.listdir(results_dir):
        if fname.endswith('.json') and fname != 'summary.json':
            with open(os.path.join(results_dir, fname)) as f:
                results[fname.replace('.json', '')] = json.load(f)
    return results


if __name__ == '__main__':
    # Test with dummy data
    dummy_acc = np.array([
        [0.99, 0.98, 0.95],
        [0.0, 0.97, 0.93],
        [0.0, 0.0, 0.96],
    ])
    plot_accuracy_matrix(dummy_acc, "Test Matrix", "test_matrix.png")
    plot_forgetting(np.array([0.04, 0.04, 0.0]), ["Task 0", "Task 1", "Task 2"], "Test Forgetting", "test_forgetting.png")
    print("Test plots generated.")
"""
Comprehensive evaluation script for LeanNGS vs baselines.
Runs full comparison with best hyperparameters across multiple seeds.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import EXPERIMENTS, ModelConfig, TrainConfig
from experiments.runner import run_all_experiments, aggregate_results
from experiments.plotting import generate_report, load_results


# Best hyperparameters found from ablation studies (param-matched: max_k=448 ~513K params)
BEST_CONFIGS = {
    'split_mnist': {
        'epochs_per_task': 5,
        'kd_weight': 2.0,
        'split_thresh': 0.01,
        'prune_thresh': 0.01,
        'top_k': 8,
        'd_latent': 32,
        'max_k': 448,
        'replay_ratio': 1.0,
        'batch_size': 256,
        'lr': 1e-3,
    },
    'split_fashion': {
        'epochs_per_task': 5,
        'kd_weight': 2.0,
        'split_thresh': 0.005,
        'prune_thresh': 0.01,
        'top_k': 8,
        'd_latent': 32,
        'max_k': 448,
        'replay_ratio': 1.0,
        'batch_size': 256,
        'lr': 1e-3,
    },
    'split_cifar10': {
        'epochs_per_task': 5,
        'kd_weight': 2.0,
        'split_thresh': 0.005,
        'prune_thresh': 0.01,
        'top_k': 8,
        'd_latent': 32,
        'max_k': 448,
        'replay_ratio': 1.0,
        'batch_size': 256,
        'lr': 1e-3,
    },
    'split_cifar100': {
        'epochs_per_task': 10,
        'kd_weight': 2.0,
        'split_thresh': 0.005,
        'prune_thresh': 0.01,
        'top_k': 8,
        'd_latent': 64,
        'max_k': 448,
        'replay_ratio': 1.0,
        'batch_size': 128,
        'lr': 1e-3,
    },
    'permuted_mnist': {
        'epochs_per_task': 5,
        'kd_weight': 2.0,
        'split_thresh': 0.005,
        'prune_thresh': 0.01,
        'top_k': 8,
        'd_latent': 32,
        'max_k': 448,
        'replay_ratio': 1.0,
        'batch_size': 256,
        'lr': 1e-3,
    },
}


def apply_best_config(exp_name: str):
    """Apply best hyperparameters to experiment config."""
    if exp_name not in EXPERIMENTS:
        return
    if exp_name not in BEST_CONFIGS:
        return
    
    config = EXPERIMENTS[exp_name]
    best = BEST_CONFIGS[exp_name]
    
    # Apply training config
    for k, v in best.items():
        if hasattr(config.train, k):
            setattr(config.train, k, v)
    
    # Apply model config
    for k, v in best.items():
        if hasattr(config.model, k):
            setattr(config.model, k, v)


def run_comprehensive_eval(
    experiments: list,
    models: list,
    seeds: list,
    output_dir: str = './results',
    plots_dir: str = './plots',
    use_best_config: bool = True
):
    """Run comprehensive evaluation with best configs."""
    
    if use_best_config:
        for exp in experiments:
            apply_best_config(exp)
            print(f"Applied best config to {exp}")
    
    print(f"\nRunning experiments: {experiments}")
    print(f"Models: {models}")
    print(f"Seeds: {seeds}")
    
    results = run_all_experiments(
        experiment_names=experiments,
        model_names=models,
        seeds=seeds,
        output_dir=output_dir,
        verbose=True
    )
    
    # Aggregate and print summary
    print("\n" + "="*60)
    print("AGGREGATED RESULTS (mean ± std across seeds)")
    print("="*60)
    aggregated = aggregate_results(results, group_by='model')
    for model, metrics in aggregated.items():
        print(f"\n{model.upper()}:")
        for metric, vals in metrics.items():
            print(f"  {metric}: {vals['mean']:.4f} ± {vals['std']:.4f}")
    
    # Also aggregate by experiment
    print("\n" + "="*60)
    print("BY EXPERIMENT")
    print("="*60)
    aggregated_exp = aggregate_results(results, group_by='experiment')
    for exp, metrics in aggregated_exp.items():
        print(f"\n{exp.upper()}:")
        for metric, vals in metrics.items():
            print(f"  {metric}: {vals['mean']:.4f} ± {vals['std']:.4f}")
    
    # Generate plots
    print(f"\nGenerating plots in {plots_dir}...")
    generate_report(results, plots_dir)
    
    # Save aggregated results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'aggregated_by_model.json'), 'w') as f:
        json.dump(aggregated, f, indent=2)
    with open(os.path.join(output_dir, 'aggregated_by_experiment.json'), 'w') as f:
        json.dump(aggregated_exp, f, indent=2)
    
    return results, aggregated, aggregated_exp


def generate_latex_table(aggregated_by_model: dict, aggregated_by_exp: dict):
    """Generate LaTeX table for paper."""
    print("\n" + "="*60)
    print("LATEX TABLE")
    print("="*60)
    
    models = list(aggregated_by_model.keys())
    metrics = ['avg_final_accuracy', 'avg_forgetting', 'bwt', 'la']
    
    # Table by model (averaged across experiments)
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Continual Learning Results (averaged across datasets and seeds)}")
    print("\\begin{tabular}{l" + "c" * len(metrics) + "}")
    print("\\toprule")
    print("Model & " + " & ".join([m.replace('_', ' ').title() for m in metrics]) + " \\\\")
    print("\\midrule")
    for model in sorted(models):
        row = [model.upper()]
        for metric in metrics:
            mean = aggregated_by_model[model][metric]['mean']
            std = aggregated_by_model[model][metric]['std']
            row.append(f"{mean:.2f} $\\pm$ {std:.2f}")
        print(" & ".join(row) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Table by experiment
    print("\n\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Results per Experiment (LeanNGS vs Best Baseline)}")
    print("\\begin{tabular}{l" + "c" * len(metrics) + "}")
    print("\\toprule")
    print("Experiment & " + " & ".join([m.replace('_', ' ').title() for m in metrics]) + " \\\\")
    print("\\midrule")
    for exp in sorted(aggregated_by_exp.keys()):
        if 'lean_ngs' in exp.lower():
            row = [exp.replace('_', '-')]
            for metric in metrics:
                mean = aggregated_by_exp[exp][metric]['mean']
                std = aggregated_by_exp[exp][metric]['std']
                row.append(f"{mean:.2f} $\\pm$ {std:.2f}")
            print(" & ".join(row) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive evaluation')
    parser.add_argument('--experiments', nargs='+', 
                        default=['split_mnist', 'split_fashion', 'split_cifar10'],
                        choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--models', nargs='+', 
                        default=['lean_ngs', 'mlp', 'er', 'ewc', 'si', 'lwf', 'lora'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--output-dir', default='./results')
    parser.add_argument('--plots-dir', default='./plots')
    parser.add_argument('--no-best-config', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    
    args = parser.parse_args()
    
    if args.plot_only:
        print(f"Loading results from {args.output_dir}...")
        results = load_results(args.output_dir)
        generate_report(results, args.plots_dir)
        aggregated = aggregate_results(results, group_by='model')
        generate_latex_table(aggregated, aggregate_results(results, group_by='experiment'))
    else:
        results, agg_model, agg_exp = run_comprehensive_eval(
            experiments=args.experiments,
            models=args.models,
            seeds=args.seeds,
            output_dir=args.output_dir,
            plots_dir=args.plots_dir,
            use_best_config=not args.no_best_config
        )
        generate_latex_table(agg_model, agg_exp)
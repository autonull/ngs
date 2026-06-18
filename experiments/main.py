#!/usr/bin/env python
"""
Main entry point for continual learning experiments.
Run: python -m experiments.main --experiments split_mnist split_fashion --models ngs_baseline mlp er ewc --seeds 42 123
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import EXPERIMENTS
from experiments.runner import run_all_experiments, aggregate_results
from experiments.plotting import generate_report


def main():
    parser = argparse.ArgumentParser(description='Continual Learning Experiments')
    parser.add_argument('--experiments', nargs='+', default=['split_mnist'],
                        choices=list(EXPERIMENTS.keys()),
                        help='Experiments to run')
    ngs_models = [
        'ngs_baseline', 'ngs_cfg_net', 'ngs_ultra_edge', 'ngs_abl_hyper',
        'ngs_baseline_lora', 'ngs_cfg_net_lora', 'ngs_abl_hyper_lora'
    ]
    all_models = ['mlp', 'er', 'ewc', 'si', 'lwf', 'lora'] + ngs_models
    
    parser.add_argument('--models', nargs='+', default=['mlp', 'er', 'ewc', 'si', 'lwf', 'lora'],
                        choices=all_models,
                        help='Models to evaluate')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                        help='Random seeds')
    parser.add_argument('--output-dir', default='./results',
                        help='Output directory for results')
    parser.add_argument('--plots-dir', default='./plots',
                        help='Output directory for plots')
    parser.add_argument('--no-verbose', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate plots from existing results')

    args = parser.parse_args()

    if args.plot_only:
        print(f"Loading results from {args.output_dir}...")
        from experiments.plotting import load_results
        results = load_results(args.output_dir)
        generate_report(results, args.plots_dir)
        return

    print(f"Running experiments: {args.experiments}")
    print(f"Models: {args.models}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {args.output_dir}")

    results = run_all_experiments(
        experiment_names=args.experiments,
        model_names=args.models,
        seeds=args.seeds,
        output_dir=args.output_dir,
        verbose=not args.no_verbose
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

    # Generate plots
    print(f"\nGenerating plots in {args.plots_dir}...")
    generate_report(results, args.plots_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
#!/usr/bin/env python
"""Generate all paper figures and LaTeX tables from experimental results."""
import json
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.plotting import (
    plot_accuracy_matrix, plot_forgetting, plot_radar_chart, generate_report, load_results
)
from ngs.visualization.visualize import (
    plot_topology_dynamics, plot_routing_heatmap, plot_3d_gaussian_means,
    plot_uncertainty_calibration, plot_subspace_alignment, plot_hypernetwork_codes,
    plot_riemannian_geodesics, plot_evolution_gif
)


def generate_latex_tables(results_dir, output_dir):
    """Generate LaTeX tables for main results, ablations, and compute."""
    results = load_results(results_dir)
    aggregated = {}
    
    for key, result in results.items():
        if 'error' in result:
            continue
        exp_name = result.get('config', 'unknown')
        model = result.get('model', 'unknown')
        if 'metrics' in result:
            m = result['metrics']
        else:
            continue
        group_key = f"{exp_name}_{model}"
        if group_key not in aggregated:
            aggregated[group_key] = {
                'avg_final_accuracy': [], 'avg_forgetting': [], 'bwt': [], 'fwt': [], 'la': []
            }
        for k in ['avg_final_accuracy', 'avg_forgetting', 'bwt', 'fwt', 'la']:
            aggregated[group_key][k].append(m[k])
    
    # Compute means and stds
    table_data = {}
    for key, vals in aggregated.items():
        table_data[key] = {k: (np.mean(v), np.std(v)) for k, v in vals.items()}
    
    # Generate main results table
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(Path(output_dir) / "main_results.tex", "w") as f:
        f.write("% Main Results Table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Continual Learning Results on Split-MNIST (5 seeds).}\n")
        f.write("\\label{tab:main_results}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Accuracy $\\uparrow$ & Forgetting $\\downarrow$ & BWT $\\uparrow$ & FWT $\\uparrow$ & LA $\\uparrow$ \\\\\n")
        f.write("\\midrule\n")
        
        for model_name in sorted(table_data.keys()):
            if 'Split-MNIST' in model_name:
                m = table_data[model_name]
                model_short = model_name.replace('Split-MNIST_', '').replace('_seed42', '')
                f.write(f"{model_short} & ")
                f.write(f"{m['avg_final_accuracy'][0]:.2f} $\\pm$ {m['avg_final_accuracy'][1]:.2f} & ")
                f.write(f"{m['avg_forgetting'][0]:.2f} $\\pm$ {m['avg_forgetting'][1]:.2f} & ")
                f.write(f"{m['bwt'][0]:.2f} $\\pm$ {m['bwt'][1]:.2f} & ")
                f.write(f"{m['fwt'][0]:.2f} $\\pm$ {m['fwt'][1]:.2f} & ")
                f.write(f"{m['la'][0]:.2f} $\\pm$ {m['la'][1]:.2f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    # Generate ablation table
    with open(Path(output_dir) / "ablation_results.tex", "w") as f:
        f.write("% Ablation Results Table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation Study on Split-MNIST.}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Variant & Accuracy $\\uparrow$ & Forgetting $\\downarrow$ \\\\\n")
        f.write("\\midrule\n")
        
        for variant in ['ngs_baseline', 'ngs_cfg_net', 'ngs_abl_hyper']:
            key = f"Split-MNIST_{variant}"
            if key in table_data:
                m = table_data[key]
                f.write(f"{variant} & ")
                f.write(f"{m['avg_final_accuracy'][0]:.2f} $\\pm$ {m['avg_final_accuracy'][1]:.2f} & ")
                f.write(f"{m['avg_forgetting'][0]:.2f} $\\pm$ {m['avg_forgetting'][1]:.2f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX tables saved to {output_dir}/")


def generate_all_figures(results_dir, plots_dir, figures_dir):
    """Generate all publication figures."""
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_results(results_dir)
    
    # 1. Generate standard experiment plots (accuracy matrices, forgetting, radar)
    print("Generating standard experiment plots...")
    generate_report(results, plots_dir)
    
    # 2. Generate radar chart
    print("Generating radar chart...")
    from scripts.generate_radar_chart import load_results as load_radar
    aggregated = load_radar(results_dir)
    plot_radar_chart(
        aggregated,
        ["avg_final_accuracy", "avg_forgetting", "la", "bwt", "fwt"],
        title="NGS Variants Comparison - Split-MNIST",
        save_path=Path(figures_dir) / "radar_comparison.png"
    )
    
    print(f"All figures saved to {figures_dir}/ and {plots_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures and tables")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--plots-dir", default="./plots")
    parser.add_argument("--figures-dir", default="./paper/figures")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    
    if not args.figures_only:
        print("Generating LaTeX tables...")
        generate_latex_tables(args.results_dir, args.figures_dir)
    
    if not args.tables_only:
        print("Generating figures...")
        generate_all_figures(args.results_dir, args.plots_dir, args.figures_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate radar chart comparing 3 NGS variants on Split-MNIST (Quick Win #3)."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.plotting import plot_radar_chart


def load_results(base_dir: str = "./results"):
    """Load Split-MNIST results for 3 NGS variants."""
    variants = {
        "NGS-Baseline": "ngs_baseline",
        "NGS-CFG": "ngs_cfg_net",
        "NGS-Hyper": "ngs_abl_hyper",
    }

    aggregated = {}
    for label, variant in variants.items():
        seed_files = [f for f in os.listdir(base_dir)
                      if f.startswith(f"Split-MNIST_{variant}_seed") and f.endswith(".json")]
        all_metrics = []
        for fname in sorted(seed_files):
            with open(os.path.join(base_dir, fname)) as f:
                data = json.load(f)
                all_metrics.append(data["metrics"])

        # Aggregate across seeds
        metrics_keys = ["avg_final_accuracy", "avg_forgetting", "la", "bwt", "fwt"]
        agg = {}
        for key in metrics_keys:
            vals = [m[key] for m in all_metrics]
            agg[key] = {"mean": sum(vals) / len(vals), "std": (sum((v - sum(vals)/len(vals))**2 for v in vals) / len(vals))**0.5}
        aggregated[label] = agg

    return aggregated


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate radar chart")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output", default="./plots/radar_comparison.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    aggregated = load_results(args.results_dir)
    metrics = ["avg_final_accuracy", "avg_forgetting", "la", "bwt", "fwt"]

    plot_radar_chart(
        aggregated,
        metrics,
        title="NGS Variants Comparison - Split-MNIST",
        save_path=args.output,
    )
    print(f"Saved radar chart to {args.output}")


if __name__ == "__main__":
    main()
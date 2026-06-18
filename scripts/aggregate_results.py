#!/usr/bin/env python
"""Enhanced result aggregation: 95% CI, Cohen's d, significance tests, LaTeX tables (Phase 1.3)."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir: str) -> Dict[str, dict]:
    """Load all JSON result files from directory."""
    results = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json") and not fname.startswith("aggregated"):
            with open(os.path.join(results_dir, fname)) as f:
                results[fname.replace(".json", "")] = json.load(f)
    return results


def group_results(results: Dict[str, dict]) -> Dict[str, Dict[str, List[float]]]:
    """Group results by (experiment, variant) and extract metric values."""
    groups = defaultdict(lambda: defaultdict(list))
    for key, data in results.items():
        if "error" in data:
            continue
        metrics = data.get("metrics", data)
        if not isinstance(metrics, dict):
            continue

        # Infer experiment name and variant from filename
        parts = key.rsplit("_seed", 1)
        if len(parts) == 2:
            name = parts[0]
            # Try to split into experiment and model
            tokens = name.split("_")
            if len(tokens) >= 2:
                # Heuristic: first N-1 tokens = experiment, last token = variant
                exp = "_".join(tokens[:-1])
                variant = tokens[-1]
            else:
                exp = name
                variant = name
        else:
            exp = key
            variant = "unknown"

        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                groups[(exp, variant)][metric_name].append(float(value))

    return groups


def compute_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval."""
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    mean = np.mean(values)
    se = stats.sem(values)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h


def cohens_d(vals_a: np.ndarray, vals_b: np.ndarray) -> float:
    """Cohen's d effect size."""
    n1, n2 = len(vals_a), len(vals_b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = np.var(vals_a, ddof=1), np.var(vals_b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return (np.mean(vals_a) - np.mean(vals_b)) / pooled


def significance_test(
    vals_a: np.ndarray, vals_b: np.ndarray
) -> Dict[str, float]:
    """Run paired t-test and Wilcoxon signed-rank test."""
    result = {}
    if len(vals_a) != len(vals_b) or len(vals_a) < 2:
        return {"t_stat": float("nan"), "t_pvalue": float("nan"),
                "wilcoxon_stat": float("nan"), "wilcoxon_pvalue": float("nan"),
                "cohens_d": float("nan")}

    t_stat, t_p = stats.ttest_rel(vals_a, vals_b)
    w_stat, w_p = stats.wilcoxon(vals_a - vals_b, alternative="two-sided")
    d = cohens_d(vals_a, vals_b)

    return {
        "t_stat": float(t_stat),
        "t_pvalue": float(t_p),
        "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
        "wilcoxon_pvalue": float(w_p),
        "cohens_d": float(d),
        "significant_005": bool(t_p < 0.05),
    }


def aggregate(groups: Dict, baseline_variant: Optional[str] = None) -> Dict:
    """Compute full aggregation with statistics."""
    aggregated = {}
    # Group by experiment
    experiments = defaultdict(dict)
    for (exp, variant), metrics in groups.items():
        experiments[exp][variant] = metrics

    for exp, variants in experiments.items():
        exp_result = {}
        for variant, metrics in variants.items():
            variant_result = {}
            for metric_name, values in metrics.items():
                arr = np.array(values)
                mean = float(np.mean(arr))
                std = float(np.std(arr, ddof=1))
                ci_lo, ci_hi = compute_ci(arr)
                variant_result[metric_name] = {
                    "mean": mean,
                    "std": std,
                    "ci_95": [ci_lo, ci_hi],
                    "n": len(arr),
                    "values": values,
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }
            exp_result[variant] = variant_result

        # Significance tests vs baseline
        if baseline_variant and baseline_variant in variants:
            baseline = variants[baseline_variant]
            for variant, metrics in variants.items():
                if variant == baseline_variant:
                    continue
                sig_tests = {}
                common_metrics = set(metrics.keys()) & set(baseline.keys())
                for m in common_metrics:
                    vals_a = np.array(metrics[m].get("values", metrics[m]) if isinstance(metrics[m], dict) else metrics[m])
                    vals_b = np.array(baseline[m].get("values", baseline[m]) if isinstance(baseline[m], dict) else baseline[m])
                    sig_tests[m] = significance_test(vals_a, vals_b)
                exp_result.setdefault("significance_tests", {})[variant] = sig_tests

        aggregated[exp] = exp_result

    return aggregated


def generate_latex(aggregated: Dict, output_dir: str):
    """Generate LaTeX tables from aggregated results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    key_metrics = ["avg_final_accuracy", "avg_forgetting", "bwt", "la"]
    metric_labels = {
        "avg_final_accuracy": "Avg. Final Acc.",
        "avg_forgetting": "Avg. Forgetting",
        "bwt": "BWT",
        "la": "LA",
    }

    for exp, variants in aggregated.items():
        exp_clean = exp.replace("_", "-").replace("Split-", "").replace("MNIST", "MNIST")
        variant_names = [v for v in variants if v != "significance_tests"]
        sig_tests = variants.get("significance_tests", {})

        lines = [
            "\\begin{table}[ht]",
            "\\centering",
            f"\\caption{{Results on {exp_clean}}}",
            f"\\label{{tab:{exp}}}",
            "\\begin{tabular}{l" + "c".join(["c"] * len(key_metrics)) + "}",
            "\\toprule",
            "Variant & " + " & ".join([metric_labels.get(m, m) for m in key_metrics]) + " \\\\",
            "\\midrule",
        ]

        for var in variant_names:
            row = [var.replace("_", " ").title()]
            for m in key_metrics:
                if m in variants[var]:
                    v = variants[var][m]
                    row.append(f"${v['mean']:.3f}\\pm{v['std']:.3f}$")
                else:
                    row.append("---")

            # Add significance star
            if var in sig_tests:
                for m in key_metrics:
                    if m in sig_tests[var] and sig_tests[var][m].get("significant_005"):
                        row[-1] = row[-1] + "$^*$"

            lines.append(" & ".join(row) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ])

        path = Path(output_dir) / f"table_{exp}.tex"
        path.write_text("\n".join(lines))
        print(f"  LaTeX: {path}")


def generate_markdown(aggregated: Dict, output_path: str):
    """Generate readable markdown summary."""
    lines = ["# Aggregated Results", "", "## Summary", ""]
    key_metrics = ["avg_final_accuracy", "avg_forgetting", "bwt", "la"]
    metric_labels = dict(
        avg_final_accuracy="Acc\\%",
        avg_forgetting="Forget\\%",
        bwt="BWT\\%",
        la="LA\\%",
    )

    for exp, variants in sorted(aggregated.items()):
        variant_names = [v for v in variants if v != "significance_tests"]
        sig_tests = variants.get("significance_tests", {})

        lines.append(f"### {exp}")
        lines.append("")
        header = "| Variant | " + " | ".join(metric_labels.values()) + " | Sig |"
        sep = "|" + "|".join(["---"] * (len(key_metrics) + 2)) + "|"
        lines.append(header)
        lines.append(sep)

        for var in variant_names:
            row = [var.replace("_", " ").title()]
            for m in key_metrics:
                if m in variants[var]:
                    v = variants[var][m]
                    ci_str = f"[{v['ci_95'][0]:.3f}, {v['ci_95'][1]:.3f}]"
                    row.append(f"{v['mean']:.3f}±{v['std']:.3f} ({ci_str})")
                else:
                    row.append("---")

            # Significance stars
            stars = ""
            if var in sig_tests:
                sig_flags = []
                for m in key_metrics:
                    if m in sig_tests[var]:
                        sig_flags.append("*" if sig_tests[var][m].get("significant_005") else "")
                if any(sig_flags):
                    stars = "/".join(sig_flags)
            row.append(stars)
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    Path(str(output_path)).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines))
    print(f"  Markdown: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate results with statistics")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--baseline", default=None, help="Baseline variant for significance tests (e.g., mlp)")
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}")
    results = load_results(args.results_dir)
    print(f"  Loaded {len(results)} result files")

    groups = group_results(results)
    print(f"  Found {len(groups)} (experiment, variant) groups")

    aggregated = aggregate(groups, baseline_variant=args.baseline)
    print(f"  Aggregated {len(aggregated)} experiments")

    # Save aggregated JSON
    for exp, data in aggregated.items():
        path = Path(args.output_dir) / f"aggregated_{exp}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  JSON: {path}")

    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    generate_latex(aggregated, args.output_dir)

    # Generate markdown summary
    md_path = Path(args.output_dir) / "aggregated_report.md"
    generate_markdown(aggregated, str(md_path))

    print(f"\nDone. All outputs in {args.output_dir}")


if __name__ == "__main__":
    main()

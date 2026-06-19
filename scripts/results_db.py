#!/usr/bin/env python
"""Results database: systematic accumulation, querying, and comparison."""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ExperimentResult:
    """Structured experiment result."""
    variant: str
    benchmark: str
    seed: int
    metrics: Dict[str, float]
    config: Dict[str, Any]
    elapsed_seconds: float
    timestamp: str


class ResultsDatabase:
    """SQLite-like interface for experiment results stored as JSON."""
    
    def __init__(self, db_dir: str = "./results_db"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.db_dir / "index.json"
        self._index = self._load_index()
    
    def _load_index(self) -> Dict:
        if self.index_path.exists():
            with open(self.index_path) as f:
                data = json.load(f)
            data["variants"] = set(data.get("variants", []))
            data["benchmarks"] = set(data.get("benchmarks", []))
            return data
        return {"experiments": [], "variants": set(), "benchmarks": set()}
    def _load_index(self) -> Dict:
        if self.index_path.exists():
            with open(self.index_path) as f:
                data = json.load(f)
            data["variants"] = set(data.get("variants", []))
            data["benchmarks"] = set(data.get("benchmarks", []))
            return data
        return {"experiments": [], "variants": set(), "benchmarks": set()}
    def _load_index(self) -> Dict:
        if self.index_path.exists():
            with open(self.index_path) as f:
                data = json.load(f)
            data["variants"] = set(data.get("variants", []))
            data["benchmarks"] = set(data.get("benchmarks", []))
            return data
        return {"experiments": [], "variants": set(), "benchmarks": set()}
    def _load_index(self) -> Dict:
        if self.index_path.exists():
            with open(self.index_path) as f:
                data = json.load(f)
            data["variants"] = set(data.get("variants", []))
            data["benchmarks"] = set(data.get("benchmarks", []))
            return data
        return {"experiments": [], "variants": set(), "benchmarks": set()}
    def _load_index(self) -> Dict:
        if self.index_path.exists():
            with open(self.index_path) as f:
                data = json.load(f)
            data["variants"] = set(data.get("variants", []))
            data["benchmarks"] = set(data.get("benchmarks", []))
            return data
        return {"experiments": [], "variants": set(), "benchmarks": set()}
    
    def _save_index(self):
        # Convert sets to lists for JSON
        idx = self._index.copy()
        idx["variants"] = list(self._index["variants"])
        idx["benchmarks"] = list(self._index["benchmarks"])
        with open(self.index_path, 'w') as f:
            json.dump(idx, f, indent=2)
    
    def add_result(self, result: Dict[str, Any], variant: str, benchmark: str, 
                   seed: int, config: Dict, elapsed: float) -> str:
        """Add a result to the database."""
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        
        exp_id = f"{variant}_{benchmark}_seed{seed}_{timestamp.replace(':', '-')}"
        result_file = self.db_dir / f"{exp_id}.json"
        
        record = {
            "id": exp_id,
            "variant": variant,
            "benchmark": benchmark,
            "seed": seed,
            "metrics": result.get("metrics", result),
            "config": config,
            "elapsed_seconds": elapsed,
            "timestamp": timestamp,
        }
        
        with open(result_file, 'w') as f:
            json.dump(record, f, indent=2, default=str)
        
        # Update index
        self._index["experiments"].append({
            "id": exp_id,
            "variant": variant,
            "benchmark": benchmark,
            "seed": seed,
            "timestamp": timestamp,
            "file": str(result_file),
        })
        self._index["variants"].add(variant)
        self._index["benchmarks"].add(benchmark)
        self._save_index()
        
        return exp_id
    
    def query(self, variant: str = None, benchmark: str = None, 
              seeds: List[int] = None, metric: str = None) -> List[Dict]:
        """Query experiments with filters."""
        results = []
        for exp in self._index["experiments"]:
            if variant and exp["variant"] != variant:
                continue
            if benchmark and exp["benchmark"] != benchmark:
                continue
            if seeds and exp["seed"] not in seeds:
                continue
            
            with open(exp["file"]) as f:
                data = json.load(f)
            
            if metric and metric not in data.get("metrics", {}):
                continue
            
            results.append(data)
        
        return results
    
    def aggregate(self, variant: str = None, benchmark: str = None,
                  metric: str = "avg_final_accuracy") -> Dict[str, Any]:
        """Aggregate metrics across seeds."""
        results = self.query(variant=variant, benchmark=benchmark)
        
        if not results:
            return {}
        
        # Group by (variant, benchmark)
        groups = defaultdict(list)
        for r in results:
            key = (r["variant"], r["benchmark"])
            metrics = r.get("metrics", {})
            if metric in metrics:
                groups[key].append(metrics[metric])
        
        aggregated = {}
        for (var, bench), values in groups.items():
            arr = np.array(values)
            aggregated[f"{var}_{bench}"] = {
                "variant": var,
                "benchmark": bench,
                "n": len(arr),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "values": values,
            }
        
        return aggregated
    
    def compare_variants(self, benchmark: str, baseline: str, 
                         variants: List[str], metric: str = "avg_final_accuracy",
                         paired: bool = True) -> Dict[str, Any]:
        """Statistical comparison of variants vs baseline."""
        baseline_results = self.query(variant=baseline, benchmark=benchmark)
        baseline_vals = [r["metrics"].get(metric, np.nan) for r in baseline_results]
        baseline_vals = [v for v in baseline_vals if not np.isnan(v)]
        
        comparisons = {}
        for var in variants:
            if var == baseline:
                continue
            var_results = self.query(variant=var, benchmark=benchmark)
            var_vals = [r["metrics"].get(metric, np.nan) for r in var_results]
            var_vals = [v for v in var_vals if not np.isnan(v)]
            
            if len(baseline_vals) < 2 or len(var_vals) < 2:
                comparisons[var] = {"error": "Insufficient data"}
                continue
            
            # Paired t-test (assuming same seeds)
            if paired and len(baseline_vals) == len(var_vals):
                t_stat, t_p = stats.ttest_rel(var_vals, baseline_vals)
            else:
                t_stat, t_p = stats.ttest_ind(var_vals, baseline_vals, equal_var=False)
            
            # Wilcoxon
            try:
                w_stat, w_p = stats.wilcoxon(var_vals, baseline_vals, alternative="two-sided")
            except:
                w_stat, w_p = np.nan, np.nan
            
            # Cohen's d
            n1, n2 = len(var_vals), len(baseline_vals)
            s1, s2 = np.var(var_vals, ddof=1), np.var(baseline_vals, ddof=1)
            pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
            d = (np.mean(var_vals) - np.mean(baseline_vals)) / pooled if pooled > 0 else 0
            
            comparisons[var] = {
                "mean_diff": float(np.mean(var_vals) - np.mean(baseline_vals)),
                "t_stat": float(t_stat),
                "t_pvalue": float(t_p),
                "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
                "wilcoxon_pvalue": float(w_p),
                "cohens_d": float(d),
                "significant_005": bool(t_p < 0.05),
                "n_baseline": len(baseline_vals),
                "n_variant": len(var_vals),
            }
        
        return {
            "benchmark": benchmark,
            "baseline": baseline,
            "metric": metric,
            "comparisons": comparisons,
        }
    
    def get_pareto_frontier(self, benchmark: str, 
                            metrics: List[str] = ["avg_final_accuracy", "avg_forgetting"],
                            maximize: List[bool] = [True, False]) -> List[Dict]:
        """Find Pareto-optimal variants for a benchmark."""
        variants = list(self._index["variants"])
        variant_scores = {}
        
        for var in variants:
            scores = {}
            for m in metrics:
                agg = self.aggregate(variant=var, benchmark=benchmark, metric=m)
                key = f"{var}_{benchmark}"
                if key in agg:
                    scores[m] = agg[key]
            if scores:
                variant_scores[var] = scores
        
        # Simple Pareto: a variant is Pareto-optimal if no other variant dominates it
        # Other dominates var if other is better/equal in ALL metrics and strictly better in at least one
        pareto = []
        for var, scores in variant_scores.items():
            dominated = False
            for other_var, other_scores in variant_scores.items():
                if var == other_var:
                    continue
                # Check if other dominates var
                other_better_or_equal_all = True
                other_strictly_better_any = False
                for m, maximize_m in zip(metrics, maximize):
                    if maximize_m:
                        if other_scores[m]["mean"] < scores[m]["mean"]:
                            other_better_or_equal_all = False
                            break
                        if other_scores[m]["mean"] > scores[m]["mean"]:
                            other_strictly_better_any = True
                    else:
                        if other_scores[m]["mean"] > scores[m]["mean"]:
                            other_better_or_equal_all = False
                            break
                        if other_scores[m]["mean"] < scores[m]["mean"]:
                            other_strictly_better_any = True
                if other_better_or_equal_all and other_strictly_better_any:
                    dominated = True
                    break
            if not dominated:
                pareto.append({"variant": var, "scores": scores})
        
        return pareto
    
    def export_csv(self, output_path: str, variant: str = None, 
                   benchmark: str = None):
        """Export results to CSV for analysis."""
        import csv
        results = self.query(variant=variant, benchmark=benchmark)
        
        if not results:
            print("No results to export")
            return
        
        # Flatten metrics
        fieldnames = ["id", "variant", "benchmark", "seed", "elapsed_seconds", "timestamp"]
        metric_keys = set()
        for r in results:
            metric_keys.update(r.get("metrics", {}).keys())
        fieldnames.extend(sorted(metric_keys))
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {
                    "id": r["id"],
                    "variant": r["variant"],
                    "benchmark": r["benchmark"],
                    "seed": r["seed"],
                    "elapsed_seconds": r["elapsed_seconds"],
                    "timestamp": r["timestamp"],
                }
                row.update(r.get("metrics", {}))
                writer.writerow(row)
        
        print(f"Exported {len(results)} records to {output_path}")
    
    def summary(self) -> Dict:
        """Get database summary."""
        return {
            "total_experiments": len(self._index["experiments"]),
            "variants": sorted(list(self._index["variants"])),
            "benchmarks": sorted(list(self._index["benchmarks"])),
            "seeds_used": sorted(set(e["seed"] for e in self._index["experiments"])),
        }


def ingest_results_dir(results_dir: str, db: ResultsDatabase):
    """Ingest existing results from a directory into the database."""
    results_dir = Path(results_dir)
    for json_file in results_dir.glob("*_aggregated.json"):
        with open(json_file) as f:
            data = json.load(f)
        
        # Parse variant and benchmark from filename
        stem = json_file.stem.replace("_aggregated", "")
        # Heuristic: last part is variant, rest is benchmark
        parts = stem.split("_")
        if len(parts) >= 2:
            variant = parts[-1]
            benchmark = "_".join(parts[:-1])
        else:
            variant = "unknown"
            benchmark = stem
        
        # Add individual seed results
        for indiv in data.get("individual", []):
            seed = indiv.get("seed", 42)
            db.add_result(
                indiv.get("metrics", {}),
                variant, benchmark, seed,
                {},  # config not stored in aggregated
                data.get("elapsed_seconds", 0)
            )
    
    print(f"Ingested results from {results_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Results database CLI")
    parser.add_argument("--db-dir", default="./results_db")
    subparsers = parser.add_subparsers(dest="command")
    
    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest results from directory")
    p_ingest.add_argument("results_dir")
    
    # query
    p_query = subparsers.add_parser("query", help="Query experiments")
    p_query.add_argument("--variant")
    p_query.add_argument("--benchmark")
    p_query.add_argument("--seeds", nargs="+", type=int)
    p_query.add_argument("--metric")
    
    # aggregate
    p_agg = subparsers.add_parser("aggregate", help="Aggregate metrics")
    p_agg.add_argument("--variant")
    p_agg.add_argument("--benchmark")
    p_agg.add_argument("--metric", default="avg_final_accuracy")
    
    # compare
    p_cmp = subparsers.add_parser("compare", help="Compare variants statistically")
    p_cmp.add_argument("benchmark")
    p_cmp.add_argument("baseline")
    p_cmp.add_argument("variants", nargs="+")
    p_cmp.add_argument("--metric", default="avg_final_accuracy")
    
    # pareto
    p_pareto = subparsers.add_parser("pareto", help="Find Pareto frontier")
    p_pareto.add_argument("benchmark")
    p_pareto.add_argument("--metrics", nargs="+", default=["avg_final_accuracy", "avg_forgetting"])
    p_pareto.add_argument("--maximize", nargs="+", type=lambda x: x.lower()=="true", default=[True, False])
    
    # export
    p_exp = subparsers.add_parser("export", help="Export to CSV")
    p_exp.add_argument("output")
    p_exp.add_argument("--variant")
    p_exp.add_argument("--benchmark")
    
    # summary
    p_sum = subparsers.add_parser("summary", help="Database summary")
    
    args = parser.parse_args()
    db = ResultsDatabase(args.db_dir)
    
    if args.command == "ingest":
        ingest_results_dir(args.results_dir, db)
    elif args.command == "query":
        results = db.query(args.variant, args.benchmark, args.seeds, args.metric)
        print(json.dumps(results, indent=2, default=str))
    elif args.command == "aggregate":
        agg = db.aggregate(args.variant, args.benchmark, args.metric)
        print(json.dumps(agg, indent=2, default=str))
    elif args.command == "compare":
        cmp = db.compare_variants(args.benchmark, args.baseline, args.variants, args.metric)
        print(json.dumps(cmp, indent=2, default=str))
    elif args.command == "pareto":
        pareto = db.get_pareto_frontier(args.benchmark, args.metrics, args.maximize)
        print(json.dumps(pareto, indent=2, default=str))
    elif args.command == "export":
        db.export_csv(args.output, args.variant, args.benchmark)
    elif args.command == "summary":
        print(json.dumps(db.summary(), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Insight dataclass and automated report generator."""
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import json
import numpy as np
from pathlib import Path


class InsightType(Enum):
    STRONG_EFFECT = "strong_effect"
    PARAM_IMPORTANCE = "param_importance"
    PARETO_SHIFT = "pareto_shift"
    DOMAIN_TRANSFER = "domain_transfer"
    FAILURE_PATTERN = "failure_pattern"
    SCALING_TREND = "scaling_trend"
    INTERACTION_EFFECT = "interaction_effect"


@dataclass
class Insight:
    """A discovered insight from experiments."""
    type: InsightType
    claim: str
    evidence: Dict[str, Any]
    confidence: float = 0.0  # 0-1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    variant: Optional[str] = None
    benchmark: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['type'] = self.type.value
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Insight':
        d = d.copy()
        d['type'] = InsightType(d['type'])
        return cls(**d)


class InsightStore:
    """Persistent storage for insights."""
    
    def __init__(self, path: str = "./insights.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.insights: List[Insight] = self._load()
    
    def _load(self) -> List[Insight]:
        if not self.path.exists():
            return []
        insights = []
        with open(self.path) as f:
            for line in f:
                try:
                    insights.append(Insight.from_dict(json.loads(line)))
                except:
                    pass
        return insights
    
    def add(self, insight: Insight):
        self.insights.append(insight)
        with open(self.path, 'a') as f:
            f.write(json.dumps(insight.to_dict()) + '\n')
    
    def get_recent(self, n: int = 10) -> List[Insight]:
        return self.insights[-n:]
    
    def get_by_type(self, type: InsightType) -> List[Insight]:
        return [i for i in self.insights if i.type == type]


class ReportGenerator:
    """Generates automated reports from experiment results and insights."""
    
    def __init__(self, results_db, insight_store: InsightStore):
        self.db = results_db
        self.insights = insight_store
    
    def generate_discovery_report(self, output_path: str = "./discovery_report.md") -> str:
        """Generate markdown discovery report."""
        lines = [
            "# NGS Continuous Discovery Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"Total experiments: {len(self.db._index['experiments'])}",
            f"Variants tested: {len(self.db._index['variants'])}",
            f"Benchmarks: {len(self.db._index['benchmarks'])}",
            f"Insights discovered: {len(self.insights.insights)}",
            "",
        ]
        
        # Pareto frontiers
        lines.append("## Pareto Frontiers (Accuracy vs Forgetting)")
        for benchmark in sorted(self.db._index['benchmarks']):
            pareto = self.db.get_pareto_frontier(benchmark, 
                                                 ['avg_final_accuracy', 'avg_forgetting'],
                                                 [True, False])
            if pareto:
                lines.append(f"### {benchmark}")
                lines.append("| Variant | Accuracy | Forgetting |")
                lines.append("|---------|----------|------------|")
                for p in pareto:
                    v = p['variant']
                    acc = p['scores']['avg_final_accuracy']['mean']
                    forget = p['scores']['avg_forgetting']['mean']
                    lines.append(f"| {v} | {acc:.4f} | {forget:.4f} |")
                lines.append("")
        
        # Statistical comparisons
        lines.append("## Statistical Comparisons (vs Baseline)")
        for benchmark in sorted(self.db._index['benchmarks']):
            variants = [v for v in self.db._index['variants'] if v != 'baseline']
            if variants:
                cmp = self.db.compare_variants(benchmark, 'baseline', variants)
                lines.append(f"### {benchmark}")
                lines.append("| Variant | Δ Accuracy | p-value | Cohen's d | Significant |")
                lines.append("|---------|------------|---------|-----------|-------------|")
                for var, comp in cmp['comparisons'].items():
                    if 'error' not in comp:
                        lines.append(f"| {var} | {comp['mean_diff']:+.4f} | "
                                   f"{comp['t_pvalue']:.2e} | {comp['cohens_d']:.2f} | "
                                   f"{'✓' if comp['significant_005'] else '✗'} |")
                lines.append("")
        
        # Parameter importance
        lines.append("## Parameter Importance (from Bayesian Optimization)")
        # This would come from optimizer state
        lines.append("*Run Bayesian optimization to populate*")
        lines.append("")
        
        # Insights
        lines.append("## Discovered Insights")
        for insight in self.insights.insights:
            lines.append(f"### {insight.type.value.replace('_', ' ').title()}")
            lines.append(f"**Claim**: {insight.claim}")
            lines.append(f"**Confidence**: {insight.confidence:.0%}")
            if insight.variant:
                lines.append(f"**Variant**: {insight.variant}")
            if insight.benchmark:
                lines.append(f"**Benchmark**: {insight.benchmark}")
            lines.append(f"**Evidence**: `{json.dumps(insight.evidence, default=str)[:200]}...`")
            lines.append("")
        
        # Recommendations
        lines.append("## Recommendations for Next Experiments")
        lines.append(self._generate_recommendations())
        
        report = "\n".join(lines)
        Path(output_path).write_text(report)
        return report
    
    def _generate_recommendations(self) -> str:
        """Generate actionable recommendations."""
        recs = []
        
        # Check for unexplored variant-benchmark pairs
        all_pairs = [(v, b) for v in self.db._index['variants'] 
                     for b in self.db._index['benchmarks']]
        completed = set((e['variant'], e['benchmark']) for e in self.db._index['experiments'])
        missing = [p for p in all_pairs if p not in completed]
        
        if missing:
            recs.append(f"1. **Complete matrix**: {len(missing)} variant-benchmark pairs not yet evaluated")
            for v, b in missing[:5]:
                recs.append(f"   - {v} on {b}")
        
        # Check for low-seed experiments
        seed_counts = {}
        for e in self.db._index['experiments']:
            key = (e['variant'], e['benchmark'])
            seed_counts[key] = seed_counts.get(key, 0) + 1
        
        low_seed = [(k, v) for k, v in seed_counts.items() if v < 3]
        if low_seed:
            recs.append(f"2. **Increase seeds**: {len(low_seed)} pairs have <3 seeds")
            for (v, b), n in low_seed[:5]:
                recs.append(f"   - {v} on {b}: {n} seeds")
        
        # Best performing configs to explore further
        recs.append("3. **Deepen exploration**: Run Bayesian optimization on top-performing pairs")
        
        return "\n".join(recs) if recs else "All caught up!"


def extract_insights(results_db, optimizers: Dict = None) -> List[Insight]:
    """Automatically extract insights from completed experiments."""
    insights = []
    
    # 1. Strong effects (Cohen's d > 0.8)
    for benchmark in results_db._index['benchmarks']:
        variants = [v for v in results_db._index['variants'] if v != 'baseline']
        if not variants:
            continue
        cmp = results_db.compare_variants(benchmark, 'baseline', variants)
        for var, comp in cmp['comparisons'].items():
            if 'error' not in comp and comp.get('cohens_d', 0) > 0.8:
                insights.append(Insight(
                    type=InsightType.STRONG_EFFECT,
                    claim=f"{var} significantly outperforms baseline on {benchmark} (d={comp['cohens_d']:.2f})",
                    evidence=comp,
                    confidence=min(0.9, comp['cohens_d'] / 2),
                    variant=var,
                    benchmark=benchmark,
                    tags=['effect_size', 'significant']
                ))
    
    # 2. Pareto shifts
    for benchmark in results_db._index['benchmarks']:
        pareto = results_db.get_pareto_frontier(benchmark, 
                                                ['avg_final_accuracy', 'avg_forgetting'],
                                                [True, False])
        pareto_variants = {p['variant'] for p in pareto}
        if 'baseline' not in pareto_variants:
            insights.append(Insight(
                type=InsightType.PARETO_SHIFT,
                claim=f"Baseline is Pareto-dominated on {benchmark} by {pareto_variants}",
                evidence={"pareto": pareto},
                confidence=0.85,
                benchmark=benchmark,
                tags=['pareto', 'dominance']
            ))
    
    # 3. Parameter importance from optimizers
    if optimizers:
        for key, opt in optimizers.items():
            importance = opt.parameter_importance()
            if importance:
                top_params = importance[:3]
                insights.append(Insight(
                    type=InsightType.PARAM_IMPORTANCE,
                    claim=f"Top params for {key}: {', '.join(f'{p}({s:.2f})' for p,s in top_params)}",
                    evidence={"importance": importance},
                    confidence=0.7,
                    tags=['bayesian_opt', 'hyperparameters']
                ))
    
    # 4. Domain transfer patterns
    domain_benchmarks = ['permuted_mnist', 'rotated_mnist', 'blurry_mnist', 'noisy_mnist']
    for variant in results_db._index['variants']:
        domain_scores = []
        class_scores = []
        for bench in domain_benchmarks:
            agg = results_db.aggregate(variant=variant, benchmark=bench, metric='avg_final_accuracy')
            key = f"{variant}_{bench}"
            if key in agg:
                domain_scores.append(agg[key]['mean'])
        for bench in ['split_mnist', 'split_fashion']:
            agg = results_db.aggregate(variant=variant, benchmark=bench, metric='avg_final_accuracy')
            key = f"{variant}_{bench}"
            if key in agg:
                class_scores.append(agg[key]['mean'])
        
        if domain_scores and class_scores:
            domain_avg = np.mean(domain_scores)
            class_avg = np.mean(class_scores)
            if domain_avg > class_avg + 0.05:
                insights.append(Insight(
                    type=InsightType.DOMAIN_TRANSFER,
                    claim=f"{variant} excels on domain-incremental ({domain_avg:.3f}) vs class-incremental ({class_avg:.3f})",
                    evidence={"domain_avg": domain_avg, "class_avg": class_avg},
                    confidence=0.75,
                    variant=variant,
                    tags=['domain_incremental', 'transfer']
                ))
    
    return insights


if __name__ == "__main__":
    # Demo
    from scripts.results_db import ResultsDatabase
    db = ResultsDatabase('./test_db7')
    
    # Add some data
    import numpy as np
    np.random.seed(42)
    for var, base_acc in [('baseline', 0.5), ('factorized', 0.55), ('hyper', 0.62)]:
        for bench in ['split_mnist', 'permuted_mnist']:
            for seed in [42, 123, 456]:
                acc = base_acc + np.random.randn() * 0.02
                forget = 0.15 - base_acc * 0.1 + np.random.randn() * 0.01
                db.add_result(
                    {'avg_final_accuracy': acc, 'avg_forgetting': max(0, forget)},
                    var, bench, seed, {}, 50
                )
    
    # Extract insights
    insights = extract_insights(db)
    print(f"Extracted {len(insights)} insights")
    for i in insights:
        print(f"  {i.type.value}: {i.claim}")
    
    # Generate report
    from scripts.callbacks import InsightStore, ReportGenerator
    store = InsightStore('./test_insights.jsonl')
    for i in insights:
        store.add(i)
    
    rg = ReportGenerator(db, store)
    report = rg.generate_discovery_report('./test_report.md')
    print(f"Report generated: {len(report)} chars")

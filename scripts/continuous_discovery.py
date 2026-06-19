#!/usr/bin/env python
"""Continuous Discovery Orchestrator - integrates all components for automated research."""
import time
import threading
import signal
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from scripts.continuous_queue import PersistentQueue, ExperimentJob, JobStatus, create_job
from scripts.bayesian_optimizer import BayesianOptimizer, MetaLearner, VARIANT_SPACES
from scripts.results_db import ResultsDatabase
from scripts.insights import InsightStore, ReportGenerator, extract_insights, Insight, InsightType


@dataclass
class DiscoveryConfig:
    """Configuration for continuous discovery."""
    budget_hours: float = 24.0
    max_concurrent: int = 4
    variants: List[str] = field(default_factory=lambda: ["baseline", "factorized", "hyper", "cfg_net"])
    benchmarks: List[str] = field(default_factory=lambda: [
        "split_mnist", "permuted_mnist", "rotated_mnist", "blurry_mnist", "noisy_mnist",
        "split_fashion", "split_cifar10"
    ])
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    epochs_per_task: int = 5
    analysis_interval_hours: float = 1.0
    checkpoint_dir: str = "./checkpoints"
    queue_db: str = "experiment_queue.db"
    results_db_dir: str = "./results_db"
    insights_path: str = "./insights.jsonl"
    report_path: str = "./discovery_report.md"


class ContinuousDiscovery:
    """Main orchestrator for continuous automated experiment discovery."""
    
    def __init__(self, config: DiscoveryConfig = None):
        self.config = config or DiscoveryConfig()
        self.queue = PersistentQueue(self.config.queue_db)
        self.db = ResultsDatabase(self.config.results_db_dir)
        self.insight_store = InsightStore(self.config.insights_path)
        self.optimizers: Dict[str, BayesianOptimizer] = {}
        self.meta_learner = MetaLearner(self.db)
        self.running = False
        self.start_time = 0
        self.last_analysis = 0
        self.worker_id = f"worker_{threading.current_thread().ident}"
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.running = False
    
    def initialize(self):
        """Initialize optimizers and populate queue with initial experiments."""
        print("Initializing Continuous Discovery...")
        
        # Create optimizers for each variant-benchmark pair
        for variant in self.config.variants:
            if variant not in VARIANT_SPACES:
                continue
            for benchmark in self.config.benchmarks:
                key = f"{variant}_{benchmark}"
                space = VARIANT_SPACES[variant]
                opt = BayesianOptimizer(variant, benchmark, space)
                # Warm-start with meta-learned prior
                prior = self.meta_learner.get_meta_prior(variant, benchmark)
                opt.set_prior(prior)
                self.optimizers[key] = opt
        
        # Populate queue with initial random exploration if empty
        if self.queue.get_pending_count() == 0:
            self._populate_initial_queue()
        
        print(f"Initialized {len(self.optimizers)} optimizers")
        print(f"Queue pending: {self.queue.get_pending_count()}")
    
    def _populate_initial_queue(self):
        """Add initial random configurations to queue."""
        n_initial = 3  # per variant-benchmark
        jobs = []
        for variant in self.config.variants:
            if variant not in VARIANT_SPACES:
                continue
            space = VARIANT_SPACES[variant]
            for benchmark in self.config.benchmarks:
                for _ in range(n_initial):
                    config = space.sample(np.random.RandomState())
                    config.update({
                        'seeds': self.config.seeds,
                        'epochs_per_task': self.config.epochs_per_task,
                    })
                    job = create_job(variant, benchmark, config, priority=1.0)
                    jobs.append(job)
        
        self.queue.submit_batch(jobs)
        print(f"Added {len(jobs)} initial exploration jobs to queue")
    
    def run(self):
        """Main discovery loop."""
        self.running = True
        self.start_time = time.time()
        self.last_analysis = time.time()
        
        print(f"Starting continuous discovery (budget: {self.config.budget_hours}h)")
        print(f"Max concurrent: {self.config.max_concurrent}")
        
        while self.running:
            # Check budget
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours >= self.config.budget_hours:
                print(f"Budget exhausted ({elapsed_hours:.1f}h / {self.config.budget_hours}h)")
                break
            
            # Reset stale jobs
            self.queue.reset_stale_jobs(max_age_hours=2)
            
            # Launch jobs up to max_concurrent
            running_count = len(self.queue.get_jobs(status=JobStatus.RUNNING))
            while running_count < self.config.max_concurrent and self.running:
                job = self.queue.pop_next(self.worker_id)
                if not job:
                    break
                
                # Execute job in background thread
                thread = threading.Thread(target=self._execute_job, args=(job,))
                thread.daemon = True
                thread.start()
                running_count += 1
            
            # Wait a bit for jobs to complete
            time.sleep(10)
            
            # Periodic analysis
            if time.time() - self.last_analysis > self.config.analysis_interval_hours * 3600:
                self._analyze_and_update()
                self.last_analysis = time.time()
            
            # Generate report periodically
            if int(elapsed_hours * 60) % 30 == 0:  # Every 30 minutes
                self._generate_report()
        
        # Final analysis and report
        self._analyze_and_update()
        self._generate_report()
        print("Continuous discovery completed")
    
    def _execute_job(self, job: ExperimentJob):
        """Execute a single experiment job."""
        print(f"Executing job {job.id}: {job.variant} on {job.benchmark}")
        
        try:
            # Import experiment runner
            from scripts.run_experiment import run_from_yaml
            from scripts.experiment_config import load_experiment_config
            import yaml
            import tempfile
            
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                config_data = {
                    'experiment': job.benchmark.replace('_', '-').title(),
                    'dataset': job.benchmark,
                    'scenario': 'class_incremental' if 'split' in job.benchmark else 'domain_incremental',
                    'n_tasks': 5,
                    'classes_per_task': 2,
                    'input_dim': 784,
                    'output_dim': 10,
                    'model': job.config,
                    'training': job.config.get('training', {}),
                    'seeds': job.config.get('seeds', [42]),
                    'device': 'cuda',
                }
                yaml.dump(config_data, f)
                config_path = f.name
            
            # Run experiment
            result = run_from_yaml(
                config_path,
                seeds=job.config.get('seeds', [42]),
                output_dir=self.config.results_db_dir,
                device='cuda',
                epochs_override=job.config.get('epochs_per_task', 5),
            )
            
            # Record result in database
            for seed_result in result.get('individual', []):
                self.db.add_result(
                    seed_result.get('metrics', {}),
                    job.variant, job.benchmark, seed_result.get('seed', 42),
                    job.config, result.get('elapsed_seconds', 0)
                )
            
            # Update optimizer
            key = f"{job.variant}_{job.benchmark}"
            if key in self.optimizers:
                # Use mean metrics across seeds
                metrics = {}
                for m in ['avg_final_accuracy', 'avg_forgetting', 'bwt', 'la']:
                    vals = [r.get('metrics', {}).get(m, 0) for r in result.get('individual', [])]
                    if vals:
                        metrics[m] = np.mean(vals)
                
                if metrics:
                    self.optimizers[key].observe(job.config, metrics)
                    print(f"  Updated optimizer for {key}: {metrics}")
            
            self.queue.complete(job.id, result)
            print(f"Job {job.id} completed successfully")
            
        except Exception as e:
            print(f"Job {job.id} failed: {e}")
            self.queue.fail(job.id, str(e))
    
    def _analyze_and_update(self):
        """Analyze results, extract insights, update priorities."""
        print("\n=== Analysis Cycle ===")
        
        # Extract new insights
        new_insights = extract_insights(self.db)
        for insight in new_insights:
            # Check if already stored
            exists = any(i.claim == insight.claim for i in self.insight_store.insights)
            if not exists:
                self.insight_store.add(insight)
                print(f"  New insight: {insight.type.value} - {insight.claim[:80]}")
        
        # Re-prioritize queue based on optimizer uncertainty
        self._reprioritize_queue()
        
        # Prune unpromising directions
        self._prune_unpromising()
        
        # Update meta-learner with new data
        self.meta_learner = MetaLearner(self.db)
    
    def _reprioritize_queue(self):
        """Update job priorities based on expected information gain."""
        # Get pending jobs
        pending = self.queue.get_jobs(status=JobStatus.PENDING, limit=100)
        
        for job in pending:
            key = f"{job.variant}_{job.benchmark}"
            if key in self.optimizers:
                opt = self.optimizers[key]
                # Higher priority for configurations with high uncertainty
                # (simplified: just boost recent suggestions)
                importance = opt.parameter_importance()
                if importance:
                    job.priority *= 1.5
                    self.queue.submit(job)  # Re-submit with new priority
    
    def _prune_unpromising(self):
        """Remove jobs for variants that consistently underperform."""
        # Compare each variant against baseline on each benchmark
        for benchmark in self.config.benchmarks:
            variants = [v for v in self.config.variants if v != 'baseline']
            if not variants:
                continue
            
            cmp = self.db.compare_variants(benchmark, 'baseline', variants)
            for var, comp in cmp['comparisons'].items():
                if 'error' not in comp and comp.get('cohens_d', 0) < -0.5:
                    # This variant is significantly worse - deprioritize
                    pending = self.queue.get_jobs(
                        status=JobStatus.PENDING, variant=var, benchmark=benchmark
                    )
                    for job in pending:
                        job.priority *= 0.1  # Heavy deprioritization
                        self.queue.submit(job)
    
    def _generate_report(self):
        """Generate discovery report."""
        rg = ReportGenerator(self.db, self.insight_store)
        report = rg.generate_discovery_report(self.config.report_path)
        print(f"Report updated: {self.config.report_path}")
    
    def status(self) -> Dict:
        """Get current status."""
        return {
            'running': self.running,
            'elapsed_hours': (time.time() - self.start_time) / 3600,
            'budget_hours': self.config.budget_hours,
            'queue_stats': self.queue.get_stats(),
            'db_summary': self.db.summary(),
            'optimizers': len(self.optimizers),
            'insights': len(self.insight_store.insights),
        }


def run_continuous_discovery(
    budget_hours: float = 24.0,
    max_concurrent: int = 4,
    variants: List[str] = None,
    benchmarks: List[str] = None,
    **kwargs
):
    """Convenience function to run continuous discovery."""
    config = DiscoveryConfig(
        budget_hours=budget_hours,
        max_concurrent=max_concurrent,
        variants=variants or ["baseline", "factorized", "hyper", "cfg_net"],
        benchmarks=benchmarks or ["split_mnist", "permuted_mnist", "rotated_mnist"],
        **kwargs
    )
    
    discovery = ContinuousDiscovery(config)
    discovery.initialize()
    discovery.run()
    
    return discovery


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Continuous Discovery")
    parser.add_argument("--budget", type=float, default=1.0, help="Budget in hours")
    parser.add_argument("--concurrent", type=int, default=2, help="Max concurrent jobs")
    parser.add_argument("--quick", action="store_true", help="Quick test (10 min)")
    args = parser.parse_args()
    
    if args.quick:
        budget = 1/6  # 10 minutes
        concurrent = 1
    else:
        budget = args.budget
        concurrent = args.concurrent
    
    discovery = run_continuous_discovery(
        budget_hours=budget,
        max_concurrent=concurrent,
    )

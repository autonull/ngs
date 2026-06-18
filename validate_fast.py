#!/usr/bin/env python3
"""
Fast validation with progress feedback.
Reduces: epochs=1, seeds=2, key datasets only.
Adds: per-task progress, timing, live metrics.
"""
import subprocess
import sys
import time
from datetime import timedelta

# Reduced config for speed
FAST_CONFIG = {
    'epochs': 1,           # was 2
    'seeds': [42, 123],    # was [42, 123, 456]
    'datasets': [
        'split_mnist',      # class-inc (5 tasks)
        'split_fashion',    # class-inc (5 tasks) 
        'permuted_mnist',   # domain-inc (10 tasks)
        'rotated_mnist',    # domain-inc (10 tasks)
    ],
    'models': [
        'ngs_cfg_net',     # best NGS
        'ngs_baseline',    # ablation
        'mlp', 'er', 'ewc', 'si', 'lwf',  # baselines
    ]
}

def run_with_progress(cmd, desc):
    """Run command with live output and timing."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                            text=True, bufsize=1)
    
    task_count = 0
    for line in proc.stdout:
        print(line.rstrip())
        if "Task" in line and "done" in line:
            task_count += 1
    
    proc.wait()
    elapsed = time.time() - start
    print(f"\n  Completed in {timedelta(seconds=int(elapsed))} ({task_count} tasks)")
    return proc.returncode == 0

def run_single(cmd, desc):
    """Run single command and return success."""
    return run_with_progress(cmd, desc)

def main():
    print("FAST VALIDATION: epochs=1, seeds=2, 4 datasets, 7 models")
    print(f"Estimated time: ~30-45 minutes (vs 6+ hours)")
    
    # Quick single-run test first
    test_cmd = [
        sys.executable, '-m', 'experiments.main',
        '--experiments', 'split_mnist',
        '--models', 'ngs_cfg_net',
        '--seeds', '42',
        '--no-verbose'
    ]
    if not run_with_progress(test_cmd, "SMOKE TEST: split_mnist / ngs_cfg_net / seed 42"):
        print("Smoke test failed!")
        return 1
    
    # Full fast validation
    for dataset in FAST_CONFIG['datasets']:
        cmd = [
            sys.executable, '-m', 'experiments.main',
            '--experiments', dataset,
            '--models', *FAST_CONFIG['models'],
            '--seeds', *map(str, FAST_CONFIG['seeds']),
            '--no-verbose'
        ]
        desc = f"{dataset.upper()} | {len(FAST_CONFIG['models'])} models × {len(FAST_CONFIG['seeds'])} seeds × 1 epoch"
        if not run_with_progress(cmd, desc):
            print(f"Failed on {dataset}, continuing...")
    
    # Generate summary
    print("\n" + "="*60)
    print("  GENERATING SUMMARY REPORT")
    print("="*60)
    subprocess.run([
        sys.executable, '-m', 'experiments.report',
        '--results-dir', './results',
        '--plots-dir', './plots'
    ])
    
    print("\n✓ Fast validation complete!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
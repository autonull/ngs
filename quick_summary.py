#!/usr/bin/env python3
"""Live summary of results as they're generated."""
import json
import os
import glob
from datetime import datetime

def summarize_results():
    results_dir = './results'
    files = glob.glob(os.path.join(results_dir, '*_seed*.json'))
    
    # Group by model
    by_model = {}
    for f in files:
        if 'summary.json' in f:
            continue
        try:
            with open(f) as fp:
                data = json.load(fp)
            if 'error' in data or 'metrics' not in data:
                continue
            model = data['model']
            exp = data['config']
            seed = data['seed']
            acc = data['metrics']['avg_final_accuracy']
            forget = data['metrics']['avg_forgetting']
            units = data['metrics']['active_units']
            key = f"{exp}_{model}"
            if key not in by_model:
                by_model[key] = []
            by_model[key].append((seed, acc, forget, units))
        except:
            pass
    
    # Print summary table
    print(f"\n{'='*100}")
    print(f"LIVE SUMMARY  {datetime.now().strftime('%H:%M:%S')}  ({len(files)} result files)")
    print(f"{'='*100}")
    print(f"{'Experiment_Model':<35} {'Seeds':<10} {'Avg Acc':<10} {'Avg Forget':<12} {'Units':<8} {'Per-seed Acc'}")
    print(f"{'-'*100}")
    
    for key, vals in sorted(by_model.items()):
        seeds = [v[0] for v in vals]
        accs = [v[1] for v in vals]
        forgets = [v[2] for v in vals]
        units = [v[3] for v in vals]
        per_seed = ' '.join(f"{s}:{a:.3f}" for s,a in zip(seeds, accs))
        print(f"{key:<35} {len(seeds):<10} {sum(accs)/len(accs):.4f}     {sum(forgets)/len(forgets):.4f}       {units[-1]:<8} {per_seed}")
    
    print(f"{'='*100}")

if __name__ == '__main__':
    summarize_results()
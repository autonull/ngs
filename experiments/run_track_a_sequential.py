#!/usr/bin/env python
"""
Sequential runner for Track A1-A6 - for debugging/verification.
Runs all tracks one by one in the same process.
"""
import sys
import os
import time

# Add project root to path
sys.path.insert(0, '/home/me/ngs')

def run_a1():
    print("\n" + "="*60)
    print("RUNNING TRACK A1: Progressive top_k")
    print("="*60)
    os.system("python experiments/track_a1_progressive_topk.py --epochs 5 --seed 42")

def run_a2():
    print("\n" + "="*60)
    print("RUNNING TRACK A2: Growing K")
    print("="*60)
    os.system("python experiments/track_a2_growing_k.py --epochs 5 --seed 42")

def run_a3():
    print("\n" + "="*60)
    print("RUNNING TRACK A3: Dense Residual")
    print("="*60)
    os.system("python experiments/track_a3_dense_residual.py --epochs 5 --seed 42")

def run_a4():
    print("\n" + "="*60)
    print("RUNNING TRACK A4: Shared Router")
    print("="*60)
    os.system("python experiments/track_a4_shared_router.py --epochs 5 --seed 42")

def run_a5():
    print("\n" + "="*60)
    print("RUNNING TRACK A5: MLP Projections")
    print("="*60)
    os.system("python experiments/track_a5_mlp_projections.py --epochs 5 --seed 42")

def run_a6():
    print("\n" + "="*60)
    print("RUNNING TRACK A6: Soft Routing")
    print("="*60)
    os.system("python ngs/experiments/track_a6_soft_routing.py --epochs 5 --seed 42")


def main():
    start = time.time()
    
    # Run all tracks
    run_a1()
    run_a2()
    run_a3()
    run_a4()
    run_a5()
    run_a6()
    
    total = time.time() - start
    print(f"\n{'='*60}")
    print(f"ALL TRACKS COMPLETE in {total:.1f}s")
    print(f"{'='*60}")
    
    # Check Gate A
    print("\n" + "="*60)
    print("GATE A CHECK")
    print("="*60)
    
    import json
    from pathlib import Path
    
    any_passed = False
    for track in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']:
        result_file = Path(f'results/track_a/{track.lower()}_results.json')
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            
            for exp in data.get('results', []):
                if 'test_accuracy' in exp and exp['test_accuracy'] >= 0.93:
                    print(f"  ✓ {track}: {exp.get('config_name', 'unknown')} = {exp['test_accuracy']:.4f} >= 93%")
                    any_passed = True
    
    if any_passed:
        print("\n  GATE A: PASSED - Proceed to Track A7")
    else:
        print("\n  GATE A: FAILED - Pivot to single-layer NGS")


if __name__ == "__main__":
    main()
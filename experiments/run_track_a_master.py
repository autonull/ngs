#!/usr/bin/env python
"""
Master Runner for Track A1-A6 Parallel Execution
Runs all 6 experiment tracks in parallel using subprocess.
"""
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


TRACK_SCRIPTS = {
    'A1': 'experiments/track_a1_progressive_topk.py',
    'A2': 'experiments/track_a2_growing_k.py',
    'A3': 'experiments/track_a3_dense_residual.py',
    'A4': 'experiments/track_a4_shared_router.py',
    'A5': 'experiments/track_a5_mlp_projections.py',
    'A6': 'ngs/experiments/track_a6_soft_routing.py',
}


def run_track(track_name, script_path, args):
    """Run a single track experiment."""
    cmd = [
        sys.executable, script_path,
        '--seed', str(args.seed),
        '--epochs', str(args.epochs),
        '--device', args.device,
        '--output', f'results/track_a/{track_name.lower()}_results.json',
    ]
    
    print(f"[{track_name}] Starting: {' '.join(cmd)}")
    start = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"[{track_name}] Completed in {elapsed:.1f}s")
            return {'track': track_name, 'status': 'success', 'time': elapsed, 'stdout': result.stdout[-2000:]}
        else:
            print(f"[{track_name}] FAILED after {elapsed:.1f}s")
            print(f"  stderr: {result.stderr[-1000:]}")
            return {'track': track_name, 'status': 'failed', 'time': elapsed, 'error': result.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"[{track_name}] TIMEOUT after {elapsed:.1f}s")
        return {'track': track_name, 'status': 'timeout', 'time': elapsed}
    except Exception as e:
        elapsed = time.time() - start
        print(f"[{track_name}] ERROR: {e}")
        return {'track': track_name, 'status': 'error', 'time': elapsed, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Master runner for Track A1-A6")
    parser.add_argument("--tracks", nargs='+', default=['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
                        help="Tracks to run (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--parallel", type=int, default=3, help="Number of parallel processes")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per track (seconds)")
    parser.add_argument("--output", default="results/track_a/master_results.json")
    args = parser.parse_args()
    
    # Validate tracks
    valid_tracks = [t for t in args.tracks if t in TRACK_SCRIPTS]
    if len(valid_tracks) != len(args.tracks):
        invalid = set(args.tracks) - set(TRACK_SCRIPTS.keys())
        print(f"WARNING: Invalid tracks ignored: {invalid}")
    
    print(f"Running {len(valid_tracks)} tracks in parallel (max {args.parallel} workers)")
    print(f"Tracks: {valid_tracks}")
    print(f"Seed: {args.seed}, Epochs: {args.epochs}, Device: {args.device}")
    
    # Ensure output directory exists
    Path('results/track_a').mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    results = []
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(run_track, track, TRACK_SCRIPTS[track], args): track 
            for track in valid_tracks
        }
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    total_time = time.time() - start_time
    
    # Save master results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'tracks_run': valid_tracks,
            'seed': args.seed,
            'epochs': args.epochs,
            'device': args.device,
            'total_time_seconds': total_time,
            'results': results,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"MASTER RUNNER COMPLETE in {total_time:.1f}s")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    for r in results:
        status_icon = {'success': '✓', 'failed': '✗', 'timeout': '⏱', 'error': '✗'}.get(r['status'], '?')
        print(f"  {status_icon} {r['track']}: {r['status']} ({r['time']:.1f}s)")
    
    # Check Gate A criteria
    print("\n" + "="*60)
    print("GATE A CHECK")
    print("="*60)
    
    # Load individual track results and check for >= 93% on MNIST
    any_passed = False
    for track in valid_tracks:
        result_file = Path(f'results/track_a/{track.lower()}_results.json')
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            
            for exp in data.get('results', []):
                if 'test_accuracy' in exp and exp['test_accuracy'] >= 0.93:
                    print(f"  ✓ {track}: {exp['config_name']} = {exp['test_accuracy']:.4f} >= 93%")
                    any_passed = True
    
    if any_passed:
        print("\n  GATE A: PASSED - At least one config achieved >= 93%")
        print("  → Proceed to Track A7 (Combined Best)")
    else:
        print("\n  GATE A: FAILED - No config achieved >= 93%")
        print("  → Pivot: NGS is fundamentally shallow; focus on single-layer apps")
    
    return 0 if any_passed else 1


if __name__ == "__main__":
    sys.exit(main())
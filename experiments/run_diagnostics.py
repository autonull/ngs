#!/usr/bin/env python
"""
Diagnostic Pipeline Orchestrator — TODO11 Phase A0

Runs all diagnostic experiments systematically, aggregates results,
and produces a unified report.

Usage:
    python experiments/run_diagnostics.py
    python experiments/run_diagnostics.py --diagnostics spectral_norm ep_vs_bp_updates
    python experiments/run_diagnostics.py --stop-on-showstopper
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch

# Add project root to path
sys.path.insert(0, '/home/me/ngs')

DIAGNOSTICS = [
    {"name": "spectral_norm", "script": "diagnose_spectral_norm.py", "priority": 1},
    {"name": "ep_vs_bp_updates", "script": "diagnose_ep_vs_bp_updates.py", "priority": 1},
    {"name": "energy_landscape", "script": "diagnose_energy_landscape.py", "priority": 1},
    {"name": "bioplausible_baseline", "script": None, "priority": 2,
     "command": "python bioplausible/mep/examples/mnist_comparison.py"},
    {"name": "ngs_vs_dense", "script": "compare_ngs_vs_dense.py", "priority": 3},
    {"name": "3dgs_hardness", "script": "diagnose_3dgs_hardness.py", "priority": 4},
    {"name": "entropy_distribution", "script": "diagnose_entropy_distribution.py", "priority": 4},
    {"name": "gaussian_overlap", "script": "diagnose_gaussian_overlap.py", "priority": 4},
]


def run_diagnostic(
    diag: Dict[str, Any],
    results_dir: Path,
    seed: int = 42,
    device: str = "auto",
    stop_on_showstopper: bool = False,
    showstopper_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Run a single diagnostic and return its results."""
    name = diag["name"]
    script = diag["script"]
    command = diag.get("command")
    
    print(f"\n{'='*60}")
    print(f"Running diagnostic: {name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = {
        "name": name,
        "script": script,
        "command": command,
        "success": False,
        "duration": 0.0,
        "output_file": None,
        "error": None,
        "showstopper": False,
    }
    
    # Set up environment
    env = {
        **dict(subprocess.os.environ),
        "PYTHONPATH": "/home/me/ngs:/home/me/ngs/bioplausible/mep:" + subprocess.os.environ.get("PYTHONPATH", ""),
        "CUDA_VISIBLE_DEVICES": "0" if device == "cuda" else "",
    }
    
    if device == "auto":
        env["CUDA_VISIBLE_DEVICES"] = "0" if torch.cuda.is_available() else ""
    
    # Determine command to run
    if command:
        cmd = command.split()
    elif script:
        cmd = [sys.executable, f"experiments/{script}"]
    else:
        result["error"] = "No script or command specified"
        result["duration"] = time.time() - start_time
        return result
    
    # Add seed if script supports it
    if script and script != "bioplausible_baseline":
        cmd.extend(["--seed", str(seed)])
    
    try:
        proc = subprocess.run(
            cmd,
            cwd="/home/me/ngs",
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per diagnostic
        )
        result["duration"] = time.time() - start_time
        
        if proc.returncode == 0:
            result["success"] = True
            print(f"  ✓ Completed in {result['duration']:.1f}s")
            
            # Try to read output file
            output_file = results_dir / f"{name}.json"
            if output_file.exists():
                with open(output_file) as f:
                    result["output"] = json.load(f)
                result["output_file"] = str(output_file)
                
                # Check for showstopper
                if stop_on_showstopper:
                    cos_sim = result["output"].get("cosine_sim_ep_vs_bp")
                    if cos_sim is not None and cos_sim < showstopper_threshold:
                        result["showstopper"] = True
                        print(f"  ⚠ SHOWSTOPPER: cosine_sim={cos_sim:.4f} < {showstopper_threshold}")
            else:
                # Try to parse from stdout
                try:
                    result["output"] = json.loads(proc.stdout.strip().split('\n')[-1])
                except:
                    result["output"] = {"stdout": proc.stdout[-500:]}
        else:
            result["error"] = proc.stderr[-1000:]
            print(f"  ✗ Failed: {result['error'][:200]}")
            
    except subprocess.TimeoutExpired:
        result["duration"] = time.time() - start_time
        result["error"] = "Timeout (600s)"
        print(f"  ✗ Timeout after {result['duration']:.1f}s")
    except Exception as e:
        result["duration"] = time.time() - start_time
        result["error"] = str(e)
        print(f"  ✗ Exception: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Run diagnostic pipeline")
    parser.add_argument(
        "--diagnostics",
        nargs="+",
        default=None,
        help="Specific diagnostics to run (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use"
    )
    parser.add_argument(
        "--stop-on-showstopper",
        action="store_true",
        help="Stop early if a showstopper result is detected"
    )
    parser.add_argument(
        "--showstopper-threshold",
        type=float,
        default=0.1,
        help="Cosine similarity threshold for showstopper"
    )
    parser.add_argument(
        "--results-dir",
        default="results/diagnostics",
        help="Directory for output files"
    )
    args = parser.parse_args()
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter diagnostics
    to_run = DIAGNOSTICS
    if args.diagnostics:
        to_run = [d for d in DIAGNOSTICS if d["name"] in args.diagnostics]
        if len(to_run) != len(args.diagnostics):
            missing = set(args.diagnostics) - {d["name"] for d in to_run}
            print(f"Warning: Unknown diagnostics: {missing}")
    
    # Sort by priority
    to_run.sort(key=lambda x: x["priority"])
    
    print(f"Running {len(to_run)} diagnostics...")
    print(f"Seed: {args.seed}, Device: {args.device}")
    print(f"Results dir: {results_dir}")
    
    # Run diagnostics
    all_results = []
    for diag in to_run:
        result = run_diagnostic(
            diag, results_dir, 
            seed=args.seed,
            device=args.device,
            stop_on_showstopper=args.stop_on_showstopper,
            showstopper_threshold=args.showstopper_threshold,
        )
        all_results.append(result)
        
        # Check for showstopper
        if args.stop_on_showstopper and result.get("showstopper"):
            print(f"\n⚠ Showstopper detected in {result['name']}. Stopping early.")
            break
    
    # Aggregate results
    report = {
        "timestamp": time.time(),
        "seed": args.seed,
        "device": args.device,
        "diagnostics": all_results,
        "summary": {
            "total": len(all_results),
            "passed": sum(1 for r in all_results if r["success"]),
            "failed": sum(1 for r in all_results if not r["success"]),
            "showstoppers": sum(1 for r in all_results if r.get("showstopper")),
        }
    }
    
    # Save unified report
    report_file = results_dir / "diagnostics_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        status = "✓" if r["success"] else "✗"
        showstopper = " ⚠ SHOWSTOPPER" if r.get("showstopper") else ""
        print(f"  {status} {r['name']:30s} ({r['duration']:.1f}s){showstopper}")
    print(f"\nTotal: {report['summary']['total']}, Passed: {report['summary']['passed']}, Failed: {report['summary']['failed']}")
    print(f"Report saved to: {report_file}")
    
    # Return exit code based on success
    if report["summary"]["failed"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
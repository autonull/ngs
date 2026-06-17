#!/usr/bin/env python
"""
Train NGS for 2D density estimation demo.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

from ngs.benchmarks.density import run_density_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run NGS density estimation demo")
    parser.add_argument("--dataset", default="moons", 
                        choices=["moons", "circles", "pinwheel", "swissroll", "checkerboard"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./density_results")
    args = parser.parse_args()
    
    results = run_density_benchmark(
        dataset=args.dataset,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    print(f"\nFinal NLL: {results['final_nll']:.4f}")
    print(f"Final K: {results['final_k']}")


if __name__ == "__main__":
    main()
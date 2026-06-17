#!/usr/bin/env python
"""
Train NGS for Few-Shot learning on Omniglot.
"""

import argparse
from ngs.benchmarks import run_fewshot_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run NGS few-shot benchmark")
    parser.add_argument("--dataset", default="omniglot", choices=["omniglot", "miniimagenet"])
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./fewshot_results")
    args = parser.parse_args()
    
    results = run_fewshot_benchmark(
        dataset=args.dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    print(f"\nTest Accuracy: {results['test_acc']:.4f}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Extended benchmark runner - avoids module import warnings."""

import sys
sys.path.insert(0, '/home/me/ngs')

from ngs.benchmarks.extended import run_extended_benchmark

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run extended NGS benchmarks")
    parser.add_argument("--domain", default="vision", choices=["vision", "nlp", "robotics"])
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./extended_results")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--k-init", type=int, default=64)
    parser.add_argument("--max-k", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    run_extended_benchmark(
        domain=args.domain,
        dataset=args.dataset,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        latent_dim=args.latent_dim,
        k_init=args.k_init,
        max_k=args.max_k,
        top_k=args.top_k,
        lr=args.lr,
        batch_size=args.batch_size,
    )
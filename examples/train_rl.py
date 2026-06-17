#!/usr/bin/env python
"""
Train NGS for RL on CartPole with domain randomization.
"""

import argparse
from ngs.benchmarks.rl import run_rl_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run NGS RL benchmark")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--domain-shift", default="none", 
                        choices=["none", "gravity", "mass", "length", "noise"])
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./rl_results")
    args = parser.parse_args()
    
    results = run_rl_benchmark(
        env_name=args.env,
        domain_shift=args.domain_shift,
        total_timesteps=args.timesteps,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    print(f"\nFinal Eval Reward: {results['final_eval_reward']:.2f}")


if __name__ == "__main__":
    main()
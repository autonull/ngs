"""Benchmark suite for NGS."""

from .density import run_density_benchmark
from .fewshot import run_fewshot_benchmark
from .rl import run_rl_benchmark
from .generative import run_generative_benchmark
from .metalearn import run_metalearn_benchmark
from .continual_rl import run_continual_rl_benchmark
from .bandit import run_bandit_benchmark
from .federated import run_federated_benchmark

__all__ = [
    "run_density_benchmark",
    "run_fewshot_benchmark",
    "run_rl_benchmark",
    "run_generative_benchmark",
    "run_metalearn_benchmark",
    "run_continual_rl_benchmark",
    "run_bandit_benchmark",
    "run_federated_benchmark",
]

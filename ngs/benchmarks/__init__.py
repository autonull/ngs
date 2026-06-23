"""Benchmark suite for NGS."""

from .density import run_density_benchmark
from .fewshot import run_fewshot_benchmark
from .rl import run_rl_benchmark
from .generative import run_generative_benchmark
from .metalearn import run_metalearn_benchmark
from .continual_rl import run_continual_rl_benchmark
from .bandit import run_bandit_benchmark
from .federated import run_federated_benchmark
from .flow_matching import run_flow_matching_benchmark
from .rapid_adaptation import run_rapid_adaptation_benchmark
from .online_cl import run_online_cl_benchmark
from .class_incremental import run_class_incremental_benchmark
from .gossip import run_gossip_benchmark
from .comparison import run_comparison_benchmark
from .extended import run_vision_benchmark, run_nlp_benchmark, run_robotics_benchmark, run_extended_benchmark
from .multimodal import run_multimodal_benchmark
from .metagaussian import run_metagaussian_benchmark

__all__ = [
    "run_density_benchmark",
    "run_fewshot_benchmark",
    "run_rl_benchmark",
    "run_generative_benchmark",
    "run_metalearn_benchmark",
    "run_continual_rl_benchmark",
    "run_bandit_benchmark",
    "run_federated_benchmark",
    "run_flow_matching_benchmark",
    "run_rapid_adaptation_benchmark",
    "run_online_cl_benchmark",
    "run_class_incremental_benchmark",
    "run_gossip_benchmark",
    "run_comparison_benchmark",
    "run_vision_benchmark",
    "run_nlp_benchmark",
    "run_robotics_benchmark",
    "run_extended_benchmark",
    "run_multimodal_benchmark",
    "run_metagaussian_benchmark",
]

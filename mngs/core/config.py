"""Configuration schema for the Modular Neural Gaussian System."""
from dataclasses import dataclass
from enum import Enum


class RoutingStrategy(Enum):
    """Routing strategy for selecting active units."""
    MONOLITHIC_MAHALANOBIS = "monolithic_mahalanobis"
    FACTORIZED_SUBSPACE = "factorized_subspace"
    LSH_APPROXIMATE = "lsh_approximate"


class ParameterStorage(Enum):
    """Strategy for storing unit parameters."""
    DIRECT_ADAPTER = "direct_adapter"
    HYPERNETWORK_GENERATED = "hypernetwork_generated"


class TopologyControl(Enum):
    """Strategy for dynamic topology adaptation."""
    DISCRETE_HEURISTIC = "discrete_heuristic"
    CONTINUOUS_DENSITY = "continuous_density"


class MemoryManagement(Enum):
    """Strategy for managing dynamic unit memory."""
    DYNAMIC_GROWTH = "dynamic_growth"
    PRE_ALLOCATED_MASKED = "pre_allocated_masked"
    STRICT_CAPACITY = "strict_capacity"


@dataclass
class MNGSConfig:
    """Configuration for a Modular Neural Gaussian System instance."""
    # Core Dimensions
    latent_dim: int = 32
    output_dim: int = 64
    k_init: int = 128
    max_k: int = 512
    top_k: int = 8
    
    # Modular Choices
    routing: RoutingStrategy = RoutingStrategy.MONOLITHIC_MAHALANOBIS
    parameter_storage: ParameterStorage = ParameterStorage.DIRECT_ADAPTER
    topology_control: TopologyControl = TopologyControl.DISCRETE_HEURISTIC
    memory_management: MemoryManagement = MemoryManagement.PRE_ALLOCATED_MASKED
    
    # Strategy-Specific Hyperparameters
    top_k_factorized: int = 2
    num_subspaces: int = 4
    hypernetwork_hidden_dim: int = 16
    hypernetwork_code_dim: int = 8
    split_threshold: float = 0.05
    prune_threshold: float = 0.01
    
    # Training hyperparameters
    lora_rank: int = 4
    tau: float = 1.0
    gamma_residual: float = 0.1
    ema_decay: float = 0.99
    diversity_weight: float = 0.01
    entropy_weight: float = 0.01

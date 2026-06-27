"""Core configuration and interfaces for NGS."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import torch


class RoutingStrategy(Enum):
    """Routing strategy for selecting active units."""
    MONOLITHIC_MAHALANOBIS = "monolithic_mahalanobis"
    FACTORIZED_SUBSPACE = "factorized_subspace"
    LSH_APPROXIMATE = "lsh_approximate"
    HIERARCHICAL = "hierarchical"
    GAUSSIAN_ATTENTION = "gaussian_attention"
    UNCERTAINTY_AWARE = "uncertainty_aware"


class ParameterStorage(Enum):
    """Strategy for storing unit parameters."""
    DIRECT_ADAPTER = "direct_adapter"
    HYPERNETWORK_GENERATED = "hypernetwork_generated"
    LORA = "lora"


class TopologyControl(Enum):
    """Strategy for dynamic topology adaptation."""
    DISCRETE_HEURISTIC = "discrete_heuristic"
    CONTINUOUS_DENSITY = "continuous_density"
    MERGE_AWARE = "merge_aware"
    META_LEARNED = "meta_learned"
    AUTOPOIETIC = "autopoietic"


class MemoryManagement(Enum):
    """Strategy for managing dynamic unit memory."""
    PRE_ALLOCATED = "pre_allocated"
    DYNAMIC = "dynamic"
    STRICT_CAPACITY = "strict_capacity"


@dataclass
class NGSConfig:
    """Configuration for a Neural Gaussian System instance."""
    # Core Dimensions
    latent_dim: int = 32
    k_init: int = 128
    max_k: int = 512
    top_k: int = 8
    
    # Modular Choices
    routing: RoutingStrategy = RoutingStrategy.MONOLITHIC_MAHALANOBIS
    parameter_storage: ParameterStorage = ParameterStorage.DIRECT_ADAPTER
    topology_control: TopologyControl = TopologyControl.DISCRETE_HEURISTIC
    memory_management: MemoryManagement = MemoryManagement.PRE_ALLOCATED
    
    # Strategy-Specific Hyperparameters
    top_k_factorized: int = 2
    num_subspaces: int = 4
    hypernetwork_hidden_dim: int = 16
    hypernetwork_code_dim: int = 8
    split_threshold: float = 0.05
    prune_threshold: float = 0.01
    
    # TODO12 Tracks
    use_mlp_projections: bool = False  # Track A5
    mlp_hidden_multiplier: int = 4
    soft_routing: bool = False  # Track A6 (ablate top-k)
    gamma_residual: float = 0.1  # Track A3
    beta_residual: float = 0.1   # Track A3

    # Hierarchical routing
    num_levels: int = 3
    level_capacity_ratio: float = 0.5
    level_top_k: int = 4
    
    # Gaussian Attention
    attention_heads: int = 4
    attention_dropout: float = 0.1
    sparse_top_k: int = 64
    
    # Uncertainty-aware routing
    evidential_prior: float = 1.0
    uncertainty_weight: float = 0.1
    
    # Merge-aware topology
    merge_threshold: float = 0.95
    merge_check_interval: int = 100
    
    # Meta-learned topology
    meta_lr: float = 1e-3
    meta_hidden_dim: int = 64
    
    # Training hyperparameters
    lora_rank: int = 4
    use_lora: bool = True
    tau: float = 1.0
    ema_decay: float = 0.99
    diversity_weight: float = 0.01
    entropy_weight: float = 0.01
    
    # Additional
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.routing, str):
            self.routing = RoutingStrategy(self.routing)
        if isinstance(self.parameter_storage, str):
            self.parameter_storage = ParameterStorage(self.parameter_storage)
        if isinstance(self.topology_control, str):
            self.topology_control = TopologyControl(self.topology_control)
        if isinstance(self.memory_management, str):
            self.memory_management = MemoryManagement(self.memory_management)


@dataclass
class RoutingOutput:
    """Standardized routing output across all strategies."""
    indices: torch.Tensor
    weights: torch.Tensor
    logits: Optional[torch.Tensor] = None
    uncertainty: Optional[torch.Tensor] = None
    level_indices: Optional[List[torch.Tensor]] = None
    level_weights: Optional[List[torch.Tensor]] = None


@dataclass
class TopologyAction:
    """Explicit topology change record."""
    action_type: str
    indices: torch.Tensor
    metadata: Dict[str, Any]
    timestamp: int


class BaseRouter(torch.nn.Module):
    """Base class for all routing strategies."""
    
    def __init__(self, config: NGSConfig):
        super().__init__()
        self.config = config
        self.max_k = config.max_k
        self.d_latent = config.latent_dim
    
    @property
    def K(self) -> int:
        """Number of currently active units."""
        raise NotImplementedError
    
    @property
    def max_units(self) -> int:
        """Maximum capacity."""
        return self.max_k
    
    def forward(self, z: torch.Tensor) -> RoutingOutput:
        """Route inputs to active units."""
        raise NotImplementedError
    
    def initialize_units(self, k_init: int):
        """Initialize the first k_init units as active."""
        raise NotImplementedError


class BaseParameterStore(torch.nn.Module):
    """Base class for all parameter storage strategies."""
    
    def __init__(self, config: NGSConfig):
        super().__init__()
        self.config = config
        self.max_k = config.max_k
        self.d_latent = config.latent_dim
    
    def forward(self, active_indices: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Generate or retrieve transformations for active units."""
        raise NotImplementedError
    
    def get_parameters_for_indices(self, indices: torch.Tensor):
        """Get transformation parameters for specific units."""
        raise NotImplementedError
    
    def expand_capacity(self, new_max_k: int):
        """Expand parameter buffers to new capacity."""
        raise NotImplementedError


class BaseTopologyManager:
    """Base class for all topology management strategies."""
    
    def __init__(self, config: NGSConfig):
        self.config = config
    
    def adapt_topology(self, model, **kwargs) -> tuple:
        """Adapt the network topology based on current state."""
        raise NotImplementedError


class BaseMemoryManager:
    """Base class for all memory management strategies."""
    
    def __init__(self, config: NGSConfig):
        self.config = config
    
    def enforce_capacity(self, model) -> int:
        """Enforce memory capacity constraints. Returns number of pruned units."""
        raise NotImplementedError
    
    def expand_buffers(self, model, new_max_k: int):
        """Expand model buffers for increased capacity."""
        raise NotImplementedError


__all__ = [
    "RoutingStrategy",
    "ParameterStorage", 
    "TopologyControl",
    "MemoryManagement",
    "NGSConfig",
    "RoutingOutput",
    "TopologyAction",
    "BaseRouter",
    "BaseParameterStore",
    "BaseTopologyManager",
    "BaseMemoryManager",
]
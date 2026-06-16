"""Core interfaces and abstract base classes for NGS library."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Optional, TypeVar
import torch
import torch.nn as nn

T = TypeVar('T', bound=nn.Module)


class RoutingStrategy(str, Enum):
    MONOLITHIC = "monolithic"
    FACTORIZED = "factorized"
    HIERARCHICAL = "hierarchical"
    LSH = "lsh"
    GAUSSIAN_ATTENTION = "gaussian_attention"


class ParameterStorage(str, Enum):
    DIRECT = "direct"
    HYPERNETWORK = "hypernetwork"
    LORA = "lora"


class TopologyControl(str, Enum):
    HEURISTIC = "heuristic"
    CONTINUOUS_DENSITY = "continuous_density"
    MERGE_AWARE = "merge_aware"
    META_LEARNED = "meta_learned"


class MemoryManagement(str, Enum):
    PRE_ALLOCATED = "pre_allocated"
    DYNAMIC = "dynamic"
    STRICT_CAPACITY = "strict_capacity"


@dataclass
class RoutingOutput:
    """Standardized routing output across all strategies."""
    indices: torch.Tensor | list[torch.Tensor]  # [B, K] or list of [B, K_s]
    weights: torch.Tensor | list[torch.Tensor]  # [B, K] or list of [B, K_s]
    aux: dict = field(default_factory=dict)  # e.g., uncertainty, attention scores


@dataclass
class TopologyAction:
    """Result of topology adaptation."""
    num_pruned: int = 0
    num_split: int = 0
    num_spawned: int = 0
    num_merged: int = 0
    merged_indices: list[tuple[int, int]] = field(default_factory=list)


class BaseRouter(nn.Module, ABC):
    """Abstract base for all routing strategies."""
    
    @property
    @abstractmethod
    def num_active_units(self) -> int:
        """Number of currently active units."""
        pass
    
    @property
    @abstractmethod
    def max_units(self) -> int:
        """Maximum capacity."""
        pass
    
    @abstractmethod
    def forward(self, z: torch.Tensor) -> RoutingOutput:
        """Route latent input to active units."""
        pass
    
    @abstractmethod
    def get_unit_params(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get parameters for specific unit indices."""
        pass
    
    @abstractmethod
    def initialize_units(self, k_init: int) -> None:
        """Initialize first k_init units as active."""
        pass


class BaseParameterStore(nn.Module, ABC):
    """Abstract base for parameter storage strategies."""
    
    @abstractmethod
    def forward(self, indices: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Generate/apply transformations for active units.
        
        Args:
            indices: [B, K] unit indices
            z: [B, d] latent features
            
        Returns:
            [B, K, d] transformed features
        """
        pass
    
    @abstractmethod
    def get_parameters(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get stored parameters for indices."""
        pass
    
    @abstractmethod
    def init_unit(self, index: int, source_index: Optional[int] = None) -> None:
        """Initialize a new unit, optionally copying from source."""
        pass
    
    @abstractmethod
    def merge_units(self, target_idx: int, source_idx: int, weight: float = 0.5) -> None:
        """Merge source into target with given weight."""
        pass


class BaseTopologyManager(ABC):
    """Abstract base for topology management strategies."""
    
    @abstractmethod
    def adapt_topology(
        self,
        model: 'NGSModel',
        z_samples: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TopologyAction:
        """Adapt topology based on current state."""
        pass
    
    @abstractmethod
    def compute_losses(self, model: 'NGSModel', **kwargs) -> dict[str, torch.Tensor]:
        """Compute topology-related losses (entropy, diversity, split-gate, etc.)."""
        pass


class BaseMemoryManager(ABC):
    """Abstract base for memory management strategies."""
    
    @abstractmethod
    def enforce_capacity(self, model: 'NGSModel') -> int:
        """Enforce capacity constraints, return number of units pruned."""
        pass
    
    @abstractmethod
    def allocate_unit(self, model: 'NGSModel') -> Optional[int]:
        """Allocate a new unit slot, return index or None if full."""
        pass
    
    @abstractmethod
    def free_unit(self, model: 'NGSModel', index: int) -> None:
        """Free a unit slot."""
        pass


@dataclass
class NGSConfig:
    """Unified configuration for NGS models."""
    # Core dimensions
    latent_dim: int = 32
    k_init: int = 128
    max_k: int = 512
    top_k: int = 8
    
    # Modular strategy choices
    routing: RoutingStrategy = RoutingStrategy.FACTORIZED
    parameter_storage: ParameterStorage = ParameterStorage.HYPERNETWORK
    topology_control: TopologyControl = TopologyControl.CONTINUOUS_DENSITY
    memory_management: MemoryManagement = MemoryManagement.PRE_ALLOCATED
    
    # Strategy-specific params
    # Factorized routing
    num_subspaces: int = 4
    top_k_factorized: int = 2
    
    # Hierarchical routing
    num_levels: int = 2
    coarse_units: int = 16
    fine_units_per_coarse: int = 32
    
    # Hypernetwork
    hypernetwork_code_dim: int = 8
    hypernetwork_hidden_dim: int = 16
    use_lora: bool = True
    lora_rank: int = 4
    
    # Topology control
    split_threshold: float = 0.05
    prune_threshold: float = 0.01
    split_gate_threshold: float = 0.65
    merge_threshold: float = 0.1  # cosine similarity threshold for merging
    density_decay: float = 0.99
    
    # Uncertainty routing
    uncertainty_method: str = "evidential"  # "evidential", "bayesian", "ensemble"
    evidential_prior: float = 1.0
    
    # Training
    tau: float = 1.0
    gamma_residual: float = 0.1
    ema_decay: float = 0.99
    diversity_weight: float = 0.01
    entropy_weight: float = 0.01
    split_gate_weight: float = 0.001
    merge_weight: float = 0.01
    
    # Advanced
    enable_self_referential: bool = False
    meta_gaussian_ratio: float = 0.1  # fraction of units that are meta
    
    def __post_init__(self):
        # Validate combinations
        if self.routing == RoutingStrategy.HIERARCHICAL:
            assert self.coarse_units * self.fine_units_per_coarse <= self.max_k
        if self.routing == RoutingStrategy.FACTORIZED:
            assert self.max_k % self.num_subspaces == 0
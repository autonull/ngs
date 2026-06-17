"""Core configuration and interfaces for NGS."""

from mngs.core.config import (
    MNGSConfig as NGSConfig,
    RoutingStrategy,
    ParameterStorage,
    TopologyControl,
    MemoryManagement,
)

__all__ = [
    "NGSConfig",
    "RoutingStrategy",
    "ParameterStorage",
    "TopologyControl",
    "MemoryManagement",
]
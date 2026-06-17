"""Core configuration and interfaces for NGS."""

from ngs.core.interfaces import (
    NGSConfig,
    RoutingStrategy,
    ParameterStorage,
    TopologyControl,
    MemoryManagement,
    RoutingOutput,
    TopologyAction,
    BaseRouter,
    BaseParameterStore,
    BaseTopologyManager,
    BaseMemoryManager,
)

__all__ = [
    "NGSConfig",
    "RoutingStrategy",
    "ParameterStorage",
    "TopologyControl",
    "MemoryManagement",
    "RoutingOutput",
    "TopologyAction",
    "BaseRouter",
    "BaseParameterStore",
    "BaseTopologyManager",
    "BaseMemoryManager",
]
"""Modular components for NGS."""

from ngs.modules.routers import (
    MonolithicRouter,
    FactorizedRouter,
    LSRRouter,
    HierarchicalRouter,
    GaussianAttentionRouter,
    UncertaintyAwareRouter,
    build_router,
    BaseRouter,
)
from ngs.modules.parameter_stores import (
    DirectAdapterStore,
    HypernetworkStore,
    LoRAStore,
    build_parameter_store,
    BaseParameterStore,
)
from ngs.modules.topology_managers import (
    HeuristicManager,
    ContinuousDensityManager,
    MergeAwareManager,
    MetaLearnedManager,
    build_topology_manager,
    BaseTopologyManager,
)
from ngs.modules.memory_managers import (
    PreAllocatedManager,
    DynamicManager,
    StrictCapacityManager,
    build_memory_manager,
    BaseMemoryManager,
)
from ngs.modules.riemannian import (
    RiemannianHypernetworkManifold,
    HypernetworkCodeManifold,
    build_riemannian_manifold,
)

__all__ = [
    "MonolithicRouter",
    "FactorizedRouter",
    "LSRRouter",
    "HierarchicalRouter",
    "GaussianAttentionRouter",
    "UncertaintyAwareRouter",
    "build_router",
    "BaseRouter",
    "DirectAdapterStore",
    "HypernetworkStore",
    "LoRAStore",
    "build_parameter_store",
    "BaseParameterStore",
    "HeuristicManager",
    "ContinuousDensityManager",
    "MergeAwareManager",
    "MetaLearnedManager",
    "build_topology_manager",
    "BaseTopologyManager",
    "PreAllocatedManager",
    "DynamicManager",
    "StrictCapacityManager",
    "build_memory_manager",
    "BaseMemoryManager",
    "RiemannianHypernetworkManifold",
    "HypernetworkCodeManifold",
    "build_riemannian_manifold",
]
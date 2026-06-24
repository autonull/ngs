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
from ngs.modules.ngs_layer import (
    NGSLayer,
    StackedNGSModel,
    MultiHeadProj,
    build_stacked_ngs,
)
from ngs.modules.riemannian import (
    RiemannianHypernetworkManifold,
    HypernetworkCodeManifold,
    build_riemannian_manifold,
)
from ngs.modules.advanced import (
    SymbolicExtractor,
    CrossModalFusion,
    MetaMetaLearner,
    MahalanobisKernel,
    LoRAKernel,
    TritonKernelConfig,
    build_symbolic_extractor,
    build_cross_modal_fusion,
    build_meta_meta_learner,
)
from ngs.modules.dynamic_head import (
    DynamicHead,
    build_dynamic_head,
    default_dynamic_config,
)

__all__ = [
    "NGSLayer",
    "StackedNGSModel",
    "MultiHeadProj",
    "build_stacked_ngs",
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
    "SymbolicExtractor",
    "CrossModalFusion",
    "MetaMetaLearner",
    "MahalanobisKernel",
    "LoRAKernel",
    "TritonKernelConfig",
    "build_symbolic_extractor",
    "build_cross_modal_fusion",
    "build_meta_meta_learner",
    "DynamicHead",
    "build_dynamic_head",
    "default_dynamic_config",
]
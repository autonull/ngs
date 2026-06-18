"""Neural Gaussian Systems (NGS) - Modular Adaptive Neural Networks."""

from ngs.core.interfaces import (
    NGSConfig,
    RoutingStrategy,
    ParameterStorage,
    TopologyControl,
    MemoryManagement,
    RoutingOutput,
    TopologyAction,
)
from ngs.models.ngs import NGSModel, build_ngs
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
from ngs.training.trainer import (
    NGSTrainer,
    TrainerConfig,
)
from ngs.visualization.visualize import (
    plot_topology_dynamics,
    plot_routing_heatmap,
    plot_3d_gaussian_means,
    plot_uncertainty_calibration,
    plot_evolution_gif,
    plot_subspace_alignment,
    plot_hypernetwork_codes,
    plot_riemannian_geodesics,
    interactive_dashboard,
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

__all__ = [
    "NGSConfig",
    "RoutingStrategy",
    "ParameterStorage",
    "TopologyControl",
    "MemoryManagement",
    "RoutingOutput",
    "TopologyAction",
    "NGSModel",
    "build_ngs",
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
    "NGSTrainer",
    "TrainerConfig",
    "plot_topology_dynamics",
    "plot_routing_heatmap",
    "plot_3d_gaussian_means",
    "plot_uncertainty_calibration",
    "plot_evolution_gif",
    "plot_subspace_alignment",
    "plot_hypernetwork_codes",
    "plot_riemannian_geodesics",
    "interactive_dashboard",
    "SymbolicExtractor",
    "CrossModalFusion",
    "MetaMetaLearner",
    "MahalanobisKernel",
    "LoRAKernel",
    "TritonKernelConfig",
    "build_symbolic_extractor",
    "build_cross_modal_fusion",
    "build_meta_meta_learner",
]

# Version
__version__ = "0.1.0"
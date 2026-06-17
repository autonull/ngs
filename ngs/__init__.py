"""Neural Gaussian Systems (NGS) - Modular Adaptive Neural Networks.

This is the new clean API namespace for NGS, re-exporting from the mngs implementation.
"""

from mngs.core.config import (
    MNGSConfig as NGSConfig,
    RoutingStrategy,
    ParameterStorage,
    TopologyControl,
    MemoryManagement,
)
from mngs.model import MNGS as NGSModel, build_mngs as build_ngs
from mngs.modules.routers import (
    MonolithicRouter,
    FactorizedRouter,
    LSRRouter,
    build_router,
)
from mngs.modules.parameter_stores import (
    DirectAdapterStore,
    HypernetworkStore,
    build_parameter_store,
)
from mngs.modules.topology_managers import (
    HeuristicManager,
    ContinuousDensityManager,
    build_topology_manager,
)
from mngs.training.trainer import (
    NGSTrainer,
    ContinualTrainer,
    TrainConfig,
    TrainerConfig,
    create_trainer,
)

__all__ = [
    "NGSConfig",
    "RoutingStrategy",
    "ParameterStorage",
    "TopologyControl",
    "MemoryManagement",
    "NGSModel",
    "build_ngs",
    "MonolithicRouter",
    "FactorizedRouter",
    "LSRRouter",
    "build_router",
    "DirectAdapterStore",
    "HypernetworkStore",
    "build_parameter_store",
    "HeuristicManager",
    "ContinuousDensityManager",
    "build_topology_manager",
    "NGSTrainer",
    "ContinualTrainer",
    "TrainConfig",
    "TrainerConfig",
    "create_trainer",
]

# Version
__version__ = "0.1.0"
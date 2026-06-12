"""Example MNGS profile configurations matching the TODO.md plan."""
from mngs.core.config import MNGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement


def Baseline_LeanNGS() -> MNGSConfig:
    """
    Profile 1: Baseline LeanNGS (The Control) - LoRA efficient version (~190K params)
    
    Reproduces the exact behavior of the original LeanNGS prototype.
    Serves as the absolute baseline for all ablation studies.
    """
    return MNGSConfig(
        # Core dimensions
        latent_dim=32,
        k_init=128,
        max_k=512,
        top_k=8,
        
        # Baseline settings
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.PRE_ALLOCATED_MASKED,
        
        # Training hyperparameters from original
        lora_rank=4,
        use_lora=True,
        tau=1.0,
        gamma_residual=0.1,
        ema_decay=0.99,
        split_threshold=0.05,
        prune_threshold=0.01,
    )


def Baseline_LeanNGS_ParamMatched() -> MNGSConfig:
    """
    Profile 1b: Baseline LeanNGS Parameter-Matched (~513K params, matches MLP baseline)
    
    Uses full adapters (no LoRA) with max_k=448 for fair comparison with MLP/EWC/SI/LwF/ER baselines.
    """
    return MNGSConfig(
        # Core dimensions - max_k=448 gives ~513K params matching MLP [512,256] (~534K)
        latent_dim=32,
        k_init=128,
        max_k=448,
        top_k=8,
        
        # Baseline settings
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.PRE_ALLOCATED_MASKED,
        
        # Full adapters for parameter-matched comparison
        lora_rank=4,
        use_lora=False,
        tau=1.0,
        gamma_residual=0.1,
        ema_decay=0.99,
        split_threshold=0.05,
        prune_threshold=0.01,
    )


def CFG_Net_Full() -> MNGSConfig:
    """
    Profile 2: CFG-Net Full (The Proposed Upgrade) - LoRA efficient version
    
    Tests the full hypothesis: factorized routing, hypernetwork storage,
    continuous density topology. Expects better accuracy with lower memory.
    """
    return MNGSConfig(
        # Core dimensions
        latent_dim=32,
        k_init=128,
        max_k=512,
        top_k=8,
        
        # Full CFG-Net settings
        routing=RoutingStrategy.FACTORIZED_SUBSPACE,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.PRE_ALLOCATED_MASKED,
        
        # CFG-Net specific
        top_k_factorized=2,
        num_subspaces=4,
        hypernetwork_code_dim=8,
        hypernetwork_hidden_dim=16,
        use_lora=True,
        
        # Adjusted thresholds for continuous topology
        split_threshold=0.02,
        prune_threshold=0.01,
    )


def CFG_Net_Full_ParamMatched() -> MNGSConfig:
    """
    Profile 2b: CFG-Net Full Parameter-Matched (~513K params)
    
    Full CFG-Net with full adapters for fair comparison.
    """
    return MNGSConfig(
        # Core dimensions
        latent_dim=32,
        k_init=128,
        max_k=448,
        top_k=8,
        
        # Full CFG-Net settings
        routing=RoutingStrategy.FACTORIZED_SUBSPACE,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.PRE_ALLOCATED_MASKED,
        
        # CFG-Net specific
        top_k_factorized=2,
        num_subspaces=4,
        hypernetwork_code_dim=8,
        hypernetwork_hidden_dim=16,
        use_lora=False,
        
        # Adjusted thresholds for continuous topology
        split_threshold=0.02,
        prune_threshold=0.01,
    )


def Ultra_Edge_Sparse() -> MNGSConfig:
    """
    Profile 3: Ultra-Edge Sparse (Decentralized Optimization)
    
    Designed for microcontrollers. Factorized routing minimizes compute.
    Hypernetwork minimizes RAM. Strict capacity ensures the model never
    exceeds its memory budget.
    """
    return MNGSConfig(
        # Smaller dimensions for edge
        latent_dim=16,
        k_init=64,
        max_k=256,
        top_k=4,
        
        # Edge-optimized settings
        routing=RoutingStrategy.FACTORIZED_SUBSPACE,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.STRICT_CAPACITY,
        
        # Tighter constraints
        top_k_factorized=1,
        num_subspaces=4,
        hypernetwork_code_dim=4,
        hypernetwork_hidden_dim=8,
        split_threshold=0.08,
        prune_threshold=0.02,
    )


def Ablation_Hypernetwork_Only() -> MNGSConfig:
    """
    Profile 4: Ablation - Hypernetwork Only (LoRA version)
    
    Isolates the value of the hypernetwork by keeping original routing
    and splitting but swapping to hypernetwork storage.
    """
    return MNGSConfig(
        # Core dimensions
        latent_dim=32,
        k_init=128,
        max_k=512,
        top_k=8,
        
        # Only parameter storage is changed
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.PRE_ALLOCATED_MASKED,
        
        # Hypernetwork settings
        hypernetwork_code_dim=8,
        hypernetwork_hidden_dim=16,
        use_lora=True,
        
        # Standard thresholds
        split_threshold=0.05,
        prune_threshold=0.01,
    )


def Ablation_Hypernetwork_Only_ParamMatched() -> MNGSConfig:
    """
    Profile 4b: Ablation - Hypernetwork Only Parameter-Matched
    
    Hypernetwork with full adapters for fair comparison.
    """
    return MNGSConfig(
        # Core dimensions
        latent_dim=32,
        k_init=128,
        max_k=448,
        top_k=8,
        
        # Only parameter storage is changed
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.PRE_ALLOCATED_MASKED,
        
        # Hypernetwork settings
        hypernetwork_code_dim=8,
        hypernetwork_hidden_dim=16,
        use_lora=False,
        
        # Standard thresholds
        split_threshold=0.05,
        prune_threshold=0.01,
    )


def profile_all() -> list:
    """Return all example configurations (LoRA efficient versions)."""
    return [
        Baseline_LeanNGS(),
        CFG_Net_Full(),
        Ultra_Edge_Sparse(),
        Ablation_Hypernetwork_Only(),
    ]


def profile_all_param_matched() -> list:
    """Return all parameter-matched configurations for fair baseline comparison."""
    return [
        Baseline_LeanNGS_ParamMatched(),
        CFG_Net_Full_ParamMatched(),
        Ablation_Hypernetwork_Only_ParamMatched(),
    ]

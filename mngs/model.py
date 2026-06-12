"""Main MNGS model integrating all modular components."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mngs.core.config import MNGSConfig, RoutingStrategy, ParameterStorage, TopologyControl
from mngs.modules.routers import MonolithicRouter, FactorizedRouter, LSRRouter
from mngs.modules.parameter_stores import DirectAdapterStore, HypernetworkStore
from mngs.modules.topology_managers import HeuristicManager, ContinuousDensityManager


class MNGS(nn.Module):
    """
    Modular Neural Gaussian System.
    
    Composable, configuration-driven framework for sparse neural networks.
    """
    
    def __init__(self, d_in: int, d_out: int, config: MNGSConfig):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.config = config
        
        self.d_latent = config.latent_dim
        self.eps = 1e-5
        
        # Projection layers (shared across all configurations)
        self.p_down = nn.Linear(d_in, self.d_latent, bias=False)
        self.p_up = nn.Linear(self.d_latent, d_out, bias=False)
        self.gamma = nn.Parameter(torch.tensor(config.gamma_residual))
        
        # Parameter Store (parameter storage strategy)
        self.param_store = self._build_param_store()
        
        # Per-subspace parameter stores for FactorizedRouter (M2.3) - build before router
        if self.config.routing == RoutingStrategy.FACTORIZED_SUBSPACE:
            self.param_stores_per_subspace = self._build_param_stores_per_subspace()
        else:
            self.param_stores_per_subspace = None
        
        # Router (routing strategy)
        self.router = self._build_router()
        
        # Topology Manager (topology control strategy)
        self.topology_manager = self._build_topology_manager()
        
        # Memory management strategy
        self.memory_management = config.memory_management
        
        # Split gate for continuous density topology
        self.register_parameter('split_gate', nn.Parameter(torch.full((config.max_k,), config.gamma_residual)))
        self.register_buffer('activation_density', torch.zeros(config.max_k))
        self.register_buffer('error_density', torch.zeros(config.max_k))
        
        # Initialize router units
        if hasattr(self.router, 'initialize_units'):
            self.router.initialize_units(config.k_init)
    
    def _build_router(self):
        """Build the routing module based on config."""
        routing = self.config.routing
        
        if routing == RoutingStrategy.MONOLITHIC_MAHALANOBIS:
            return MonolithicRouter(
                max_k=self.config.max_k,
                d_latent=self.d_latent,
                top_k=self.config.top_k,
                tau=self.config.tau,
                ema_decay=self.config.ema_decay
            )
        elif routing == RoutingStrategy.FACTORIZED_SUBSPACE:
            units_per_space = -(-self.config.k_init // self.config.num_subspaces)  # ceiling div
            router = FactorizedRouter(
                d_latent=self.d_latent,
                num_subspaces=self.config.num_subspaces,
                units_per_space=units_per_space,
                top_k=self.config.top_k_factorized,
                tau=self.config.tau
            )
            if self.param_stores_per_subspace is not None:
                router.set_param_stores(self.param_stores_per_subspace)
            return router
        elif routing == RoutingStrategy.LSH_APPROXIMATE:
            return LSRRouter(
                d_latent=self.d_latent,
                num_buckets=self.config.max_k // 4,
                num_hash_functions=4,
                top_k=self.config.top_k
            )
        else:
            raise ValueError(f"Unknown routing strategy: {routing}")
    
    def _build_param_store(self):
        """Build the parameter store based on config."""
        storage = self.config.parameter_storage
        
        if storage == ParameterStorage.DIRECT_ADAPTER:
            return DirectAdapterStore(
                max_k=self.config.max_k,
                d_latent=self.d_latent,
                use_lora=self.config.use_lora,
                lora_rank=self.config.lora_rank
            )
        elif storage == ParameterStorage.HYPERNETWORK_GENERATED:
            return HypernetworkStore(
                max_k=self.config.max_k,
                d_latent=self.d_latent,
                code_dim=self.config.hypernetwork_code_dim,
                hidden_dim=self.config.hypernetwork_hidden_dim,
                use_lora=self.config.use_lora
            )
        else:
            raise ValueError(f"Unknown parameter storage: {storage}")
    
    def _build_param_stores_per_subspace(self):
        """Build parameter stores for each subspace (used by FactorizedRouter)."""
        storage = self.config.parameter_storage
        units_per_space = -(-self.config.k_init // self.config.num_subspaces)
        num_subspaces = self.config.num_subspaces
        
        param_stores = []
        for s in range(num_subspaces):
            if storage == ParameterStorage.DIRECT_ADAPTER:
                param_stores.append(DirectAdapterStore(
                    max_k=units_per_space,
                    d_latent=self.d_latent,
                    use_lora=self.config.use_lora,
                    lora_rank=self.config.lora_rank
                ))
            elif storage == ParameterStorage.HYPERNETWORK_GENERATED:
                param_stores.append(HypernetworkStore(
                    max_k=units_per_space,
                    d_latent=self.d_latent,
                    code_dim=self.config.hypernetwork_code_dim,
                    hidden_dim=self.config.hypernetwork_hidden_dim,
                    use_lora=self.config.use_lora
                ))
            else:
                raise ValueError(f"Unknown parameter storage: {storage}")
        return param_stores
    
    def _build_topology_manager(self):
        """Build the topology manager based on config."""
        topology = self.config.topology_control
        
        if topology == TopologyControl.DISCRETE_HEURISTIC:
            return HeuristicManager(
                split_threshold=self.config.split_threshold,
                prune_threshold=self.config.prune_threshold,
                ema_decay=self.config.ema_decay
            )
        elif topology == TopologyControl.CONTINUOUS_DENSITY:
            return ContinuousDensityManager(
                split_threshold=self.config.split_threshold,
                prune_threshold=self.config.prune_threshold,
                density_decay=self.config.ema_decay
            )
        else:
            raise ValueError(f"Unknown topology control: {topology}")
    
    @property
    def K(self):
        """Number of active units."""
        if hasattr(self.router, 'active_mask'):
            return self.router.active_mask.sum().item()
        if hasattr(self.router, 'num_subspaces'):
            return self.router.num_subspaces * self.router.units_per_space
        return self.config.max_k
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MNGS.
        
        Args:
            x: [B, d_in] input features
            
        Returns:
            [B, d_out] output features
        """
        z = self.p_down(x)  # [B, d]
        
        # Route to active units
        routing_output = self.router(z)
        
        if isinstance(routing_output, tuple) and len(routing_output) == 2:
            if isinstance(routing_output[0], list):
                # Factorized routing: list of subspace indices and weights
                subspace_indices, subspace_weights = routing_output
                out = self._forward_factorized(z, subspace_indices, subspace_weights)
            else:
                # Standard routing: [B, K] indices and weights
                active_indices, routing_weights = routing_output
                out = self._forward_standard(z, active_indices, routing_weights)
        else:
            raise ValueError(f"Unexpected routing output format: {type(routing_output)}")
        
        # Density tracking for continuous density topology
        if isinstance(self.topology_manager, ContinuousDensityManager) and hasattr(self.router, 'active_mask'):
            active_idx = self.router.active_mask.nonzero(as_tuple=True)[0]
            if active_idx.numel() > 0:
                with torch.no_grad():
                    # Activation density: which units were selected
                    act = torch.zeros(self.config.max_k, device=out.device)
                    act[active_idx] = 1.0
                    self.activation_density = (
                        self.topology_manager.density_decay * self.activation_density
                        + (1 - self.topology_manager.density_decay) * act
                    )
        
        return out
    
    def _forward_standard(self, z: torch.Tensor, active_indices: torch.Tensor, 
                          routing_weights: torch.Tensor) -> torch.Tensor:
        """Forward pass for monolithic routing."""
        # Apply transformations for active units
        local_out = self.param_store(active_indices, z)  # [B, K, d]
        
        # Weighted combination
        weights = routing_weights.unsqueeze(-1)  # [B, K, 1]
        blended = (weights * local_out).sum(dim=1)  # [B, d]
        
        # Up projection with residual
        return self.p_up(blended + self.gamma * z)
    
    def _forward_factorized(self, z: torch.Tensor, subspace_indices: list, 
                           subspace_weights: list) -> torch.Tensor:
        """Forward pass for factorized routing."""
        B = z.shape[0]
        num_subspaces = len(subspace_indices)
        
        all_activations = []
        for s in range(num_subspaces):
            indices = subspace_indices[s]  # [B, top_k] (local within subspace)
            weights = subspace_weights[s]    # [B, top_k]
            
            # Use per-subspace parameter store if available (M2.3)
            if self.param_stores_per_subspace is not None:
                param_store = self.param_stores_per_subspace[s]
                local_out = param_store(indices, z)  # [B, top_k, d]
            else:
                # Fallback: convert to global indices (for backwards compat with PRE_ALLOCATED_MASKED)
                units_per_space = -(-self.config.k_init // self.config.num_subspaces)
                global_indices = indices + s * units_per_space
                local_out = self.param_store(global_indices, z)
            
            # Weighted combination within subspace
            w = weights.unsqueeze(-1)  # [B, top_k, 1]
            subspace_out = (w * local_out).sum(dim=1)  # [B, d]
            all_activations.append(subspace_out)
        
        # Combine across subspaces
        combined = torch.stack(all_activations, dim=1).mean(dim=1)  # [B, d]
        
        return self.p_up(combined + self.gamma * z)
    
    def entropy_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute entropy regularization loss."""
        z = self.p_down(x)
        routing_output = self.router(z)
        
        # Determine if factorized or monolithic routing
        if isinstance(routing_output[0], list):
            # Factorized routing: compute entropy per subspace
            entropy = 0.0
            for weights in routing_output[1]:  # subspace_weights
                p = weights  # [B, top_k]
                entropy += -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
            return entropy / len(routing_output[1])
        else:
            # Monolithic routing
            weights = routing_output[1]  # [B, top_k]
            p = weights
            return -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
    
    def update_grad_ema(self):
        """Update gradient EMA for topology management.
        
        Deprecated: automatic via register_hook in MonolithicRouter.
        Kept for backward compatibility.
        """
    
    def adapt_density(self, **kwargs) -> tuple:
        """
        Apply topology management (split/prune/spawn), respecting memory strategy.
        
        Args:
            **kwargs: Passed to topology manager
            
        Returns:
            Tuple of (num_pruned, num_split, num_spawned)
        """
        from mngs.core.config import MemoryManagement
        result = self.topology_manager.adapt_topology(self, **kwargs)
        
        if self.memory_management == MemoryManagement.STRICT_CAPACITY and hasattr(self.router, 'active_mask'):
            # Enforce capacity cap: prune lowest-alpha units if over budget
            while self.K > self.config.max_k:
                active_idx = self.router.active_mask.nonzero(as_tuple=True)[0]
                alphas = torch.sigmoid(self.router.log_alpha[active_idx])
                worst = alphas.argmin()
                prune_idx = active_idx[worst:worst+1]
                self.router.active_mask[prune_idx] = False
                self.router.grad_mu_ema[prune_idx] = 0
                result = (result[0] + 1, result[1], result[2])
        
        return result
    
    def split_gate_loss(self) -> torch.Tensor:
        """Compute regularization loss for continuous density split gates."""
        sig = torch.sigmoid(self.split_gate)
        # Push gamma to {0, 1} with strength proportional to error density
        err = self.error_density / (self.error_density.sum() + 1e-8)
        return -(sig * (1 - sig) * err).sum()
    
    def diversity_loss(self) -> torch.Tensor:
        """Push Gaussian means apart to encourage coverage."""
        if not hasattr(self.router, 'active_mask'):
            return torch.tensor(0.0, device=self.p_down.weight.device)
        active_idx = self.router.active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) < 2:
            return torch.tensor(0.0, device=self.p_down.weight.device)
        mu = self.router.mu[active_idx]  # [K, d]
        dist = torch.cdist(mu, mu, p=2)  # [K, K]
        mask = ~torch.eye(len(active_idx), dtype=torch.bool, device=mu.device)
        min_dist = dist[mask].min()
        return -min_dist


def build_mngs(d_in: int, d_out: int, config: MNGSConfig) -> MNGS:
    """
    Factory function to build MNGS from configuration.
    
    Args:
        d_in: Input dimension
        d_out: Output dimension
        config: MNGSConfig instance
        
    Returns:
        Configured MNGS model
    """
    return MNGS(d_in, d_out, config)

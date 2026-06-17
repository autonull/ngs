"""Main MNGS model integrating all modular components."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Dict, Optional
from types import SimpleNamespace

from mngs.core.config import MNGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
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
        
        # Cached routing output for entropy_loss
        self._last_routing_output = None
        
        # Memory management strategy
        self.memory_management = config.memory_management
        
        # Split gate for continuous density topology
        self.register_parameter('split_gate', nn.Parameter(torch.full((config.max_k,), config.gamma_residual)))
        self.register_buffer('activation_density', torch.zeros(config.max_k))
        self.register_buffer('error_density', torch.zeros(config.max_k))
        
        # Cached routing info for error density estimation
        self._last_active_indices = None
        self._last_routing_weights = None
        
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
            units_per_space = self.config.max_k // self.config.num_subspaces
            active_per_space = -(-self.config.k_init // self.config.num_subspaces)  # ceiling div
            router = FactorizedRouter(
                d_latent=self.d_latent,
                num_subspaces=self.config.num_subspaces,
                units_per_space=units_per_space,
                top_k=self.config.top_k_factorized,
                tau=self.config.tau
            )
            # Only activate k_init units (rest are free slots for growth)
            router.active_mask.fill_(False)
            for s in range(self.config.num_subspaces):
                start = s * units_per_space
                router.active_mask[start:start + active_per_space] = True
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
        units_per_space = self.config.max_k // self.config.num_subspaces
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
    def K(self) -> int:
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
        
        # Cache routing output for entropy_loss
        self._last_routing_output = SimpleNamespace(latent=z, indices=None, weights=None)
        
        if isinstance(routing_output, tuple) and len(routing_output) == 2:
            if isinstance(routing_output[0], list):
                # Factorized routing: list of subspace indices and weights
                subspace_indices, subspace_weights = routing_output
                out = self._forward_factorized(z, subspace_indices, subspace_weights)
                # Router now returns global flat indices directly — just concatenate
                flat_indices = torch.cat(subspace_indices, dim=1)
                flat_weights = torch.cat(subspace_weights, dim=1)
                self._last_active_indices = flat_indices
                self._last_routing_weights = flat_weights
                self._last_routing_output.indices = flat_indices
                self._last_routing_output.weights = flat_weights
            else:
                # Standard routing: [B, K] indices and weights
                active_indices, routing_weights = routing_output
                out = self._forward_standard(z, active_indices, routing_weights)
                self._last_active_indices = active_indices
                self._last_routing_weights = routing_weights
                self._last_routing_output.indices = active_indices
                self._last_routing_output.weights = routing_weights
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
    
    def _forward_factorized(self, z: torch.Tensor, subspace_indices: List[torch.Tensor], 
                           subspace_weights: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass for factorized routing."""
        B = z.shape[0]
        num_subspaces = len(subspace_indices)
        units_per_space = self.config.max_k // self.config.num_subspaces
        
        all_activations = []
        for s in range(num_subspaces):
            global_indices = subspace_indices[s]  # [B, top_k] (global flat indices)
            weights = subspace_weights[s]    # [B, top_k]
            
            # Convert global flat indices back to local subspace indices
            local_indices = global_indices - s * units_per_space
            
            # Use per-subspace parameter store if available (M2.3)
            if self.param_stores_per_subspace is not None:
                param_store = self.param_stores_per_subspace[s]
                local_out = param_store(local_indices, z)  # [B, top_k, d]
            else:
                # Fallback: use global param store
                local_out = self.param_store(global_indices, z)
            
            # Weighted combination within subspace
            w = weights.unsqueeze(-1)  # [B, top_k, 1]
            subspace_out = (w * local_out).sum(dim=1)  # [B, d]
            all_activations.append(subspace_out)
        
        # Combine across subspaces
        combined = torch.stack(all_activations, dim=1).mean(dim=1)  # [B, d]
        
        return self.p_up(combined + self.gamma * z)
    
    def entropy_loss(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute entropy regularization loss."""
        if x is not None:
            z = self.p_down(x)
        elif self._last_routing_output is not None:
            z = self._last_routing_output.latent  # Use cached latent
        else:
            return torch.tensor(0.0, device=self.p_down.weight.device)
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
    
    def update_unit_errors(self, logits: torch.Tensor, targets: torch.Tensor,
                           decay: float = 0.99) -> None:
        """Update per-unit error density from training loss.
        
        Attributes loss to units based on their routing weights.
        Called from the training loop after computing the loss.
        """
        if self._last_active_indices is None or self._last_routing_weights is None:
            return
        
        with torch.no_grad():
            per_sample_loss = F.cross_entropy(logits, targets, reduction='none')  # [B]
            weights = self._last_routing_weights  # [B, K]
            active_idx = self._last_active_indices  # [B, K]
            
            # Weighted loss attribution: for each unit, average loss weighted by routing weight
            unit_loss = (weights * per_sample_loss.unsqueeze(-1)).mean(dim=0)  # [K]
            
            # Map back to global indices and update with EMA
            for k in range(active_idx.shape[1]):
                global_idx = active_idx[:, k]  # [B] (same index for all samples in this slot)
                unique_idx = global_idx.unique()
                for uid in unique_idx:
                    mask = global_idx == uid
                    self.error_density[uid] = (
                        decay * self.error_density[uid]
                        + (1 - decay) * unit_loss[k].item()
                    )
    
    def update_grad_ema(self) -> None:
        """Update gradient EMA for topology management.
        
        Deprecated: automatic via register_hook in MonolithicRouter.
        Kept for backward compatibility.
        """
    
    def adapt_density(self, **kwargs) -> Tuple[int, int, int]:
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
        """Compute regularization loss for continuous density split gates.
        
        Pushes gates toward 1 (split) for high-error units and 0 (keep) for low-error units.
        """
        import torch.nn.functional as F
        sig = torch.sigmoid(self.split_gate)
        # Normalize error density to [0, 1] as target
        err = self.error_density / (self.error_density.max() + 1e-8)
        # BCE: pushes gate toward 1 where error is high, toward 0 where error is low
        return F.binary_cross_entropy(sig, err.detach())
    
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

    def compute_topology_losses(self) -> Dict[str, torch.Tensor]:
        """Compute all topology-related losses."""
        losses = {}
        losses['entropy'] = self.entropy_loss(torch.randn(1, self.d_in, device=self.p_down.weight.device))
        losses['diversity'] = self.diversity_loss()
        if self.config.topology_control == TopologyControl.CONTINUOUS_DENSITY:
            losses['split_gate'] = self.split_gate_loss()
        return losses


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

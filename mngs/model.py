"""Main MNGS model integrating all modular components."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mngs.core.config import MNGSConfig
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
        
        # Router (routing strategy)
        self.router = self._build_router()
        
        # Parameter Store (parameter storage strategy)
        self.param_store = self._build_param_store()
        
        # Topology Manager (topology control strategy)
        self.topology_manager = self._build_topology_manager()
        
        # Gradient tracking for topology adaptation
        self.register_buffer('grad_mu_ema', torch.zeros(config.max_k))
        self.ema_decay = config.ema_decay
        
        # Initialize router units
        if hasattr(self.router, 'initialize_units'):
            self.router.initialize_units(config.k_init)
    
    def _build_router(self):
        """Build the routing module based on config."""
        routing = self.config.routing
        
        if routing.name == "MONOLITHIC_MAHALANOBIS":
            return MonolithicRouter(
                max_k=self.config.max_k,
                d_latent=self.d_latent,
                top_k=self.config.top_k,
                tau=self.config.tau
            )
        elif routing.name == "FACTORIZED_SUBSPACE":
            units_per_space = self.config.k_init // self.config.num_subspaces
            return FactorizedRouter(
                d_latent=self.d_latent,
                num_subspaces=self.config.num_subspaces,
                units_per_space=units_per_space,
                top_k=self.config.top_k_factorized,
                tau=self.config.tau
            )
        elif routing.name == "LSH_APPROXIMATE":
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
        
        if storage.name == "DIRECT_ADAPTER":
            return DirectAdapterStore(
                max_k=self.config.max_k,
                d_latent=self.d_latent,
                use_lora=True,
                lora_rank=self.config.lora_rank
            )
        elif storage.name == "HYPERNETWORK_GENERATED":
            return HypernetworkStore(
                max_k=self.config.max_k,
                d_latent=self.d_latent,
                code_dim=self.config.hypernetwork_code_dim,
                hidden_dim=self.config.hypernetwork_hidden_dim,
                use_lora=True
            )
        else:
            raise ValueError(f"Unknown parameter storage: {storage}")
    
    def _build_topology_manager(self):
        """Build the topology manager based on config."""
        topology = self.config.topology_control
        
        if topology.name == "DISCRETE_HEURISTIC":
            return HeuristicManager(
                split_threshold=self.config.split_threshold,
                prune_threshold=self.config.prune_threshold,
                ema_decay=self.config.ema_decay
            )
        elif topology.name == "CONTINUOUS_DENSITY":
            manager = ContinuousDensityManager(
                split_threshold=self.config.split_threshold,
                prune_threshold=self.config.prune_threshold
            )
            # Initialize gates
            manager.initialize_gates(self.config.max_k, self.p_down.weight.device)
            return manager
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
        units_per_space = self.config.k_init // self.config.num_subspaces
        
        all_activations = []
        for s in range(num_subspaces):
            indices = subspace_indices[s]  # [B, top_k] (local within subspace)
            weights = subspace_weights[s]    # [B, top_k]
            
            # Convert to global unit indices
            global_indices = indices + s * units_per_space  # [B, top_k]
            
            # Apply parameter store
            local_out = self.param_store(global_indices, z)  # [B, top_k, d]
            
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
            p = F.softmax(weights, dim=-1)
            return -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
    
    def update_grad_ema(self):
        """Update gradient EMA for topology management."""
        if not hasattr(self.router, 'mu') or not hasattr(self.router, 'active_mask'):
            return
        if self.router.mu.grad is None:
            return
        active_mask = self.router.active_mask
        grad_mag = self.router.mu.grad.norm(dim=-1)
        self.grad_mu_ema[active_mask] = (
            self.ema_decay * self.grad_mu_ema[active_mask]
            + (1 - self.ema_decay) * grad_mag[active_mask]
        )
    
    def adapt_density(self, **kwargs) -> tuple:
        """
        Apply topology management (split/prune/spawn).
        
        Args:
            **kwargs: Passed to topology manager
            
        Returns:
            Tuple of (num_pruned, num_split, num_spawned)
        """
        return self.topology_manager.adapt_topology(self, **kwargs)
    
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

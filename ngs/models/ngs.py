"""Main NGS model integrating all modular components."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

from ngs.core.interfaces import (
    NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement,
    BaseRouter, BaseParameterStore, BaseTopologyManager, BaseMemoryManager,
    RoutingOutput, TopologyAction
)
from ngs.modules.routers import build_router
from ngs.modules.parameter_stores import build_parameter_store
from ngs.modules.topology_managers import build_topology_manager
from ngs.modules.memory_managers import build_memory_manager


@dataclass
class NGSOutput:
    """Complete NGS forward output."""
    logits: torch.Tensor
    routing: RoutingOutput
    latent: torch.Tensor


class NGSModel(nn.Module):
    """
    Neural Gaussian System - unified adaptive neural representation.
    
    Modular architecture with four independent strategy dimensions:
    1. Routing: How to select active units (monolithic, factorized, hierarchical, attention)
    2. Parameter Storage: How to store unit parameters (direct, hypernetwork, LoRA)
    3. Topology Control: How to adapt structure (heuristic, continuous density, merge-aware)
    4. Memory Management: How to handle capacity (pre-allocated, dynamic, strict)
    """

    def __init__(self, d_in: int, d_out: int, config: NGSConfig):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.config = config
        self.d_latent = config.latent_dim
        self.eps = 1e-5

        # Projection layers
        self.p_down = nn.Linear(d_in, self.d_latent, bias=False)
        self.p_up = nn.Linear(self.d_latent, d_out, bias=False)
        self.gamma = nn.Parameter(torch.tensor(config.gamma_residual))

        # Modular components
        self.router = build_router(config)
        self.param_store = build_parameter_store(config)
        self.topology_manager = build_topology_manager(config)
        self.memory_manager = build_memory_manager(config)

        # Continuous density topology state
        if config.topology_control in ['continuous_density', 'merge_aware', 'meta_learned']:
            self.register_parameter('split_gate', nn.Parameter(torch.full((config.max_k,), 0.0)))
            self.register_buffer('activation_density', torch.zeros(config.max_k))
            self.register_buffer('error_density', torch.zeros(config.max_k))
        else:
            self.register_buffer('split_gate', torch.zeros(config.max_k))
            self.register_buffer('activation_density', torch.zeros(config.max_k))
            self.register_buffer('error_density', torch.zeros(config.max_k))

        # Self-referential growth (meta-Gaussians)
        if config.enable_self_referential:
            self._init_self_referential()

        # Cached routing info
        self._last_routing_output: Optional[RoutingOutput] = None

        # Initialize units
        if hasattr(self.router, 'initialize_units'):
            self.router.initialize_units(config.k_init)

    def _init_self_referential(self):
        """Initialize meta-Gaussians that control hyperparameters."""
        n_meta = int(self.config.max_k * self.config.meta_gaussian_ratio)
        self.register_buffer('is_meta', torch.zeros(self.config.max_k, dtype=torch.bool))
        self.is_meta[:n_meta] = True
        # Meta-Gaussians predict routing temperature, split thresholds, etc.
        self.meta_head = nn.Sequential(
            nn.Linear(self.d_latent, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # tau, split_thresh, prune_thresh, merge_thresh
        )

    def forward(self, x: torch.Tensor) -> NGSOutput:
        """Forward pass through NGS."""
        z = self.p_down(x)

        # Self-referential: meta-Gaussians predict hyperparameters
        if self.config.enable_self_referential and hasattr(self, 'meta_head'):
            self._apply_meta_control(z)

        # Route to active units
        routing_output = self.router(z)
        self._last_routing_output = routing_output

        # Apply parameter transformations
        if isinstance(routing_output.indices, list):
            # Factorized routing
            out = self._forward_factorized(z, routing_output)
        else:
            # Standard routing
            out = self._forward_standard(z, routing_output)

        # Track activation density
        self._update_activation_density(routing_output)

        # Up projection with residual
        logits = self.p_up(out + self.gamma * z)

        return NGSOutput(logits=logits, routing=routing_output, latent=z)

    def _forward_standard(self, z: torch.Tensor, routing: RoutingOutput) -> torch.Tensor:
        local_out = self.param_store(routing.indices, z)
        weights = routing.weights.unsqueeze(-1)
        blended = (weights * local_out).sum(dim=1)
        return blended

    def _forward_factorized(self, z: torch.Tensor, routing: RoutingOutput) -> torch.Tensor:
        """Forward for factorized routing with per-subspace parameter stores."""
        num_subspaces = len(routing.indices)
        all_activations = []

        for s in range(num_subspaces):
            indices = routing.indices[s]
            weights = routing.weights[s]

            # Use global parameter store with global indices
            local_out = self.param_store(indices, z)
            w = weights.unsqueeze(-1)
            subspace_out = (w * local_out).sum(dim=1)
            all_activations.append(subspace_out)

        # Combine across subspaces (mean by default)
        combined = torch.stack(all_activations, dim=1).mean(dim=1)
        return combined

    def _update_activation_density(self, routing: RoutingOutput) -> None:
        if self.config.topology_control not in ['continuous_density', 'merge_aware', 'meta_learned']:
            return

        if hasattr(self.router, 'active_mask'):
            active_idx = self.router.active_mask.nonzero(as_tuple=True)[0]
            if active_idx.numel() > 0:
                with torch.no_grad():
                    act = torch.zeros(self.config.max_k, device=self.activation_density.device)
                    if isinstance(routing.indices, list):
                        for idx in routing.indices:
                            act[idx.flatten()] = 1.0
                    else:
                        act[routing.indices.flatten()] = 1.0
                    self.activation_density = (
                        self.config.density_decay * self.activation_density
                        + (1 - self.config.density_decay) * act
                    )

    def _apply_meta_control(self, z: torch.Tensor) -> None:
        """Apply meta-Gaussian control over hyperparameters."""
        if not hasattr(self, 'meta_head'):
            return

        # Use mean latent to predict hyperparameters
        z_mean = z.mean(dim=0)
        meta_params = self.meta_head(z_mean)
        tau, split_thresh, prune_thresh, merge_thresh = meta_params.unbind(-1)

        # Apply with sigmoid to keep in valid ranges
        self.router.tau.data = torch.sigmoid(tau) * 2.0 + 0.1
        # Could update topology manager thresholds similarly

    def entropy_loss(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute entropy regularization loss."""
        if x is not None:
            z = self.p_down(x)
        elif self._last_routing_output is not None:
            # Use cached routing
            pass
        else:
            return torch.tensor(0.0, device=self.p_down.weight.device)

        routing = self.router(z)

        if isinstance(routing.weights, list):
            entropy = 0.0
            for w in routing.weights:
                p = w
                entropy += -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
            return entropy / len(routing.weights)
        else:
            p = routing.weights
            return -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()

    def diversity_loss(self) -> torch.Tensor:
        """Push Gaussian means apart to encourage coverage."""
        if not hasattr(self.router, 'active_mask') and not hasattr(self.router, 'coarse_active'):
            return torch.tensor(0.0, device=self.p_down.weight.device)

        if hasattr(self.router, 'active_mask'):
            active_idx = self.router.active_mask.nonzero(as_tuple=True)[0]
            if len(active_idx) < 2:
                return torch.tensor(0.0, device=self.p_down.weight.device)
            mu = self.router.mu[active_idx]
        elif hasattr(self.router, 'coarse_active'):
            # Hierarchical: use fine units
            active_idx = self.router.fine_active.nonzero(as_tuple=True)[0]
            if len(active_idx) < 2:
                return torch.tensor(0.0, device=self.p_down.weight.device)
            mu = self.router.fine_mu[active_idx]

        dist = torch.cdist(mu, mu, p=2)
        mask = ~torch.eye(len(active_idx), dtype=torch.bool, device=mu.device)
        min_dist = dist[mask].min()
        return -min_dist

    def split_gate_loss(self) -> torch.Tensor:
        """Split gate regularization for continuous density."""
        sig = torch.sigmoid(self.split_gate)
        err = self.error_density / (self.error_density.max() + 1e-8)
        return F.binary_cross_entropy(sig, err.detach())

    def update_unit_errors(self, logits: torch.Tensor, targets: torch.Tensor, decay: float = 0.99) -> None:
        """Update per-unit error density from training loss."""
        if self._last_routing_output is None:
            return

        with torch.no_grad():
            per_sample_loss = F.cross_entropy(logits, targets, reduction='none')
            routing = self._last_routing_output

            if isinstance(routing.indices, list):
                # Factorized: flatten
                all_indices = torch.cat(routing.indices, dim=1)
                all_weights = torch.cat(routing.weights, dim=1)
            else:
                all_indices = routing.indices
                all_weights = routing.weights

            # Weighted loss attribution
            unit_loss = (all_weights * per_sample_loss.unsqueeze(-1)).mean(dim=0)

            for k in range(all_indices.shape[1]):
                global_idx = all_indices[:, k]
                unique_idx = global_idx.unique()
                for uid in unique_idx:
                    mask = global_idx == uid
                    self.error_density[uid] = (
                        decay * self.error_density[uid]
                        + (1 - decay) * unit_loss[k].item()
                    )

    def adapt_density(self, z_samples: Optional[torch.Tensor] = None, **kwargs) -> TopologyAction:
        """Apply topology management."""
        result = self.topology_manager.adapt_topology(self, z_samples=z_samples, **kwargs)

        # Enforce memory capacity
        if self.config.memory_management == MemoryManagement.STRICT_CAPACITY:
            pruned = self.memory_manager.enforce_capacity(self)
            if pruned > 0:
                result = TopologyAction(
                    num_pruned=result.num_pruned + pruned,
                    num_split=result.num_split,
                    num_spawned=result.num_spawned,
                    num_merged=result.num_merged,
                    merged_indices=result.merged_indices
                )

        return result

    def compute_topology_losses(self) -> Dict[str, torch.Tensor]:
        """Compute all topology-related losses."""
        return self.topology_manager.compute_losses(self)

    @property
    def K(self) -> int:
        return self.router.num_active_units

    def get_active_units(self) -> torch.Tensor:
        """Get indices of active units."""
        if hasattr(self.router, 'active_mask'):
            return self.router.active_mask.nonzero(as_tuple=True)[0]
        elif hasattr(self.router, 'fine_active'):
            return self.router.fine_active.nonzero(as_tuple=True)[0]
        return torch.tensor([], dtype=torch.long, device=self.p_down.weight.device)


def build_ngs(d_in: int, d_out: int, config: NGSConfig) -> NGSModel:
    """Factory function to build NGS model."""
    model = NGSModel(d_in, d_out, config)
    return model

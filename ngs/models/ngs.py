"""Unified NGS Model integrating all modular components."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from types import SimpleNamespace
from pathlib import Path
import json

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.modules.routers import build_router
from ngs.modules.parameter_stores import build_parameter_store
from ngs.modules.topology_managers import build_topology_manager
from ngs.modules.memory_managers import build_memory_manager


class NGSModel(nn.Module):
    """
    Neural Gaussian System (NGS) model.
    
    Composable, configuration-driven framework combining:
    - Router (routing strategy)
    - Parameter Store (parameter storage strategy)
    - Topology Manager (dynamic topology adaptation)
    - Memory Manager (capacity management)
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
        
        # Core components
        self.param_store = build_parameter_store(config)
        self.router = build_router(config)
        self.topology_manager = build_topology_manager(config)
        self.memory_manager = build_memory_manager(config)
        
        # Per-subspace parameter stores for FactorizedRouter
        if config.routing == RoutingStrategy.FACTORIZED_SUBSPACE:
            self.param_stores_per_subspace = self._build_param_stores_per_subspace()
        else:
            self.param_stores_per_subspace = None
        
        # Cached routing output
        self._last_routing_output = None
        self._last_active_indices = None
        self._last_routing_weights = None
        
        # Continuous density tracking
        self.register_parameter('split_gate', nn.Parameter(torch.full((config.max_k,), config.gamma_residual)))
        self.register_buffer('activation_density', torch.zeros(config.max_k))
        self.register_buffer('error_density', torch.zeros(config.max_k))
        
        # Initialize router units (lazy data-dependent init on first forward)
        self._router_initialized = False
        self._k_init = config.k_init
    
    def _build_param_stores_per_subspace(self):
        """Build parameter stores for each subspace (used by FactorizedRouter)."""
        num_subspaces = self.config.num_subspaces
        units_per_space = self.config.max_k // self.config.num_subspaces
        
        param_stores = []
        for s in range(num_subspaces):
            from ngs.modules.parameter_stores import DirectAdapterStore, HypernetworkStore, LoRAStore
            storage = self.config.parameter_storage
            
            if storage == ParameterStorage.DIRECT_ADAPTER:
                store = DirectAdapterStore(self.config)
                # Override max_k for per-subspace store
                store.max_k = units_per_space
            elif storage == ParameterStorage.HYPERNETWORK_GENERATED:
                store = HypernetworkStore(self.config)
                store.max_k = units_per_space
            else:
                store = LoRAStore(self.config)
                store.max_k = units_per_space
            
            param_stores.append(store)
        
        return nn.ModuleList(param_stores)
    
    @property
    def K(self) -> int:
        """Number of active units."""
        if not self._router_initialized:
            return self._k_init
        if hasattr(self.router, 'active_mask'):
            return self.router.active_mask.sum().item()
        if hasattr(self.router, 'K'):
            return self.router.K
        return self.config.max_k
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through NGS.
        
        Args:
            x: [B, d_in] input features
            
        Returns:
            output object with logits and routing info
        """
        z = self.p_down(x)  # [B, d]
        
        # Lazy data-dependent router initialization
        if not self._router_initialized and hasattr(self.router, 'initialize_units'):
            self.router.initialize_units(self._k_init, z)
            self._router_initialized = True
        
        # Route to active units
        routing_output = self.router(z)
        
        # Cache routing output
        self._last_routing_output = SimpleNamespace(latent=z, indices=None, weights=None)
        
        # Process routing output
        if hasattr(routing_output, 'indices') and hasattr(routing_output, 'weights'):
            # Standard routing
            if routing_output.level_indices is not None and routing_output.level_weights is not None:
                # Factorized routing
                out = self._forward_factorized(z, routing_output.level_indices, routing_output.level_weights)
                flat_indices = torch.cat(routing_output.level_indices, dim=1)
                flat_weights = torch.cat(routing_output.level_weights, dim=1)
                self._last_active_indices = flat_indices
                self._last_routing_weights = flat_weights
                self._last_routing_output.indices = flat_indices
                self._last_routing_output.weights = flat_weights
            else:
                # Monolithic routing
                out = self._forward_standard(z, routing_output.indices, routing_output.weights)
                self._last_active_indices = routing_output.indices
                self._last_routing_weights = routing_output.weights
                self._last_routing_output.indices = routing_output.indices
                self._last_routing_output.weights = routing_output.weights
        else:
            # Fallback for older router formats
            if isinstance(routing_output, tuple):
                active_indices, routing_weights = routing_output
                out = self._forward_standard(z, active_indices, routing_weights)
                self._last_active_indices = active_indices
                self._last_routing_weights = routing_weights
        
        # Density tracking for continuous density topology
        if self.config.topology_control == TopologyControl.CONTINUOUS_DENSITY and hasattr(self.router, 'active_mask'):
            active_idx = self.router.active_mask.nonzero(as_tuple=True)[0]
            if active_idx.numel() > 0:
                with torch.no_grad():
                    act = torch.zeros(self.config.max_k, device=out.device)
                    act[active_idx] = 1.0
                    self.activation_density = (
                        self.config.ema_decay * self.activation_density
                        + (1 - self.config.ema_decay) * act
                    )
        
        # Return structured output
        return SimpleNamespace(
            logits=out,
            routing_output=routing_output,
            latent=z
        )
    
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
        
        all_activations = []
        for s in range(num_subspaces):
            global_indices = subspace_indices[s]  # [B, top_k]
            weights = subspace_weights[s]    # [B, top_k]
            
            # Use per-subspace parameter store if available
            if self.param_stores_per_subspace is not None:
                param_store = self.param_stores_per_subspace[s]
                # Convert global to local indices
                if hasattr(self.router, 'units_per_space'):
                    units_per_space = self.router.units_per_space
                    local_indices = global_indices - s * units_per_space
                else:
                    local_indices = global_indices
                local_out = param_store(local_indices, z)  # [B, top_k, d]
            else:
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
            z = self._last_routing_output.latent
        else:
            return torch.tensor(0.0, device=self.p_down.weight.device)
        
        routing_output = self.router(z)
        
        if hasattr(routing_output, 'weights'):
            if routing_output.level_weights is not None:
                # Factorized routing: weight entropy by active units per subspace
                entropy = 0.0
                weights_list = []
                for weights in routing_output.level_weights:
                    if weights.sum() > 0:
                        p = weights / (weights.sum(dim=-1, keepdim=True).clamp(min=1e-8))
                        weights_list.append(p)
                num_levels = len(weights_list)
                if num_levels > 0:
                    for p in weights_list:
                        entropy += -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
                    return entropy / num_levels
                return torch.tensor(0.0, device=self.p_down.weight.device)
            else:
                # Monolithic routing
                weights = routing_output.weights
                if weights.sum() > 0:
                    p = weights / (weights.sum(dim=-1, keepdim=True).clamp(min=1e-8))
                    return -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
        
        return torch.tensor(0.0, device=self.p_down.weight.device)
    
    def update_unit_errors(self, logits: torch.Tensor, targets: torch.Tensor,
                          decay: float = 0.99) -> None:
        """Update per-unit error density from training loss."""
        if self._last_active_indices is None or self._last_routing_weights is None:
            return
        
        with torch.no_grad():
            per_sample_loss = F.cross_entropy(logits, targets, reduction='none')  # [B]
            weights = self._last_routing_weights  # [B, K]
            active_idx = self._last_active_indices  # [B, K]
            
            # Weighted loss attribution
            unit_loss = (weights * per_sample_loss.unsqueeze(-1)).mean(dim=0)  # [K]
            
            # Map back to global indices and update with EMA
            for k in range(active_idx.shape[1]):
                global_idx = active_idx[:, k]
                unique_idx = global_idx.unique()
                for uid in unique_idx:
                    mask = global_idx == uid
                    self.error_density[uid] = (
                        decay * self.error_density[uid]
                        + (1 - decay) * unit_loss[k].item()
                    )
    
    def adapt_density(self, **kwargs) -> Tuple[int, int, int]:
        """
        Apply topology management (split/prune/spawn), respecting memory strategy.
        
        Returns:
            Tuple of (num_pruned, num_split, num_spawned)
        """
        result = self.topology_manager.adapt_topology(self, **kwargs)
        
        # Enforce memory capacity
        num_pruned_memory = self.memory_manager.enforce_capacity(self)
        
        if num_pruned_memory > 0:
            result = (result[0] + num_pruned_memory, result[1], result[2])
        
        return result
    
    def split_gate_loss(self) -> torch.Tensor:
        """Compute regularization loss for continuous density split gates."""
        sig = torch.sigmoid(self.split_gate)
        # Normalize error density to [0, 1] as target
        err = self.error_density / (self.error_density.max() + 1e-8)
        # Clamp targets slightly to avoid log(0) in BCE if implemented custom later
        err = err.clamp(min=1e-6, max=1.0 - 1e-6)
        # Use sigmoid output matching for BCE (both in [0,1])
        return F.binary_cross_entropy(sig, err.detach())
    
    def diversity_loss(self) -> torch.Tensor:
        """Push Gaussian means apart to encourage coverage."""
        router = self.router
        if not hasattr(router, 'active_mask') or not hasattr(router, 'mu'):
            return torch.tensor(0.0, device=self.p_down.weight.device)

        active_idx = router.active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) < 2:
            return torch.tensor(0.0, device=self.p_down.weight.device)

        # FactorizedRouter: mu is (num_subspaces, units_per_space, d_sub)
        if hasattr(router, 'num_subspaces') and hasattr(router, 'units_per_space'):
            losses = []
            for s in range(router.num_subspaces):
                start = s * router.units_per_space
                end = start + router.units_per_space
                sub_active = router.active_mask[start:end].nonzero(as_tuple=True)[0]
                if len(sub_active) < 2:
                    continue
                mu_s = router.mu[s][sub_active]
                dist = torch.cdist(mu_s, mu_s, p=2)
                eye = ~torch.eye(len(sub_active), dtype=torch.bool, device=mu_s.device)
                if eye.any():
                    # Repel all close pairs (softmin) to provide smoother gradients
                    close_dists = dist[eye]
                    if len(close_dists) > 0:
                        soft_min = -torch.logsumexp(-close_dists, dim=0)
                        losses.append(-soft_min)
            if not losses:
                return torch.tensor(0.0, device=self.p_down.weight.device)
            return torch.stack(losses).mean()

        # Monolithic / GaussianAttention / UncertaintyAware: mu is (max_k, latent_dim)
        mu = router.mu[active_idx]
        dist = torch.cdist(mu, mu, p=2)
        mask = ~torch.eye(len(active_idx), dtype=torch.bool, device=mu.device)
        if not mask.any():
            return torch.tensor(0.0, device=self.p_down.weight.device)
        close_dists = dist[mask]
        soft_min = -torch.logsumexp(-close_dists, dim=0)
        return -soft_min
    
    def compute_topology_losses(self) -> Dict[str, torch.Tensor]:
        """Compute all topology-related losses."""
        losses = {}
        losses['entropy'] = self.entropy_loss(torch.randn(1, self.d_in, device=self.p_down.weight.device))
        losses['diversity'] = self.diversity_loss()
        if self.config.topology_control == TopologyControl.CONTINUOUS_DENSITY:
            losses['split_gate'] = self.split_gate_loss()
        return losses
    
    def expand_capacity(self, new_max_k: int):
        """Expand model capacity."""
        self.memory_manager.expand_buffers(self, new_max_k)
        self.config.max_k = new_max_k

    def save_checkpoint(self, path: str | Path, optimizer: torch.optim.Optimizer | None = None, 
                        epoch: int | None = None, metadata: dict | None = None) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__,
            'd_in': self.d_in,
            'd_out': self.d_out,
            'epoch': epoch,
            'metadata': metadata or {},
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = 'cpu', 
                        optimizer: torch.optim.Optimizer | None = None,
                        strict: bool = True) -> tuple['NGSModel', dict]:
        """Load model checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        config = NGSConfig(**checkpoint['config'])
        model = cls(checkpoint['d_in'], checkpoint['d_out'], config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        metadata = checkpoint.get('metadata', {})
        epoch = checkpoint.get('epoch')
        
        return model, {'epoch': epoch, 'metadata': metadata}

    def export_onnx(self, path: str | Path, input_shape: tuple = (1, 784), 
                    opset_version: int = 17) -> None:
        """Export model to ONNX format.
        
        Raises:
            RuntimeError: If ONNX export fails (e.g., due to PyTorch version incompatibility).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.eval()
        dummy_input = torch.randn(*input_shape)
        
        try:
            torch.onnx.export(
                self,
                dummy_input,
                path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['logits'],
                dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
            )
        except Exception as e:
            raise RuntimeError(
                f"ONNX export failed. This may be due to PyTorch version incompatibility "
                f"(detected: {torch.__version__}). Error: {e}"
            ) from e


def build_ngs(d_in: int, d_out: int, config: NGSConfig) -> NGSModel:
    """
    Factory function to build NGS from configuration.
    
    Args:
        d_in: Input dimension
        d_out: Output dimension
        config: NGSConfig instance
        
    Returns:
        Configured NGS model
    """
    return NGSModel(d_in, d_out, config)


__all__ = [
    'NGSModel',
    'build_ngs',
]

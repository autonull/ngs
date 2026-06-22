"""Composable NGSLayer — a drop-in replacement for nn.Linear with Gaussian routing.

Architecture (from TODO2.md):
    z = self.input_proj(x)         # project input to latent
    routing = self.router(z)       # sparse gate with top-K
    out = self.experts(z, routing) # expert mixture d_latent -> d_out
    if self.residual:
        out = out + self.norm(x)   # bypass with normalized original signal

All components are optional/configurable. The layer can be stacked to form deep NGS networks.
Key insight: residual keeps a direct gradient path from input to each layer,
solving the gradient collapse observed in deep NGS architectures.
"""
from __future__ import annotations
from typing import Optional, List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl
from ngs.modules.routers import MonolithicRouter, BaseRouter
from ngs.modules.topology_managers import HeuristicManager


class MultiHeadProj(nn.Module):
    """Multi-head input projection.

    M independent heads each project the full input into a latent subspace.
    Router sees the concatenation of all head outputs, enabling richer routing.

    Each head receives a full gradient (not gated by router), so all heads learn
    useful features independently. Critical fix for CIFAR-scale inputs where
    a single projection receives only sparse gradient from top-K router.
    """

    def __init__(self, d_in: int, d_per_head: int, n_heads: int = 4, bias: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.d_per_head = d_per_head
        self.total_latent = n_heads * d_per_head
        self.heads = nn.ModuleList([
            nn.Linear(d_in, d_per_head, bias=bias) for _ in range(n_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([h(x) for h in self.heads], dim=-1)

    def extra_repr(self) -> str:
        return f'n_heads={self.n_heads}, d_per_head={self.d_per_head}, total_latent={self.total_latent}'


class NGSLayer(nn.Module):
    """Composable NGSLayer — replaces nn.Linear with Gaussian routing.

    The expert mixture directly maps d_latent -> d_out (no separate output projection),
    matching the TODO2 Phase 1 specification exactly.

    Forward: [B, d_in] -> [B, d_out]

    Configurable components:
    - input_proj: Linear or MultiHeadProj (n_heads > 1)
    - router: Gaussian mixture with top-K sparsity
    - experts: Direct weight matrix [n_experts, d_latent, d_out]
    - norm: LayerNorm on residual path (prevents collapse)
    - residual: identity shortcut when d_in == d_out
    """

    def __init__(
        self,
        d_in: int,
        d_latent: int,
        d_out: int,
        n_experts: int = 256,
        n_heads: int = 1,
        top_k: int = 8,
        use_residual: bool = True,
        use_norm: bool = True,
        tau: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_latent = d_latent
        self.d_out = d_out
        self.n_experts = n_experts
        self.use_residual = use_residual and (d_in == d_out)
        self.use_norm = use_norm
        self.top_k = top_k
        self.n_heads = n_heads

        # Input projection
        if n_heads > 1:
            self.input_proj = MultiHeadProj(d_in, d_latent, n_heads)
            effective_latent = n_heads * d_latent
        else:
            self.input_proj = nn.Linear(d_in, d_latent, bias=False)
            effective_latent = d_latent

        # Norm for residual path (only used when residual is active)
        if use_norm:
            self.norm = nn.LayerNorm(d_in)
        else:
            self.norm = nn.Identity()

        # Build NGSConfig for router
        self._config = NGSConfig(
            latent_dim=effective_latent,
            max_k=n_experts,
            k_init=n_experts,
            top_k=top_k,
            tau=tau,
        )

        # Gaussian mixture router
        self.router = MonolithicRouter(self._config)

        # Re-init router means with smaller scale so distances are input-driven
        # Default MonolithicRouter init uses N(0,1) which makes pairwise distances
        # ~sqrt(2*d) — this drowns input differences and causes nearly identical routing
        nn.init.normal_(self.router.mu, mean=0.0, std=0.1)

        # Expert weights: directly map effective_latent -> d_out
        # Xavier-style init: std = sqrt(2 / (d_latent + d_out))
        xavier_std = (2.0 / (effective_latent + d_out)) ** 0.5
        self.expert_weights = nn.Parameter(
            torch.randn(n_experts, effective_latent, d_out) * xavier_std * 1.5
        )

        # Output bias: baseline logit per class, critical when routing is random
        self.out_bias = nn.Parameter(torch.zeros(d_out))

        # Learnable residual gate
        self.residual_gate = nn.Parameter(torch.tensor(1.0))

        # Ablation flags (for controlled experiments)
        self._ablation_remove_out_bias = kwargs.get('_remove_out_bias', False)
        self._ablation_router_mu_std = kwargs.get('_router_mu_std', 0.1)

        # Re-init router means with ablation-specified std
        nn.init.normal_(self.router.mu, mean=0.0, std=self._ablation_router_mu_std)

        # Enable all units from the start
        self.router.active_mask[:] = True

        # Topology manager (optional, off by default)
        self._topology_manager = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_in] input features
        Returns:
            [B, d_out] output features
        """
        # Input projection to latent space
        z = self.input_proj(x)  # [B, effective_latent]

        # Route to top-K experts
        routing = self.router(z)  # indices [B, K], weights [B, K]

        # Expert weights for selected experts
        W_k = self.expert_weights[routing.indices]  # [B, K, d_latent, d_out]

        # Apply expert transformations: z @ W_k for each expert
        expert_out = torch.einsum('bd,bkdo->bko', z, W_k)  # [B, K, d_out]

        # Blend weighted experts
        w = routing.weights.unsqueeze(-1)  # [B, K, 1]
        out = (w * expert_out).sum(dim=1)  # [B, d_out]

        # Output bias (baseline per-class) — ablatable
        if not self._ablation_remove_out_bias:
            out = out + self.out_bias

        # Residual connection (when d_in == d_out)
        if self.use_residual:
            out = out + self.residual_gate * self.norm(x)

        return out

    @property
    def K(self) -> int:
        """Number of active experts."""
        return self.router.K if hasattr(self.router, 'K') else 0

    def enable_topology_adaptation(self, split_thresh: float = 0.05, prune_thresh: float = 0.01):
        """Enable topology management (split/prune/spawn) on this layer."""
        config = NGSConfig(
            latent_dim=self._config.latent_dim,
            max_k=self._config.max_k,
            split_threshold=split_thresh,
            prune_threshold=prune_thresh,
            topology_control=TopologyControl.DISCRETE_HEURISTIC,
        )
        self._topology_manager = HeuristicManager(config)

    def adapt_density(self, **kwargs) -> Tuple[int, int, int]:
        """Apply topology adaptation. Returns (pruned, split, spawned)."""
        if self._topology_manager is not None:
            return self._topology_manager.adapt_topology(self, **kwargs)
        return 0, 0, 0

    def extra_repr(self) -> str:
        effective = self.n_heads * self.d_latent if self.n_heads > 1 else self.d_latent
        return (
            f'd_in={self.d_in}, d_latent={self.d_latent}'
            f'{"(x" + str(self.n_heads) + ")" if self.n_heads > 1 else ""}'
            f'={effective}, d_out={self.d_out}, '
            f'n_experts={self.n_experts}, top_k={self.top_k}, '
            f'residual={self.use_residual}, norm={self.use_norm}'
        )


class StackedNGSModel(nn.Module):
    """Deep NGS model composed of stacked NGSLayers.

    Supports arbitrary depth with configurable per-layer dimensions.
    Automatically enables residual on layers where d_in == d_out.

    Examples:
        StackedNGSModel(3072, 10, n_layers=2, d_latent=128)
        -> NGSLayer(3072->128) -> NGSLayer(128->10)

        StackedNGSModel(3072, 10, n_layers=3, d_latent=128)
        -> NGSLayer(3072->128) -> NGSLayer(128->128) -> NGSLayer(128->10)
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        layer_dims: Optional[List[int]] = None,
        n_layers: int = 2,
        d_latent: int = 128,
        n_experts: int = 256,
        n_heads: int = 1,
        top_k: int = 8,
        use_residual: bool = True,
        use_norm: bool = True,
        tau: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        # Build layer dimensions
        if layer_dims is not None:
            dims = layer_dims
        else:
            if n_layers <= 1:
                dims = [d_in, d_out]
            else:
                dims = [d_in] + [d_latent] * (n_layers - 1) + [d_out]

        self.layer_dims = dims
        self.num_layers = len(dims) - 1

        layers = []
        for i in range(self.num_layers):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            layers.append(NGSLayer(
                d_in=in_dim,
                d_latent=d_latent if i < self.num_layers - 1 else min(d_latent, out_dim),
                d_out=out_dim,
                n_experts=n_experts,
                n_heads=n_heads,
                top_k=top_k,
                use_residual=use_residual,
                use_norm=use_norm,
                tau=tau,
                **kwargs,
            ))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def K(self) -> int:
        return sum(l.K for l in self.layers)

    def get_layer_metrics(self) -> Dict[str, int]:
        return {f'layer_{i}_K': l.K for i, l in enumerate(self.layers)}

    def enable_topology_adaptation(self, **kwargs):
        for l in self.layers:
            l.enable_topology_adaptation(**kwargs)

    def adapt_density_all(self, **kwargs) -> List[Tuple[int, int, int]]:
        return [l.adapt_density(**kwargs) for l in self.layers]


def build_stacked_ngs(
    d_in: int,
    d_out: int,
    n_layers: int = 2,
    d_latent: int = 128,
    n_experts: int = 256,
    n_heads: int = 1,
    top_k: int = 8,
    use_residual: bool = True,
    use_norm: bool = True,
    **kwargs,
) -> StackedNGSModel:
    """Factory function for StackedNGSModel.

    Args:
        d_in: Input dimension
        d_out: Output dimension (number of classes)
        n_layers: Number of NGSLayers to stack
        d_latent: Latent dimension per layer
        n_experts: Max experts per layer
        n_heads: Multi-head projection heads (1 = single head)
        top_k: Active experts per sample
        use_residual: Enable residual connections on matching layers
        use_norm: Enable LayerNorm on residual path
        **kwargs: Passed to NGSLayer (e.g., ablation flags)
    """
    return StackedNGSModel(
        d_in=d_in, d_out=d_out, n_layers=n_layers, d_latent=d_latent,
        n_experts=n_experts, n_heads=n_heads, top_k=top_k,
        use_residual=use_residual, use_norm=use_norm,
        **kwargs,
    )


__all__ = [
    "MultiHeadProj",
    "NGSLayer",
    "StackedNGSModel",
    "build_stacked_ngs",
]

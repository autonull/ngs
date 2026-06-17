"""Router implementations for NGS."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
from ngs.core.interfaces import NGSConfig, RoutingStrategy, RoutingOutput, BaseRouter


class MonolithicRouter(BaseRouter):
    """Monolithic Mahalanobis routing (original LeanNGS)."""

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.top_k = config.top_k
        self.tau = nn.Parameter(torch.tensor(config.tau))
        self.eps = 1e-5
        
        self.mu = nn.Parameter(torch.randn(config.max_k, config.latent_dim) * 1.0)
        self.log_s = nn.Parameter(torch.zeros(config.max_k, config.latent_dim))
        self.log_alpha = nn.Parameter(torch.zeros(config.max_k))
        
        self.register_buffer('active_mask', torch.zeros(config.max_k, dtype=torch.bool))
        self.register_buffer('grad_mu_ema', torch.zeros(config.max_k))
        self.ema_decay = config.ema_decay
        self.mu.register_hook(self._update_mu_grad_ema)

    @property
    def K(self) -> int:
        return self.active_mask.sum().item()

    def initialize_units(self, k_init: int):
        self.active_mask[:k_init] = True

    def forward(self, z: torch.Tensor) -> RoutingOutput:
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        mu = self.mu[active_idx]
        log_s = self.log_s[active_idx]
        log_alpha = self.log_alpha[active_idx]
        
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        s_sq = torch.exp(2 * log_s) + self.eps
        mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1)
        
        log_w = log_alpha - (0.5 / self.tau) * mahalanobis_sq
        k_actual = min(self.top_k, self.K)
        topk_vals, topk_rel_idx = torch.topk(log_w, k_actual, dim=-1)
        topk_idx = active_idx[topk_rel_idx]
        topk_weights = F.softmax(topk_vals, dim=-1)
        
        return RoutingOutput(indices=topk_idx, weights=topk_weights)

    def _update_mu_grad_ema(self, grad):
        if self.active_mask.any():
            active_mask = self.active_mask
            grad_mag = grad.norm(dim=-1)
            self.grad_mu_ema[active_mask] = (
                self.ema_decay * self.grad_mu_ema[active_mask]
                + (1 - self.ema_decay) * grad_mag[active_mask]
            )
        return grad


class FactorizedRouter(BaseRouter):
    """Factorized subspace routing (CFG-Net)."""

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.num_subspaces = config.num_subspaces
        self.units_per_space = config.max_k // config.num_subspaces
        self.top_k = config.top_k = config.top_k_factorized
        self.tau = nn.Parameter(torch.tensor(config.tau))
        self.eps = 1e-5
        
        total_units = self.num_subspaces * self.units_per_space
        d_sub = max(config.latent_dim // self.num_subspaces, 1)
        self.d_sub = d_sub
        self.subspace_projectors = nn.ModuleList([
            nn.Linear(config.latent_dim, d_sub, bias=False)
            for _ in range(self.num_subspaces)
        ])
        
        self.mu = nn.Parameter(torch.randn(self.num_subspaces, self.units_per_space, d_sub) * 1.0)
        self.log_s = nn.Parameter(torch.zeros(self.num_subspaces, self.units_per_space, d_sub))
        self.log_alpha = nn.Parameter(torch.zeros(self.num_subspaces, self.units_per_space))
        
        self.register_buffer('active_mask', torch.zeros(total_units, dtype=torch.bool))
        self.register_buffer('grad_mu_ema', torch.zeros(total_units))

    @property
    def K(self) -> int:
        return self.active_mask.sum().item()

    def initialize_units(self, k_init: int):
        active_per_space = -(-k_init // self.num_subspaces)
        for s in range(self.num_subspaces):
            start = s * self.units_per_space
            self.active_mask[start:start + active_per_space] = True

    def forward(self, z: torch.Tensor) -> RoutingOutput:
        B = z.shape[0]
        all_indices = []
        all_weights = []
        
        for s in range(self.num_subspaces):
            start = s * self.units_per_space
            end = start + self.units_per_space
            
            sub_active = self.active_mask[start:end]
            active_local = sub_active.nonzero(as_tuple=True)[0]
            
            if len(active_local) == 0:
                all_indices.append(torch.zeros(B, self.top_k, dtype=torch.long, device=z.device))
                all_weights.append(torch.zeros(B, self.top_k, device=z.device))
                continue
            
            z_s = self.subspace_projectors[s](z)
            mu_s = self.mu[s][active_local]
            log_s_s = self.log_s[s][active_local]
            log_alpha_s = self.log_alpha[s][active_local]
            
            diff = z_s.unsqueeze(1) - mu_s.unsqueeze(0)
            s_sq = torch.exp(2 * log_s_s) + self.eps
            mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1)
            
            log_w = log_alpha_s - (0.5 / self.tau) * mahalanobis_sq
            k_actual = min(self.top_k, len(active_local))
            topk_vals, topk_rel_idx = torch.topk(log_w, k_actual, dim=-1)
            topk_weights = F.softmax(topk_vals, dim=-1)
            
            topk_global = active_local[topk_rel_idx] + start
            all_indices.append(topk_global)
            all_weights.append(topk_weights)
        
        flat_indices = torch.cat(all_indices, dim=1)
        flat_weights = torch.cat(all_weights, dim=1)
        
        return RoutingOutput(
            indices=flat_indices, 
            weights=flat_weights,
            level_indices=all_indices,
            level_weights=all_weights
        )


class LSRRouter(BaseRouter):
    """Locality-Sensitive Hashing router (extensibility hook)."""

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.num_buckets = config.max_k // 4
        self.num_hash_functions = 4
        self.top_k = config.top_k
        
        self.hash_projections = nn.Parameter(
            torch.randn(self.num_hash_functions, config.latent_dim, self.num_buckets)
        )
        self.register_buffer('active_mask', torch.ones(self.num_buckets, dtype=torch.bool))

    @property
    def K(self) -> int:
        return self.active_mask.sum().item()

    def initialize_units(self, k_init: int):
        self.active_mask[:min(k_init, self.num_buckets)] = True

    def forward(self, z: torch.Tensor) -> RoutingOutput:
        B = z.shape[0]
        signatures = []
        for h in range(self.num_hash_functions):
            proj = z @ self.hash_projections[h]
            sig = torch.argmax(proj, dim=-1)
            signatures.append(sig)
        signatures = torch.stack(signatures, dim=-1)
        
        pseudo_dist = torch.randn(B, self.num_buckets, device=z.device)
        topk_vals, topk_idx = torch.topk(pseudo_dist, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)
        
        return RoutingOutput(indices=topk_idx, weights=topk_weights)


class HierarchicalRouter(BaseRouter):
    """Hierarchical multi-scale routing."""

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.num_levels = config.num_levels
        self.level_capacity_ratio = config.level_capacity_ratio
        self.level_top_k = config.level_top_k
        self.tau = nn.Parameter(torch.tensor(config.tau))
        self.eps = 1e-5
        
        # Compute capacities per level
        self.level_capacities = []
        remaining = config.max_k
        for l in range(self.num_levels):
            cap = int(remaining * self.level_capacity_ratio)
            self.level_capacities.append(max(cap, 1))
            remaining -= cap
        
        # Level-specific parameters - use ParameterList for proper registration
        self.level_mu = nn.ParameterList()
        self.level_log_s = nn.ParameterList()
        self.level_log_alpha = nn.ParameterList()
        
        for i, cap in enumerate(self.level_capacities):
            self.level_mu.append(nn.Parameter(torch.randn(cap, config.latent_dim) * 1.0))
            self.level_log_s.append(nn.Parameter(torch.zeros(cap, config.latent_dim)))
            self.level_log_alpha.append(nn.Parameter(torch.zeros(cap)))
            self.register_buffer(f'level_{i}_active_mask', 
                               torch.zeros(cap, dtype=torch.bool))

    @property
    def K(self) -> int:
        total = 0
        for l in range(self.num_levels):
            mask = getattr(self, f'level_{l}_active_mask')
            total += mask.sum().item()
        return total

    def initialize_units(self, k_init: int):
        per_level = -(-k_init // self.num_levels)
        for l in range(self.num_levels):
            mask = getattr(self, f'level_{l}_active_mask')
            mask[:min(per_level, len(mask))] = True

    def forward(self, z: torch.Tensor) -> RoutingOutput:
        B = z.shape[0]
        all_indices = []
        all_weights = []
        level_indices = []
        level_weights = []
        
        for l in range(self.num_levels):
            mask = getattr(self, f'level_{l}_active_mask')
            active_idx = mask.nonzero(as_tuple=True)[0]
            
            if len(active_idx) == 0:
                level_indices.append(torch.zeros(B, self.level_top_k, dtype=torch.long, device=z.device))
                level_weights.append(torch.zeros(B, self.level_top_k, device=z.device))
                continue
            
            mu = self.level_mu[l][active_idx]
            log_s = self.level_log_s[l][active_idx]
            log_alpha = self.level_log_alpha[l][active_idx]
            
            diff = z.unsqueeze(1) - mu.unsqueeze(0)
            s_sq = torch.exp(2 * log_s) + self.eps
            mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1)
            
            log_w = log_alpha - (0.5 / self.tau) * mahalanobis_sq
            k_actual = min(self.level_top_k, len(active_idx))
            topk_vals, topk_rel_idx = torch.topk(log_w, k_actual, dim=-1)
            topk_weights = F.softmax(topk_vals, dim=-1)
            topk_idx = active_idx[topk_rel_idx]
            
            level_indices.append(topk_idx)
            level_weights.append(topk_weights)
            all_indices.append(topk_idx)
            all_weights.append(topk_weights)
        
        flat_indices = torch.cat(all_indices, dim=1)
        flat_weights = torch.cat(all_weights, dim=1)
        
        return RoutingOutput(
            indices=flat_indices,
            weights=flat_weights,
            level_indices=level_indices,
            level_weights=level_weights
        )


class GaussianAttentionRouter(BaseRouter):
    """Gaussian Attention routing with Mahalanobis attention."""

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.attention_heads = config.attention_heads
        self.sparse_top_k = config.sparse_top_k
        self.tau = nn.Parameter(torch.tensor(config.tau))
        self.dropout = nn.Dropout(config.attention_dropout)
        self.eps = 1e-5
        
        self.mu = nn.Parameter(torch.randn(config.max_k, config.latent_dim) * 1.0)
        self.log_s = nn.Parameter(torch.zeros(config.max_k, config.latent_dim))
        self.log_alpha = nn.Parameter(torch.zeros(config.max_k))
        
        # Query/Key projections for attention
        self.q_proj = nn.Linear(config.latent_dim, config.latent_dim)
        self.k_proj = nn.Linear(config.latent_dim, config.latent_dim)
        
        self.register_buffer('active_mask', torch.zeros(config.max_k, dtype=torch.bool))

    @property
    def K(self) -> int:
        return self.active_mask.sum().item()

    def initialize_units(self, k_init: int):
        self.active_mask[:k_init] = True

    def forward(self, z: torch.Tensor) -> RoutingOutput:
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        
        if len(active_idx) == 0:
            B = z.shape[0]
            return RoutingOutput(
                indices=torch.zeros(B, self.sparse_top_k, dtype=torch.long, device=z.device),
                weights=torch.zeros(B, self.sparse_top_k, device=z.device)
            )
        
        mu = self.mu[active_idx]
        log_s = self.log_s[active_idx]
        log_alpha = self.log_alpha[active_idx]
        
        # Project queries and keys
        q = self.q_proj(z)  # [B, d]
        k = self.k_proj(mu)  # [K, d]
        
        # Mahalanobis attention scores
        diff = q.unsqueeze(1) - k.unsqueeze(0)  # [B, K, d]
        s_sq = torch.exp(2 * log_s) + self.eps  # [K, d]
        scores = -0.5 * ((diff ** 2) / s_sq).sum(dim=-1)  # [B, K]
        
        # Add alpha (opacity)
        scores = scores + log_alpha.unsqueeze(0)
        
        # Sparse top-k
        k_actual = min(self.sparse_top_k, self.K)
        topk_vals, topk_rel_idx = torch.topk(scores, k_actual, dim=-1)
        topk_idx = active_idx[topk_rel_idx]
        topk_weights = F.softmax(topk_vals / self.tau, dim=-1)
        topk_weights = self.dropout(topk_weights)
        
        return RoutingOutput(indices=topk_idx, weights=topk_weights)


class UncertaintyAwareRouter(BaseRouter):
    """Uncertainty-aware routing with evidential uncertainty."""

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.top_k = config.top_k
        self.tau = nn.Parameter(torch.tensor(config.tau))
        self.evidential_prior = config.evidential_prior
        self.uncertainty_weight = config.uncertainty_weight
        self.eps = 1e-5
        
        self.mu = nn.Parameter(torch.randn(config.max_k, config.latent_dim) * 1.0)
        self.log_s = nn.Parameter(torch.zeros(config.max_k, config.latent_dim))
        self.log_alpha = nn.Parameter(torch.zeros(config.max_k))
        
        # Evidential head: predicts Dirichlet parameters
        self.evidential_head = nn.Sequential(
            nn.Linear(config.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # alpha parameters for Dirichlet
        )
        
        self.register_buffer('active_mask', torch.zeros(config.max_k, dtype=torch.bool))

    @property
    def K(self) -> int:
        return self.active_mask.sum().item()

    def initialize_units(self, k_init: int):
        self.active_mask[:k_init] = True

    def forward(self, z: torch.Tensor) -> RoutingOutput:
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        
        if len(active_idx) == 0:
            B = z.shape[0]
            return RoutingOutput(
                indices=torch.zeros(B, self.top_k, dtype=torch.long, device=z.device),
                weights=torch.zeros(B, self.top_k, device=z.device),
                uncertainty=torch.ones(B, device=z.device)
            )
        
        mu = self.mu[active_idx]
        log_s = self.log_s[active_idx]
        log_alpha = self.log_alpha[active_idx]
        
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        s_sq = torch.exp(2 * log_s) + self.eps
        mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1)
        
        log_w = log_alpha - (0.5 / self.tau) * mahalanobis_sq
        k_actual = min(self.top_k, self.K)
        topk_vals, topk_rel_idx = torch.topk(log_w, k_actual, dim=-1)
        topk_idx = active_idx[topk_rel_idx]
        topk_weights = F.softmax(topk_vals, dim=-1)
        
        # Evidential uncertainty
        evidence = self.evidential_head(z)  # [B, 4]
        alpha = F.softplus(evidence) + self.evidential_prior
        total_alpha = alpha.sum(dim=-1, keepdim=True)
        uncertainty = alpha.size(-1) / total_alpha.squeeze(-1)  # [B]
        
        return RoutingOutput(
            indices=topk_idx,
            weights=topk_weights,
            uncertainty=uncertainty
        )


def build_router(config: NGSConfig) -> BaseRouter:
    """Factory function to build router from config."""
    routing = config.routing
    
    if routing == RoutingStrategy.MONOLITHIC_MAHALANOBIS:
        return MonolithicRouter(config)
    elif routing == RoutingStrategy.FACTORIZED_SUBSPACE:
        return FactorizedRouter(config)
    elif routing == RoutingStrategy.LSH_APPROXIMATE:
        return LSRRouter(config)
    elif routing == RoutingStrategy.HIERARCHICAL:
        return HierarchicalRouter(config)
    elif routing == RoutingStrategy.GAUSSIAN_ATTENTION:
        return GaussianAttentionRouter(config)
    elif routing == RoutingStrategy.UNCERTAINTY_AWARE:
        return UncertaintyAwareRouter(config)
    else:
        raise ValueError(f"Unknown routing strategy: {routing}")


__all__ = [
    "MonolithicRouter",
    "FactorizedRouter", 
    "LSRRouter",
    "HierarchicalRouter",
    "GaussianAttentionRouter",
    "UncertaintyAwareRouter",
    "build_router",
    "BaseRouter",
]

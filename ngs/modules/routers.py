"""Router implementations for NGS."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from ngs.core.interfaces import (
    BaseRouter, RoutingOutput, RoutingStrategy, NGSConfig
)


class MonolithicRouter(BaseRouter):
    """Monolithic Mahalanobis routing (original LeanNGS)."""

    def __init__(self, config: NGSConfig):
        super().__init__()
        self.config = config
        self.max_k = config.max_k
        self.d_latent = config.latent_dim
        self.top_k = config.top_k
        self.tau = nn.Parameter(torch.tensor(config.tau))
        self.eps = 1e-5

        # Gaussian unit parameters
        self.mu = nn.Parameter(torch.randn(config.max_k, config.latent_dim) * 1.0)
        self.log_s = nn.Parameter(torch.zeros(config.max_k, config.latent_dim))
        self.log_alpha = nn.Parameter(torch.zeros(config.max_k))

        # Active mask for pre-allocated memory
        self.register_buffer('active_mask', torch.zeros(config.max_k, dtype=torch.bool))

        # Gradient EMA for topology adaptation
        self.register_buffer('grad_mu_ema', torch.zeros(config.max_k))
        self.ema_decay = config.ema_decay
        self.mu.register_hook(self._update_mu_grad_ema)

    @property
    def num_active_units(self) -> int:
        return self.active_mask.sum().item()

    @property
    def max_units(self) -> int:
        return self.max_k

    def initialize_units(self, k_init: int) -> None:
        self.active_mask[:k_init] = True

    def forward(self, z: torch.Tensor) -> RoutingOutput:
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]

        if len(active_idx) == 0:
            B = z.shape[0]
            device = z.device
            return RoutingOutput(
                indices=torch.zeros(B, self.top_k, dtype=torch.long, device=device),
                weights=torch.zeros(B, self.top_k, device=device)
            )

        mu = self.mu[active_idx]
        log_s = self.log_s[active_idx]
        log_alpha = self.log_alpha[active_idx]

        # Diagonal Mahalanobis distance
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        s_sq = torch.exp(2 * log_s) + self.eps
        mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1)

        # Log-weights
        log_w = log_alpha - (0.5 / self.tau) * mahalanobis_sq

        # Top-K selection
        k_actual = min(self.top_k, len(active_idx))
        topk_vals, topk_rel_idx = torch.topk(log_w, k_actual, dim=-1)

        # Convert to global indices
        topk_idx = active_idx[topk_rel_idx]

        # Softmax weights
        topk_weights = F.softmax(topk_vals, dim=-1)

        return RoutingOutput(
            indices=topk_idx,
            weights=topk_weights,
            aux={'mahalanobis_sq': mahalanobis_sq, 'log_alpha': log_alpha}
        )

    def get_unit_params(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            'mu': self.mu[indices],
            'log_s': self.log_s[indices],
            'log_alpha': self.log_alpha[indices],
        }

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
    """Factorized subspace routing (CFG-Net style)."""

    def __init__(self, config: NGSConfig):
        super().__init__()
        self.config = config
        self.d_latent = config.latent_dim
        self.num_subspaces = config.num_subspaces
        self.units_per_space = config.max_k // config.num_subspaces
        self.top_k = config.top_k_factorized
        self.tau = nn.Parameter(torch.tensor(config.tau))
        self.eps = 1e-5

        total_units = config.num_subspaces * self.units_per_space

        # Project latent space into subspaces
        d_sub = max(config.latent_dim // config.num_subspaces, 1)
        self.d_sub = d_sub
        self.subspace_projectors = nn.ModuleList([
            nn.Linear(config.latent_dim, d_sub, bias=False)
            for _ in range(config.num_subspaces)
        ])

        # Gaussian units per subspace
        self.mu = nn.Parameter(torch.randn(config.num_subspaces, self.units_per_space, d_sub) * 1.0)
        self.log_s = nn.Parameter(torch.zeros(config.num_subspaces, self.units_per_space, d_sub))
        self.log_alpha = nn.Parameter(torch.zeros(config.num_subspaces, self.units_per_space))

        # Active mask
        self.register_buffer('active_mask', torch.ones(total_units, dtype=torch.bool))
        self.register_buffer('grad_mu_ema', torch.zeros(total_units))

    @property
    def num_active_units(self) -> int:
        return self.active_mask.sum().item()

    @property
    def max_units(self) -> int:
        return self.config.num_subspaces * self.units_per_space

    def initialize_units(self, k_init: int) -> None:
        active_per_space = -(-k_init // self.config.num_subspaces)
        self.active_mask.fill_(False)
        for s in range(self.config.num_subspaces):
            start = s * self.units_per_space
            self.active_mask[start:start + active_per_space] = True

    @property
    def flat_mu(self):
        return self.mu.view(-1, self.d_sub)

    @property
    def flat_log_s(self):
        return self.log_s.view(-1, self.d_sub)

    @property
    def flat_log_alpha(self):
        return self.log_alpha.view(-1)

    def forward(self, z: torch.Tensor) -> RoutingOutput:
        B = z.shape[0]
        all_indices = []
        all_weights = []

        for s in range(self.config.num_subspaces):
            start = s * self.units_per_space
            end = start + self.units_per_space

            sub_active = self.active_mask[start:end]
            active_local = sub_active.nonzero(as_tuple=True)[0]

            if len(active_local) == 0:
                all_indices.append(torch.zeros(B, self.top_k, dtype=torch.long, device=z.device))
                all_weights.append(torch.zeros(B, self.top_k, device=z.device))
                continue

            # Project into subspace
            z_s = self.subspace_projectors[s](z)

            mu_s = self.mu[s][active_local]
            log_s_s = self.log_s[s][active_local]
            log_alpha_s = self.log_alpha[s][active_local]

            # Diagonal Mahalanobis in subspace
            diff = z_s.unsqueeze(1) - mu_s.unsqueeze(0)
            s_sq = torch.exp(2 * log_s_s) + self.eps
            mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1)

            # Log-weights
            log_w = log_alpha_s - (0.5 / self.tau) * mahalanobis_sq

            # Top-K per subspace
            k_actual = min(self.top_k, len(active_local))
            topk_vals, topk_rel_idx = torch.topk(log_w, k_actual, dim=-1)
            topk_weights = F.softmax(topk_vals, dim=-1)

            # Convert to global flat indices
            topk_global = active_local[topk_rel_idx] + start
            all_indices.append(topk_global)
            all_weights.append(topk_weights)

        return RoutingOutput(
            indices=all_indices,
            weights=all_weights,
            aux={'subspace_indices': all_indices}
        )

    def get_unit_params(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        # indices are global flat indices
        s = indices // self.units_per_space
        local = indices % self.units_per_space
        return {
            'mu': self.mu[s, local],
            'log_s': self.log_s[s, local],
            'log_alpha': self.log_alpha[s, local],
        }


class HierarchicalRouter(BaseRouter):
    """Hierarchical coarse-to-fine routing for multi-scale representation."""

    def __init__(self, config: NGSConfig):
        super().__init__()
        self.config = config
        self.d_latent = config.latent_dim
        self.num_levels = config.num_levels
        self.coarse_units = config.coarse_units
        self.fine_units_per_coarse = config.fine_units_per_coarse
        self.top_k = config.top_k
        self.tau = nn.Parameter(torch.tensor(config.tau))
        self.eps = 1e-5

        # Level 0: Coarse routers
        self.coarse_mu = nn.Parameter(torch.randn(config.coarse_units, config.latent_dim) * 1.0)
        self.coarse_log_s = nn.Parameter(torch.zeros(config.coarse_units, config.latent_dim))
        self.coarse_log_alpha = nn.Parameter(torch.zeros(config.coarse_units))
        self.register_buffer('coarse_active', torch.zeros(config.coarse_units, dtype=torch.bool))

        # Level 1: Fine routers per coarse unit
        total_fine = config.coarse_units * config.fine_units_per_coarse
        self.fine_mu = nn.Parameter(torch.randn(total_fine, config.latent_dim) * 1.0)
        self.fine_log_s = nn.Parameter(torch.zeros(total_fine, config.latent_dim))
        self.fine_log_alpha = nn.Parameter(torch.zeros(total_fine))
        self.register_buffer('fine_active', torch.zeros(total_fine, dtype=torch.bool))

        # Grad EMA
        self.register_buffer('coarse_grad_ema', torch.zeros(config.coarse_units))
        self.register_buffer('fine_grad_ema', torch.zeros(total_fine))

    @property
    def num_active_units(self) -> int:
        return self.fine_active.sum().item()

    @property
    def max_units(self) -> int:
        return self.config.coarse_units * self.config.fine_units_per_coarse

    def initialize_units(self, k_init: int) -> None:
        # Activate coarse units first
        n_coarse = min(self.config.coarse_units, max(1, k_init // self.config.fine_units_per_coarse))
        self.coarse_active[:n_coarse] = True

        # Activate fine units under active coarse units
        for c in range(n_coarse):
            start = c * self.config.fine_units_per_coarse
            end = start + self.config.fine_units_per_coarse
            n_fine = min(self.config.fine_units_per_coarse, k_init - c * self.config.fine_units_per_coarse)
            if n_fine > 0:
                self.fine_active[start:start + n_fine] = True

    def forward(self, z: torch.Tensor) -> RoutingOutput:
        B = z.shape[0]

        # Level 0: Route to coarse units
        coarse_idx = self.coarse_active.nonzero(as_tuple=True)[0]
        if len(coarse_idx) == 0:
            return RoutingOutput(
                indices=torch.zeros(B, self.top_k, dtype=torch.long, device=z.device),
                weights=torch.zeros(B, self.top_k, device=z.device)
            )

        coarse_mu = self.coarse_mu[coarse_idx]
        coarse_log_s = self.coarse_log_s[coarse_idx]
        coarse_log_alpha = self.coarse_log_alpha[coarse_idx]

        coarse_diff = z.unsqueeze(1) - coarse_mu.unsqueeze(0)
        coarse_s_sq = torch.exp(2 * coarse_log_s) + self.eps
        coarse_mahal = ((coarse_diff ** 2) / coarse_s_sq).sum(dim=-1)
        coarse_log_w = coarse_log_alpha - (0.5 / self.tau) * coarse_mahal

        # Select top coarse units
        k_coarse = min(4, len(coarse_idx))
        coarse_topk_vals, coarse_topk_rel = torch.topk(coarse_log_w, k_coarse, dim=-1)
        coarse_topk_idx = coarse_idx[coarse_topk_rel]
        coarse_weights = F.softmax(coarse_topk_vals, dim=-1)  # [B, k_coarse]

        # Level 1: Route within selected coarse units' fine units
        all_fine_indices = []
        all_fine_weights = []

        for b in range(B):
            batch_fine_idx = []
            batch_fine_w = []
            for c_idx, c_weight in zip(coarse_topk_idx[b], coarse_weights[b]):
                if c_weight < 0.01:  # Skip very low weight coarse units
                    continue
                start = c_idx * self.config.fine_units_per_coarse
                end = start + self.config.fine_units_per_coarse
                fine_active_local = self.fine_active[start:end].nonzero(as_tuple=True)[0]
                if len(fine_active_local) == 0:
                    continue

                fine_mu = self.fine_mu[start:end][fine_active_local]
                fine_log_s = self.fine_log_s[start:end][fine_active_local]
                fine_log_alpha = self.fine_log_alpha[start:end][fine_active_local]

                fine_diff = z[b:b+1].unsqueeze(1) - fine_mu.unsqueeze(0)
                fine_s_sq = torch.exp(2 * fine_log_s) + self.eps
                fine_mahal = ((fine_diff ** 2) / fine_s_sq).sum(dim=-1)
                fine_log_w = fine_log_alpha - (0.5 / self.tau) * fine_mahal

                k_fine = min(self.top_k, len(fine_active_local))
                fine_topk_vals, fine_topk_rel = torch.topk(fine_log_w, k_fine, dim=-1)
                fine_topk_global = (fine_active_local + start)[fine_topk_rel]
                fine_topk_weights = F.softmax(fine_topk_vals, dim=-1) * c_weight

                batch_fine_idx.append(fine_topk_global)
                batch_fine_w.append(fine_topk_weights)

            if batch_fine_idx:
                combined_idx = torch.cat(batch_fine_idx, dim=-1)
                combined_w = torch.cat(batch_fine_w, dim=-1)
                # Re-normalize and take top-k
                combined_w = F.softmax(torch.log(combined_w + 1e-8), dim=-1)
                final_k = min(self.top_k, combined_idx.shape[-1])
                final_vals, final_rel = torch.topk(combined_w, final_k, dim=-1)
                all_fine_indices.append(combined_idx[final_rel])
                all_fine_weights.append(final_vals)
            else:
                all_fine_indices.append(torch.zeros(self.top_k, dtype=torch.long, device=z.device))
                all_fine_weights.append(torch.zeros(self.top_k, device=z.device))

        final_indices = torch.stack(all_fine_indices)
        final_weights = torch.stack(all_fine_weights)

        return RoutingOutput(
            indices=final_indices,
            weights=final_weights,
            aux={
                'coarse_indices': coarse_topk_idx,
                'coarse_weights': coarse_weights
            }
        )

    def get_unit_params(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            'mu': self.fine_mu[indices],
            'log_s': self.fine_log_s[indices],
            'log_alpha': self.fine_log_alpha[indices],
        }


class GaussianAttentionRouter(BaseRouter):
    """Gaussian Attention: Mahalanobis-based soft routing over dynamic key Gaussians."""

    def __init__(self, config: NGSConfig):
        super().__init__()
        self.config = config
        self.max_k = config.max_k
        self.d_latent = config.latent_dim
        self.top_k = config.top_k
        self.tau = nn.Parameter(torch.tensor(config.tau))
        self.eps = 1e-5

        # Key Gaussians (the "memory")
        self.key_mu = nn.Parameter(torch.randn(config.max_k, config.latent_dim) * 1.0)
        self.key_log_s = nn.Parameter(torch.zeros(config.max_k, config.latent_dim))
        self.key_log_alpha = nn.Parameter(torch.zeros(config.max_k))

        # Value projections (generated per key)
        self.value_proj = nn.Linear(config.latent_dim, config.latent_dim, bias=False)

        self.register_buffer('active_mask', torch.zeros(config.max_k, dtype=torch.bool))
        self.register_buffer('grad_mu_ema', torch.zeros(config.max_k))

    @property
    def num_active_units(self) -> int:
        return self.active_mask.sum().item()

    @property
    def max_units(self) -> int:
        return self.max_k

    def initialize_units(self, k_init: int) -> None:
        self.active_mask[:k_init] = True

    def forward(self, z: torch.Tensor) -> RoutingOutput:
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]

        if len(active_idx) == 0:
            B = z.shape[0]
            return RoutingOutput(
                indices=torch.zeros(B, self.top_k, dtype=torch.long, device=z.device),
                weights=torch.zeros(B, self.top_k, device=z.device)
            )

        key_mu = self.key_mu[active_idx]
        key_log_s = self.key_log_s[active_idx]
        key_log_alpha = self.key_log_alpha[active_idx]

        # Mahalanobis similarity as attention scores
        diff = z.unsqueeze(1) - key_mu.unsqueeze(0)
        s_sq = torch.exp(2 * key_log_s) + self.eps
        mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1)

        # Attention logits: log_alpha - 0.5/tau * mahalanobis
        attn_logits = key_log_alpha - (0.5 / self.tau) * mahalanobis_sq

        # Soft attention over all active keys (not top-k)
        attn_weights = F.softmax(attn_logits / math.sqrt(self.d_latent), dim=-1)

        # Top-k for sparse output (but can use full attention)
        k_actual = min(self.top_k, len(active_idx))
        topk_vals, topk_rel_idx = torch.topk(attn_weights, k_actual, dim=-1)
        topk_idx = active_idx[topk_rel_idx]

        # Values from value projection
        values = self.value_proj(z)  # [B, d]

        return RoutingOutput(
            indices=topk_idx,
            weights=topk_vals,
            aux={
                'full_attention': attn_weights,
                'values': values,
                'mahalanobis_sq': mahalanobis_sq
            }
        )

    def get_unit_params(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            'key_mu': self.key_mu[indices],
            'key_log_s': self.key_log_s[indices],
            'key_log_alpha': self.key_log_alpha[indices],
        }


class UncertaintyAwareRouter(BaseRouter):
    """Router with evidential uncertainty quantification."""

    def __init__(self, config: NGSConfig):
        super().__init__()
        self.config = config
        self.base_router = MonolithicRouter(config)
        # Evidential head: predicts Dirichlet concentration parameters
        self.evidential_head = nn.SequentialHead(config.latent_dim, config.max_k * 4)  # 4 params per unit
        self.evidential_prior = config.evidential_prior

    @property
    def num_active_units(self) -> int:
        return self.base_router.num_active_units

    @property
    def max_units(self) -> int:
        return self.base_router.max_units

    def initialize_units(self, k_init: int) -> None:
        self.base_router.initialize_units(k_init)

    def forward(self, z: torch.Tensor) -> RoutingOutput:
        # Get base routing
        base_out = self.base_router(z)

        # Evidential uncertainty from latent features
        B = z.shape[0]
        active_idx = self.base_router.active_mask.nonzero(as_tuple=True)[0]
        K = len(active_idx)

        if K == 0:
            return RoutingOutput(
                indices=base_out.indices,
                weights=base_out.weights,
                aux={'uncertainty': torch.ones(B, 1, device=z.device)}
            )

        # Predict Dirichlet parameters for each active unit
        evidential_params = self.evidential_head(z)  # [B, K*4]
        evidential_params = evidential_params.view(B, K, 4)
        alpha = F.softplus(evidential_params) + self.evidential_prior  # [B, K, 4]

        # Dirichlet uncertainty: total evidence S = sum(alpha), uncertainty = K/S
        S = alpha.sum(dim=-1)  # [B, K]
        uncertainty = K / S  # [B, K]

        # Expected probability
        expected_p = alpha / S.unsqueeze(-1)  # [B, K, 4]
        # Use class 1 probability as routing confidence
        routing_conf = expected_p[:, :, 1]

        # Combine with base routing weights
        combined_weights = base_out.weights * routing_conf.gather(1, base_out.indices - active_idx[0])
        combined_weights = F.softmax(combined_weights, dim=-1)

        return RoutingOutput(
            indices=base_out.indices,
            weights=combined_weights,
            aux={
                'uncertainty': uncertainty.gather(1, base_out.indices - active_idx[0]),
                'evidential_alpha': alpha,
                'base_weights': base_out.weights
            }
        )

    def get_unit_params(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.base_router.get_unit_params(indices)


def build_router(config: NGSConfig) -> BaseRouter:
    """Factory function to build router from config."""
    if config.routing == RoutingStrategy.MONOLITHIC:
        return MonolithicRouter(config)
    elif config.routing == RoutingStrategy.FACTORIZED:
        return FactorizedRouter(config)
    elif config.routing == RoutingStrategy.HIERARCHICAL:
        return HierarchicalRouter(config)
    elif config.routing == RoutingStrategy.GAUSSIAN_ATTENTION:
        return GaussianAttentionRouter(config)
    elif config.routing == RoutingStrategy.LSH:
        # Fallback to monolithic for now
        return MonolithicRouter(config)
    else:
        raise ValueError(f"Unknown routing strategy: {config.routing}")
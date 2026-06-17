"""Router implementations for MNGS."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, List


class BaseRouter(nn.Module, ABC):
    """Base class for all routing strategies."""
    
    @abstractmethod
    def forward(self, z: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Route inputs to active units.
        
        Args:
            z: Latent input [B, d]
            
        Returns:
            Tuple of (active_indices, routing_weights)
            - active_indices: [B, k_actual] or list of [B, k_actual] per subspace
            - routing_weights: [B, k_actual] or list of [B, k_actual] per subspace
        """
        pass


class MonolithicRouter(BaseRouter):
    """
    Monolithic Mahalanobis routing (original LeanNGS).
    
    Computes O(N) distance to all units and selects Top-K.
    """
    
    def __init__(self, max_k: int, d_latent: int, top_k: int, tau: float = 1.0, ema_decay: float = 0.99):
        super().__init__()
        self.max_k = max_k
        self.d_latent = d_latent
        self.top_k = top_k
        self.tau = nn.Parameter(torch.tensor(tau))
        self.eps = 1e-5
        
        # Gaussian unit parameters
        self.mu = nn.Parameter(torch.randn(max_k, d_latent) * 1.0)
        self.log_s = nn.Parameter(torch.zeros(max_k, d_latent))
        self.log_alpha = nn.Parameter(torch.zeros(max_k))
        
        # Active mask for pre-allocated memory
        self.register_buffer('active_mask', torch.zeros(max_k, dtype=torch.bool))
        
        # Gradient EMA for topology adaptation (auto-updated via hook)
        self.register_buffer('grad_mu_ema', torch.zeros(max_k))
        self.ema_decay = ema_decay
        self.mu.register_hook(self._update_mu_grad_ema)
        
    def initialize_units(self, k_init: int):
        """Initialize the first k_init units as active."""
        self.active_mask[:k_init] = True
        
    @property
    def K(self) -> int:
        """Number of currently active units."""
        return self.active_mask.sum().item()

    @property
    def max_units(self) -> int:
        """Maximum capacity."""
        return self.max_k
    
    def forward(self, z: torch.Tensor) -> tuple:
        """
        Compute monolithic Mahalanobis routing.
        
        Args:
            z: [B, d_latent]
            
        Returns:
            (topk_idx, topk_weights): both [B, k_actual]
        """
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        
        mu = self.mu[active_idx]                           # [K, d]
        log_s = self.log_s[active_idx]                     # [K, d]
        log_alpha = self.log_alpha[active_idx]             # [K]
        
        # Diagonal Mahalanobis distance
        diff = z.unsqueeze(1) - mu.unsqueeze(0)            # [B, K, d]
        s_sq = torch.exp(2 * log_s) + self.eps             # [K, d]
        mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1)  # [B, K]
        
        # Log-weights
        log_w = log_alpha - (0.5 / self.tau) * mahalanobis_sq  # [B, K]
        
        # Top-K selection
        k_actual = min(self.top_k, self.K)
        topk_vals, topk_rel_idx = torch.topk(log_w, k_actual, dim=-1)  # [B, k_actual]
        
        # Convert to global indices
        topk_idx = active_idx[topk_rel_idx]              # [B, k_actual]
        
        # Softmax weights
        topk_weights = F.softmax(topk_vals, dim=-1)      # [B, k_actual]
        
        return topk_idx, topk_weights
    
    def get_active_params(self, active_idx: torch.Tensor):
        """Get parameters for active units."""
        return {
            'mu': self.mu[active_idx],
            'log_s': self.log_s[active_idx],
            'log_alpha': self.log_alpha[active_idx],
        }
    
    def _update_mu_grad_ema(self, grad):
        """Hook that auto-updates grad_mu_ema after backward."""
        if self.active_mask.any():
            active_mask = self.active_mask
            grad_mag = grad.norm(dim=-1)
            self.grad_mu_ema[active_mask] = (
                self.ema_decay * self.grad_mu_ema[active_mask]
                + (1 - self.ema_decay) * grad_mag[active_mask]
            )
        return grad


class FactorizedRouter(BaseRouter):
    """
    Factorized subspace routing (CFG-Net).
    
    Projects input into S orthogonal subspaces and selects Top-K per subspace.
    Complexity: O(S * M) where N = S * M total units.
    """
    
    def __init__(self, d_latent: int, num_subspaces: int, units_per_space: int, 
                 top_k: int = 2, tau: float = 1.0, param_stores: list = None):
        super().__init__()
        self.d_latent = d_latent
        self.num_subspaces = num_subspaces
        self.units_per_space = units_per_space
        self.top_k = top_k
        self.tau = nn.Parameter(torch.tensor(tau))
        self.eps = 1e-5
        
        total_units = num_subspaces * units_per_space
        
        # Project latent space into subspaces
        d_sub = max(d_latent // num_subspaces, 1)
        self.d_sub = d_sub
        self.subspace_projectors = nn.ModuleList([
            nn.Linear(d_latent, d_sub, bias=False)
            for _ in range(num_subspaces)
        ])
        
        # Gaussian units per subspace
        self.mu = nn.Parameter(torch.randn(num_subspaces, units_per_space, d_sub) * 1.0)
        self.log_s = nn.Parameter(torch.zeros(num_subspaces, units_per_space, d_sub))
        self.log_alpha = nn.Parameter(torch.zeros(num_subspaces, units_per_space))
        
        # Active mask for topology adaptation (flat over all subspaces)
        self.register_buffer('active_mask', torch.zeros(total_units, dtype=torch.bool))
        self.register_buffer('grad_mu_ema', torch.zeros(total_units))
        self.ema_decay = 0.99
        
        # Per-subspace parameter stores (optional, for DYNAMIC_GROWTH support)
        if param_stores is not None:
            self.param_stores = nn.ModuleList(param_stores)
        else:
            self.param_stores = None

    @property
    def K(self) -> int:
        """Number of currently active units."""
        return self.active_mask.sum().item()

    @property
    def max_units(self) -> int:
        """Maximum capacity."""
        return self.num_subspaces * self.units_per_space

    def initialize_units(self, k_init: int):
        """Initialize the first k_init units as active."""
        self.active_mask[:k_init] = True
    
    def set_param_stores(self, param_stores):
        """Set per-subspace parameter stores."""
        self.param_stores = nn.ModuleList(param_stores)
    
    @property
    def flat_mu(self):
        return self.mu.view(-1, self.d_sub)
    
    @property
    def flat_log_s(self):
        return self.log_s.view(-1, self.d_sub)
    
    @property
    def flat_log_alpha(self):
        return self.log_alpha.view(-1)
        
    def forward(self, z: torch.Tensor) -> tuple:
        """
        Factorized subspace routing.
        
        Args:
            z: [B, d_latent]
            
        Returns:
            (subspace_indices, subspace_weights)
            - subspace_indices: list of [B, top_k] tensors (global flat indices)
            - subspace_weights: list of [B, top_k] tensors
        """
        B = z.shape[0]
        subspace_indices = []
        subspace_weights = []
        
        for s in range(self.num_subspaces):
            start = s * self.units_per_space
            end = start + self.units_per_space
            
            # Get active units in this subspace
            sub_active = self.active_mask[start:end]
            active_local = sub_active.nonzero(as_tuple=True)[0]
            
            if len(active_local) == 0:
                # No active units — return zeros
                subspace_indices.append(torch.zeros(B, self.top_k, dtype=torch.long, device=z.device))
                subspace_weights.append(torch.zeros(B, self.top_k, device=z.device))
                continue
            
            # Project into subspace
            z_s = self.subspace_projectors[s](z)  # [B, d_sub]
            
            # Get active subspace units
            mu_s = self.mu[s][active_local]         # [K_active, d_sub]
            log_s_s = self.log_s[s][active_local]   # [K_active, d_sub]
            log_alpha_s = self.log_alpha[s][active_local]  # [K_active]
            
            # Diagonal Mahalanobis in subspace
            diff = z_s.unsqueeze(1) - mu_s.unsqueeze(0)   # [B, K_active, d_sub]
            s_sq = torch.exp(2 * log_s_s) + self.eps
            mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1)  # [B, K_active]
            
            # Log-weights
            log_w = log_alpha_s - (0.5 / self.tau) * mahalanobis_sq
            
            # Top-K per subspace
            k_actual = min(self.top_k, len(active_local))
            topk_vals, topk_rel_idx = torch.topk(log_w, k_actual, dim=-1)
            topk_weights = F.softmax(topk_vals, dim=-1)
            
            # Convert local indices to global flat indices
            topk_global = active_local[topk_rel_idx] + start
            subspace_indices.append(topk_global)
            subspace_weights.append(topk_weights)
        
        return subspace_indices, subspace_weights


class LSRRouter(BaseRouter):
    """
    Locality-Sensitive Hashing router (extensibility hook).
    
    Hashes latent inputs into buckets to enable sub-linear routing.
    Future implementation.
    """
    
    def __init__(self, d_latent: int, num_buckets: int, num_hash_functions: int, top_k: int = 8):
        super().__init__()
        self.d_latent = d_latent
        self.num_buckets = num_buckets
        self.num_hash_functions = num_hash_functions
        self.top_k = top_k
        
        # Random projection matrices for LSH
        self.hash_projections = nn.Parameter(
            torch.randn(num_hash_functions, d_latent, num_buckets)
        )
        
        # Active mask for compatibility
        self.register_buffer('active_mask', torch.ones(num_buckets, dtype=torch.bool))
        
    @property
    def K(self) -> int:
        """Number of currently active units."""
        return self.active_mask.sum().item()

    @property
    def max_units(self) -> int:
        """Maximum capacity."""
        return self.num_buckets

    def initialize_units(self, k_init: int):
        """Initialize the first k_init units as active."""
        self.active_mask[:min(k_init, self.num_buckets)] = True
        
    def forward(self, z: torch.Tensor) -> tuple:
        """
        LSH-based approximate routing.
        
        Args:
            z: [B, d_latent]
            
        Returns:
            (indices, weights)
        """
        # Compute hash signatures
        # z: [B, d], projections: [H, d, num_buckets]
        # We want signatures per hash function: [B, H]
        # For each hash function h: z @ W_h where W_h: [d, num_buckets] -> [B, num_buckets]
        # Then take argmax over buckets for each h
        B = z.shape[0]
        # z: [B, d], hash_projections: [H, d, num_buckets]
        # projections[h] = z @ hash_projections[h] -> [B, num_buckets]
        # We want signatures: [B, H]
        signatures = []
        for h in range(self.num_hash_functions):
            proj = z @ self.hash_projections[h]  # [B, num_buckets]
            sig = torch.argmax(proj, dim=-1)  # [B]
            signatures.append(sig)
        signatures = torch.stack(signatures, dim=-1)  # [B, H]
        
        # Create pseudo-distances based on hash collisions (placeholder)
        pseudo_dist = torch.randn(B, self.num_buckets, device=z.device)
        topk_vals, topk_idx = torch.topk(pseudo_dist, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)
        
        return topk_idx, topk_weights


def build_router(config, d_latent: int = None, max_k: int = None, top_k: int = None):
    """Factory function to build router from config."""
    from mngs.core.config import RoutingStrategy
    
    routing = config.routing if hasattr(config, 'routing') else config
    max_k = max_k or config.max_k
    d_latent = d_latent or config.latent_dim
    top_k = top_k or config.top_k
    tau = getattr(config, 'tau', 1.0)
    ema_decay = getattr(config, 'ema_decay', 0.99)
    
    if routing == RoutingStrategy.MONOLITHIC_MAHALANOBIS:
        return MonolithicRouter(
            max_k=max_k,
            d_latent=d_latent,
            top_k=top_k,
            tau=tau,
            ema_decay=ema_decay
        )
    elif routing == RoutingStrategy.FACTORIZED_SUBSPACE:
        num_subspaces = getattr(config, 'num_subspaces', 4)
        units_per_space = max_k // num_subspaces
        top_k_factorized = getattr(config, 'top_k_factorized', 2)
        return FactorizedRouter(
            d_latent=d_latent,
            num_subspaces=num_subspaces,
            units_per_space=units_per_space,
            top_k=top_k_factorized,
            tau=tau
        )
    elif routing == RoutingStrategy.LSH_APPROXIMATE:
        return LSRRouter(
            d_latent=d_latent,
            num_buckets=max_k // 4,
            num_hash_functions=4,
            top_k=top_k
        )
    else:
        raise ValueError(f"Unknown routing strategy: {routing}")

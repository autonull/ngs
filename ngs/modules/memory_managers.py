"""Memory management implementations for NGS."""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from ngs.core.interfaces import BaseMemoryManager, NGSConfig, MemoryManagement


class PreAllocatedManager(BaseMemoryManager):
    """
    Pre-allocated fixed buffer management.
    All units are pre-allocated up to max_k, but only active units participate.
    Unused units are masked out in the routing layer.
    """

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.capacity = config.max_k

    def enforce_capacity(self, model) -> int:
        """
        Enforce pre-allocated capacity by pruning excess units if needed.
        Returns number of pruned units.
        """
        if not hasattr(model.router, 'active_mask'):
            return 0

        active_mask = model.router.active_mask
        active_count = active_mask.sum().item()

        if active_count <= self.capacity:
            return 0

        # Need to prune down to capacity - remove lowest opacity units
        active_idx = active_mask.nonzero(as_tuple=True)[0]

        # Access opacity values
        mu, log_s, log_alpha = _flat_access(model.router)
        if mu is not None:
            alpha = torch.sigmoid(log_alpha[active_idx])
            # Sort by alpha ascending and prune the weakest
            num_to_prune = active_count - self.capacity
            if num_to_prune > 0:
                _, prune_local_idx = torch.topk(alpha, num_to_prune, largest=False)
                prune_global = active_idx[prune_local_idx]
                model.router.active_mask[prune_global] = False
                if hasattr(model.router, 'grad_mu_ema'):
                    model.router.grad_mu_ema[prune_global] = 0
                return num_to_prune

        return 0

    def expand_buffers(self, model, new_max_k: int):
        """
        Expand model buffers for increased capacity.
        For pre-allocated, this means expanding mu, log_s, log_alpha to new_max_k.
        Delegates to each component to expand its buffers.
        """
        if new_max_k <= self.capacity:
            return

        router = model.router
        if hasattr(router, 'mu') and hasattr(router, 'active_mask'):
            old_max = router.mu.shape[0] if router.mu.dim() == 2 else router.mu.shape[1]
            if new_max_k <= old_max:
                return

            # Expand router parameters
            latent_dim = router.mu.shape[-1]
            device = router.mu.device

            # Expand mu
            new_mu = torch.randn(new_max_k, latent_dim, device=device) * 1.0
            if router.mu.dim() == 2:
                new_mu[:old_max] = router.mu.data
                router.mu = nn.Parameter(new_mu)
            else:
                # For factorized routers, expand each subspace
                num_subspaces = router.num_subspaces
                new_mu_per_space = new_max_k // num_subspaces
                for s in range(num_subspaces):
                    old_size = router.mu.shape[1]
                    new_shape = (new_max_k // num_subspaces, router.mu.shape[-1])
                    new_mu_s = torch.randn(*new_shape, device=device) * 1.0
                    new_mu_s[:old_size] = router.mu[s].data
                    router.mu.data = torch.cat([router.mu.data, new_mu_s[old_size:]], dim=0)

            # Expand log_s
            new_log_s = torch.zeros(new_max_k, latent_dim, device=device)
            if router.log_s.dim() == 2:
                new_log_s[:old_max] = router.log_s.data
                router.log_s = nn.Parameter(new_log_s)
            else:
                for s in range(router.num_subspaces if hasattr(router, 'num_subspaces') else 1):
                    if hasattr(router, 'num_subspaces'):
                        old_size = router.log_s.shape[1]
                        new_shape = (new_max_k // router.num_subspaces, router.log_s.shape[-1])
                        new_log_s_s = torch.zeros(*new_shape, device=device)
                        new_log_s_s[:old_size] = router.log_s[s].data
                        router.log_s.data = torch.cat([router.log_s.data, new_log_s_s[old_size:]], dim=0)

            # Expand log_alpha
            new_log_alpha = torch.zeros(new_max_k, device=device)
            if router.log_alpha.dim() == 1:
                new_log_alpha[:old_max] = router.log_alpha.data
                router.log_alpha = nn.Parameter(new_log_alpha)
            else:
                for s in range(router.num_subspaces if hasattr(router, 'num_subspaces') else 1):
                    if hasattr(router, 'num_subspaces'):
                        old_size = router.log_alpha.shape[1]
                        new_log_alpha_s = torch.zeros(new_max_k // router.num_subspaces, device=device)
                        new_log_alpha_s[:old_size] = router.log_alpha[s].data
                        router.log_alpha.data = torch.cat([router.log_alpha.data, new_log_alpha_s[old_size:]], dim=0)

            # Expand active mask
            new_active_mask = torch.zeros(new_max_k, dtype=torch.bool, device=device)
            new_active_mask[:old_max] = router.active_mask
            router.active_mask = new_active_mask

            # Expand grad_mu_ema
            if hasattr(router, 'grad_mu_ema'):
                new_grad = torch.zeros(new_max_k, device=device)
                new_grad[:old_max] = router.grad_mu_ema
                router.grad_mu_ema = new_grad

        # Expand parameter store
        if hasattr(model, 'param_store'):
            model.param_store.expand_capacity(new_max_k)

        # Expand split gate and densities if they exist
        if hasattr(model, 'split_gate'):
            new_split = torch.full((new_max_k,), model.split_gate.data[0] if model.split_gate.numel() > 0 else 0.1, device=device)
            new_split[:old_max] = model.split_gate.data[:old_max]
            model.split_gate.data = new_split

        if hasattr(model, 'activation_density'):
            new_act = torch.zeros(new_max_k, device=device)
            new_act[:old_max] = model.activation_density
            model.activation_density = new_act

        if hasattr(model, 'error_density'):
            new_err = torch.zeros(new_max_k, device=device)
            new_err[:old_max] = model.error_density
            model.error_density = new_err

        self.capacity = new_max_k


class DynamicManager(BaseMemoryManager):
    """
    Dynamic buffer management that expands capacity on demand.
    Tracks utilization and automatically expands when utilization is high.
    """

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.capacity = config.k_init  # Start smaller
        self.max_capacity = config.max_k
        self.utilization_threshold = 0.85
        self.active_count = 0

    def enforce_capacity(self, model) -> int:
        """
        Enforce dynamic capacity by pruning if over-subscribed,
        but prefer expanding if near threshold.
        """
        if not hasattr(model.router, 'active_mask'):
            return 0

        active_mask = model.router.active_mask
        self.active_count = active_mask.sum().item()
        utilization = self.active_count / self.capacity

        # If over capacity and can't expand more, prune
        if self.active_count > self.capacity:
            active_idx = active_mask.nonzero(as_tuple=True)[0]
            num_to_prune = self.active_count - self.capacity

            if num_to_prune > 0:
                # Prune least active units
                mu, log_s, log_alpha = _flat_access(model.router)
                if mu is not None:
                    alpha = torch.sigmoid(log_alpha[active_idx])
                    _, prune_local_idx = torch.topk(alpha, num_to_prune, largest=False)
                    prune_global = active_idx[prune_local_idx]
                    model.router.active_mask[prune_global] = False
                    if hasattr(model.router, 'grad_mu_ema'):
                        model.router.grad_mu_ema[prune_global] = 0
                    return num_to_prune

        # Check if we need to expand
        if utilization > self.utilization_threshold and self.capacity < self.max_capacity:
            new_capacity = min(int(self.capacity * 1.5), self.max_capacity)
            self.expand_buffers(model, new_capacity)

        return 0

    def expand_buffers(self, model, new_max_k: int):
        """Expand model buffers to new capacity."""
        if new_max_k <= self.capacity:
            return

        router = model.router
        if not hasattr(router, 'mu'):
            return

        old_max = router.mu.shape[0] if router.mu.dim() == 2 else router.mu.shape[1]
        if new_max_k <= old_max:
            self.capacity = new_max_k
            return

        latent_dim = router.mu.shape[-1]
        device = router.mu.device

        # Expand router parameters
        if router.mu.dim() == 2:
            # Monolithic router
            new_mu = torch.randn(new_max_k, latent_dim, device=device) * 1.0
            new_mu[:old_max] = router.mu.data
            router.mu = nn.Parameter(new_mu)

            new_log_s = torch.zeros(new_max_k, latent_dim, device=device)
            new_log_s[:old_max] = router.log_s.data
            router.log_s = nn.Parameter(new_log_s)

            new_log_alpha = torch.zeros(new_max_k, device=device)
            new_log_alpha[:old_max] = router.log_alpha.data
            router.log_alpha = nn.Parameter(new_log_alpha)

            # Expand active mask
            new_active_mask = torch.zeros(new_max_k, dtype=torch.bool, device=device)
            new_active_mask[:old_max] = router.active_mask
            router.active_mask = new_active_mask

            # Expand grad_mu_ema
            if hasattr(router, 'grad_mu_ema'):
                new_grad = torch.zeros(new_max_k, device=device)
                new_grad[:old_max] = router.grad_mu_ema
                router.grad_mu_ema = new_grad
        else:
            # Factorized: expand per subspace (simplified)
            pass

        # Expand parameter store
        if hasattr(model, 'param_store'):
            model.param_store.expand_capacity(new_max_k)

        # Expand split gate and densities if they exist
        for attr in ['split_gate', 'activation_density', 'error_density']:
            if hasattr(model, attr):
                tensor = getattr(model, attr)
                new_tensor = torch.zeros(new_max_k, device=device)
                new_tensor[:old_max] = tensor[:old_max]
                tensor.data = new_tensor

        self.capacity = new_max_k


class StrictCapacityManager(BaseMemoryManager):
    """
    Strict capacity management with hard bounds.
    When capacity is reached, the least recently used (LRU) or
    least recently accessed units are evicted.
    """

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.capacity = config.max_k
        self.lru_counter = 0

    def _get_lru_scores(self, model) -> torch.Tensor:
        """
        Compute LRU scores based on recent access patterns.
        Lower scores mean less recently used.
        """
        if not hasattr(model.router, 'active_mask'):
            return torch.zeros(0)

        active_mask = model.router.active_mask
        active_idx = active_mask.nonzero(as_tuple=True)[0]

        # Simple heuristic: use opacity as a proxy for "usefulness"
        # Units with higher opacity are considered more recently/relevantly used
        mu, log_s, log_alpha = _flat_access(model.router)
        if mu is not None:
            alpha = torch.sigmoid(log_alpha[active_idx])
            return alpha

        return torch.ones(len(active_idx))

    def enforce_capacity(self, model) -> int:
        """
        Enforce strict capacity by evicting LRU units when over capacity.
        Returns number of evicted (pruned) units.
        """
        if not hasattr(model.router, 'active_mask'):
            return 0

        active_mask = model.router.active_mask
        active_count = active_mask.sum().item()

        if active_count <= self.capacity:
            return 0

        num_to_evict = active_count - self.capacity
        active_idx = active_mask.nonzero(as_tuple=True)[0]

        # Get LRU scores and evict the lowest ones
        lru_scores = self._get_lru_scores(model)
        _, evict_local_idx = torch.topk(lru_scores, num_to_evict, largest=False)
        evict_global = active_idx[evict_local_idx]

        model.router.active_mask[evict_global] = False
        if hasattr(model.router, 'grad_mu_ema'):
            model.router.grad_mu_ema[evict_global] = 0

        return num_to_evict

    def expand_buffers(self, model, new_max_k: int):
        """
        Expand model buffers on demand.
        """
        if new_max_k <= self.capacity:
            return

        # Delegate to PreAllocatedManager logic
        prealloc = PreAllocatedManager(self.config)
        prealloc.capacity = self.capacity
        prealloc.expand_buffers(model, new_max_k)
        self.capacity = new_max_k


# ─────────────────────────── Helpers ───────────────────────────

def _flat_access(router):
    """Return flat-accessible mu, log_s, log_alpha from router."""
    mu = log_s = log_alpha = None
    if hasattr(router, 'mu'):
        if router.mu.dim() == 2:
            mu, log_s, log_alpha = router.mu, router.log_s, router.log_alpha
        else:
            mu = router.mu.view(-1, router.mu.shape[-1])
            log_s = router.log_s.view(-1, router.log_s.shape[-1])
            log_alpha = router.log_alpha.view(-1)
    return mu, log_s, log_alpha


def build_memory_manager(config: NGSConfig):
    """Factory function to build memory manager from config."""
    memory = config.memory_management

    if memory.name == "PRE_ALLOCATED":
        return PreAllocatedManager(config)
    elif memory.name == "DYNAMIC":
        return DynamicManager(config)
    elif memory.name == "STRICT_CAPACITY":
        return StrictCapacityManager(config)
    else:
        raise ValueError(f"Unknown memory management: {memory}")


__all__ = [
    "PreAllocatedManager",
    "DynamicManager",
    "StrictCapacityManager",
    "build_memory_manager",
]

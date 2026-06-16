"""Memory management implementations for NGS."""
import torch
from typing import Optional

from ngs.core.interfaces import BaseMemoryManager, MemoryManagement, NGSConfig, BaseRouter


class PreAllocatedMemoryManager(BaseMemoryManager):
    """Pre-allocated masked memory (original approach)."""

    def __init__(self, config: NGSConfig):
        self.config = config

    def enforce_capacity(self, model: 'NGSModel') -> int:
        router = model.router
        if not hasattr(router, 'active_mask'):
            return 0
        active_mask = router.active_mask
        num_active = active_mask.sum().item()
        if num_active > self.config.max_k:
            # Prune lowest alpha
            mu, log_s, log_alpha = self._get_flat_attrs(router)
            active_idx = active_mask.nonzero(as_tuple=True)[0]
            alphas = torch.sigmoid(log_alpha[active_idx])
            n_prune = num_active - self.config.max_k
            _, worst_idx = torch.topk(alphas, n_prune, largest=False)
            prune_idx = active_idx[worst_idx]
            active_mask[prune_idx] = False
            if hasattr(router, 'grad_mu_ema'):
                router.grad_mu_ema[prune_idx] = 0
            return n_prune
        return 0

    def allocate_unit(self, model: 'NGSModel') -> Optional[int]:
        router = model.router
        if not hasattr(router, 'active_mask'):
            return None
        active_mask = router.active_mask
        free = (~active_mask).nonzero(as_tuple=True)[0]
        if len(free) > 0:
            return free[0].item()
        return None

    def free_unit(self, model: 'NGSModel', index: int) -> None:
        router = model.router
        if hasattr(router, 'active_mask'):
            router.active_mask[index] = False
            if hasattr(router, 'grad_mu_ema'):
                router.grad_mu_ema[index] = 0

    def _get_flat_attrs(self, router: BaseRouter):
        if hasattr(router, 'flat_mu'):
            return router.flat_mu, router.flat_log_s, router.flat_log_alpha
        if hasattr(router, 'mu'):
            return router.mu, router.log_s, router.log_alpha
        if hasattr(router, 'coarse_mu'):
            return (torch.cat([router.coarse_mu, router.fine_mu], dim=0),
                    torch.cat([router.coarse_log_s, router.fine_log_s], dim=0),
                    torch.cat([router.coarse_log_alpha, router.fine_log_alpha], dim=0))
        raise AttributeError("Router has no accessible Gaussian parameters")


class DynamicMemoryManager(BaseMemoryManager):
    """Dynamic growth without pre-allocation."""

    def __init__(self, config: NGSConfig):
        self.config = config
        self.current_max = config.k_init

    def enforce_capacity(self, model: 'NGSModel') -> int:
        # No hard limit, but warn if growing too large
        return 0

    def allocate_unit(self, model: 'NGSModel') -> Optional[int]:
        router = model.router
        if hasattr(router, 'active_mask'):
            active_mask = router.active_mask
            if active_mask.sum() < len(active_mask):
                free = (~active_mask).nonzero(as_tuple=True)[0]
                if len(free) > 0:
                    return free[0].item()
            # Need to expand buffers
            self._expand_buffers(model)
            new_idx = len(active_mask)
            return new_idx
        return None

    def free_unit(self, model: 'NGSModel', index: int) -> None:
        router = model.router
        if hasattr(router, 'active_mask'):
            router.active_mask[index] = False

    def _expand_buffers(self, model: 'NGSModel') -> None:
        router = model.router
        old_size = router.active_mask.shape[0]
        new_size = old_size * 2

        # Expand router buffers
        if hasattr(router, 'active_mask'):
            router.active_mask = torch.cat([
                router.active_mask,
                torch.zeros(new_size - old_size, dtype=torch.bool, device=router.active_mask.device)
            ])
            router.grad_mu_ema = torch.cat([
                router.grad_mu_ema,
                torch.zeros(new_size - old_size, device=router.grad_mu_ema.device)
            ])

        if hasattr(router, 'mu'):
            router.mu = torch.nn.Parameter(torch.cat([
                router.mu,
                torch.randn(new_size - old_size, router.d_latent, device=router.mu.device)
            ]))
            router.log_s = torch.nn.Parameter(torch.cat([
                router.log_s,
                torch.zeros(new_size - old_size, router.d_latent, device=router.log_s.device)
            ]))
            router.log_alpha = torch.nn.Parameter(torch.cat([
                router.log_alpha,
                torch.zeros(new_size - old_size, device=router.log_alpha.device)
            ]))

        # Expand parameter store
        param_store = model.param_store
        if hasattr(param_store, 'codes'):
            param_store.codes = torch.nn.Parameter(torch.cat([
                param_store.codes,
                torch.randn(new_size - old_size, param_store.code_dim, device=param_store.codes.device) * 0.1
            ]))
        elif hasattr(param_store, 'lora_A'):
            param_store.lora_A = torch.nn.Parameter(torch.cat([
                param_store.lora_A,
                torch.randn(new_size - old_size, param_store.d_latent, param_store.lora_rank, device=param_store.lora_A.device) * 1e-2
            ]))
            param_store.lora_B = torch.nn.Parameter(torch.cat([
                param_store.lora_B,
                torch.randn(new_size - old_size, param_store.lora_rank, param_store.d_latent, device=param_store.lora_B.device) * 1e-2
            ]))


class StrictCapacityManager(BaseMemoryManager):
    """Strict capacity enforcement for edge deployment."""

    def __init__(self, config: NGSConfig):
        self.config = config

    def enforce_capacity(self, model: 'NGSModel') -> int:
        return PreAllocatedMemoryManager(self.config).enforce_capacity(model)

    def allocate_unit(self, model: 'NGSModel') -> Optional[int]:
        router = model.router
        if not hasattr(router, 'active_mask'):
            return None
        active_mask = router.active_mask
        if active_mask.sum() >= self.config.max_k:
            return None  # At capacity
        free = (~active_mask).nonzero(as_tuple=True)[0]
        if len(free) > 0:
            return free[0].item()
        return None

    def free_unit(self, model: 'NGSModel', index: int) -> None:
        PreAllocatedMemoryManager(self.config).free_unit(model, index)


def build_memory_manager(config: NGSConfig) -> BaseMemoryManager:
    """Factory function to build memory manager from config."""
    if config.memory_management == MemoryManagement.PRE_ALLOCATED:
        return PreAllocatedMemoryManager(config)
    elif config.memory_management == MemoryManagement.DYNAMIC:
        return DynamicMemoryManager(config)
    elif config.memory_management == MemoryManagement.STRICT_CAPACITY:
        return StrictCapacityManager(config)
    else:
        raise ValueError(f"Unknown memory management: {config.memory_management}")

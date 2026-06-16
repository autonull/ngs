"""Topology management implementations for NGS."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from ngs.core.interfaces import BaseTopologyManager, TopologyAction, NGSConfig, BaseRouter, BaseParameterStore


def _get_flat_attrs(router: BaseRouter) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get flat-accessible mu, log_s, log_alpha from router."""
    if hasattr(router, 'flat_mu'):
        return router.flat_mu, router.flat_log_s, router.flat_log_alpha
    if hasattr(router, 'mu'):
        return router.mu, router.log_s, router.log_alpha
    if hasattr(router, 'coarse_mu'):
        # Hierarchical: concatenate coarse and fine
        return (torch.cat([router.coarse_mu, router.fine_mu], dim=0),
                torch.cat([router.coarse_log_s, router.fine_log_s], dim=0),
                torch.cat([router.coarse_log_alpha, router.fine_log_alpha], dim=0))
    raise AttributeError("Router has no accessible Gaussian parameters")


def _get_active_mask(router: BaseRouter) -> torch.Tensor:
    """Get active mask from router."""
    if hasattr(router, 'active_mask'):
        return router.active_mask
    if hasattr(router, 'coarse_active'):
        # Hierarchical: combine coarse and fine
        coarse = router.coarse_active
        fine = router.fine_active
        return torch.cat([coarse, fine], dim=0)
    raise AttributeError("Router has no active_mask")


def _set_active_mask(router: BaseRouter, mask: torch.Tensor) -> None:
    """Set active mask on router."""
    if hasattr(router, 'active_mask'):
        router.active_mask.copy_(mask)
    elif hasattr(router, 'coarse_active'):
        coarse_size = router.coarse_active.shape[0]
        router.coarse_active.copy_(mask[:coarse_size])
        router.fine_active.copy_(mask[coarse_size:])


class HeuristicManager(BaseTopologyManager):
    """Discrete heuristic topology control (original LeanNGS)."""

    def __init__(self, config: NGSConfig):
        self.split_threshold = config.split_threshold
        self.prune_threshold = config.prune_threshold
        self.split_scale = 0.5
        self.noise_std = 0.01
        self.ema_decay = config.ema_decay

    def adapt_topology(
        self,
        model: 'NGSModel',
        z_samples: Optional[torch.Tensor] = None,
        spawn_thresh: float = -5.0,
        max_spawn_per_call: int = 10,
        **kwargs
    ) -> TopologyAction:
        router = model.router
        param_store = model.param_store

        if not hasattr(router, 'grad_mu_ema'):
            return TopologyAction()

        active_mask = _get_active_mask(router)
        mu, log_s, log_alpha = _get_flat_attrs(router)
        grad_ema = router.grad_mu_ema

        active_idx = active_mask.nonzero(as_tuple=True)[0]
        K = len(active_idx)
        if K == 0:
            return TopologyAction()

        alpha = torch.sigmoid(log_alpha[active_idx])
        max_s = torch.exp(log_s[active_idx]).max(dim=-1).values
        active_grad_ema = grad_ema[active_idx]

        num_pruned = 0
        num_split = 0
        num_spawned = 0
        num_merged = 0
        merged_indices = []

        # Prune low-opacity units
        prune_mask = alpha < self.prune_threshold
        if prune_mask.any():
            prune_idx = active_idx[prune_mask]
            new_mask = active_mask.clone()
            new_mask[prune_idx] = False
            _set_active_mask(router, new_mask)
            grad_ema[prune_idx] = 0
            num_pruned = prune_mask.sum().item()

        # Recompute after pruning
        active_mask = _get_active_mask(router)
        active_idx = active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) == 0:
            return TopologyAction(num_pruned=num_pruned)

        alpha = torch.sigmoid(log_alpha[active_idx])
        active_grad_ema = grad_ema[active_idx]
        max_s = torch.exp(log_s[active_idx]).max(dim=-1).values

        # Split high-gradient units
        split_mask = (active_grad_ema > self.split_threshold) & (max_s > self.split_threshold)

        # Adaptive fallback
        if not split_mask.any() and len(active_idx) > 0 and max_spawn_per_call > 0:
            max_grad = active_grad_ema.max()
            if max_grad > self.split_threshold * 0.1:
                n_fallback = min(2, len(active_idx) // 4)
                n_fallback = max(n_fallback, 1)
                _, top_indices = torch.topk(active_grad_ema, n_fallback)
                split_mask = torch.zeros(len(active_idx), dtype=torch.bool, device=active_grad_ema.device)
                split_mask[top_indices] = True

        if split_mask.any():
            split_idx = active_idx[split_mask]
            free_slots = (~active_mask).nonzero(as_tuple=True)[0]
            n_available = len(free_slots)
            n_split = min(len(split_idx), n_available)

            if n_split > 0:
                split_idx = split_idx[:n_split]
                new_idx = free_slots[:n_split]

                mu.data[new_idx] = mu[split_idx].clone()
                noise = torch.randn_like(mu[split_idx]) * self.noise_std
                mu.data[new_idx] += noise

                log_s.data[new_idx] = log_s[split_idx].clone()
                log_s.data[new_idx] += torch.log(torch.tensor(self.split_scale, device=mu.device))
                log_s.data[split_idx] += torch.log(torch.tensor(self.split_scale, device=mu.device))

                log_alpha.data[new_idx] = log_alpha[split_idx].clone()
                alpha_new = torch.sigmoid(log_alpha.data[new_idx])
                log_alpha.data[new_idx] = torch.logit(alpha_new * 0.5, eps=1e-8)
                alpha_split = torch.sigmoid(log_alpha.data[split_idx])
                log_alpha.data[split_idx] = torch.logit(alpha_split * 0.5, eps=1e-8)

                grad_ema[new_idx] = 0
                grad_ema[split_idx] = 0

                new_mask = active_mask.clone()
                new_mask[new_idx] = True
                _set_active_mask(router, new_mask)

                # Initialize parameter store for new units
                param_store.init_unit(new_idx[0].item(), split_idx[0].item())
                for i in range(1, n_split):
                    param_store.init_unit(new_idx[i].item(), split_idx[i].item())

                num_split = n_split

        # Spawn in uncovered regions
        if z_samples is not None:
            active_mask = _get_active_mask(router)
            free_slots = (~active_mask).nonzero(as_tuple=True)[0]
            if len(free_slots) > 0:
                z_samples = z_samples.to(mu.device)
                active_idx = active_mask.nonzero(as_tuple=True)[0]
                mu_active = mu[active_idx]
                log_s_active = log_s[active_idx]
                s_sq = torch.exp(2 * log_s_active) + 1e-5

                diff = z_samples.unsqueeze(1) - mu_active.unsqueeze(0)
                mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1)
                log_w = log_alpha[active_idx] - 0.5 * mahalanobis_sq
                max_log_w, _ = log_w.max(dim=-1)

                uncovered_mask = max_log_w < spawn_thresh
                if uncovered_mask.any():
                    uncovered_z = z_samples[uncovered_mask]
                    n_spawn = min(len(uncovered_z), len(free_slots), max_spawn_per_call)
                    if n_spawn > 0:
                        spawn_idx = free_slots[:n_spawn]
                        mu.data[spawn_idx] = uncovered_z[:n_spawn]
                        log_s.data[spawn_idx].fill_(0.0)
                        log_alpha.data[spawn_idx].fill_(0.0)
                        grad_ema[spawn_idx] = 0

                        new_mask = active_mask.clone()
                        new_mask[spawn_idx] = True
                        _set_active_mask(router, new_mask)

                        param_store.init_unit(spawn_idx[0].item())
                        for i in range(1, n_spawn):
                            param_store.init_unit(spawn_idx[i].item())

                        num_spawned = n_spawn

        return TopologyAction(
            num_pruned=num_pruned,
            num_split=num_split,
            num_spawned=num_spawned,
            num_merged=num_merged,
            merged_indices=merged_indices
        )

    def compute_losses(self, model: 'NGSModel', **kwargs) -> dict[str, torch.Tensor]:
        return {
            'entropy': model.entropy_loss(),
            'diversity': model.diversity_loss(),
        }


class ContinuousDensityManager(BaseTopologyManager):
    """Continuous density-driven topology with differentiable split gates."""

    def __init__(self, config: NGSConfig):
        self.split_threshold = config.split_threshold
        self.prune_threshold = config.prune_threshold
        self.density_decay = config.density_decay
        self.split_gate_threshold = config.split_gate_threshold
        self.spawn_threshold = -5.0
        self.noise_std = 0.01

    def adapt_topology(
        self,
        model: 'NGSModel',
        z_samples: Optional[torch.Tensor] = None,
        max_spawn_per_call: int = 5,
        **kwargs
    ) -> TopologyAction:
        router = model.router
        param_store = model.param_store

        if not hasattr(model, 'split_gate'):
            return TopologyAction()

        active_mask = _get_active_mask(router)
        mu, log_s, log_alpha = _get_flat_attrs(router)
        active_idx = active_mask.nonzero(as_tuple=True)[0]
        alpha = torch.sigmoid(log_alpha[active_idx])

        num_pruned = 0
        num_split = 0
        num_spawned = 0
        num_merged = 0
        merged_indices = []

        # Prune
        prune_mask = alpha < self.prune_threshold
        if prune_mask.any():
            prune_idx = active_idx[prune_mask]
            new_mask = active_mask.clone()
            new_mask[prune_idx] = False
            _set_active_mask(router, new_mask)
            if hasattr(router, 'grad_mu_ema'):
                router.grad_mu_ema[prune_idx] = 0
            num_pruned = prune_mask.sum().item()

        # Split gate execution
        active_mask = _get_active_mask(router)
        active_idx = active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) > 0:
            gamma = torch.sigmoid(model.split_gate[active_idx])
            err_density = model.error_density[active_idx]
            split_mask = (gamma > self.split_gate_threshold) & (err_density > 1e-3)

            if split_mask.any():
                free_slots = (~active_mask).nonzero(as_tuple=True)[0]
                n_available = len(free_slots)
                split_idx = active_idx[split_mask]
                n_split = min(len(split_idx), n_available)

                if n_split > 0:
                    split_idx = split_idx[:n_split]
                    new_idx = free_slots[:n_split]

                    mu.data[new_idx] = mu[split_idx].clone()
                    noise = torch.randn_like(mu[split_idx]) * self.noise_std
                    mu.data[new_idx] += noise

                    log_s.data[new_idx] = log_s[split_idx].clone()
                    half_scale = torch.log(torch.tensor(0.5, device=mu.device))
                    log_s.data[new_idx] += half_scale
                    log_s.data[split_idx] += half_scale

                    log_alpha.data[new_idx] = log_alpha[split_idx].clone()

                    # Reset gates for both parent and child
                    model.split_gate.data[new_idx] = 0.0
                    model.split_gate.data[split_idx] = 0.0
                    model.activation_density[new_idx] = 0.0
                    model.error_density[new_idx] = 0.0

                    new_mask = active_mask.clone()
                    new_mask[new_idx] = True
                    _set_active_mask(router, new_mask)

                    param_store.init_unit(new_idx[0].item(), split_idx[0].item())
                    for i in range(1, n_split):
                        param_store.init_unit(new_idx[i].item(), split_idx[i].item())

                    num_split = n_split

        # Spawn
        if z_samples is not None:
            active_mask = _get_active_mask(router)
            free_slots = (~active_mask).nonzero(as_tuple=True)[0]
            if len(free_slots) > 0:
                z_samples = z_samples.to(mu.device)
                active_idx = active_mask.nonzero(as_tuple=True)[0]
                if len(active_idx) > 0:
                    mu_active = mu[active_idx]
                    log_s_active = log_s[active_idx]
                    s_sq = torch.exp(2 * log_s_active) + 1e-5

                    diff = z_samples.unsqueeze(1) - mu_active.unsqueeze(0)
                    mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1)
                    log_w = log_alpha[active_idx] - 0.5 * mahalanobis_sq
                    max_log_w, _ = log_w.max(dim=-1)

                    uncovered_mask = max_log_w < self.spawn_threshold
                    if uncovered_mask.any():
                        uncovered_z = z_samples[uncovered_mask]
                        n_spawn = min(len(uncovered_z), len(free_slots), max_spawn_per_call)
                        if n_spawn > 0:
                            spawn_idx = free_slots[:n_spawn]
                            mu.data[spawn_idx] = uncovered_z[:n_spawn]
                            log_s.data[spawn_idx].fill_(0.0)
                            log_alpha.data[spawn_idx].fill_(0.0)
                            model.split_gate.data[spawn_idx] = 0.0
                            model.activation_density[spawn_idx] = 0.0
                            model.error_density[spawn_idx] = 0.0
                            if hasattr(router, 'grad_mu_ema'):
                                router.grad_mu_ema[spawn_idx] = 0

                            new_mask = active_mask.clone()
                            new_mask[spawn_idx] = True
                            _set_active_mask(router, new_mask)

                            param_store.init_unit(spawn_idx[0].item())
                            for i in range(1, n_spawn):
                                param_store.init_unit(spawn_idx[i].item())

                            num_spawned = n_spawn

        return TopologyAction(
            num_pruned=num_pruned,
            num_split=num_split,
            num_spawned=num_spawned,
            num_merged=num_merged,
            merged_indices=merged_indices
        )

    def compute_losses(self, model: 'NGSModel', **kwargs) -> dict[str, torch.Tensor]:
        losses = {
            'entropy': model.entropy_loss(),
            'diversity': model.diversity_loss(),
        }
        if hasattr(model, 'split_gate') and hasattr(model, 'error_density'):
            losses['split_gate'] = model.split_gate_loss()
        return losses


class MergeAwareManager(ContinuousDensityManager):
    """Topology manager with differentiable merge operator."""

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.merge_threshold = config.merge_threshold
        self.merge_weight = config.merge_weight

    def adapt_topology(
        self,
        model: 'NGSModel',
        z_samples: Optional[torch.Tensor] = None,
        max_spawn_per_call: int = 5,
        **kwargs
    ) -> TopologyAction:
        # First run standard adaptation
        action = super().adapt_topology(model, z_samples, max_spawn_per_call, **kwargs)

        # Then attempt merges
        router = model.router
        param_store = model.param_store

        active_mask = _get_active_mask(router)
        mu, log_s, log_alpha = _get_flat_attrs(router)
        active_idx = active_mask.nonzero(as_tuple=True)[0]

        if len(active_idx) < 2:
            return action

        # Compute pairwise cosine similarity of means
        mu_active = mu[active_idx]
        mu_norm = F.normalize(mu_active, p=2, dim=-1)
        sim_matrix = torch.mm(mu_norm, mu_norm.t())

        # Mask diagonal
        mask = torch.eye(len(active_idx), dtype=torch.bool, device=mu.device)
        sim_matrix.masked_fill_(mask, -1)

        # Find pairs above threshold
        max_sim, max_idx = sim_matrix.max(dim=-1)
        merge_candidates = max_sim > (1 - self.merge_threshold)

        num_merged = 0
        merged_indices = []

        if merge_candidates.any():
            for i in merge_candidates.nonzero(as_tuple=True)[0]:
                j = max_idx[i].item()
                if i >= j:  # Avoid double counting
                    continue
                if max_sim[i] > (1 - self.merge_threshold):
                    target = active_idx[i].item()
                    source = active_idx[j].item()

                    # Weighted merge in parameter space
                    w = 0.5  # Equal weight
                    mu.data[target].lerp_(mu[source], w)
                    log_s.data[target].lerp_(log_s[source], w)
                    log_alpha.data[target].lerp_(log_alpha[source], w)

                    # Merge parameter store
                    param_store.merge_units(target, source, w)

                    # Deactivate source
                    new_mask = active_mask.clone()
                    new_mask[source] = False
                    _set_active_mask(router, new_mask)
                    if hasattr(router, 'grad_mu_ema'):
                        router.grad_mu_ema[source] = 0

                    num_merged += 1
                    merged_indices.append((target, source))

        return TopologyAction(
            num_pruned=action.num_pruned,
            num_split=action.num_split,
            num_spawned=action.num_spawned,
            num_merged=num_merged,
            merged_indices=merged_indices
        )

    def compute_losses(self, model: 'NGSModel', **kwargs) -> dict[str, torch.Tensor]:
        losses = super().compute_losses(model, **kwargs)
        # Add merge regularization loss
        losses['merge_reg'] = self._merge_regularization(model)
        return losses

    def _merge_regularization(self, model: 'NGSModel') -> torch.Tensor:
        """Encourage similar units to merge."""
        router = model.router
        active_mask = _get_active_mask(router)
        mu, _, _ = _get_flat_attrs(router)
        active_idx = active_mask.nonzero(as_tuple=True)[0]

        if len(active_idx) < 2:
            return torch.tensor(0.0, device=mu.device)

        mu_active = mu[active_idx]
        mu_norm = F.normalize(mu_active, p=2, dim=-1)
        sim_matrix = torch.mm(mu_norm, mu_norm.t())

        mask = ~torch.eye(len(active_idx), dtype=torch.bool, device=mu.device)
        # Penalize high similarity (encourage merge or separation)
        high_sim = sim_matrix[mask]
        return -high_sim.mean() * self.merge_weight


def build_topology_manager(config: NGSConfig) -> BaseTopologyManager:
    """Factory function to build topology manager from config."""
    if config.topology_control == 'heuristic':
        return HeuristicManager(config)
    elif config.topology_control == 'continuous_density':
        return ContinuousDensityManager(config)
    elif config.topology_control == 'merge_aware':
        return MergeAwareManager(config)
    else:
        raise ValueError(f"Unknown topology control: {config.topology_control}")

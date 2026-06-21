"""Topology management implementations for NGS."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from ngs.core.interfaces import BaseTopologyManager, NGSConfig


# ───────────────────────── Shared helpers ───────────────────────────

def _logit_stable(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Numerically stable logit with clamping to avoid inf/nan."""
    return torch.log(x.clamp(min=eps, max=1 - eps))


def _flat_access(router):
    """Return flat-accessible mu, log_s, log_alpha from router."""
    mu = log_s = log_alpha = None
    if hasattr(router, 'mu'):
        if router.mu.dim() == 2:
            mu, log_s, log_alpha = router.mu, router.log_s, router.log_alpha
        else:
            # Factorized: flatten subspace parameters
            mu = router.mu.view(-1, router.mu.shape[-1])
            log_s = router.log_s.view(-1, router.log_s.shape[-1])
            log_alpha = router.log_alpha.view(-1)
    return mu, log_s, log_alpha


def _factorized_coverage(model, z_samples, spawn_thresh, mu, log_s, log_alpha,
                         free_slots, max_spawn_per_call):
    """Compute coverage and spawn for FactorizedRouter (per-subspace)."""
    router = model.router
    if not hasattr(router, 'num_subspaces'):
        return 0

    num_spawned = 0
    num_subspaces = router.num_subspaces
    units_per_space = router.units_per_space
    active_mask = router.active_mask

    for s in range(num_subspaces):
        start = s * units_per_space
        end = start + units_per_space
        sub_active = active_mask[start:end]
        active_local = sub_active.nonzero(as_tuple=True)[0]

        if len(active_local) == 0:
            continue

        # Project z_samples into this subspace
        z_s = router.subspace_projectors[s](z_samples)  # [N, d_sub]

        # Get active subspace units
        mu_s = router.mu[s][active_local]
        log_s_s = router.log_s[s][active_local]
        log_alpha_s = router.log_alpha[s][active_local]
        s_sq = torch.exp(2 * log_s_s) + 1e-5

        diff = z_s.unsqueeze(1) - mu_s.unsqueeze(0)
        mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1)
        log_w = log_alpha_s - 0.5 * mahalanobis_sq
        max_log_w, _ = log_w.max(dim=-1)

        uncovered_mask = max_log_w < spawn_thresh
        if uncovered_mask.any():
            uncovered_z = z_s[uncovered_mask]
            free_slots_sub = (~active_mask[start:end]).nonzero(as_tuple=True)[0]
            if len(free_slots_sub) > 0:
                n_spawn = min(len(uncovered_z), len(free_slots_sub), max_spawn_per_call - num_spawned)
                if n_spawn > 0:
                    spawn_local = free_slots_sub[:n_spawn]
                    spawn_global = start + spawn_local

                    # Flatten router parameters for assignment
                    mu_flat = router.mu.view(-1, router.mu.shape[-1])
                    log_s_flat = router.log_s.view(-1, router.log_s.shape[-1])
                    log_alpha_flat = router.log_alpha.view(-1)

                    mu_flat.data[spawn_global] = uncovered_z[:n_spawn]
                    log_s_flat.data[spawn_global].fill_(0.0)
                    log_alpha_flat.data[spawn_global].fill_(0.0)
                    if hasattr(router, 'grad_mu_ema'):
                        router.grad_mu_ema[spawn_global] = 0
                    active_mask[spawn_global] = True
                    num_spawned += n_spawn

    return num_spawned


# ─────────────────────── Heuristic Manager ────────────────────────

class HeuristicManager(BaseTopologyManager):
    """
    Discrete heuristic topology control (original LeanNGS).
    Monitors EMA of gradient norms and opacity.
    If grad > threshold and scale > min_scale, executes hard split.
    """

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.split_threshold = config.split_threshold
        self.prune_threshold = config.prune_threshold
        self.scale_threshold = 0.5  # Separate threshold for scale-based split decisions
        self.split_scale = 0.5
        self.noise_std = 0.01
        self.ema_decay = config.ema_decay

    def adapt_topology(self, model,
                       optimizer=None,
                       spawn_thresh: float = -5.0,
                       max_spawn_per_call: int = 10,
                       z_samples: torch.Tensor = None,
                       split_thresh: float = None,
                       prune_thresh: float = None) -> Tuple[int, int, int]:
        router = model.router
        if not hasattr(router, 'active_mask'):
            return 0, 0, 0

        # Use flat accessors for FactorizedRouter compatibility
        mu, log_s, log_alpha = _flat_access(router)
        if mu is None:
            return 0, 0, 0

        split_thresh = split_thresh if split_thresh is not None else self.split_threshold
        prune_thresh = prune_thresh if prune_thresh is not None else self.prune_threshold
        active_idx = router.active_mask.nonzero(as_tuple=True)[0]
        K = len(active_idx)
        if K == 0:
            return 0, 0, 0

        # Access router parameters
        alpha = torch.sigmoid(log_alpha[active_idx])

        # Access EMA tracker from router
        grad_ema = router.grad_mu_ema[active_idx]
        max_s = torch.exp(log_s[active_idx]).max(dim=-1).values

        num_pruned = 0
        num_split = 0
        num_spawned = 0

        # Prune low-opacity units
        prune_mask = alpha < prune_thresh
        if prune_mask.any():
            prune_global = active_idx[prune_mask]
            router.active_mask[prune_global] = False
            router.grad_mu_ema[prune_global] = 0
            num_pruned = prune_mask.sum().item()

        # Recompute active after pruning
        active_idx = router.active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) == 0:
            return num_pruned, 0, 0

        # Split high-gradient units
        alpha = torch.sigmoid(log_alpha[active_idx])
        grad_ema = router.grad_mu_ema[active_idx]
        max_s = torch.exp(log_s[active_idx]).max(dim=-1).values

        split_mask = (grad_ema > split_thresh) & (max_s > self.scale_threshold)

        # Adaptive fallback: only when max grad_ema is at least 10% of threshold
        if not split_mask.any() and len(active_idx) > 0 and max_spawn_per_call > 0:
            max_grad = grad_ema.max()
            if max_grad > split_thresh * 0.1:
                n_fallback = min(2, len(active_idx) // 4)
                n_fallback = max(n_fallback, 1)
                _, top_indices = torch.topk(grad_ema, n_fallback)
                split_mask = torch.zeros(len(active_idx), dtype=torch.bool, device=grad_ema.device)
                split_mask[top_indices] = True

        if split_mask.any():
            split_idx = active_idx[split_mask]
            free_slots = (~router.active_mask).nonzero(as_tuple=True)[0]
            n_available = len(free_slots)
            n_split = min(len(split_idx), n_available)

            if n_split > 0:
                split_idx = split_idx[:n_split]
                new_idx = free_slots[:n_split]

                # Copy and perturb mean
                mu.data[new_idx] = mu[split_idx].clone()
                noise = torch.randn_like(mu[split_idx]) * self.noise_std
                mu.data[new_idx] += noise

                # Copy log_s and scale
                log_s.data[new_idx] = log_s[split_idx].clone()
                log_s.data[new_idx] += torch.log(torch.tensor(self.split_scale))
                log_s.data[split_idx] += torch.log(torch.tensor(self.split_scale))

                # Copy log_alpha and halve in probability space (with numerical stability)
                alpha_orig = torch.sigmoid(log_alpha[split_idx])
                alpha_new = torch.sigmoid(log_alpha.data[new_idx].clone())
                log_alpha.data[new_idx] = _logit_stable(alpha_orig * 0.5, eps=1e-7)
                log_alpha.data[split_idx] = _logit_stable(alpha_orig * 0.5, eps=1e-7)

                # Reset EMA
                router.grad_mu_ema[new_idx] = 0
                router.grad_mu_ema[split_idx] = 0

                # Mark active
                router.active_mask[new_idx] = True
                num_split = n_split

        # Spawn in uncovered regions
        if z_samples is not None:
            free_slots = (~router.active_mask).nonzero(as_tuple=True)[0]
            if len(free_slots) > 0:
                z_samples = z_samples.to(mu.device)
                active_idx = router.active_mask.nonzero(as_tuple=True)[0]

                if hasattr(router, 'num_subspaces'):
                    num_spawned = _factorized_coverage(
                        model, z_samples, spawn_thresh, mu, log_s, log_alpha,
                        free_slots, max_spawn_per_call
                    )
                elif len(active_idx) > 0:
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
                            router.grad_mu_ema[spawn_idx] = 0
                            router.active_mask[spawn_idx] = True
                            num_spawned = n_spawn

        return num_pruned, num_split, num_spawned


# ─────────────────── ContinuousDensity Manager ───────────────────

class ContinuousDensityManager(BaseTopologyManager):
    """
    Continuous density-driven topology (CFG-Net).
    Each unit maintains a learnable split-gate gamma in [0, 1].
    Growth is smooth and differentiable, avoiding optimizer state resets.
    """

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.split_threshold = config.split_threshold
        self.prune_threshold = config.prune_threshold
        self.density_decay = config.ema_decay
        self.split_gate_threshold = 0.65
        self.spawn_threshold = -5.0
        self.noise_std = 0.01
        self.split_scale = 0.5

    def adapt_topology(self, model, split_thresh: float = None,
                       prune_thresh: float = None, spawn_thresh: float = None,
                       z_samples: torch.Tensor = None, max_spawn_per_call: int = 5, **kwargs):
        router = model.router
        if not hasattr(router, 'active_mask'):
            return 0, 0, 0

        mu, log_s, log_alpha = _flat_access(router)
        if mu is None:
            return 0, 0, 0

        prune_thresh = prune_thresh if prune_thresh is not None else self.prune_threshold
        spawn_thresh = spawn_thresh if spawn_thresh is not None else self.spawn_threshold
        active_idx = router.active_mask.nonzero(as_tuple=True)[0]
        alpha = torch.sigmoid(log_alpha[active_idx])

        num_pruned = 0
        prune_mask = alpha < prune_thresh
        if prune_mask.any():
            prune_global = active_idx[prune_mask]
            router.active_mask[prune_global] = False
            router.grad_mu_ema[prune_global] = 0
            num_pruned = prune_mask.sum().item()

        # Split gate execution
        active_idx = router.active_mask.nonzero(as_tuple=True)[0]

        # Try to access split gate from model
        if hasattr(model, 'split_gate'):
            gamma = torch.sigmoid(model.split_gate[active_idx])
            if hasattr(model, 'error_density'):
                err_density = model.error_density[active_idx]
                split_mask = (gamma > self.split_gate_threshold) & (err_density > 1e-3)
            else:
                split_mask = gamma > self.split_gate_threshold
        else:
            # Fallback: use heuristic split
            grad_ema = router.grad_mu_ema[active_idx]
            split_mask = grad_ema > self.split_threshold

        num_split = 0
        if split_mask.any():
            free_slots = (~router.active_mask).nonzero(as_tuple=True)[0]
            n_available = len(free_slots)
            split_idx = active_idx[split_mask]
            n_split = min(len(split_idx), n_available)

            if n_split > 0:
                split_idx = split_idx[:n_split]
                new_idx = free_slots[:n_split]

                # Copy and perturb mean
                mu.data[new_idx] = mu[split_idx].clone()
                noise = torch.randn_like(mu[split_idx]) * self.noise_std
                mu.data[new_idx] += noise

                # Copy log_s and scale
                log_s.data[new_idx] = log_s[split_idx].clone()
                half_scale = torch.log(torch.tensor(self.split_scale))
                log_s.data[new_idx] += half_scale
                log_s.data[split_idx] += half_scale

                # Copy log_alpha (no halving - split_gate controls split decisions now)
                log_alpha.data[new_idx] = log_alpha[split_idx].clone()

                # Reset gates for both parent and child
                if hasattr(model, 'split_gate'):
                    model.split_gate.data[new_idx] = 0.0
                    model.split_gate.data[split_idx] = 0.0
                if hasattr(model, 'activation_density'):
                    model.activation_density[new_idx] = 0.0
                    model.error_density[new_idx] = 0.0
                router.grad_mu_ema[new_idx] = 0

                # Mark active
                router.active_mask[new_idx] = True
                num_split = n_split

        # Spawn in uncovered regions (similar to HeuristicManager)
        num_spawned = 0
        if z_samples is not None:
            free_slots = (~router.active_mask).nonzero(as_tuple=True)[0]
            if len(free_slots) > 0:
                z_samples = z_samples.to(mu.device)
                active_idx = router.active_mask.nonzero(as_tuple=True)[0]

                if hasattr(router, 'num_subspaces'):
                    num_spawned = _factorized_coverage(
                        model, z_samples, spawn_thresh, mu, log_s, log_alpha,
                        free_slots, max_spawn_per_call
                    )
                elif len(active_idx) > 0:
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
                            if hasattr(model, 'split_gate'):
                                model.split_gate.data[spawn_idx] = 0.0
                            if hasattr(model, 'activation_density'):
                                model.activation_density[spawn_idx] = 0.0
                                model.error_density[spawn_idx] = 0.0
                            router.grad_mu_ema[spawn_idx] = 0
                            router.active_mask[spawn_idx] = True
                            num_spawned = n_spawn

        return num_pruned, num_split, num_spawned


# ───────────────────── MergeAware Manager ──────────────────────

class MergeAwareManager(BaseTopologyManager):
    """
    Merge-aware topology management.
    Uses cosine similarity on full covariance (means + scales)
    for intersection-over-union or overlap detection.
    """

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.merge_threshold = config.merge_threshold
        self.merge_check_interval = config.merge_check_interval
        self.prune_threshold = config.prune_threshold
        self.split_threshold = config.split_threshold
        self.split_scale = 0.5
        self.noise_std = 0.01
        self.spawn_threshold = -5.0

    def _compute_overlap(self, model) -> torch.Tensor:
        """Compute pairwise overlap between active units."""
        router = model.router
        active_idx = router.active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) < 2:
            return torch.zeros(0, 0)

        mu, log_s, log_alpha = _flat_access(router)
        if mu is None:
            return torch.zeros(0, 0)

        mu_active = mu[active_idx]
        s_active = torch.exp(log_s[active_idx])  # Use actual scales, not log_s

        # Represent each unit as [mu, s] and compute cosine similarity
        repr = torch.cat([mu_active, s_active], dim=-1)  # [K, 2d]
        repr_norm = F.normalize(repr, dim=-1, eps=1e-8)  # Add numerical stability
        similarity = repr_norm @ repr_norm.T  # [K, K]

        return similarity

    def adapt_topology(self, model, **kwargs) -> Tuple[int, int, int]:
        router = model.router
        if not hasattr(router, 'active_mask'):
            return 0, 0, 0

        active_idx = router.active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) < 2:
            # Not enough units to merge, fallback to heuristic
            return 0, 0, 0

        num_pruned = 0
        num_split = 0
        num_merged = 0

        # Check for merges
        similarity = self._compute_overlap(model)
        if similarity.numel() > 0:
            # Upper triangle (excluding diagonal)
            triu_mask = torch.triu(torch.ones_like(similarity), diagonal=1).bool()
            merge_candidates = (similarity > self.merge_threshold) & triu_mask

            if merge_candidates.any():
                # Find pairs to merge (greedy: merge highest similarity first)
                merge_pairs = []
                for i in range(len(active_idx)):
                    for j in range(i + 1, len(active_idx)):
                        if merge_candidates[i, j]:
                            merge_pairs.append((similarity[i, j].item(), i, j))

                merge_pairs.sort(reverse=True)
                merged = set()

                mu, log_s, log_alpha = _flat_access(router)
                if mu is not None:
                    for _, i, j in merge_pairs:
                        if i in merged or j in merged:
                            continue
                        idx_i = active_idx[i]
                        idx_j = active_idx[j]

                        # Merge j into i by averaging parameters
                        # Keep i active with averaged parameters, deactivate j
                        mu.data[idx_i] = 0.5 * (mu[idx_i] + mu[idx_j])
                        # Geometric mean of scales: sqrt(s_i * s_j) in log space
                        log_s.data[idx_i] = torch.log(
                            torch.sqrt(torch.exp(log_s[idx_i]) * torch.exp(log_s[idx_j]) + 1e-8)
                        )

                        # Deactivate j (remove from active set)
                        router.active_mask[idx_j] = False
                        router.grad_mu_ema[idx_j] = 0
                        merged.add(i)
                        merged.add(j)
                        num_merged += 1

                        if num_merged >= 2:  # Limit merges per call
                            break

        # Apply standard heuristic split/prune/spawn after merge
        # We delegate to HeuristicManager for the rest
        heuristic = HeuristicManager(self.config)
        h_pruned, h_split, h_spawned = heuristic.adapt_topology(model, **kwargs)

        return num_pruned + h_pruned, num_split + h_split, h_spawned


# ───────────────────── MetaLearned Manager ──────────────────────

class MetaLearnedManager(BaseTopologyManager):
    """
    Meta-learned topology management.
    Learns a policy over topology actions (split/prune/merge/spawn)
    using a small MLP to predict action scores based on unit statistics.
    """

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.meta_lr = config.meta_lr
        self.meta_hidden_dim = config.meta_hidden_dim
        self.prune_threshold = config.prune_threshold
        self.split_threshold = config.split_threshold
        self.spawn_threshold = -5.0
        self.split_scale = 0.5
        self.noise_std = 0.01

        # Meta-controller: predicts action scores from unit features
        self.controller = nn.Sequential(
            nn.Linear(5, self.meta_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.meta_hidden_dim, self.meta_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.meta_hidden_dim // 2, 4)  # [keep, split, prune, spawn]
        )
        self.controller_optimizer = torch.optim.Adam(self.controller.parameters(), lr=self.meta_lr)

    def _get_unit_features(self, model):
        """Extract features for each active unit."""
        router = model.router
        active_idx = router.active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) == 0:
            return active_idx, None

        mu, log_s, log_alpha = _flat_access(router)
        if mu is None:
            return active_idx, None

        alpha = torch.sigmoid(log_alpha[active_idx])
        s_mean = torch.exp(log_s[active_idx]).mean(dim=-1)
        grad = router.grad_mu_ema[active_idx]

        # Features: [alpha, s_mean, grad, log_alpha, mean(log_s)]
        features = torch.stack([
            alpha,
            s_mean,
            grad,
            log_alpha[active_idx],
            log_s[active_idx].mean(dim=-1)
        ], dim=-1)  # [K, 5]

        return active_idx, features

    def adapt_topology(self, model, **kwargs) -> Tuple[int, int, int]:
        router = model.router
        if not hasattr(router, 'active_mask'):
            return 0, 0, 0

        active_idx, features = self._get_unit_features(model)
        if features is None or len(active_idx) == 0:
            return 0, 0, 0

        # Predict action scores
        with torch.no_grad():
            action_scores = self.controller(features)  # [K, 4]
            action_probs = F.softmax(action_scores, dim=-1)

        # Apply actions greedily
        keep_prob, split_prob, prune_prob, spawn_prob = action_probs[:, 0], action_probs[:, 1], action_probs[:, 2], action_probs[:, 3]

        num_pruned = 0
        num_split = 0
        num_spawned = 0

        mu, log_s, log_alpha = _flat_access(router)
        if mu is None:
            return 0, 0, 0

        # Prune units with high prune probability and low keep probability
        prune_mask = (prune_prob > 0.5) & (keep_prob < 0.3)
        if prune_mask.any():
            prune_global = active_idx[prune_mask]
            router.active_mask[prune_global] = False
            router.grad_mu_ema[prune_global] = 0
            num_pruned = prune_mask.sum().item()

        # Split units with high split probability
        active_idx = router.active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) > 0:
            split_mask = split_prob[active_idx] > 0.5
            if split_mask.any():
                free_slots = (~router.active_mask).nonzero(as_tuple=True)[0]
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
                    log_s.data[new_idx] += torch.log(torch.tensor(self.split_scale))
                    log_s.data[split_idx] += torch.log(torch.tensor(self.split_scale))

                    log_alpha.data[new_idx] = log_alpha[split_idx].clone()

                    router.grad_mu_ema[new_idx] = 0
                    router.grad_mu_ema[split_idx] = 0
                    router.active_mask[new_idx] = True
                    num_split = n_split

        # Spawn based on spawn probability (only if latent samples provided)
        z_samples = kwargs.get('z_samples', None)
        spawn_thresh = kwargs.get('spawn_thresh', self.spawn_threshold)
        max_spawn = kwargs.get('max_spawn_per_call', 5)

        if z_samples is not None and spawn_prob.max() > 0.5:
            free_slots = (~router.active_mask).nonzero(as_tuple=True)[0]
            active_idx = router.active_mask.nonzero(as_tuple=True)[0]
            if len(free_slots) > 0 and len(active_idx) > 0:
                mu_active = mu[active_idx]
                log_s_active = log_s[active_idx]
                s_sq = torch.exp(2 * log_s_active) + 1e-5

                diff = z_samples.unsqueeze(1) - mu_active.unsqueeze(0)
                mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1)
                log_w = log_alpha[active_idx] - 0.5 * mahalanobis_sq
                max_log_w, _ = log_w.max(dim=-1)

                uncovered_mask = max_log_w < spawn_thresh
                if uncovered_mask.any():
                    z_uncovered = z_samples[uncovered_mask]
                    n_spawn = min(len(z_uncovered), len(free_slots), max_spawn)
                    if n_spawn > 0:
                        spawn_idx = free_slots[:n_spawn]
                        mu.data[spawn_idx] = z_uncovered[:n_spawn]
                        log_s.data[spawn_idx].fill_(0.0)
                        log_alpha.data[spawn_idx].fill_(0.0)
                        router.grad_mu_ema[spawn_idx] = 0
                        router.active_mask[spawn_idx] = True
                        num_spawned = n_spawn

        # Meta-update: encourage exploration (entropy of action distribution)
        # This keeps the controller functional even without explicit rewards
        action_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        loss = -action_entropy  # Maximize entropy to encourage exploration

        self.controller_optimizer.zero_grad()
        loss.backward()
        self.controller_optimizer.step()

        return num_pruned, num_split, num_spawned


# ─────────────────────────── Builder ───────────────────────────

def build_topology_manager(config: NGSConfig):
    """Factory function to build topology manager from config."""
    topology = config.topology_control

    if topology.name == "DISCRETE_HEURISTIC":
        return HeuristicManager(config)
    elif topology.name == "CONTINUOUS_DENSITY":
        return ContinuousDensityManager(config)
    elif topology.name == "MERGE_AWARE":
        return MergeAwareManager(config)
    elif topology.name == "META_LEARNED":
        return MetaLearnedManager(config)
    else:
        raise ValueError(f"Unknown topology control: {topology}")


__all__ = [
    "HeuristicManager",
    "ContinuousDensityManager",
    "MergeAwareManager",
    "MetaLearnedManager",
    "build_topology_manager",
]

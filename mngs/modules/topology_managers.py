"""Topology management implementations for MNGS."""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


class BaseTopologyManager(ABC):
    """Base class for all topology management strategies."""
    
    @abstractmethod
    def adapt_topology(self, model, **kwargs):
        """
        Adapt the network topology based on current state.
        
        Args:
            model: The MNGS model instance
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Tuple of (num_pruned, num_split, num_spawned)
        """
        pass


class HeuristicManager(BaseTopologyManager):
    """
    Discrete heuristic topology control (original LeanNGS).
    
    Monitors EMA of gradient norms and opacity.
    If grad > threshold and scale > min_scale, executes hard split.
    """
    
    def __init__(self, 
                 split_threshold: float = 0.05,
                 prune_threshold: float = 0.01,
                 split_scale: float = 0.5,
                 noise_std: float = 0.01,
                 ema_decay: float = 0.99):
        self.split_threshold = split_threshold
        self.prune_threshold = prune_threshold
        self.split_scale = split_scale
        self.noise_std = noise_std
        self.ema_decay = ema_decay
    
    def adapt_topology(self, model, 
                       optimizer=None,
                       spawn_thresh: float = -5.0,
                       max_spawn_per_call: int = 10,
                       z_samples: torch.Tensor = None,
                       split_thresh: float = None,
                       prune_thresh: float = None) -> Tuple[int, int, int]:
        """
        Apply heuristic topology changes.
        
        Args:
            model: MNGS model with active_mask, mu, log_s, etc.
            optimizer: Optional optimizer for state manipulation
            spawn_thresh: Threshold for spawning new units
            max_spawn_per_call: Max units to spawn at once
            z_samples: [N, d] latent samples for coverage analysis
            split_thresh: Overrides self.split_threshold if provided
            prune_thresh: Overrides self.prune_threshold if provided
            
        Returns:
            (num_pruned, num_split, num_spawned)
        """
        if not hasattr(model.router, 'active_mask'):
            return 0, 0, 0
        
        # Use flat accessors for FactorizedRouter compatibility
        if hasattr(model.router, 'flat_mu'):
            mu, log_s, log_alpha = model.router.flat_mu, model.router.flat_log_s, model.router.flat_log_alpha
        else:
            mu, log_s, log_alpha = model.router.mu, model.router.log_s, model.router.log_alpha
        
        split_thresh = split_thresh if split_thresh is not None else self.split_threshold
        prune_thresh = prune_thresh if prune_thresh is not None else self.prune_threshold
        active_idx = model.router.active_mask.nonzero(as_tuple=True)[0]
        K = len(active_idx)
        if K == 0:
            return 0, 0, 0
        
        # Access router parameters
        alpha = torch.sigmoid(log_alpha[active_idx])
        
        # Access EMA tracker from router
        grad_ema = model.router.grad_mu_ema[active_idx]
        max_s = torch.exp(log_s[active_idx]).max(dim=-1).values
        
        num_pruned = 0
        num_split = 0
        num_spawned = 0
        
        # Prune low-opacity units
        prune_mask = alpha < prune_thresh
        if prune_mask.any():
            prune_idx = active_idx[prune_mask]
            model.router.active_mask[prune_idx] = False
            model.router.grad_mu_ema[prune_idx] = 0
            num_pruned = prune_mask.sum().item()
        
        # Recompute active after pruning
        active_idx = model.router.active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) == 0:
            return num_pruned, 0, 0
        
        # Split high-gradient units
        alpha = torch.sigmoid(log_alpha[active_idx])
        grad_ema = model.router.grad_mu_ema[active_idx]
        max_s = torch.exp(log_s[active_idx]).max(dim=-1).values
        
        split_mask = (grad_ema > split_thresh) & (max_s > split_thresh)
        
        # Adaptive fallback: only when max grad_ema is at least 10% of threshold
        # (meaning gradients exist but are too small for the absolute threshold)
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
            free_slots = (~model.router.active_mask).nonzero(as_tuple=True)[0]
            n_available = len(free_slots)
            n_split = min(len(split_idx), n_available)
            
            if n_split > 0:
                split_idx = split_idx[:n_split]
                new_idx = free_slots[:n_split]
                
                # Copy and perturb mean
                mu.data[new_idx] = mu[split_idx].clone()
                noise = torch.randn_like(mu[split_idx]) * self.noise_std
                mu.data[new_idx] += noise
                
                # Copy log_s and scale down
                log_s.data[new_idx] = log_s[split_idx].clone()
                log_s.data[new_idx] += torch.log(torch.tensor(self.split_scale))
                log_s.data[split_idx] += torch.log(torch.tensor(self.split_scale))
                
                # Copy and halve alpha in probability space
                log_alpha.data[new_idx] = log_alpha[split_idx].clone()
                alpha_new = torch.sigmoid(log_alpha.data[new_idx])
                log_alpha.data[new_idx] = torch.logit(alpha_new * 0.5, eps=1e-8)
                alpha_split = torch.sigmoid(log_alpha.data[split_idx])
                log_alpha.data[split_idx] = torch.logit(alpha_split * 0.5, eps=1e-8)
                
                # Reset EMA
                model.router.grad_mu_ema[new_idx] = 0
                model.router.grad_mu_ema[split_idx] = 0
                
                # Mark active
                model.router.active_mask[new_idx] = True
                num_split = n_split
        
        # Spawn in uncovered regions
        if z_samples is not None:
            free_slots = (~model.router.active_mask).nonzero(as_tuple=True)[0]
            if len(free_slots) > 0:
                z_samples = z_samples.to(mu.device)
                active_idx = model.router.active_mask.nonzero(as_tuple=True)[0]
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
                        model.router.grad_mu_ema[spawn_idx] = 0
                        model.router.active_mask[spawn_idx] = True
                        num_spawned = n_spawn
        
        return num_pruned, num_split, num_spawned


class ContinuousDensityManager(BaseTopologyManager):
    """
    Continuous density-driven topology (CFG-Net).
    
    Each unit maintains a learnable split-gate gamma in [0, 1].
    Growth is smooth and differentiable, avoiding optimizer state resets.
    """
    
    def __init__(self,
                 split_threshold: float = 0.05,
                 prune_threshold: float = 0.01,
                 density_decay: float = 0.99,
                 split_gate_threshold: float = 0.7,
                 noise_std: float = 0.01):
        self.split_threshold = split_threshold
        self.prune_threshold = prune_threshold
        self.density_decay = density_decay
        self.split_gate_threshold = split_gate_threshold
        self.noise_std = noise_std
    
    @staticmethod
    def _get_flat_attrs(router):
        """Get flat-accessible mu, log_s, log_alpha from router."""
        if hasattr(router, 'flat_mu'):
            return router.flat_mu, router.flat_log_s, router.flat_log_alpha
        return router.mu, router.log_s, router.log_alpha
    
    def adapt_topology(self, model, split_thresh: float = None,
                       prune_thresh: float = None, **kwargs):
        """
        Differentiable topology adaptation via split gates.
        
        Executes a split when split_gate > threshold and error density is high.
        """
        if not hasattr(model.router, 'active_mask'):
            return 0, 0, 0
        
        prune_thresh = prune_thresh if prune_thresh is not None else self.prune_threshold
        active_idx = model.router.active_mask.nonzero(as_tuple=True)[0]
        mu, log_s, log_alpha = self._get_flat_attrs(model.router)
        alpha = torch.sigmoid(log_alpha[active_idx])
        
        num_pruned = 0
        prune_mask = alpha < prune_thresh
        if prune_mask.any():
            prune_idx = active_idx[prune_mask]
            model.router.active_mask[prune_idx] = False
            model.router.grad_mu_ema[prune_idx] = 0
            num_pruned = prune_mask.sum().item()
        
        # Split gate execution
        active_idx = model.router.active_mask.nonzero(as_tuple=True)[0]
        gamma = torch.sigmoid(model.split_gate[active_idx])
        err_density = model.error_density[active_idx]
        split_mask = (gamma > self.split_gate_threshold) & (err_density > 1e-3)
        
        if split_mask.any():
            free_slots = (~model.router.active_mask).nonzero(as_tuple=True)[0]
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
                half_scale = torch.log(torch.tensor(0.5))
                log_s.data[new_idx] += half_scale
                log_s.data[split_idx] += half_scale
                
                # Copy log_alpha
                log_alpha.data[new_idx] = log_alpha[split_idx].clone()
                
                # Reset gates for both parent and child
                model.split_gate.data[new_idx] = 0.0
                model.split_gate.data[split_idx] = 0.0
                model.activation_density[new_idx] = 0.0
                model.error_density[new_idx] = 0.0
                
                # Mark active
                model.router.active_mask[new_idx] = True
                
                return num_pruned, n_split, 0
        
        return num_pruned, 0, 0

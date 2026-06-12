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
        split_thresh = split_thresh if split_thresh is not None else self.split_threshold
        prune_thresh = prune_thresh if prune_thresh is not None else self.prune_threshold
        active_idx = model.router.active_mask.nonzero(as_tuple=True)[0]
        K = len(active_idx)
        if K == 0:
            return 0, 0, 0
        
        # Access router parameters
        log_alpha = model.router.log_alpha[active_idx]
        alpha = torch.sigmoid(log_alpha)
        
        # Access EMA tracker from model
        grad_ema = model.grad_mu_ema[active_idx]
        max_s = torch.exp(model.router.log_s[active_idx]).max(dim=-1).values
        
        num_pruned = 0
        num_split = 0
        num_spawned = 0
        
        # Prune low-opacity units
        prune_mask = alpha < prune_thresh
        if prune_mask.any():
            prune_idx = active_idx[prune_mask]
            model.router.active_mask[prune_idx] = False
            model.grad_mu_ema[prune_idx] = 0
            num_pruned = prune_mask.sum().item()
        
        # Recompute active after pruning
        active_idx = model.router.active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) == 0:
            return num_pruned, 0, 0
        
        # Split high-gradient units
        log_alpha = model.router.log_alpha[active_idx]
        alpha = torch.sigmoid(log_alpha)
        grad_ema = model.grad_mu_ema[active_idx]
        max_s = torch.exp(model.router.log_s[active_idx]).max(dim=-1).values
        
        split_mask = (grad_ema > split_thresh) & (max_s > split_thresh)
        if split_mask.any():
            split_idx = active_idx[split_mask]
            free_slots = (~model.router.active_mask).nonzero(as_tuple=True)[0]
            n_available = len(free_slots)
            n_split = min(len(split_idx), n_available)
            
            if n_split > 0:
                split_idx = split_idx[:n_split]
                new_idx = free_slots[:n_split]
                
                # Copy and perturb mean
                model.router.mu.data[new_idx] = model.router.mu[split_idx].clone()
                noise = torch.randn_like(model.router.mu[split_idx]) * self.noise_std
                model.router.mu.data[new_idx] += noise
                
                # Copy log_s and scale down
                model.router.log_s.data[new_idx] = model.router.log_s[split_idx].clone()
                model.router.log_s.data[new_idx] += torch.log(torch.tensor(self.split_scale))
                model.router.log_s.data[split_idx] += torch.log(torch.tensor(self.split_scale))
                
                # Copy and halve alpha
                model.router.log_alpha.data[new_idx] = model.router.log_alpha[split_idx].clone()
                model.router.log_alpha.data[new_idx] += torch.log(torch.tensor(0.5))
                model.router.log_alpha.data[split_idx] += torch.log(torch.tensor(0.5))
                
                # Reset EMA
                model.grad_mu_ema[new_idx] = 0
                model.grad_mu_ema[split_idx] = 0
                
                # Mark active
                model.router.active_mask[new_idx] = True
                num_split = n_split
        
        # Spawn in uncovered regions
        if z_samples is not None:
            free_slots = (~model.router.active_mask).nonzero(as_tuple=True)[0]
            if len(free_slots) > 0:
                z_samples = z_samples.to(model.router.mu.device)
                active_idx = model.router.active_mask.nonzero(as_tuple=True)[0]
                mu_active = model.router.mu[active_idx]
                log_s_active = model.router.log_s[active_idx]
                s_sq = torch.exp(2 * log_s_active) + 1e-5
                
                diff = z_samples.unsqueeze(1) - mu_active.unsqueeze(0)
                mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1)
                log_w = model.router.log_alpha[active_idx] - 0.5 * mahalanobis_sq
                max_log_w, _ = log_w.max(dim=-1)
                
                uncovered_mask = max_log_w < spawn_thresh
                if uncovered_mask.any():
                    uncovered_z = z_samples[uncovered_mask]
                    n_spawn = min(len(uncovered_z), len(free_slots), max_spawn_per_call)
                    if n_spawn > 0:
                        spawn_idx = free_slots[:n_spawn]
                        model.router.mu.data[spawn_idx] = uncovered_z[:n_spawn]
                        model.router.log_s.data[spawn_idx].fill_(0.0)
                        model.router.log_alpha.data[spawn_idx].fill_(0.0)
                        model.grad_mu_ema[spawn_idx] = 0
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
                 gamma_init: float = 0.0):
        self.split_threshold = split_threshold
        self.prune_threshold = prune_threshold
        self.density_decay = density_decay
        self.gamma_init = gamma_init
    
    def initialize_gates(self, max_k: int, device: torch.device):
        """Initialize split gates parameters."""
        self.split_gate = nn.Parameter(
            torch.full((max_k,), self.gamma_init, device=device)
        )
        self.activation_density = torch.zeros(max_k, device=device)
        self.error_density = torch.zeros(max_k, device=device)
    
    def adapt_topology(self, model, split_thresh: float = None,
                       prune_thresh: float = None, **kwargs):
        """
        Differentiable topology adaptation via split gates.
        
        The forward pass blends parent and child outputs using gamma.
        Loss regularizer pushes gamma to 0 or 1.
        """
        if not hasattr(model.router, 'active_mask'):
            return 0, 0, 0
        prune_thresh = prune_thresh if prune_thresh is not None else self.prune_threshold
        active_idx = model.router.active_mask.nonzero(as_tuple=True)[0]
        log_alpha = model.router.log_alpha[active_idx]
        alpha = torch.sigmoid(log_alpha)
        
        num_pruned = 0
        prune_mask = alpha < prune_thresh
        if prune_mask.any():
            prune_idx = active_idx[prune_mask]
            model.router.active_mask[prune_idx] = False
            model.grad_mu_ema[prune_idx] = 0
            num_pruned = prune_mask.sum().item()
        
        return num_pruned, 0, 0

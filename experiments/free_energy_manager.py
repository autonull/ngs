"""
FreeEnergyManager: Thermodynamic Self-Regulation for NGS

Extends AutopoieticManager with Free Energy Principle:
- Free Energy = Routing Entropy + lambda * Complexity (K)
- Network grows/shrinks to minimize free energy
- Implements active inference: minimize surprise = maximize evidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/home/me/ngs')

from ngs.modules.topology_managers import AutopoieticManager, _flat_access, HeuristicManager
from ngs.core.interfaces import NGSConfig, BaseRouter
from ngs.modules.routers import MonolithicRouter


class FreeEnergyManager(HeuristicManager):
    """
    Thermodynamic topology control via Free Energy minimization.
    
    Free Energy F = E_routing + lambda * K
    
    Where:
    - E_routing = -H(routing) = Σ w_i log w_i  (negative entropy = surprise)
    - K = number of active Gaussians (complexity cost)
    - lambda = inverse temperature / compute budget
    
    Split/merge/spawn decisions driven by dF/dK < 0 (free energy reduction).
    """
    
    def __init__(self, config: NGSConfig, 
                 free_energy_lambda: float = 0.1,
                 target_free_energy: float = None):
        super().__init__(config)
        self.free_energy_lambda = free_energy_lambda
        self.target_free_energy = target_free_energy
        
        # Override thresholds for FE-based control
        self.tau_split = 2.0  # Higher entropy threshold
        self.tau_merge = 0.8  # Lower overlap threshold
        
        # Track free energy history
        self.free_energy_history = []
        self.K_history = []
        self.split_history = []
        self.merge_history = []
        self.spawn_history = []
        
    def compute_free_energy(self, routing_output) -> torch.Tensor:
        """
        Compute variational free energy.
        
        F = -H(w) + lambda * K
        = Σ w_i log w_i + lambda * K
        
        Lower F = better (less surprise, lower complexity)
        """
        weights = routing_output.weights  # [B, K]
        
        # Routing entropy (surprise)
        # H = -Σ w log w, so -H = Σ w log w
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
        
        # Complexity penalty
        K = routing_output.weights.shape[1]
        complexity = self.free_energy_lambda * K
        
        free_energy = entropy + complexity
        return free_energy
    
    def compute_free_energy_per_gaussian(self, routing_output) -> torch.Tensor:
        """Free energy contribution per Gaussian."""
        weights = routing_output.weights  # [B, K]
        
        # Per-Gaussian: w_i log w_i + lambda
        per_gauss = weights * (weights + 1e-8).log()  # [B, K]
        per_gauss = per_gauss.mean(dim=0) + self.free_energy_lambda  # [K]
        
        return per_gauss
    
    def should_split(self, router, grad_mu_ema) -> tuple:
        """
        Split if it reduces free energy.
        
        Split increases K by 1, reduces routing entropy (more specific Gaussians).
        """
        if router.K >= self.config.max_k:
            return False, -1
        
        # Find Gaussian with highest per-Gaussian free energy
        if not hasattr(self, '_last_routing') or self._last_routing is None:
            return False, -1
            
        per_gauss_fe = self.compute_free_energy_per_gaussian(self._last_routing)
        
        # High per-Gaussian FE = this Gaussian covers too much space
        max_fe_idx = per_gauss_fe.argmax().item()
        max_fe = per_gauss_fe[max_fe_idx].item()
        
        # Threshold: split if per-Gaussian FE > lambda (complexity cost)
        if max_fe > self.free_energy_lambda * 2:
            # Map to active index
            active_idx = router.active_mask.nonzero(as_tuple=True)[0]
            if max_fe_idx < len(active_idx):
                return True, active_idx[max_fe_idx].item()
        
        return False, -1
    
    def should_merge(self, router) -> tuple:
        """
        Merge if it reduces free energy.
        
        Merge decreases K by 1, increases routing entropy.
        """
        if router.K <= self.config.k_init:
            return False, (-1, -1)
        
        mu, log_s, log_alpha = _flat_access(router)
        if mu is None:
            return False, (-1, -1)
        
        active_mu = mu[router.active_mask]
        active_log_alpha = log_alpha[router.active_mask]
        
        if len(active_mu) < 2:
            return False, (-1, -1)
        
        # Find pair with highest overlap (most redundant)
        best_pair = (-1, -1)
        best_overlap = 0
        
        for i in range(len(active_mu)):
            for j in range(i+1, len(active_mu)):
                diff = active_mu[i] - active_mu[j]
                sigma_sq_i = torch.exp(2 * log_s[router.active_mask][i])
                sigma_sq_j = torch.exp(2 * log_s[router.active_mask][j])
                overlap = torch.exp(-0.5 * (diff**2 / (sigma_sq_i + sigma_sq_j + 1e-6)).sum())
                overlap = overlap * active_log_alpha[i].exp() * active_log_alpha[j].exp()
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    active_indices = router.active_mask.nonzero(as_tuple=True)[0]
                    best_pair = (active_indices[i].item(), active_indices[j].item())
        
        # Merge if overlap is high (redundant)
        if best_overlap > 0.7:
            return True, best_pair
        
        return False, (-1, -1)
    
    def should_spawn(self, router, grad_mu_ema) -> bool:
        """
        Spawn if free energy is too high (underfitting).
        """
        if router.K >= self.config.max_k:
            return False
        
        if not hasattr(self, '_last_routing') or self._last_routing is None:
            return False
            
        fe = self.compute_free_energy(self._last_routing).item()
        
        if self.target_free_energy is not None:
            return fe > self.target_free_energy * 1.2
        else:
            weights = self._last_routing.weights
            avg_entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean().item()
            return avg_entropy > 2.0 and router.K < self.config.max_k * 0.8
    
    def step(self, router, grad_mu_ema, routing_output=None):
        """
        Thermodynamic step: minimize free energy via split/merge/spawn.
        """
        self._last_routing = routing_output
        
        # Record state
        fe = self.compute_free_energy(routing_output).item() if routing_output is not None else 0
        self.free_energy_history.append(fe)
        self.K_history.append(router.K)
        
        # FE-based checks
        split_flag, split_idx = self.should_split(router, grad_mu_ema)
        merge_flag, merge_pair = self.should_merge(router)
        spawn_flag = self.should_spawn(router, grad_mu_ema)
        
        actions = []
        
        if split_flag:
            # Use parent class split logic
            self._split_gaussian(router, split_idx)
            self.split_history.append((len(self.free_energy_history)-1, split_idx))
            actions.append(f"split {split_idx}")
            
        if merge_flag and merge_pair != (-1, -1):
            self._merge_gaussians(router, merge_pair[0], merge_pair[1])
            self.merge_history.append((len(self.free_energy_history)-1, merge_pair))
            actions.append(f"merge {merge_pair}")
            
        if spawn_flag:
            self._spawn_gaussian(router)
            self.spawn_history.append(len(self.free_energy_history)-1)
            actions.append("spawn")
            
        return actions
    
    def _split_gaussian(self, router, idx):
        """Split a Gaussian into two."""
        free_slots = (~router.active_mask).nonzero(as_tuple=True)[0]
        if len(free_slots) == 0:
            return
        new_idx = free_slots[0].item()
        
        # Copy and perturb
        if hasattr(router, 'mu') and router.mu.dim() == 2:
            router.mu.data[new_idx] = router.mu[idx].clone()
            router.mu.data[new_idx] += torch.randn_like(router.mu[idx]) * 0.1
            router.log_s.data[new_idx] = router.log_s[idx].clone() - 0.693  # half scale
            router.log_s.data[idx] -= 0.693
            router.log_alpha.data[new_idx] = router.log_alpha[idx].clone()
        router.active_mask[new_idx] = True
    
    def _merge_gaussians(self, router, idx1, idx2):
        """Merge two Gaussians."""
        if hasattr(router, 'mu') and router.mu.dim() == 2:
            # Average parameters
            router.mu.data[idx1] = (router.mu[idx1] + router.mu[idx2]) / 2
            router.log_s.data[idx1] = (router.log_s[idx1] + router.log_s[idx2]) / 2 + 0.693  # double scale
            router.log_alpha.data[idx1] = (router.log_alpha[idx1] + router.log_alpha[idx2]) / 2
        router.active_mask[idx2] = False
    
    def _spawn_gaussian(self, router):
        """Spawn new Gaussian in unexplored region."""
        free_slots = (~router.active_mask).nonzero(as_tuple=True)[0]
        if len(free_slots) == 0:
            return
        new_idx = free_slots[0].item()
        
        if hasattr(router, 'mu') and router.mu.dim() == 2:
            # Place far from existing Gaussians
            active_mu = router.mu[router.active_mask]
            if len(active_mu) > 0:
                new_mu = active_mu.mean(dim=0) + torch.randn_like(active_mu[0]) * 2.0
            else:
                new_mu = torch.randn_like(router.mu[0]) * 2.0
            router.mu.data[new_idx] = new_mu
            router.log_s.data[new_idx].fill_(0.0)
            router.log_alpha.data[new_idx].fill_(0.0)
        router.active_mask[new_idx] = True
    
    def get_thermodynamic_stats(self) -> dict:
        """Return thermodynamic statistics."""
        return {
            'free_energy': self.free_energy_history[-1] if self.free_energy_history else 0,
            'avg_free_energy': np.mean(self.free_energy_history) if self.free_energy_history else 0,
            'K': self.K_history[-1] if self.K_history else 0,
            'K_trajectory': self.K_history,
            'n_splits': len(self.split_history),
            'n_merges': len(self.merge_history),
            'n_spawns': len(self.spawn_history),
        }
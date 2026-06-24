"""
EqNGS Layer — Equilibrium Propagation for Neural Gaussian Splatting.

Custom EP step for NGS's Mahalanobis routing energy.
The internal energy IS the Mahalanobis distance: E = Σ w_i ||z - μ_i||² / σ_i²
Gaussian states (μ, log_s) settle to minimize this energy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple

from ngs.models.ngs import NGSModel
from ngs.core.interfaces import NGSConfig, RoutingOutput
from ngs.optim.eqprop_wrapper import SpectralConstraint


class EqNGSLayer(nn.Module):
    """
    EqNGS wrapper around NGSModel with custom Equilibrium Propagation.
    
    Instead of standard backprop:
    1. FREE PHASE:   Settle Gaussian router + ParamStore to equilibrium
       Energy = Σ w_i * Mahalanobis(z, μ_i, σ_i²) + 0 (no target nudge)
    2. NUDGED PHASE: Apply output nudge (β * ∇L), re-settle
       Energy = Σ w_i * Mahalanobis(z, μ_i, σ_i²) + β * CE(output, target)
    3. LOCAL UPDATE: Δθ ∝ (θ_nudged - θ_free) for Gaussian params
    
    Memory: O(1) activation graph — only final equilibrium states stored.
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        config: NGSConfig,
        ep_beta: float = 0.5,
        ep_settle_steps: int = 10,
        ep_settle_lr: float = 0.2,
        ep_momentum: float = 0.9,
        spectral_gamma: float = 0.95,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.config = config
        self.ep_beta = ep_beta
        self.ep_settle_steps = ep_settle_steps
        self.ep_settle_lr = ep_settle_lr
        self.ep_momentum = ep_momentum
        self.spectral_gamma = spectral_gamma
        
        # Base NGS model
        self.ngs = NGSModel(d_in, d_out, config)
        
        # Parameters to adapt in EP (Gaussian router + adapter params)
        self.ep_params = []
        self.ep_param_names = []
        for name, param in self.ngs.named_parameters():
            if 'router' in name or 'param_store' in name:
                if param.requires_grad:
                    self.ep_params.append(param)
                    self.ep_param_names.append(name)
        
        # Momentum buffers for EP updates (register as buffers so .to(device) works)
        self.ep_buffer_names = []
        for i, p in enumerate(self.ep_params):
            buf = torch.zeros_like(p)
            self.register_buffer(f'ep_buffer_{i}', buf)
            self.ep_buffer_names.append(f'ep_buffer_{i}')
        
        # Spectral constraint for router projections (contraction guarantee)
        self.spectral_constraints = []
        self._register_spectral_constraints()
        
        # Track if we're in EP training mode
        self._ep_training = False
        
    @property
    def ep_buffers(self):
        """Dynamically access registered buffers (moved by .to(device))."""
        return [getattr(self, name) for name in self.ep_buffer_names]
    
    def _register_spectral_constraints(self):
        """Register SpectralConstraint on router projection layers."""
        for name, param in self.ngs.named_parameters():
            if 'router' in name and 'mu' in name and param.ndim >= 2:
                constraint = SpectralConstraint(gamma=self.spectral_gamma, timing='post_update')
                self.spectral_constraints.append((name, param, constraint))
    
    def enforce_spectral_constraints(self):
        """Enforce spectral norm constraints post-update."""
        for name, param, constraint in self.spectral_constraints:
            constraint.enforce(param, {}, {})
    
    def forward(self, x: torch.Tensor) -> Any:
        """Standard forward (uses base NGS forward)."""
        return self.ngs(x)
    
    def _compute_routing_energy(
        self, 
        z: torch.Tensor, 
        router_output: RoutingOutput,
        target: Optional[torch.Tensor] = None,
        beta: float = 0.0
    ) -> torch.Tensor:
        """
        Compute Mahalanobis routing energy + optional nudge term.
        
        E = Σ w_i ||z - μ_i||² / σ_i² + β * CE(output, target)
        """
        # z: [B, d_latent]
        # router_output.weights: [B, K]
        # router_output.indices: [B, K]
        
        B, K = router_output.weights.shape
        device = z.device
        
        # Get active Gaussian parameters
        active_indices = router_output.indices  # [B, K]
        
        # For each active Gaussian, compute Mahalanobis distance
        # z shape: [B, d], active_indices: [B, K]
        # Need to get μ and σ for active Gaussians
        
        # Access router parameters
        router = self.ngs.router
        if hasattr(router, 'mu') and router.mu.dim() == 2:
            # Monolithic router
            mu = router.mu[active_indices]  # [B, K, d]
            log_s = router.log_s[active_indices]  # [B, K, d]
            log_alpha = router.log_alpha[active_indices]  # [B, K
        else:
            # Fallback - use flat access
            from ngs.modules.topology_managers import _flat_access
            mu, log_s, log_alpha = _flat_access(router)
            if mu is None:
                return torch.tensor(0.0, device=device)
            mu_active = mu[router.active_mask]
            log_s_active = log_s[router.active_mask]
            # This is simplified - in practice need per-batch indexing
            return torch.tensor(0.0, device=device)
        
        # Compute Mahalanobis distances
        # diff: [B, K, d]
        diff = z.unsqueeze(1) - mu  # [B, K, d]
        
        # σ² = exp(2 * log_s) + eps
        sigma_sq = torch.exp(2 * log_s) + 1e-5  # [B, K, d]
        
        # Mahalanobis squared: Σ (diff² / σ²) over d
        mahalanobis_sq = (diff ** 2 / sigma_sq).sum(dim=-1)  # [B, K]
        
        # Weights from router
        weights = router_output.weights  # [B, K]
        
        # Internal energy: Σ w_i * Mahalanobis_sq
        internal_energy = (weights * mahalanobis_sq).sum(dim=-1).mean()  # scalar
        
        # Nudge term: β * CE(output, target)
        nudge_energy = torch.tensor(0.0, device=device)
        if target is not None and beta > 0:
            # Compute output logits from latent z via param_store and p_up
            # z: [B, d_latent], router_out.weights: [B, K]
            # Get param store output for active Gaussians
            active_indices = router_output.indices  # [B, K]
            param_out = self.ngs.param_store(active_indices, z)  # [B, K, d]
            
            # Weighted sum: Σ w_i * param_out_i
            weighted_out = (router_output.weights.unsqueeze(-1) * param_out).sum(dim=1)  # [B, d]
            
            # Project to output
            logits = self.ngs.p_up(weighted_out)  # [B, d_out]
            ce_loss = F.cross_entropy(logits, target)
            nudge_energy = beta * ce_loss
        
        total_energy = internal_energy + nudge_energy
        return total_energy.to(z.device)
    
    def _router_forward(self, z: torch.Tensor) -> RoutingOutput:
        """Forward pass through router only (expects latent z)."""
        return self.ngs.router(z)
    
    def _settle_free_phase(
        self, 
        z: torch.Tensor, 
        steps: int = 10
    ) -> Tuple[RoutingOutput, List[torch.Tensor]]:
        """
        Free phase settling: minimize internal energy without target nudge.
        Returns final router output and saved parameter states.
        """
        # Save initial parameter states
        param_states = [p.clone() for p in self.ep_params]
        
        for _ in range(steps):
            # Router forward to get routing
            router_out = self._router_forward(z)
            
            if router_out is None:
                break
                
            # Compute energy (no nudge in free phase)
            energy = self._compute_routing_energy(z, router_out, target=None, beta=0.0)
            energy = energy.to(z.device)
            
            # Gradient w.r.t EP parameters
            grads = torch.autograd.grad(
                energy, self.ep_params, 
                retain_graph=True, create_graph=False,
                allow_unused=True
            )
            # Ensure gradients are on same device as parameters
            grads = [g.to(p.device) if g is not None else None for g, p in zip(grads, self.ep_params)]
            # Momentum update (skip None gradients)
            with torch.no_grad():
                for p, buf, g in zip(self.ep_params, self.ep_buffers, grads):
                    if g is not None:
                        buf.mul_(self.ep_momentum).add_(g)
                        p.sub_(buf, alpha=self.ep_settle_lr)
        
        # Final router forward to get settled router output
        router_out = self._router_forward(z)
        
        return router_out, param_states
    
    def _settle_nudged_phase(
        self, 
        z: torch.Tensor, 
        target: torch.Tensor,
        steps: int = 10
    ) -> RoutingOutput:
        """
        Nudged phase settling: minimize energy with target nudge β.
        """
        for _ in range(steps):
            router_out = self._router_forward(z)
            
            if router_out is None:
                break
                
            # Compute energy with nudge
            energy = self._compute_routing_energy(
                z, router_out, target=target, beta=self.ep_beta
            )
            
            # Gradient w.r.t EP parameters
            grads = torch.autograd.grad(
                energy, self.ep_params,
                retain_graph=True, create_graph=False,
                allow_unused=True
            )
            
            # Momentum update (skip None gradients)
            with torch.no_grad():
                for p, buf, g in zip(self.ep_params, self.ep_buffers, grads):
                    if g is not None:
                        buf.mul_(self.ep_momentum).add_(g)
                        p.sub_(buf, alpha=self.ep_settle_lr)
        
        # Final router forward to get settled router output
        return self._router_forward(z)
    
    def ep_step(
        self, 
        x: torch.Tensor, 
        target: torch.Tensor, 
        task_id: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Perform one EP optimization step:
        1. Free phase settling (no target)
        2. Nudged phase settling (with target nudge β)
        3. Contrastive update: Δθ ∝ (θ_nudged - θ_free)
        """
        self._ep_training = True
        self.ngs.train()
        
        # Flatten input
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Project to latent space
        z = self.ngs.p_down(x)
        
        # --- FREE PHASE ---
        router_free, params_free = self._settle_free_phase(z, self.ep_settle_steps)
        
        # --- NUDGED PHASE ---
        router_nudged = self._settle_nudged_phase(z, target, self.ep_settle_steps)
        
        # --- CONTRASTIVE UPDATE ---
        # Δθ = (θ_nudged - θ_free) * lr
        with torch.no_grad():
            for p, p_free in zip(self.ep_params, params_free):
                p_nudged = p.clone()  # Current state after nudged phase
                delta = p_nudged - p_free
                p.add_(delta, alpha=self.ep_settle_lr)
        
        # Enforce spectral constraints
        self.enforce_spectral_constraints()
        
        # Compute metrics
        with torch.no_grad():
            out = self.ngs(x)
            logits = out.logits if hasattr(out, 'logits') else out
            pred = logits.argmax(1)
            acc = (pred == target).float().mean().item()
            loss = F.cross_entropy(logits, target).item()
        
        self._ep_training = False
        
        return {
            'loss': loss,
            'accuracy': acc,
        }
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.ngs.train(mode)
        return self
    
    def eval(self):
        self.ngs.eval()
        return super().eval()
    
    def parameters(self, recurse: bool = True):
        return self.ngs.parameters(recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.ngs.named_parameters(prefix, recurse)
    
    # Delegate NGS attributes
    @property
    def router(self):
        return self.ngs.router
    
    @property
    def param_store(self):
        return self.ngs.param_store
    
    @property
    def topology_manager(self):
        return self.ngs.topology_manager
    
    @property
    def K(self):
        return self.ngs.K


def create_eqngs(
    d_in: int,
    d_out: int,
    config: NGSConfig,
    ep_beta: float = 0.5,
    ep_settle_steps: int = 10,
    ep_settle_lr: float = 0.2,
    ep_momentum: float = 0.9,
    spectral_gamma: float = 0.95,
) -> EqNGSLayer:
    """Factory function to create EqNGS layer."""
    return EqNGSLayer(
        d_in=d_in,
        d_out=d_out,
        config=config,
        ep_beta=ep_beta,
        ep_settle_steps=ep_settle_steps,
        ep_settle_lr=ep_settle_lr,
        ep_momentum=ep_momentum,
        spectral_gamma=spectral_gamma,
    )
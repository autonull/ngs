"""
EPOptimizer Wrapper for NGS — TODO11 Phase C6

Wraps NGSModel and delegates EP training to bioplausible EPOptimizer.
The EPOptimizer computes MSE-based internal energy via forward hooks,
completely bypassing the Mahalanobis energy function in EqNGSLayer.

Key insight: NGS has a natural 3-layer structure:
  1. p_down: x -> z (projection to latent)
  2. router + param_store: z -> blended (sparse routing + param generation)
  3. p_up: blended + gamma*z -> logits

For EP, we treat these as 3 states: z, blended, logits
And the energy is MSE(z, h_1(x)) + MSE(blended, h_2(z)) + MSE(logits, h_3(blended))
"""
import sys
from pathlib import Path

_bioplausible_path = Path(__file__).parent.parent.parent / 'bioplausible' / 'mep'
if _bioplausible_path.exists():
    sys.path.insert(0, str(_bioplausible_path))

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ngs.models.ngs import NGSModel
from ngs.core.interfaces import NGSConfig, RoutingStrategy


class EPOptimizerNGSWrapper(nn.Module):
    """
    NGSModel wrapper that properly exposes states for EPOptimizer.
    
    The EPOptimizer hooks into forward pass and captures states.
    We define the "layers" as:
    - Layer 1: p_down (Linear) - state = z
    - Layer 2: router+param_store (custom) - state = blended  
    - Layer 3: p_up (Linear) - state = logits
    
    For EP, internal energy = MSE between state and forward computation of that layer.
    """
    
    def __init__(self, d_in: int, d_out: int, config: NGSConfig):
        super().__init__()
        self.ngs = NGSModel(d_in, d_out, config)
        self._d_in = d_in
        self._d_out = d_out
        self._config = config
        
        # Expose submodules for EPOptimizer inspector
        self.p_down = self.ngs.p_down
        self.p_up = self.ngs.p_up
        self.router = self.ngs.router
        self.param_store = self.ngs.param_store
        self.gamma = self.ngs.gamma
        
        # Track states for custom energy
        self._states = {}
    
    def initialize(self, k_init: int = 8, z_init: torch.Tensor = None):
        """Initialize router units, optionally from data."""
        if hasattr(self.router, 'initialize_units'):
            self.router.initialize_units(k_init, z_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Layer 1: p_down
        z = self.p_down(x)
        self._states['z'] = z
        
        # Layer 2: router + param_store
        routing_output = self.router(z)
        if hasattr(routing_output, 'indices') and hasattr(routing_output, 'weights'):
            active_indices = routing_output.indices
            weights = routing_output.weights
            local_out = self.param_store(active_indices, z)
            blended = (weights.unsqueeze(-1) * local_out).sum(dim=1)
        else:
            blended = z  # fallback
        self._states['blended'] = blended
        
        # Layer 3: p_up with residual
        logits = self.p_up(blended + self.gamma * z)
        self._states['logits'] = logits
        
        return logits
    
    def get_states(self) -> dict:
        """Get captured states for custom energy computation."""
        return self._states
    
    def compute_ep_energy(self, target: torch.Tensor = None, beta: float = 0.0) -> torch.Tensor:
        """
        Compute EP energy using MSE between states and their forward computations.
        
        E_internal = 0.5 * MSE(z, p_down(x))  # Actually z IS p_down(x), so this is 0
        Wait - bioplausible EP treats the FORWARD PASS OUTPUT as the "target" for each state.
        The state is a separate variable that settles to match the forward pass.
        
        In bioplausible: E = 0.5 * MSE(h, state) for each layer
        where h = layer(state_prev) and state is the variable.
        
        For NGS:
        - state_1 = z, h_1 = p_down(x)
        - state_2 = blended, h_2 = router+param_store(z)  
        - state_3 = logits, h_3 = p_up(blended + gamma*z)
        
        But in our forward pass, z = p_down(x) exactly, so MSE(z, p_down(x)) = 0.
        The bioplausible approach uses a DIFFERENT variable for state that settles.
        
        For this wrapper, we need to implement the bioplausible EP properly:
        The EPOptimizer captures states via hooks, then computes:
        E = 0.5 * MSE(h, state) for each layer
        
        We need to expose the layers so EPOptimizer can hook them.
        """
        pass
    
    @property
    def K(self):
        return self.ngs.K


def create_epopt_ngs(
    d_in: int,
    d_out: int,
    config: NGSConfig,
    mode: str = 'ep',
    **optimizer_kwargs,
):
    """
    Create an EPOptimizerNGSWrapper with an EPOptimizer attached.
    """
    from mep.optimizers.ep_optimizer import EPOptimizer

    model = EPOptimizerNGSWrapper(d_in, d_out, config)

    optimizer = EPOptimizer(
        model.parameters(),
        model=model,
        mode=mode,
        **optimizer_kwargs,
    )

    return model, optimizer


# Custom EP optimizer for NGS that uses the correct energy function
class NGS_EPOptimizer:
    """
    Custom EP optimizer for NGS that implements the correct energy function.
    
    NGS states: z (latent), blended (routed), logits (output)
    Energy: E = MSE(z, p_down(x)) + MSE(blended, router(z)) + MSE(logits, p_up(blended+gamma*z))
            + beta * CE(logits, target)
    
    This matches the bioplausible MSE energy structure.
    """
    
    def __init__(self, model: EPOptimizerNGSWrapper, lr: float = 0.01, beta: float = 0.5,
                 settle_steps: int = 10, settle_lr: float = 0.2, momentum: float = 0.9):
        self.model = model
        self.lr = lr
        self.beta = beta
        self.settle_steps = settle_steps
        self.settle_lr = settle_lr
        self.momentum = momentum
        
        # Buffers for settling
        self.state_buffers = {}
        self.param_buffers = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    
    def step(self, x: torch.Tensor, target: torch.Tensor):
        """EP step: free phase settling, nudged phase settling, contrastive update."""
        device = x.device
        batch_size = x.size(0)
        
        # --- FREE PHASE ---
        # Initialize states from forward pass (no target)
        with torch.no_grad():
            out = self.model(x)  # captures states in model._states
            z_free = self.model._states['z'].clone().detach().requires_grad_(True)
            blended_free = self.model._states['blended'].clone().detach().requires_grad_(True)
            logits_free = self.model._states['logits'].clone().detach().requires_grad_(True)
        
        free_states = [z_free, blended_free, logits_free]
        free_momentum = [torch.zeros_like(s) for s in free_states]
        
        for _ in range(self.settle_steps):
            energy = self._compute_energy(x, free_states, target=None, beta=0.0)
            grads = torch.autograd.grad(energy, free_states, retain_graph=True, allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(s) for g, s in zip(grads, free_states)]
            
            with torch.no_grad():
                for i, (s, buf, g) in enumerate(zip(free_states, free_momentum, grads)):
                    buf.mul_(0.5).add_(g)
                    s.sub_(buf, alpha=self.settle_lr)
        
        # --- NUDGED PHASE ---
        with torch.no_grad():
            out = self.model(x)
            z_nudged = self.model._states['z'].clone().detach().requires_grad_(True)
            blended_nudged = self.model._states['blended'].clone().detach().requires_grad_(True)
            logits_nudged = self.model._states['logits'].clone().detach().requires_grad_(True)
        
        nudged_states = [z_nudged, blended_nudged, logits_nudged]
        nudged_momentum = [torch.zeros_like(s) for s in nudged_states]
        
        for _ in range(self.settle_steps):
            energy = self._compute_energy(x, nudged_states, target=target, beta=self.beta)
            grads = torch.autograd.grad(energy, nudged_states, retain_graph=True, allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(s) for g, s in zip(grads, nudged_states)]
            
            with torch.no_grad():
                for i, (s, buf, g) in enumerate(zip(nudged_states, nudged_momentum, grads)):
                    buf.mul_(0.5).add_(g)
                    s.sub_(buf, alpha=self.settle_lr)
        
        # --- CONTRASTIVE UPDATE ---
        # Update model parameters using (theta_nudged - theta_free) / beta
        # But we need gradients w.r.t parameters, not states.
        # Use autograd on the energy difference w.r.t parameters.
        
        # Recompute energies with parameters
        z_f, bl_f, lg_f = free_states
        z_n, bl_n, lg_n = nudged_states
        
        E_free = self._compute_energy_with_params(x, z_f, bl_f, lg_f, target=None, beta=0.0)
        E_nudged = self._compute_energy_with_params(x, z_n, bl_n, lg_n, target=target, beta=self.beta)
        
        contrast_loss = (E_nudged - E_free) / self.beta
        
        # Gradient w.r.t parameters
        params = list(self.model.parameters())
        grads = torch.autograd.grad(contrast_loss, params, retain_graph=False, allow_unused=True)
        
        with torch.no_grad():
            for p, g, buf in zip(params, grads, self.param_buffers.values()):
                if g is not None:
                    buf.mul_(self.momentum).add_(g)
                    p.sub_(buf, alpha=self.lr)
        
        # Return metrics
        with torch.no_grad():
            final_logits = self.model(x)
            acc = (final_logits.argmax(1) == target).float().mean().item()
            loss = F.cross_entropy(final_logits, target).item()
        
        return {'loss': loss, 'accuracy': acc}
    
    def _compute_energy(self, x: torch.Tensor, states: List[torch.Tensor], 
                        target: torch.Tensor = None, beta: float = 0.0) -> torch.Tensor:
        """Compute energy from given states (for settling)."""
        z, blended, logits = states
        batch_size = x.size(0)
        
        # Internal energy: MSE between state and forward computation
        # Layer 1: z vs p_down(x)
        h1 = self.model.p_down(x)
        e1 = 0.5 * F.mse_loss(h1, z, reduction='mean')
        
        # Layer 2: blended vs router+param_store(z)
        routing_output = self.model.router(z)
        if hasattr(routing_output, 'indices') and hasattr(routing_output, 'weights'):
            active_indices = routing_output.indices
            weights = routing_output.weights
            local_out = self.model.param_store(active_indices, z)
            h2 = (weights.unsqueeze(-1) * local_out).sum(dim=1)
        else:
            h2 = z
        e2 = 0.5 * F.mse_loss(h2, blended, reduction='mean')
        
        # Layer 3: logits vs p_up(blended + gamma*z)
        h3 = self.model.p_up(blended + self.model.gamma * z)
        e3 = 0.5 * F.mse_loss(h3, logits, reduction='mean')
        
        E_internal = e1 + e2 + e3
        
        # Nudge energy
        E_nudge = torch.tensor(0.0, device=x.device)
        if target is not None and beta > 0:
            E_nudge = beta * F.cross_entropy(logits, target)
        
        return E_internal + E_nudge
    
    def _compute_energy_with_params(self, x: torch.Tensor, z: torch.Tensor, 
                                    blended: torch.Tensor, logits: torch.Tensor,
                                    target: torch.Tensor = None, beta: float = 0.0) -> torch.Tensor:
        """Compute energy using model parameters (for contrastive update)."""
        batch_size = x.size(0)
        
        # Same as _compute_energy but uses model parameters directly
        h1 = self.model.p_down(x)
        e1 = 0.5 * F.mse_loss(h1, z, reduction='mean')
        
        routing_output = self.model.router(z)
        if hasattr(routing_output, 'indices') and hasattr(routing_output, 'weights'):
            active_indices = routing_output.indices
            weights = routing_output.weights
            local_out = self.model.param_store(active_indices, z)
            h2 = (weights.unsqueeze(-1) * local_out).sum(dim=1)
        else:
            h2 = z
        e2 = 0.5 * F.mse_loss(h2, blended, reduction='mean')
        
        h3 = self.model.p_up(blended + self.model.gamma * z)
        e3 = 0.5 * F.mse_loss(h3, logits, reduction='mean')
        
        E_internal = e1 + e2 + e3
        
        E_nudge = torch.tensor(0.0, device=x.device)
        if target is not None and beta > 0:
            E_nudge = beta * F.cross_entropy(logits, target)
        
        return E_internal + E_nudge


def train_ep_ngs(
    d_in: int,
    d_out: int,
    config: NGSConfig,
    train_loader,
    test_loader,
    epochs: int = 10,
    lr: float = 0.01,
    beta: float = 0.5,
    settle_steps: int = 10,
    settle_lr: float = 0.2,
    device: str = 'cuda',
) -> dict:
    """Train NGS with custom EP optimizer."""
    model = EPOptimizerNGSWrapper(d_in, d_out, config).to(device)
    
    # Initialize with data - pass through p_down to get latent representation
    x_init, _ = next(iter(train_loader))
    x_init = x_init.to(device)
    with torch.no_grad():
        z_init = model.ngs.p_down(x_init.view(x_init.size(0), -1))
    model.initialize(config.k_init, z_init)
    
    optimizer = NGS_EPOptimizer(
        model, lr=lr, beta=beta, settle_steps=settle_steps, settle_lr=settle_lr
    )
    
    results = {'epochs': [], 'final_accuracy': 0.0}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            result = optimizer.step(x, y)
            epoch_loss += result['loss']
            epoch_acc += result['accuracy']
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches
        
        # Test
        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_correct += (out.argmax(1) == y).sum().item()
                test_total += x.size(0)
        test_acc = test_correct / test_total
        
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, train_acc={avg_acc:.4f}, test_acc={test_acc:.4f}")
        results['epochs'].append({
            'epoch': epoch, 'loss': avg_loss, 'train_acc': avg_acc, 'test_acc': test_acc
        })
        results['final_accuracy'] = test_acc
    
    return results
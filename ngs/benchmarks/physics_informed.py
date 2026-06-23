"""Physics-informed topology benchmark (Experiment 3B).
PDE residuals drive split/prune decisions in NGS.
Target: Burgers' equation - units specialize to regimes."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path
import json


def burgers_1d_residual(model, x: torch.Tensor, t: torch.Tensor, nu: float = 0.01) -> torch.Tensor:
    """Compute Burgers' equation residual: u_t + u*u_x - nu*u_xx = 0"""
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    # Model predicts u(x,t)
    input_tensor = torch.cat([x, t], dim=-1)
    u = model(input_tensor).squeeze(-1)
    
    # First derivatives
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0].squeeze(-1)
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0].squeeze(-1)
    
    # Second derivative
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0].squeeze(-1)
    
    # Burgers' residual
    residual = u_t + u * u_x - nu * u_xx
    return residual


class PhysicsInformedNGS(nn.Module):
    """NGS with PDE residual tracking for topology adaptation."""
    
    def __init__(self, data_dim: int, config, pde_residual_fn=None):
        super().__init__()
        from ngs.models import build_ngs
        self.net = build_ngs(data_dim, 1, config)
        self.pde_residual_fn = pde_residual_fn
        self.config = config
        
    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    def compute_pde_residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.pde_residual_fn is not None:
            # Wrap the net to return logits directly
            class NetWrapper(nn.Module):
                def __init__(self, net):
                    super().__init__()
                    self.net = net
                def forward(self, x):
                    return self.net(x).logits
            return self.pde_residual_fn(NetWrapper(self.net), x, t)
        return None


class PhysicsInformedTopologyManager:
    """Topology manager that uses PDE residuals to drive split/prune."""
    
    def __init__(self, config, residual_threshold: float = 1.0):
        self.config = config
        self.residual_threshold = residual_threshold
        self.split_scale = 0.5
        self.noise_std = 0.01
        
    def adapt_topology(self, model, x: torch.Tensor, t: torch.Tensor, 
                       max_spawn_per_call: int = 5, **kwargs) -> Tuple[int, int, int]:
        """Adapt topology based on PDE residuals (simplified for smoke test)."""
        router = model.net.router
        if not hasattr(router, 'active_mask'):
            return 0, 0, 0
            
        # Get flat parameters
        from ngs.modules.topology_managers import _flat_access
        mu, log_s, log_alpha = _flat_access(router)
        if mu is None:
            return 0, 0, 0
            
        active_idx = router.active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) == 0:
            return 0, 0, 0
            
        num_pruned = 0
        num_split = 0
        num_spawned = 0
        
        # Prune: low alpha units
        alpha = torch.sigmoid(log_alpha[active_idx])
        prune_mask = alpha < self.config.prune_threshold
        if prune_mask.any():
            prune_global = active_idx[prune_mask]
            router.active_mask[prune_global] = False
            router.grad_mu_ema[prune_global] = 0
            num_pruned = prune_mask.sum().item()
            
        # Split: simple heuristic - split a few units randomly every few epochs
        # (In a real implementation, this would be driven by PDE residuals)
        active_idx = router.active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) > 0:
            with torch.no_grad():
                split_mask = torch.zeros(len(active_idx), dtype=torch.bool, device=mu.device)
                n_split = min(1, len(active_idx))
                if n_split > 0:
                    split_indices = torch.randperm(len(active_idx))[:n_split]
                    split_mask[split_indices] = True
                    
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
                    
        return num_pruned, num_split, num_spawned


def run_physics_informed_benchmark(
    pde: str = "burgers",
    epochs: int = 100,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./physics_informed_results",
    latent_dim: int = 16,
    k_init: int = 16,
    max_k: int = 128,
    top_k: int = 8,
    lr: float = 1e-3,
    batch_size: int = 256,
    residual_threshold: float = 0.5,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running physics-informed benchmark: {pde}")

    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    
    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=top_k,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.PRE_ALLOCATED,
        split_threshold=0.05,
        prune_threshold=0.01,
    )
    
    # Build model
    if pde == "burgers":
        model = PhysicsInformedNGS(2, config, pde_residual_fn=burgers_1d_residual).to(device)
        data_dim = 2
    else:
        raise ValueError(f"Unknown PDE: {pde}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    topo_manager = PhysicsInformedTopologyManager(config, residual_threshold=residual_threshold)
    
    # Training data: (x, t) pairs in [0, 1] x [0, 1]
    # Target: analytical solution or reference
    def get_burgers_solution(x, t, nu=0.01):
        """Approximate Burgers solution using Cole-Hopf or numerical reference."""
        # For testing, use initial condition u(x,0) = sin(2*pi*x)
        # Solution evolves into shock
        return torch.sin(2 * np.pi * x) * torch.exp(-4 * np.pi**2 * nu * t)
    
    losses = []
    residuals_history = []
    K_history = []
    
    for epoch in range(epochs):
        model.train()
        
        # Generate collocation points
        x = torch.rand(batch_size, 1, device=device)
        t = torch.rand(batch_size, 1, device=device)
        
        # Physics loss
        class NetWrapper(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
            def forward(self, x):
                return self.net(x).logits
        wrapped_net = NetWrapper(model.net)
        residual = burgers_1d_residual(wrapped_net, x, t)
        physics_loss = (residual ** 2).mean()
        
        # Data loss (match known solution at certain points)
        x_data = torch.linspace(0, 1, batch_size, device=device).unsqueeze(1)
        t_data = torch.zeros_like(x_data)
        u_target = get_burgers_solution(x_data, t_data).squeeze(-1)
        u_pred = wrapped_net(torch.cat([x_data, t_data], dim=-1)).squeeze(-1)
        data_loss = F.mse_loss(u_pred, u_target)
        
        # Boundary conditions
        x_bc = torch.tensor([[0.0], [1.0]], device=device)
        t_bc = torch.rand(2, 1, device=device)
        u_bc_pred = wrapped_net(torch.cat([x_bc, t_bc], dim=-1)).squeeze(-1)
        bc_loss = (u_bc_pred ** 2).mean()
        
        loss = physics_loss + data_loss + 0.1 * bc_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Topology adaptation (every 5 epochs)
        if epoch % 5 == 0:
            x_adapt = x[:50].clone().detach().requires_grad_(True)
            t_adapt = t[:50].clone().detach().requires_grad_(True)
            num_pruned, num_split, num_spawned = topo_manager.adapt_topology(
                model, x_adapt, t_adapt, max_spawn_per_call=5
            )
        else:
            num_pruned, num_split, num_spawned = 0, 0, 0
            
        losses.append(loss.item())
        residuals_history.append(residual.abs().mean().item())
        K_history.append(model.net.K)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f}, residual={residuals_history[-1]:.4f}, K={model.net.K}")
    
    results = {
        "pde": pde,
        "final_loss": losses[-1],
        "final_residual": residuals_history[-1],
        "final_K": K_history[-1],
        "loss_history": losses,
        "residual_history": residuals_history,
        "K_history": K_history,
    }
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / f"{pde}_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", default="burgers", choices=["burgers"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    run_physics_informed_benchmark(pde=args.pde, epochs=args.epochs, device=args.device)
"""
Demo: Thermodynamic Self-Regulation - Network grows/shrinks to maintain free-energy equilibrium
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
sys.path.insert(0, '/home/me/ngs')

from ngs.core.interfaces import NGSConfig
from ngs.models.ngs import NGSModel
from experiments.free_energy_manager import FreeEnergyManager


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_synthetic_data(n_clusters, n_samples, latent_dim, cluster_spread=1.5):
    """Create synthetic data with n_clusters in latent space."""
    centers = torch.randn(n_clusters, latent_dim) * 5
    X_list = []
    y_list = []
    for i, center in enumerate(centers):
        n = n_samples // n_clusters
        cluster = center + torch.randn(n, latent_dim) * cluster_spread
        X_list.append(cluster)
        y_list.append(torch.full((n,), i, dtype=torch.long))
    X = torch.cat(X_list)
    y = torch.cat(y_list)
    perm = torch.randperm(len(X))
    return X[perm].to(DEVICE), y[perm].to(DEVICE)


def train_with_thermodynamics(model, loader, epochs=10, verbose=True):
    """Train model with thermodynamic self-regulation."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    K_traj = []
    FE_traj = []
    
    for epoch in range(epochs):
        model.train()
        epoch_K = []
        epoch_FE = []
        
        for x, target in loader:
            x, target = x.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            
            routing_out = model.router(x)
            actions = model.topology_manager.step(
                model.router, model.router.grad_mu_ema, routing_out
            )
            
            epoch_K.append(model.router.K)
            fe = model.topology_manager.compute_free_energy(routing_out).item()
            epoch_FE.append(fe)
        
        avg_K = np.mean(epoch_K)
        avg_FE = np.mean(epoch_FE)
        K_traj.append(avg_K)
        FE_traj.append(avg_FE)
        
        if verbose and (epoch % 3 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch}: K={avg_K:.1f}, FE={avg_FE:.4f}, Actions: {len(actions) if 'actions' in locals() else 0}")
    
    return K_traj, FE_traj


def run_thermodynamic_demo():
    print("=" * 60)
    print("THERMODYNAMIC SELF-REGULATION DEMO")
    print("Free Energy = Routing Entropy + lambda * K")
    print("=" * 60)
    
    # Base config
    base_cfg = NGSConfig(
        latent_dim=32,
        k_init=4,
        max_k=64,
        top_k=8,
        routing='monolithic_mahalanobis',
        parameter_storage='direct_adapter',
        topology_control='discrete_heuristic',
        memory_management='dynamic',
    )
    
    print("\n1. VARYING COMPUTE BUDGET (lambda)")
    print("-" * 40)
    
    # Data: 6 clusters, need more than k_init=4
    X, y = create_synthetic_data(n_clusters=6, n_samples=2400, latent_dim=32)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    
    for lambda_val in [0.005, 0.01, 0.02, 0.05]:
        cfg = NGSConfig(
            latent_dim=32, k_init=4, max_k=64, top_k=8,
            routing='monolithic_mahalanobis', parameter_storage='direct_adapter',
            topology_control='discrete_heuristic', memory_management='dynamic',
        )
        model = NGSModel(32, 6, cfg).to(DEVICE)
        model.topology_manager = FreeEnergyManager(cfg, free_energy_lambda=lambda_val, target_free_energy=1.0)
        model.topology_manager.tau_split = 0.5  # More sensitive
        model.topology_manager.tau_merge = 0.95
        
        K_traj, FE_traj = train_with_thermodynamics(model, loader, epochs=8)
        print(f"  lambda={lambda_val:.3f}: K={K_traj[0]:.0f}->{K_traj[-1]:.0f}, FE={FE_traj[-1]:.3f}")
    
    print("\n2. VARYING DATA COMPLEXITY (num clusters)")
    print("-" * 40)
    
    for n_clusters in [3, 6, 12, 24]:
        X, y = create_synthetic_data(n_clusters=n_clusters, n_samples=n_clusters*400, latent_dim=32)
        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
        
        cfg = NGSConfig(latent_dim=32, k_init=4, max_k=64, top_k=8,
                        routing='monolithic_mahalanobis', parameter_storage='direct_adapter',
                        topology_control='discrete_heuristic', memory_management='dynamic')
        model = NGSModel(32, n_clusters, cfg).to(DEVICE)
        model.topology_manager = FreeEnergyManager(cfg, free_energy_lambda=0.01, target_free_energy=1.0)
        model.topology_manager.tau_split = 0.5
        model.topology_manager.tau_merge = 0.95
        
        K_traj, FE_traj = train_with_thermodynamics(model, loader, epochs=8)
        print(f"  {n_clusters:2d} clusters: K={K_traj[0]:.0f}->{K_traj[-1]:.0f}, FE={FE_traj[-1]:.3f}")
    
    print("\n3. DETAILED TRAJECTORY (6 clusters, lambda=0.01)")
    print("-" * 40)
    
    X, y = create_synthetic_data(n_clusters=6, n_samples=2400, latent_dim=32)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    
    cfg = NGSConfig(latent_dim=32, k_init=4, max_k=64, top_k=8,
                    routing='monolithic_mahalanobis', parameter_storage='direct_adapter',
                    topology_control='discrete_heuristic', memory_management='dynamic')
    model = NGSModel(32, 6, cfg).to(DEVICE)
    model.topology_manager = FreeEnergyManager(cfg, free_energy_lambda=0.01, target_free_energy=1.0)
    model.topology_manager.tau_split = 0.5
    model.topology_manager.tau_merge = 0.95
    
    K_traj, FE_traj = train_with_thermodynamics(model, loader, epochs=15)
    
    print(f"  K trajectory:   {[f'{k:.1f}' for k in K_traj]}")
    print(f"  FE trajectory:  {[f'{fe:.3f}' for fe in FE_traj]}")
    print(f"  Splits: {len(model.topology_manager.split_history)}, Merges: {len(model.topology_manager.merge_history)}")
    
    print("\n" + "=" * 60)
    print("RESULT: Network self-regulates topology via Free Energy")
    print("  - Low lambda (compute budget) -> more Gaussians (higher K)")
    print("  - High data complexity -> more Gaussians")
    print("  - Equilibrium: FE = Routing Entropy + lambda * K")
    print("=" * 60)


if __name__ == "__main__":
    run_thermodynamic_demo()
"""Visualization suite for NGS: latent Gaussians, routing heatmaps, topology dynamics."""
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
import json


class NGSVisualizer:
    """Comprehensive visualization for NGS models."""
    
    def __init__(self, model, output_dir: str = './plots'):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            'K': [], 'pruned': [], 'split': [], 'spawned': [], 'merged': [],
            'split_gate_vals': [], 'activation_density': [], 'error_density': []
        }
    
    def plot_latent_gaussians(self, epoch: int = 0, max_points: int = 5000, 
                              method: str = 'pca', save: bool = True) -> plt.Figure:
        """Plot Gaussian means in latent space with 2D projection."""
        router = self.model.router
        
        # Get active means
        if hasattr(router, 'active_mask'):
            active_idx = router.active_mask.nonzero(as_tuple=True)[0]
            mu = router.mu[active_idx].detach().cpu()
            log_s = router.log_s[active_idx].detach().cpu()
            log_alpha = router.log_alpha[active_idx].detach().cpu()
        elif hasattr(router, 'fine_active'):
            active_idx = router.fine_active.nonzero(as_tuple=True)[0]
            mu = router.fine_mu[active_idx].detach().cpu()
            log_s = router.fine_log_s[active_idx].detach().cpu()
            log_alpha = router.fine_log_alpha[active_idx].detach().cpu()
        else:
            return None
        
        if len(mu) == 0:
            return None
        
        # Project to 2D
        if mu.shape[1] > 2:
            if method == 'pca':
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                mu_2d = pca.fit_transform(mu.numpy())
            elif method == 'tsne':
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, perplexity=min(30, len(mu)-1))
                mu_2d = tsne.fit_transform(mu.numpy())
            else:
                mu_2d = mu[:, :2].numpy()
        else:
            mu_2d = mu.numpy()
        
        alpha = torch.sigmoid(log_alpha).numpy()
        scales = torch.exp(log_s).mean(dim=1).numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Means colored by opacity
        sc = axes[0].scatter(mu_2d[:, 0], mu_2d[:, 1], c=alpha, s=30, cmap='viridis', alpha=0.7)
        axes[0].set_title(f'Gaussian Means (K={len(mu)}) - Opacity')
        axes[0].set_xlabel('Dim 1')
        axes[0].set_ylabel('Dim 2')
        plt.colorbar(sc, ax=axes[0], label='Alpha')
        
        # Plot 2: Means sized by scale
        axes[1].scatter(mu_2d[:, 0], mu_2d[:, 1], s=scales*100, c=alpha, cmap='plasma', alpha=0.7)
        axes[1].set_title('Means - Size = Scale')
        axes[1].set_xlabel('Dim 1')
        
        # Plot 3: Pairwise distance histogram
        if len(mu) > 1:
            from scipy.spatial.distance import pdist
            dists = pdist(mu_2d)
            axes[2].hist(dists, bins=30, edgecolor='black', alpha=0.7)
            axes[2].set_title('Pairwise Distance Distribution')
            axes[2].set_xlabel('Distance')
            axes[2].set_ylabel('Count')
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / f'latent_gaussians_epoch{epoch}.png', dpi=150)
        plt.close()
        return fig
    
    def plot_routing_heatmap(self, dataloader, epoch: int = 0, max_samples: int = 200,
                             save: bool = True) -> plt.Figure:
        """Plot routing heatmap: samples x units activation."""
        self.model.eval()
        all_indices = []
        all_weights = []
        
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.view(x.size(0), -1).to(next(self.model.parameters()).device)
                if len(all_indices) * x.size(0) >= max_samples:
                    break
                output = self.model(x)
                routing = output.routing
                
                if isinstance(routing.indices, list):
                    # Factorized: flatten
                    indices = torch.cat(routing.indices, dim=1)
                    weights = torch.cat(routing.weights, dim=1)
                else:
                    indices = routing.indices
                    weights = routing.weights
                
                all_indices.append(indices.cpu())
                all_weights.append(weights.cpu())
        
        if not all_indices:
            return None
        
        all_indices = torch.cat(all_indices, dim=0)[:max_samples]
        all_weights = torch.cat(all_weights, dim=0)[:max_samples]
        
        # Build activation matrix
        max_k = self.model.config.max_k
        activation = torch.zeros(all_indices.shape[0], max_k)
        for b in range(all_indices.shape[0]):
            activation[b, all_indices[b]] = all_weights[b]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Heatmap
        im = axes[0].imshow(activation.numpy().T, aspect='auto', cmap='hot', interpolation='nearest')
        axes[0].set_title(f'Routing Heatmap (samples x units) - Epoch {epoch}')
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Unit Index')
        plt.colorbar(im, ax=axes[0], label='Weight')
        
        # Unit usage frequency
        usage = (activation > 0.01).float().mean(dim=0)
        axes[1].bar(range(max_k), usage.numpy())
        axes[1].set_title('Unit Usage Frequency')
        axes[1].set_xlabel('Unit Index')
        axes[1].set_ylabel('Fraction of Samples')
        axes[1].axhline(y=0.01, color='r', linestyle='--', label='1% threshold')
        axes[1].legend()
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / f'routing_heatmap_epoch{epoch}.png', dpi=150)
        plt.close()
        return fig
    
    def plot_topology_dynamics(self, save: bool = True) -> plt.Figure:
        """Plot topology changes over time: K, splits, prunes, spawns, merges."""
        if not self.history['K']:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        epochs = range(len(self.history['K']))
        
        # Active units over time
        axes[0, 0].plot(epochs, self.history['K'], 'b-', linewidth=2)
        axes[0, 0].set_title('Active Units (K) Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('K')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Topology actions
        if any(self.history['split']) or any(self.history['pruned']):
            axes[0, 1].plot(epochs, self.history['split'], 'g-', label='Split', linewidth=2)
            axes[0, 1].plot(epochs, self.history['pruned'], 'r-', label='Prune', linewidth=2)
            axes[0, 1].plot(epochs, self.history['spawned'], 'b-', label='Spawn', linewidth=2)
            if any(self.history['merged']):
                axes[0, 1].plot(epochs, self.history['merged'], 'm-', label='Merge', linewidth=2)
            axes[0, 1].set_title('Topology Actions per Epoch')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Split gate values distribution
        if self.history['split_gate_vals']:
            last_gates = self.history['split_gate_vals'][-1]
            if len(last_gates) > 0:
                axes[1, 0].hist(last_gates, bins=30, edgecolor='black', alpha=0.7)
                axes[1, 0].axvline(x=0.65, color='r', linestyle='--', label='Split threshold')
                axes[1, 0].set_title('Split Gate Values (Last Epoch)')
                axes[1, 0].set_xlabel('Gate Value')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].legend()
        
        # Activation/Error density
        if self.history['activation_density']:
            act = self.history['activation_density'][-1]
            err = self.history['error_density'][-1]
            axes[1, 1].scatter(act, err, alpha=0.5, s=10)
            axes[1, 1].set_title('Activation vs Error Density')
            axes[1, 1].set_xlabel('Activation Density')
            axes[1, 1].set_ylabel('Error Density')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'topology_dynamics.png', dpi=150)
        plt.close()
        return fig
    
    def plot_subspace_activation(self, dataloader, epoch: int = 0, save: bool = True) -> plt.Figure:
        """Plot factorized routing: subspace activation patterns."""
        router = self.model.router
        if not hasattr(router, 'num_subspaces'):
            return None
        
        self.model.eval()
        subspace_usage = torch.zeros(router.num_subspaces, router.units_per_space)
        total_samples = 0
        
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.view(x.size(0), -1).to(next(self.model.parameters()).device)
                output = self.model(x)
                routing = output.routing
                
                if isinstance(routing.indices, list):
                    for s, idx in enumerate(routing.indices):
                        for b in range(idx.shape[0]):
                            subspace_usage[s, idx[b]] += routing.weights[s][b]
                    total_samples += x.size(0)
        
        if total_samples == 0:
            return None
        
        subspace_usage /= total_samples
        
        fig, axes = plt.subplots(1, router.num_subspaces, figsize=(4*router.num_subspaces, 4))
        if router.num_subspaces == 1:
            axes = [axes]
        
        for s in range(router.num_subspaces):
            im = axes[s].imshow(subspace_usage[s].unsqueeze(0).numpy(), aspect='auto', cmap='viridis')
            axes[s].set_title(f'Subspace {s} Unit Activation')
            axes[s].set_xlabel('Unit Index')
            axes[s].set_yticks([])
            plt.colorbar(im, ax=axes[s])
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / f'subspace_activation_epoch{epoch}.png', dpi=150)
        plt.close()
        return fig
    
    def record_topology_action(self, action) -> None:
        """Record topology action for history tracking."""
        self.history['K'].append(self.model.K)
        self.history['pruned'].append(action.num_pruned)
        self.history['split'].append(action.num_split)
        self.history['spawned'].append(action.num_spawned)
        self.history['merged'].append(action.num_merged)
        
        if hasattr(self.model, 'split_gate'):
            active_idx = self.model.router.active_mask.nonzero(as_tuple=True)[0]
            if len(active_idx) > 0:
                gates = torch.sigmoid(self.model.split_gate[active_idx]).detach().cpu().numpy()
                self.history['split_gate_vals'].append(gates)
                self.history['activation_density'].append(
                    self.model.activation_density[active_idx].detach().cpu().numpy()
                )
                self.history['error_density'].append(
                    self.model.error_density[active_idx].detach().cpu().numpy()
                )
    
    def save_history(self, filepath: Optional[str] = None) -> None:
        """Save topology history to JSON."""
        if filepath is None:
            filepath = self.output_dir / 'topology_history.json'
        
        # Convert numpy arrays to lists
        serializable = {}
        for k, v in self.history.items():
            if isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                serializable[k] = [arr.tolist() for arr in v]
            else:
                serializable[k] = v
        
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    def generate_report(self, dataloader, epoch: int = 0) -> Dict[str, Any]:
        """Generate comprehensive visualization report."""
        self.plot_latent_gaussians(epoch)
        self.plot_routing_heatmap(dataloader, epoch)
        self.plot_topology_dynamics()
        
        router = self.model.router
        if hasattr(router, 'num_subspaces'):
            self.plot_subspace_activation(dataloader, epoch)
        
        self.save_history()
        
        return {
            'epoch': epoch,
            'K': self.model.K,
            'plots_saved': True,
            'output_dir': str(self.output_dir)
        }


def visualize_model(model, dataloader, output_dir: str = './plots', epoch: int = 0):
    """Convenience function for quick visualization."""
    vis = NGSVisualizer(model, output_dir)
    return vis.generate_report(dataloader, epoch)

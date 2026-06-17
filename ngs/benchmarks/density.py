"""Density Estimation Benchmark for NGS.

Tests NGS as adaptive Gaussian Mixture Model on 2D toy densities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path
import json


@dataclass
class DensityBenchmarkConfig:
    """Configuration for density estimation benchmark."""
    # Toy dataset
    dataset: str = "moons"  # "moons", "circles", "pinwheel", "swissroll", "checkerboard"
    n_samples: int = 5000
    noise: float = 0.05
    
    # NGS model config
    latent_dim: int = 16
    k_init: int = 16
    max_k: int = 128
    top_k: int = 8
    routing: str = "factorized"
    parameter_storage: str = "hypernetwork"
    topology_control: str = "continuous_density"
    memory_management: str = "pre_allocated"
    
    # Training
    epochs: int = 500
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    
    # Topology adaptation
    adapt_every: int = 10
    split_thresh: float = 0.05
    prune_th: float = 0.01
    merge_th: float = 0.1
    
    # Evaluation
    n_eval_samples: int = 10000
    grid_size: int = 100


def generate_moons(n_samples: int = 5000, noise: float = 0.05) -> torch.Tensor:
    """Generate moons dataset."""
    from sklearn.datasets import make_moons
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return torch.tensor(X, dtype=torch.float32)


def generate_circles(n_samples: int = 5000, noise: float = 0.05) -> torch.Tensor:
    """Generate circles dataset."""
    from sklearn.datasets import make_circles
    X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    return torch.tensor(X, dtype=torch.float32)


def generate_pinwheel(n_samples: int = 5000, noise: float = 0.05) -> torch.Tensor:
    """Generate pinwheel dataset."""
    radial_std = 0.3
    tangential_std = 0.1
    num_classes = 5
    num_per_class = n_samples // num_classes
    
    rate = 0.25
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
    
    X = []
    for i, rad in enumerate(rads):
        for _ in range(num_per_class):
            r = np.random.randn() * radial_std + 1.0
            t = np.random.randn() * tangential_std + rate * r
            x = r * np.cos(rad + t)
            y = r * np.sin(rad + t)
            X.append([x, y])
    
    X = np.array(X[:n_samples])
    X += np.random.randn(*X.shape) * noise
    return torch.tensor(X, dtype=torch.float32)


def generate_swissroll(n_samples: int = 5000, noise: float = 0.05) -> torch.Tensor:
    """Generate swiss roll dataset (2D projection)."""
    from sklearn.datasets import make_swiss_roll
    X, _ = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=42)
    return torch.tensor(X[:, [0, 2]], dtype=torch.float32)


def generate_checkerboard(n_samples: int = 5000, noise: float = 0.05) -> torch.Tensor:
    """Generate checkerboard dataset."""
    x1 = np.random.rand(n_samples) * 4 - 2
    x2 = np.random.rand(n_samples) * 4 - 2
    mask = (np.floor(x1) + np.floor(x2)) % 2 == 0
    X = np.stack([x1[mask], x2[mask]], axis=1)
    if len(X) > n_samples:
        X = X[:n_samples]
    X += np.random.randn(*X.shape) * noise
    return torch.tensor(X, dtype=torch.float32)


DATASET_GENERATORS = {
    "moons": generate_moons,
    "circles": generate_circles,
    "pinwheel": generate_pinwheel,
    "swissroll": generate_swissroll,
    "checkerboard": generate_checkerboard,
}


class DensityEstimationModel(nn.Module):
    """NGS adapted for density estimation (no classification head)."""
    
    def __init__(self, config: DensityBenchmarkConfig):
        super().__init__()
        self.config = config
        self.d_latent = config.latent_dim
        
        # Encoder: 2D -> latent
        self.encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_latent),
        )
        
        # Decoder: latent -> 2D (mean + log_scale for Gaussian)
        self.decoder = nn.Sequential(
            nn.Linear(self.d_latent, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 2 for mean, 2 for log_scale
        )
        
        # We'll use the NGS routing internally for mixture components
        from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
        from ngs.models.ngs import build_ngs
        
        ngs_config = NGSConfig(
            latent_dim=config.latent_dim,
            k_init=config.k_init,
            max_k=config.max_k,
            top_k=config.top_k,
            routing=RoutingStrategy(config.routing),
            parameter_storage=ParameterStorage(config.parameter_storage),
            topology_control=TopologyControl(config.topology_control),
            memory_management=MemoryManagement(config.memory_management),
        )
        
        self.ngs = build_ngs(2, 2, ngs_config)  # 2D in, 2D out (will use latent)
        self.ngs.p_down = self.encoder
        self.ngs.p_up = nn.Linear(self.d_latent, 2, bias=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (means, log_scales) for mixture components."""
        z = self.encoder(x)
        
        # Get routing
        routing = self.ngs.router(z)
        
        if isinstance(routing.indices, list):
            # Factorized routing
            all_indices = torch.cat(routing.indices, dim=1)
            all_weights = torch.cat(routing.weights, dim=1)
        else:
            all_indices = routing.indices
            all_weights = routing.weights
            
        # Get parameters for active units
        B, K = all_indices.shape
        params = self.ngs.param_store(all_indices.view(-1), z.repeat_interleave(K, dim=0))
        params = params.view(B, K, -1)
        
        # Decode to get means and log_scales
        decoded = self.decoder(params.view(-1, self.d_latent))
        means = decoded[:, :2].view(B, K, 2)
        log_scales = decoded[:, 2:].view(B, K, 2).clamp(-5, 3)
        
        return means, log_scales, all_weights


def compute_nll(means: torch.Tensor, log_scales: torch.Tensor, weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute negative log-likelihood for Gaussian mixture."""
    # means: [B, K, 2], log_scales: [B, K, 2], weights: [B, K], x: [B, 2]
    B, K, _ = means.shape
    
    # Compute log prob for each component
    diff = x.unsqueeze(1) - means  # [B, K, 2]
    var = (2 * log_scales).exp()  # [B, K, 2]
    
    # Diagonal Gaussian log prob
    log_prob = -0.5 * ((diff ** 2) / var + 2 * log_scales + np.log(2 * np.pi)).sum(dim=-1)  # [B, K]
    log_prob = log_prob + torch.log(weights + 1e-8)
    
    # Log-sum-exp over components
    log_likelihood = torch.logsumexp(log_prob, dim=1)  # [B]
    return -log_likelihood.mean()


class DensityBenchmark:
    """Density estimation benchmark runner."""
    
    def __init__(self, config: DensityBenchmarkConfig, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results = {}
        
    def run(self, seed: int = 42) -> Dict:
        """Run density estimation benchmark."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate data
        generator = DATASET_GENERATORS.get(self.config.dataset)
        if generator is None:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
            
        train_data = generator(self.config.n_samples, self.config.noise).to(self.device)
        
        # Create model
        model = DensityEstimationModel(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.epochs)
        
        # Training loop
        losses = []
        nlls = []
        k_history = []
        
        for epoch in range(self.config.epochs):
            model.train()
            
            # Shuffle and batch
            perm = torch.randperm(len(train_data))
            epoch_losses = []
            
            for i in range(0, len(train_data), self.config.batch_size):
                batch_idx = perm[i:i + self.config.batch_size]
                batch = train_data[batch_idx]
                
                optimizer.zero_grad()
                means, log_scales, weights = model(batch)
                loss = compute_nll(means, log_scales, weights, batch)
                
                # Add topology losses
                if self.config.topology_control in ["continuous_density", "merge_aware"]:
                    topo_losses = model.ngs.compute_topology_losses()
                    for name, val in topo_losses.items():
                        if name == "entropy":
                            loss += self.config.entropy_weight * val
                        elif name == "diversity":
                            loss += self.config.diversity_weight * val
                        elif name == "split_gate":
                            loss += self.config.split_gate_weight * val
                        elif name == "merge":
                            loss += self.config.merge_weight * val
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())
                
            scheduler.step()
            losses.append(np.mean(epoch_losses))
            k_history.append(model.ngs.K)
            
            # Topology adaptation
            if epoch % self.config.adapt_every == 0 and epoch > 0:
                model.eval()
                with torch.no_grad():
                    z_samples = model.encoder(train_data[torch.randperm(len(train_data))[:500]])
                    model.ngs.adapt_density(z_samples, split_thresh=self.config.split_th, 
                                           prune_thresh=self.config.prune_th, merge_thresh=self.config.merge_th)
            
            # Evaluation NLL
            if epoch % 50 == 0:
                model.eval()
                with torch.no_grad():
                    eval_data = generator(self.config.n_eval_samples, self.config.noise).to(self.device)
                    means, log_scales, weights = model(eval_data)
                    nll = compute_nll(means, log_scales, weights, eval_data)
                    nlls.append((epoch, nll.item()))
                    
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss={losses[-1]:.4f}, K={k_history[-1]}, NLL={nlls[-1][1] if nlls else 'N/A':.4f}")
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            eval_data = generator(self.config.n_eval_samples, self.config.noise).to(self.device)
            means, log_scales, weights = model(eval_data)
            final_nll = compute_nll(means, log_scales, weights, eval_data).item()
            
            # Sample quality: generate samples
            samples = self._generate_samples(model, self.config.n_eval_samples)
            
        self.results = {
            "config": self.config.__dict__,
            "seed": seed,
            "final_nll": final_nll,
            "loss_history": losses,
            "nll_history": nlls,
            "k_history": k_history,
            "final_k": model.ngs.K,
        }
        
        return self.results
    
    def _generate_samples(self, model: DensityEstimationModel, n_samples: int) -> torch.Tensor:
        """Generate samples from the learned mixture."""
        model.eval()
        with torch.no_grad():
            # Sample from mixture
            means, log_scales, weights = model(torch.randn(n_samples, 2).to(self.device))
            
            # Sample component for each point
            comp_idx = torch.multinomial(weights, 1).squeeze(-1)  # [B]
            
            batch_idx = torch.arange(n_samples, device=self.device)
            selected_means = means[batch_idx, comp_idx]
            selected_scales = (2 * log_scales[batch_idx, comp_idx]).exp().sqrt()
            
            samples = selected_means + selected_scales * torch.randn_like(selected_means)
            return samples.cpu()
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot density estimation results."""
        if not self.results:
            print("No results to plot. Run benchmark first.")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss curve
        axes[0, 0].plot(self.results["loss_history"])
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("NLL")
        axes[0, 0].set_yscale("log")
        
        # NLL over time
        if self.results["nll_history"]:
            epochs, nlls = zip(*self.results["nll_history"])
            axes[0, 1].plot(epochs, nlls, 'o-')
        axes[0, 1].set_title("Evaluation NLL")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("NLL")
        
        # K over time
        axes[0, 2].plot(self.results["k_history"])
        axes[0, 2].set_title("Active Units (K)")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("K")
        
        # Data vs Samples
        generator = DATASET_GENERATORS[self.config.dataset]
        true_data = generator(self.config.n_eval_samples, self.config.noise).numpy()
        
        axes[1, 0].scatter(true_data[:, 0], true_data[:, 1], s=1, alpha=0.5, label="True")
        axes[1, 0].set_title(f"True {self.config.dataset}")
        axes[1, 0].set_aspect('equal')
        
        # We'd need to store samples for this
        axes[1, 1].text(0.5, 0.5, "Samples\n(not stored)", ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Generated Samples")
        axes[1, 1].set_aspect('equal')
        
        # Final NLL
        axes[1, 2].text(0.5, 0.5, f"Final NLL: {self.results['final_nll']:.4f}\nFinal K: {self.results['final_k']}", 
                        ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=14)
        axes[1, 2].set_title("Summary")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
        plt.close()


def run_density_benchmark(
    dataset: str = "moons",
    epochs: int = 500,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./density_results"
) -> Dict:
    """Run density estimation benchmark with default config."""
    config = DensityBenchmarkConfig(
        dataset=dataset,
        epochs=epochs,
    )
    
    benchmark = DensityBenchmark(config, device)
    results = benchmark.run(seed)
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / f"{dataset}_seed{seed}_results.json", "w") as f:
        # Convert non-serializable items
        serializable = {k: v for k, v in results.items() if k != "config"}
        serializable["config"] = config.__dict__
        json.dump(serializable, f, indent=2)
    
    # Plot
    benchmark.plot_results(save_path=str(Path(output_dir) / f"{dataset}_seed{seed}_plot.png"))
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run density estimation benchmark")
    parser.add_argument("--dataset", default="moons", choices=list(DATASET_GENERATORS.keys()))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./density_results")
    args = parser.parse_args()
    
    run_density_benchmark(args.dataset, args.epochs, args.device, args.seed, args.output_dir)
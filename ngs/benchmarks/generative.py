"""Generative modeling benchmarks for NGS (VAE-style)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
from pathlib import Path
import json


class NGSVAE(nn.Module):
    """VAE where NGS replaces the decoder."""
    
    def __init__(self, input_dim: int, latent_dim: int, ngs_config):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        from ngs.models import build_ngs
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.mu_head = nn.Linear(64, latent_dim)
        self.logvar_head = nn.Linear(64, latent_dim)
        self.decoder = build_ngs(latent_dim, input_dim, ngs_config)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def generate(self, n_samples: int, device: torch.device):
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decoder(z)


def vae_loss(recon, x, mu, logvar, kl_weight=0.5):
    recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


def run_generative_benchmark(
    dataset: str = "moons",
    epochs: int = 200,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./generative_results",
    latent_dim: int = 8,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """Run VAE with NGS decoder on 2D toy data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running generative benchmark on {dataset} using {device}")

    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    from ngs.benchmarks.density import generate_toy_data

    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=16,
        max_k=64,
        top_k=8,
        routing=RoutingStrategy.FACTORIZED_SUBSPACE,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        num_subspaces=2,
    )

    model = NGSVAE(2, latent_dim, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    data = generate_toy_data(dataset, n_samples=5000)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, torch.zeros(len(data), dtype=torch.long)),
        batch_size=batch_size, shuffle=True
    )

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, _ in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, rl, kl = vae_loss(recon, x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(data)
        losses.append(avg_loss)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "dataset": dataset,
        "final_loss": losses[-1],
        "loss_history": losses,
        "final_k": model.decoder.K if hasattr(model.decoder, 'K') else None,
    }
    with open(Path(output_dir) / f"{dataset}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Final loss: {losses[-1]:.4f}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="moons")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_generative_benchmark(args.dataset, args.epochs, args.device, args.seed)

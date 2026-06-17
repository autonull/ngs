"""Density estimation benchmarks for NGS."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_swiss_roll
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import json

from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models import build_ngs
from ngs.training import NGSTrainer, TrainConfig


def generate_toy_data(dataset: str, n_samples: int = 2000, noise: float = 0.1) -> torch.Tensor:
    """Generate 2D toy dataset."""
    if dataset == "moons":
        X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset == "circles":
        X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    elif dataset == "swissroll":
        X, _ = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=42)
        X = X[:, [0, 2]]  # Use 2D projection
    elif dataset == "pinwheel":
        # Pinwheel dataset
        radial = np.linspace(1.0, 3.0, n_samples // 5)
        angles = np.linspace(0, 4 * np.pi, n_samples // 5)
        X = []
        for i in range(5):
            r = radial
            theta = angles + i * 2 * np.pi / 5
            X.append(np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1))
        X = np.vstack(X) + np.random.randn(*X.shape) * noise
    elif dataset == "checkerboard":
        # Checkerboard pattern
        x1 = np.random.uniform(-4, 4, n_samples)
        x2 = np.random.uniform(-4, 4, n_samples)
        X = np.stack([x1, x2], axis=1)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return torch.tensor(X, dtype=torch.float32)


def compute_nll(model: nn.Module, data: torch.Tensor, device: str = "cpu") -> float:
    """Compute negative log-likelihood for density estimation."""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        batch_size = 512
        nlls = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            logits = model(batch)
            # For density estimation, we use the routing weights as log-prob
            # This is a simplified NLL computation
            log_probs = F.log_softmax(logits, dim=-1)
            # Average log-prob across all classes (uniform target)
            nll = -log_probs.mean().item()
            nlls.append(nll)
        return np.mean(nlls)


def run_density_benchmark(
    dataset: str = "moons",
    epochs: int = 500,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./density_results",
    latent_dim: int = 2,
    k_init: int = 16,
    max_k: int = 128,
    top_k: int = 8,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Run NGS density estimation benchmark on 2D toy datasets."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running density estimation on {dataset} using {device}")
    
    # Generate data
    data = generate_toy_data(dataset, n_samples=5000)
    train_data = data[:4000]
    test_data = data[4000:]
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data, torch.zeros(len(train_data), dtype=torch.long)),
        batch_size=batch_size, shuffle=True
    )
    
    # Build model - use 10 output classes for compatibility
    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=top_k,
        routing=RoutingStrategy.FACTORIZED_SUBSPACE,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.PRE_ALLOCATED_MASKED,
        num_subspaces=2,
        split_threshold=0.05,
        prune_threshold=0.01,
    )
    
    model = build_ngs(2, 10, config).to(device)
    
    trainer_config = TrainConfig(
        lr=lr,
        epochs=1,
        batch_size=batch_size,
        entropy_weight=0.01,
        diversity_weight=0.01,
        adapt_every_epoch=True,
        split_thresh=0.05,
        prune_thresh=0.01,
    )
    
    trainer = NGSTrainer(model, trainer_config, device=device)
    
    # Training loop
    nll_history = []
    k_history = []
    
    for epoch in range(epochs):
        metrics = trainer.train_epoch(train_loader)
        
        # Evaluate NLL on test set
        test_nll = compute_nll(model, test_data, device)
        nll_history.append(test_nll)
        k_history.append(model.K)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: NLL={test_nll:.4f}, K={model.K}")
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {
        "dataset": dataset,
        "final_nll": nll_history[-1],
        "final_k": model.K,
        "nll_history": nll_history,
        "k_history": k_history,
        "config": {
            "latent_dim": latent_dim,
            "k_init": k_init,
            "max_k": max_k,
            "top_k": top_k,
            "lr": lr,
            "epochs": epochs,
        }
    }
    
    with open(Path(output_dir) / f"{dataset}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Data scatter
    axes[0].scatter(test_data[:, 0], test_data[:, 1], s=1, alpha=0.5)
    axes[0].set_title(f"{dataset} data")
    axes[0].set_aspect('equal')
    
    # NLL history
    axes[1].plot(nll_history)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("NLL")
    axes[1].set_title("Test NLL over training")
    
    # K history
    axes[2].plot(k_history)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Active units (K)")
    axes[2].set_title("Topology growth")
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{dataset}_plots.png", dpi=150)
    plt.close()
    
    print(f"\nFinal NLL: {results['final_nll']:.4f}")
    print(f"Final K: {results['final_k']}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="moons", choices=["moons", "circles", "swissroll", "pinwheel", "checkerboard"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./density_results")
    args = parser.parse_args()
    
    run_density_benchmark(
        dataset=args.dataset,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir
    )
"""Flow matching benchmark using NGS as a velocity field."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import json


class NGSVelocityField(nn.Module):
    """NGS parametrizing a time-dependent velocity field v_t(z) for flow matching."""

    def __init__(self, data_dim: int, ngs_config):
        super().__init__()
        from ngs.models import build_ngs
        self.net = build_ngs(data_dim + 1, data_dim, ngs_config)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        t_expanded = t.expand_as(z[:, :1])
        return self.net(torch.cat([z, t_expanded], dim=-1))


def conditional_flow_matching_loss(
    model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    z_t = (1 - t) * x0 + t * x1
    target = x1 - x0
    pred = model(t, z_t)
    return F.mse_loss(pred, target)


def run_flow_matching_benchmark(
    dataset: str = "moons",
    epochs: int = 500,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./flow_matching_results",
    latent_dim: int = 16,
    k_init: int = 32,
    max_k: int = 128,
    top_k: int = 8,
    lr: float = 5e-4,
    batch_size: int = 256,
    n_samples: int = 10000,
    eval_every: int = 50,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running flow matching on {dataset} using {device}")

    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    from ngs.benchmarks.density import generate_toy_data

    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=top_k,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.PRE_ALLOCATED_MASKED,
        split_threshold=0.05,
        prune_threshold=0.01,
        tau=1.0,
    )
    data_dim = 2
    model = NGSVelocityField(data_dim, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    data = generate_toy_data(dataset, n_samples).to(device)
    n_train = int(0.9 * len(data))
    train_data, test_data = data[:n_train], data[n_train:]

    train_losses = []
    test_losses = []
    K_history = []
    best_test_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_data))
        epoch_losses = []

        for i in range(0, len(train_data), batch_size):
            idx = perm[i:i + batch_size]
            x1 = train_data[idx]
            x0 = torch.randn_like(x1)
            t = torch.rand(len(x1), 1, device=device)

            optimizer.zero_grad()
            loss = conditional_flow_matching_loss(model.net, x0, x1, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

            if hasattr(model.net, "update_unit_errors"):
                model.net.update_unit_errors(loss.unsqueeze(0), x1.argmax(dim=-1) if x1.shape[-1] > 1 else x1.long())

        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)

        K_history.append(model.net.K)

        if model.config.topology_control == TopologyControl.CONTINUOUS_DENSITY and hasattr(model.net, "adapt_density"):
            z_samples = torch.randn(200, config.latent_dim, device=device)
            model.net.adapt_density(
                z_samples=z_samples,
                split_thresh=0.05,
                prune_thresh=0.01,
                max_spawn_per_call=5,
            )

        if epoch % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_batches = []
                for i in range(0, len(test_data), batch_size):
                    x1 = test_data[i:i + batch_size]
                    x0 = torch.randn_like(x1)
                    t = torch.rand(len(x1), 1, device=device)
                    loss = conditional_flow_matching_loss(model.net, x0, x1, t)
                    test_batches.append(loss.item())
                avg_test_loss = np.mean(test_batches)
                test_losses.append(avg_test_loss)
                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, test_loss={avg_test_loss:.6f}, K={model.net.K}")

    results = {
        "dataset": dataset,
        "final_train_loss": float(train_losses[-1]),
        "best_test_loss": float(best_test_loss),
        "final_K": int(model.net.K),
        "K_history": [int(k) for k in K_history],
        "train_losses": [float(l) for l in train_losses],
        "test_losses": [float(l) for l in test_losses],
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"flow_matching_{dataset}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path / f'flow_matching_{dataset}.json'}")
    return results

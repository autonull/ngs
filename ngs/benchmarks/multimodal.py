"""Multimodal fusion benchmark for NGS FactorizedRouter."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from copy import deepcopy


def compute_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                modalities, y = batch
                x = modalities
            else:
                x, y = batch
            x = x.to(device)
            y = y.to(device)
            if x.dim() == 3:  # [B, M, D]
                B, M, D = x.shape
                x = x.view(B, -1)
            out_obj = model(x)
            logits = out_obj.logits
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return correct / total


def run_multimodal_benchmark(
    modality_types: tuple = ("original", "permuted"),
    epochs: int = 10,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./multimodal_results",
    latent_dim: int = 64,
    k_init: int = 64,
    max_k: int = 512,
    lr: float = 1e-3,
    batch_size: int = 256,
    num_subspaces: int = 2,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running multimodal benchmark with modalities: {modality_types}")
    print(f"Using FactorizedRouter with {num_subspaces} subspaces")

    from experiments.datasets import get_multimodal_loaders
    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    from ngs.models import build_ngs
    from ngs.training import NGSTrainer, TrainerConfig

    train_loader, test_loader = get_multimodal_loaders(
        modality_types=modality_types, batch_size=batch_size, seed=seed
    )

    # Get input dimension
    for x, y in train_loader:
        d_in = x.shape[1] * x.shape[2] if x.dim() == 3 else x.shape[1]
        n_classes = 10
        break

    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=8,
        top_k_factorized=4,
        routing=RoutingStrategy.FACTORIZED_SUBSPACE,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.DYNAMIC,
        split_threshold=0.05,
        prune_threshold=0.01,
        num_subspaces=num_subspaces,
        hypernetwork_code_dim=16,
        hypernetwork_hidden_dim=32,
        tau=1.0,
    )

    model = build_ngs(d_in, n_classes, config).to(device)

    trainer_config = TrainerConfig(
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        kd_weight=0.0,
        adapt_every_epoch=True,
        split_thresh=0.05,
        prune_thresh=0.01,
        max_spawn_per_call=5,
    )
    trainer = NGSTrainer(model, trainer_config, device=device)

    # Track subspace utilization
    subspace_usage = {s: [] for s in range(num_subspaces)}

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if x.dim() == 3:
                B, M, D = x.shape
                x = x.view(B, -1)

            optimizer = trainer.optimizer
            optimizer.zero_grad()
            out_obj = model(x)
            logits = out_obj.logits
            ce_loss = F.cross_entropy(logits, y)

            # Entropy regularization
            entropy_loss = model.entropy_loss(x)
            diversity_loss = model.diversity_loss()

            loss = ce_loss + 0.01 * entropy_loss + 0.01 * diversity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Track subspace routing
        if hasattr(model.router, 'num_subspaces'):
            with torch.no_grad():
                for x_test, _ in test_loader:
                    x_test = x_test.to(device)
                    if x_test.dim() == 3:
                        B, M, D = x_test.shape
                        x_test = x_test.view(B, -1)
                    z = model.p_down(x_test)
                    routing = model.router(z)
                    if hasattr(routing, 'level_weights'):
                        for s, w in enumerate(routing.level_weights):
                            if w.sum() > 0:
                                subspace_usage[s].append(w.mean().item())
                    break

        if hasattr(model, "adapt_density"):
            with torch.no_grad():
                z_samples = []
                for x, _ in train_loader:
                    x = x.to(device)
                    if x.dim() == 3:
                        B, M, D = x.shape
                        x = x.view(B, -1)
                    z = model.p_down(x)
                    z_samples.append(z)
                    if len(torch.cat(z_samples)) >= 200:
                        break
                if z_samples:
                    z_samples = torch.cat(z_samples)[:200]
                    model.adapt_density(
                        z_samples=z_samples, split_thresh=0.05, prune_thresh=0.01, max_spawn_per_call=5
                    )

        test_acc = compute_accuracy(model, test_loader, device)
        print(f"Epoch {epoch}: test_acc={test_acc:.4f}, K={model.K}")

    final_acc = compute_accuracy(model, test_loader, device)

    # Compare with monolithic baseline
    print("Running monolithic baseline for comparison...")
    mono_config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.DYNAMIC,
        split_threshold=0.05,
        prune_threshold=0.01,
        hypernetwork_code_dim=16,
        hypernetwork_hidden_dim=32,
        tau=1.0,
    )
    mono_model = build_ngs(d_in, n_classes, mono_config).to(device)
    mono_trainer = NGSTrainer(mono_model, trainer_config, device=device)

    for epoch in range(epochs):
        mono_model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if x.dim() == 3:
                B, M, D = x.shape
                x = x.view(B, -1)
            optimizer = mono_trainer.optimizer
            optimizer.zero_grad()
            out_obj = mono_model(x)
            logits = out_obj.logits
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mono_model.parameters(), 1.0)
            optimizer.step()
        
        if hasattr(mono_model, "adapt_density"):
            with torch.no_grad():
                z_samples = []
                for x, _ in train_loader:
                    x = x.to(device)
                    if x.dim() == 3:
                        B, M, D = x.shape
                        x = x.view(B, -1)
                    z = mono_model.p_down(x)
                    z_samples.append(z)
                    if len(torch.cat(z_samples)) >= 200:
                        break
                if z_samples:
                    z_samples = torch.cat(z_samples)[:200]
                    mono_model.adapt_density(
                        z_samples=z_samples, split_thresh=0.05, prune_thresh=0.01, max_spawn_per_call=5
                    )

    mono_acc = compute_accuracy(mono_model, test_loader, device)
    print(f"Monolithic baseline: {mono_acc:.4f}")

    results = {
        "modality_types": list(modality_types),
        "num_subspaces": num_subspaces,
        "factorized_acc": float(final_acc),
        "monolithic_acc": float(mono_acc),
        "improvement": float(final_acc - mono_acc),
        "final_K": int(model.K),
        "subspace_usage": {str(k): np.mean(v) if v else 0.0 for k, v in subspace_usage.items()},
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"multimodal_{'_'.join(modality_types)}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path / f'multimodal_{'_'.join(modality_types)}.json'}")
    return results


if __name__ == "__main__":
    run_multimodal_benchmark()
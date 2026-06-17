"""Rapid adaptation benchmark measuring epochs to target accuracy."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass


@dataclass
class AdaptationResult:
    model_name: str
    epochs_to_90: Optional[float]
    final_acc: float
    accuracy_curve: list


def train_epoch(model, optimizer, loader, device):
    model.train()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    return correct / total


def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            out = model(x)
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return correct / total


def run_rapid_adaptation_benchmark(
    dataset: str = "mnist",
    epochs: int = 50,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./rapid_adaptation_results",
    target_acc: float = 0.90,
    latent_dim: int = 32,
    k_init: int = 32,
    max_k: int = 128,
    lr: float = 1e-3,
    batch_size: int = 128,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running rapid adaptation on {dataset} using {device}")

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset, TensorDataset

    if dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
        d_in = 28 * 28
        d_out = 10
    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        train_set = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
        d_in = 3 * 32 * 32
        d_out = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    from ngs.models import build_ngs

    n_runs = 3
    results = {}

    for run in range(n_runs):
        torch.manual_seed(seed + run)

        # NGS model
        ngs_config = NGSConfig(
            latent_dim=latent_dim,
            k_init=k_init,
            max_k=max_k,
            top_k=8,
            routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
            parameter_storage=ParameterStorage.DIRECT_ADAPTER,
            topology_control=TopologyControl.DISCRETE_HEURISTIC,
            memory_management=MemoryManagement.DYNAMIC_GROWTH,
            split_threshold=0.05,
            prune_threshold=0.01,
            tau=1.0,
        )
        ngs_model = build_ngs(d_in, d_out, ngs_config).to(device)
        ngs_optimizer = torch.optim.AdamW(ngs_model.parameters(), lr=lr, weight_decay=1e-4)

        # Baseline MLP
        mlp_model = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(),
            nn.Linear(256, d_out),
        ).to(device)
        mlp_optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=lr, weight_decay=1e-4)

        ngs_accs, mlp_accs = [], []
        ngs_epochs_90, mlp_epochs_90 = None, None

        for epoch in range(epochs):
            train_acc_ngs = train_epoch(ngs_model, ngs_optimizer, train_loader, device)
            test_acc_ngs = evaluate(ngs_model, test_loader, device)
            ngs_accs.append(test_acc_ngs)
            if test_acc_ngs >= target_acc and ngs_epochs_90 is None:
                ngs_epochs_90 = epoch + 1

            train_acc_mlp = train_epoch(mlp_model, mlp_optimizer, train_loader, device)
            test_acc_mlp = evaluate(mlp_model, test_loader, device)
            mlp_accs.append(test_acc_mlp)
            if test_acc_mlp >= target_acc and mlp_epochs_90 is None:
                mlp_epochs_90 = epoch + 1

            if epoch % 10 == 0:
                print(f"Run {run+1}/{n_runs} Epoch {epoch}: NGS={test_acc_ngs:.4f} MLP={test_acc_mlp:.4f}")

        def make_result(name, e90, final, curve):
            return {"model_name": name, "epochs_to_90": e90, "final_acc": final, "accuracy_curve": curve}

        results[f"run_{run}"] = {
            "ngs": make_result("NGS", ngs_epochs_90, ngs_accs[-1], ngs_accs),
            "mlp": make_result("MLP", mlp_epochs_90, mlp_accs[-1], mlp_accs),
        }

    summary = {
        "dataset": dataset,
        "target_acc": target_acc,
        "ngs_mean_epochs": float(np.mean([r["ngs"]["epochs_to_90"] if r["ngs"]["epochs_to_90"] else epochs for r in results.values()])),
        "mlp_mean_epochs": float(np.mean([r["mlp"]["epochs_to_90"] if r["mlp"]["epochs_to_90"] else epochs for r in results.values()])),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"rapid_adaptation_{dataset}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output_path / f'rapid_adaptation_{dataset}.json'}")
    return summary

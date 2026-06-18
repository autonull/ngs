"""Extended real-world domain benchmarks for NGS (vision, NLP, robotics)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import warnings

from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models import build_ngs
from ngs.training import NGSTrainer, TrainerConfig

try:
    from torchvision import datasets, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    datasets = None
    transforms = None


# ---------------------------------------------------------------------------
# 1. VISION BENCHMARKS
# ---------------------------------------------------------------------------

class SimpleCNNBackbone(nn.Module):
    """Lightweight CNN backbone for vision benchmarks."""

    def __init__(self, in_channels: int = 3, feat_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_vision_transform(dataset: str, train: bool = True):
    """Get image transforms for vision benchmarks."""
    if dataset == "cifar10" or dataset == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif dataset == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif dataset == "fashion_mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


def run_vision_benchmark(
    dataset: str = "cifar10",
    backbone_type: str = "cnn",
    epochs: int = 10,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./vision_results",
    latent_dim: int = 64,
    k_init: int = 64,
    max_k: int = 256,
    top_k: int = 8,
    lr: float = 1e-3,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """
    Vision benchmark: train NGS head on top of CNN backbone for image classification.

    Datasets: cifar10 (10 classes), cifar100 (100 classes), fashion_mnist.
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required for vision benchmarks")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[Vision Benchmark] Dataset={dataset}, device={device}")

    # Load dataset
    dataset_cls = {
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
        "fashion_mnist": datasets.FashionMNIST,
        "mnist": datasets.MNIST,
    }[dataset]

    num_classes = {"cifar10": 10, "cifar100": 100, "fashion_mnist": 10, "mnist": 10}[dataset]
    in_channels = 1 if dataset in ("mnist", "fashion_mnist") else 3

    train_dataset = dataset_cls(
        root="./data", train=True, download=True,
        transform=get_vision_transform(dataset, train=True),
    )
    test_dataset = dataset_cls(
        root="./data", train=False, download=True,
        transform=get_vision_transform(dataset, train=False),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Build backbone + NGS
    backbone = SimpleCNNBackbone(in_channels=in_channels, feat_dim=latent_dim).to(device)

    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=top_k,
        routing=RoutingStrategy.FACTORIZED_SUBSPACE,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.PRE_ALLOCATED,
        num_subspaces=4,
        split_threshold=0.05,
        prune_threshold=0.01,
    )
    ngs_model = build_ngs(latent_dim, num_classes, config).to(device)
    model = nn.Sequential(backbone, ngs_model)

    trainer_config = TrainerConfig(
        lr=lr,
        epochs=1,
        batch_size=batch_size,
        entropy_weight=0.01,
        diversity_weight=0.01,
        adapt_every_epoch=True,
        split_thresh=0.05,
        prune_thresh=0.01,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Training loop
    train_losses, test_accs, k_history = [], [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            logits = output.logits if hasattr(output, 'logits') else output
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Topology adaptation
        if hasattr(ngs_model, 'adapt_density'):
            ngs_model.adapt_density(split_thresh=0.05, prune_thresh=0.01)

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                logits = output.logits if hasattr(output, 'logits') else output
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

        acc = correct / total
        train_losses.append(epoch_loss / len(train_loader))
        test_accs.append(acc)
        k_history.append(ngs_model.K)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={train_losses[-1]:.4f}, acc={acc:.4f}, K={ngs_model.K}")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "benchmark": "vision",
        "dataset": dataset,
        "num_classes": num_classes,
        "final_test_acc": test_accs[-1],
        "best_test_acc": max(test_accs),
        "final_k": ngs_model.K,
        "test_acc_history": test_accs,
        "train_loss_history": train_losses,
        "k_history": k_history,
        "config": {
            "latent_dim": latent_dim, "k_init": k_init, "max_k": max_k,
            "top_k": top_k, "lr": lr, "epochs": epochs,
        },
    }

    with open(Path(output_dir) / f"vision_{dataset}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Vision {dataset}] Best test acc: {results['best_test_acc']:.4f}, Final K: {results['final_k']}")
    return results


# ---------------------------------------------------------------------------
# 2. NLP BENCHMARKS
# ---------------------------------------------------------------------------

class TextMLPBackbone(nn.Module):
    """Simple text embedding backbone for NLP benchmarks."""

    def __init__(self, vocab_size: int, embed_dim: int = 64, feat_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len] or [B, seq_len, embed_dim]
        if x.dtype == torch.long:
            emb = self.embedding(x).mean(dim=1)
        else:
            emb = x.mean(dim=1)
        return self.proj(emb)


def generate_text_dataset(
    dataset: str,
    n_train: int = 5000,
    n_test: int = 1000,
    vocab_size: int = 1000,
    seq_len: int = 32,
    num_classes: int = 4,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic text classification data."""
    rng = np.random.RandomState(seed)
    x_train = rng.randint(0, vocab_size, size=(n_train, seq_len))
    y_train = rng.randint(0, num_classes, size=n_train)
    x_test = rng.randint(0, vocab_size, size=(n_test, seq_len))
    y_test = rng.randint(0, num_classes, size=n_test)

    return (
        torch.tensor(x_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(x_test, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
    )


def run_nlp_benchmark(
    dataset: str = "ag_news",
    epochs: int = 20,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./nlp_results",
    latent_dim: int = 64,
    k_init: int = 32,
    max_k: int = 128,
    top_k: int = 8,
    lr: float = 5e-4,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    NLP benchmark: NGS adapter for text classification.

    Uses a learned embedding + NGS head on synthetic/generated data.
    For real NLP datasets, install datasets library and swap data loading.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[NLP Benchmark] Dataset={dataset}, device={device}")

    # Generate text data
    num_classes = {"ag_news": 4, "imdb": 2, "dbpedia": 14, "synthetic": 4}.get(dataset, 4)
    vocab_size = 1000
    seq_len = 32

    x_train, y_train, x_test, y_test = generate_text_dataset(
        dataset, vocab_size=vocab_size, seq_len=seq_len,
        num_classes=num_classes, seed=seed,
    )

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Build embedding + NGS
    backbone = TextMLPBackbone(vocab_size=vocab_size, feat_dim=latent_dim).to(device)

    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=top_k,
        routing=RoutingStrategy.FACTORIZED_SUBSPACE,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.PRE_ALLOCATED,
        num_subspaces=2,
        hypernetwork_code_dim=8,
        split_threshold=0.05,
        prune_threshold=0.01,
    )
    ngs_model = build_ngs(latent_dim, num_classes, config).to(device)
    model = nn.Sequential(backbone, ngs_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Training loop
    train_losses, test_accs, k_history = [], [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            logits = output.logits if hasattr(output, 'logits') else output
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Topology adaptation
        if hasattr(ngs_model, 'adapt_density'):
            ngs_model.adapt_density(split_thresh=0.05, prune_thresh=0.01)

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                logits = output.logits if hasattr(output, 'logits') else output
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

        acc = correct / total
        train_losses.append(epoch_loss / len(train_loader))
        test_accs.append(acc)
        k_history.append(ngs_model.K)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={train_losses[-1]:.4f}, acc={acc:.4f}, K={ngs_model.K}")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "benchmark": "nlp",
        "dataset": dataset,
        "num_classes": num_classes,
        "final_test_acc": test_accs[-1],
        "best_test_acc": max(test_accs),
        "final_k": ngs_model.K,
        "test_acc_history": test_accs,
        "train_loss_history": train_losses,
        "k_history": k_history,
        "config": {
            "latent_dim": latent_dim, "k_init": k_init, "max_k": max_k,
            "top_k": top_k, "lr": lr, "epochs": epochs,
        },
    }

    with open(Path(output_dir) / f"nlp_{dataset}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[NLP {dataset}] Best test acc: {results['best_test_acc']:.4f}, Final K: {results['final_k']}")
    return results


# ---------------------------------------------------------------------------
# 3. ROBOTICS BENCHMARKS
# ---------------------------------------------------------------------------

class MLPPolicy(nn.Module):
    """Simple MLP policy with NGS head for continuous control."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generate_control_dataset(
    n_samples: int = 5000,
    obs_dim: int = 8,
    action_dim: int = 2,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic control dataset (regression targets)."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n_samples, obs_dim).astype(np.float32)
    # Non-linear target: mixture of sinusoids
    y = (
        np.sin(x[:, 0:1]) * 0.5
        + np.cos(x[:, 1:2]) * 0.3
        + x[:, 2:3] * 0.1
        + np.tanh(x[:, 3:4]) * 0.5
        + rng.randn(n_samples, action_dim).astype(np.float32) * 0.05
    )
    return torch.tensor(x), torch.tensor(y)


def run_robotics_benchmark(
    env: str = "synthetic_control",
    epochs: int = 30,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./robotics_results",
    latent_dim: int = 32,
    k_init: int = 32,
    max_k: int = 128,
    top_k: int = 8,
    lr: float = 1e-3,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """
    Robotics benchmark: NGS for continuous control / regression.

    Uses synthetic control data as proxy for real robotics domains.
    For real environments (MuJoCo, Gym), swap data loading with
    `gym.make(env).sample_trajectories()` or offline RL datasets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[Robotics Benchmark] Env={env}, device={device}")

    # Generate control data
    obs_dim, action_dim = 8, 2
    x_train, y_train = generate_control_dataset(5000, obs_dim, action_dim, seed)
    x_test, y_test = generate_control_dataset(1000, obs_dim, action_dim, seed + 1)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Build policy + NGS
    backbone = MLPPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=latent_dim).to(device)

    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=top_k,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.PRE_ALLOCATED,
        split_threshold=0.05,
        prune_threshold=0.01,
    )
    ngs_model = build_ngs(latent_dim, action_dim, config).to(device)
    model = nn.Sequential(backbone, ngs_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Training loop (regression)
    train_losses, test_losses, k_history = [], [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            pred = output.logits if hasattr(output, 'logits') else output
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Topology adaptation
        if hasattr(ngs_model, 'adapt_density'):
            ngs_model.adapt_density(split_thresh=0.05, prune_thresh=0.01)

        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                pred = output.logits if hasattr(output, 'logits') else output
                test_loss += F.mse_loss(pred, y, reduction='sum').item()

        test_loss /= len(test_dataset)
        train_losses.append(epoch_loss / len(train_loader))
        test_losses.append(test_loss)
        k_history.append(ngs_model.K)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: train_loss={train_losses[-1]:.6f}, "
                  f"test_loss={test_loss:.6f}, K={ngs_model.K}")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "benchmark": "robotics",
        "env": env,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "final_test_mse": test_losses[-1],
        "best_test_mse": min(test_losses),
        "final_k": ngs_model.K,
        "test_loss_history": test_losses,
        "train_loss_history": train_losses,
        "k_history": k_history,
        "config": {
            "latent_dim": latent_dim, "k_init": k_init, "max_k": max_k,
            "top_k": top_k, "lr": lr, "epochs": epochs,
        },
    }

    with open(Path(output_dir) / f"robotics_{env}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Robotics {env}] Best test MSE: {results['best_test_mse']:.6f}, Final K: {results['final_k']}")
    return results


# ---------------------------------------------------------------------------
# 4. UNIFIED BENCHMARK RUNNER
# ---------------------------------------------------------------------------

def run_extended_benchmark(
    domain: str = "vision",
    dataset: str = "cifar10",
    epochs: int = 10,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./extended_results",
    **kwargs,
) -> Dict[str, Any]:
    """
    Run an extended benchmark in the specified domain.

    Args:
        domain: One of "vision", "nlp", "robotics"
        dataset: Dataset name within domain
        epochs: Number of training epochs
        device: Device to run on
        seed: Random seed
        output_dir: Output directory for results
        **kwargs: Additional config overrides

    Returns:
        Results dictionary
    """
    domain_map = {
        "vision": run_vision_benchmark,
        "nlp": run_nlp_benchmark,
        "robotics": run_robotics_benchmark,
    }

    if domain not in domain_map:
        raise ValueError(f"Unknown domain: {domain}. Choose from {list(domain_map.keys())}")

    run_fn = domain_map[domain]
    return run_fn(
        dataset=dataset,
        epochs=epochs,
        device=device,
        seed=seed,
        output_dir=output_dir,
        **{k: v for k, v in kwargs.items()
           if k in ['latent_dim', 'k_init', 'max_k', 'top_k', 'lr', 'batch_size']},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run extended NGS benchmarks")
    parser.add_argument("--domain", default="vision", choices=["vision", "nlp", "robotics"])
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./extended_results")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--k-init", type=int, default=64)
    parser.add_argument("--max-k", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    run_extended_benchmark(
        domain=args.domain,
        dataset=args.dataset,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        latent_dim=args.latent_dim,
        k_init=args.k_init,
        max_k=args.max_k,
        top_k=args.top_k,
        lr=args.lr,
        batch_size=args.batch_size,
    )

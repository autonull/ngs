"""
Experiment runner for NGSLayer and StackedNGSModel architectures.
Verifies the architectural improvements from TODO2 Phase 1.

Supports:
- Single NGSLayer with residual + norm
- Stacked NGSLayer (2, 4, 8 layers)
- Multi-head projection (M=1,4,8)
- Standard classification datasets (Digits, MNIST, Fashion-MNIST, CIFAR10, CIFAR100)
"""
import os
import json
import time
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ngs.modules.ngs_layer import StackedNGSModel, build_stacked_ngs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- Dataset helpers (minimal, no continual learning) ---

def _load_standard_dataset(name: str):
    """Load a standard classification dataset.
    
    Returns (train_dataset, test_dataset, input_dim, output_dim, n_train).
    """
    from torchvision import datasets, transforms
    import torch.utils.data as data
    
    if name == 'digits':
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        digits = load_digits()
        X = digits.data.astype(np.float32) / 16.0
        y = digits.target.astype(np.int64)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        test_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_test)
        )
        return train_ds, test_ds, 64, 10, len(X_train)
    
    elif name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
        return train_ds, test_ds, 784, 10, len(train_ds)
    
    elif name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        train_ds = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        return train_ds, test_ds, 784, 10, len(train_ds)
    
    elif name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        return train_ds, test_ds, 3072, 10, len(train_ds)
    
    elif name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        train_ds = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
        return train_ds, test_ds, 3072, 100, len(train_ds)
    
    else:
        raise ValueError(f"Unknown dataset: {name}")


# --- Experiment configurations ---

@dataclass
class NgsLayerExpConfig:
    """Configuration for an NGSLayer experiment."""
    name: str
    dataset: str
    n_layers: int
    d_latent: int
    n_heads: int
    n_experts: int
    top_k: int
    use_residual: bool
    use_norm: bool
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    expected_accuracy: Optional[float] = None


# Experiment catalog matching TODO2 1.4.x
NGS_LAYER_EXPERIMENTS = {
    'ngsl_digits': NgsLayerExpConfig(
        name='NGSLayer+Residual on Digits',
        dataset='digits',
        n_layers=1, d_latent=32, n_heads=1,
        n_experts=256, top_k=8,
        use_residual=True, use_norm=True,
        epochs=10, expected_accuracy=0.88,
    ),
    'ngsl_mnist': NgsLayerExpConfig(
        name='NGSLayer+Residual on MNIST',
        dataset='mnist',
        n_layers=1, d_latent=32, n_heads=1,
        n_experts=256, top_k=8,
        use_residual=True, use_norm=True,
        epochs=10, expected_accuracy=0.93,
    ),
    'ngsl_fashion': NgsLayerExpConfig(
        name='NGSLayer+Residual on FashionMNIST',
        dataset='fashion_mnist',
        n_layers=1, d_latent=32, n_heads=1,
        n_experts=256, top_k=8,
        use_residual=True, use_norm=True,
        epochs=10, expected_accuracy=0.78,
    ),
    'ngsl_cifar10_2l': NgsLayerExpConfig(
        name='2-layer StackedNGSLayer on CIFAR10',
        dataset='cifar10',
        n_layers=2, d_latent=128, n_heads=1,
        n_experts=256, top_k=8,
        use_residual=True, use_norm=True,
        epochs=20, expected_accuracy=0.45,
    ),
    'ngsl_cifar10_4l': NgsLayerExpConfig(
        name='4-layer StackedNGSLayer on CIFAR10',
        dataset='cifar10',
        n_layers=4, d_latent=128, n_heads=1,
        n_experts=256, top_k=8,
        use_residual=True, use_norm=True,
        epochs=20, expected_accuracy=0.55,
    ),
    'ngsl_cifar100_2l': NgsLayerExpConfig(
        name='2-layer StackedNGSLayer on CIFAR100',
        dataset='cifar100',
        n_layers=2, d_latent=128, n_heads=1,
        n_experts=256, top_k=8,
        use_residual=True, use_norm=True,
        epochs=20, expected_accuracy=0.25,
    ),
    'ngsl_cifar10_4h': NgsLayerExpConfig(
        name='4-head MultiHeadProj on CIFAR10',
        dataset='cifar10',
        n_layers=1, d_latent=32, n_heads=4,
        n_experts=256, top_k=8,
        use_residual=True, use_norm=True,
        epochs=20, expected_accuracy=0.50,
    ),
    'ngsl_cifar10_8h': NgsLayerExpConfig(
        name='8-head MultiHeadProj on CIFAR10',
        dataset='cifar10',
        n_layers=1, d_latent=16, n_heads=8,
        n_experts=256, top_k=8,
        use_residual=True, use_norm=True,
        epochs=20, expected_accuracy=0.55,
    ),
}


# --- Training ---

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def train_ngs_layer_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True,
) -> Dict:
    """Train and evaluate an NGSLayer/StackedNGSModel.
    
    Returns full metrics dict with accuracy history.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    test_accuracies = []
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # Evaluate
        acc = evaluate_accuracy(model, test_loader, device)
        test_accuracies.append(acc)
        best_acc = max(best_acc, acc)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, test_acc={acc:.4f}")

    wall_time = time.time() - start_time

    # Peak memory
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(device)

    # Layer metrics
    layer_metrics = {}
    if hasattr(model, 'get_layer_metrics'):
        layer_metrics = model.get_layer_metrics()

    return {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'final_accuracy': test_accuracies[-1],
        'best_accuracy': best_acc,
        'wall_time_seconds': wall_time,
        'peak_gpu_memory_bytes': peak_mem,
        'total_params': sum(p.numel() for p in model.parameters()),
        'active_units_per_layer': layer_metrics,
        'final_loss': train_losses[-1],
    }


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cpu',
) -> float:
    """Evaluate model accuracy on a test set."""
    model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def run_ngs_layer_experiment(
    config: NgsLayerExpConfig,
    seed: int = 42,
    output_dir: str = './results/ngs_layer',
    verbose: bool = True,
) -> Dict:
    """Run a single NGSLayer experiment with given config and seed."""
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    train_ds, test_ds, input_dim, output_dim, n_train = _load_standard_dataset(config.dataset)
    # Adaptive batch size: aim for ~50 batches/epoch on small datasets
    batch_size = min(config.batch_size, max(32, n_train // 50))
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0
    )

    # Build model
    model = build_stacked_ngs(
        d_in=input_dim,
        d_out=output_dim,
        n_layers=config.n_layers,
        d_latent=config.d_latent,
        n_experts=config.n_experts,
        n_heads=config.n_heads,
        top_k=config.top_k,
        use_residual=config.use_residual,
        use_norm=config.use_norm,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {config.name}")
        print(f"Dataset: {config.dataset} ({input_dim} -> {output_dim})")
        print(f"Model: {config.n_layers}x NGSLayer (d_latent={config.d_latent}, "
              f"n_heads={config.n_heads}, n_experts={config.n_experts})")
        print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Device: {device}, Seed: {seed}")
        print(f"{'='*60}")

    # Train
    metrics = train_ngs_layer_model(
        model, train_loader, test_loader,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        device=device,
        verbose=verbose,
    )

    if verbose:
        print(f"\n  Final accuracy: {metrics['final_accuracy']:.4f}")
        print(f"  Best accuracy:  {metrics['best_accuracy']:.4f}")
        print(f"  Wall time:      {metrics['wall_time_seconds']:.1f}s")
        if config.expected_accuracy:
            hit = metrics['final_accuracy'] >= config.expected_accuracy
            print(f"  Expected ≥{config.expected_accuracy:.0%}: {'PASS' if hit else 'FAIL'}")

    # Package result
    result = {
        'config': asdict(config),
        'seed': seed,
        **metrics,
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    safe_name = config.name.replace(' ', '_').replace(',', '')
    result_file = os.path.join(output_dir, f"{safe_name}_seed{seed}.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result


def run_ngs_layer_experiments(
    experiment_keys: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    output_dir: str = './results/ngs_layer',
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Run multiple NGSLayer experiments across seeds."""
    if experiment_keys is None:
        experiment_keys = list(NGS_LAYER_EXPERIMENTS.keys())
    if seeds is None:
        seeds = [42]

    all_results = {}

    for key in experiment_keys:
        if key not in NGS_LAYER_EXPERIMENTS:
            print(f"Unknown experiment: {key}")
            continue

        config = NGS_LAYER_EXPERIMENTS[key]

        for seed in seeds:
            full_key = f"{key}_seed{seed}"
            try:
                result = run_ngs_layer_experiment(config, seed, output_dir, verbose)
                all_results[full_key] = result
            except Exception as e:
                print(f"Error in {full_key}: {e}")
                all_results[full_key] = {'error': str(e)}

    return all_results


def aggregate_ngs_layer_results(results: Dict[str, Dict]) -> Dict:
    """Aggregate experiment results across seeds."""
    from experiments.metrics import compute_confidence_interval

    grouped = {}
    for key, result in results.items():
        if 'error' in result:
            continue
        # Extract experiment key (before _seedN)
        exp_key = key.rsplit('_seed', 1)[0]
        if exp_key not in grouped:
            grouped[exp_key] = []
        grouped[exp_key].append(result)

    aggregated = {}
    for exp_key, seed_results in grouped.items():
        accs = np.array([r['final_accuracy'] for r in seed_results])
        bests = np.array([r['best_accuracy'] for r in seed_results])
        times = np.array([r['wall_time_seconds'] for r in seed_results])
        params = seed_results[0]['total_params']

        config = seed_results[0]['config']
        expected = config.get('expected_accuracy')

        aggregated[exp_key] = {
            'name': config['name'],
            'dataset': config['dataset'],
            'n_layers': config['n_layers'],
            'n_heads': config['n_heads'],
            'final_accuracy_mean': float(np.mean(accs)),
            'final_accuracy_std': float(np.std(accs)),
            'final_accuracy_ci95': compute_confidence_interval(accs),
            'best_accuracy_mean': float(np.mean(bests)),
            'best_accuracy_std': float(np.std(bests)),
            'wall_time_mean': float(np.mean(times)),
            'total_params': int(params),
            'n_seeds': len(seed_results),
            'expected_accuracy': expected,
            'final_accuracies': accs.tolist(),
        }
    return aggregated


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run NGSLayer experiments')
    parser.add_argument('--exps', nargs='+', default=None,
                        help='Experiment keys (default: all)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                        help='Random seeds')
    parser.add_argument('--output', default='./results/ngs_layer',
                        help='Output directory')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    results = run_ngs_layer_experiments(args.exps, args.seeds, args.output, args.verbose)
    agg = aggregate_ngs_layer_results(results)

    print(f"\n{'='*60}")
    print("Aggregated Results")
    print(f"{'='*60}")
    for exp_key, data in agg.items():
        status = ""
        if data['expected_accuracy']:
            hit = data['final_accuracy_mean'] >= data['expected_accuracy']
            status = "PASS" if hit else "FAIL"
        print(f"\n{data['name']}:")
        print(f"  Final Acc:  {data['final_accuracy_mean']:.4f} ± {data['final_accuracy_std']:.4f} "
              f"(expected ≥{data['expected_accuracy']:.4f}) [{status}]")
        print(f"  Best Acc:   {data['best_accuracy_mean']:.4f} ± {data['best_accuracy_std']:.4f}")
        print(f"  Params:     {data['total_params']:,}")
        print(f"  Time:       {data['wall_time_mean']:.1f}s")

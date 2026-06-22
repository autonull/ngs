"""
Ablation experiments for NGSLayer — validates each architectural component.
Uses fast pre-loaded data to eliminate PIL/transform overhead (~66% speedup).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

from ngs.modules.ngs_layer import build_stacked_ngs
from experiments.fast_data import get_fast_loaders
from experiments.ngs_layer_runner import train_ngs_layer_model


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_model(config: Dict[str, Any], input_dim: int, output_dim: int) -> nn.Module:
    """Build model from config dict."""
    # Extract ablation kwargs (prefixed with _)
    ablation_kwargs = {k: v for k, v in config.items() if k.startswith('_')}
    model_config = {k: v for k, v in config.items() if not k.startswith('_')}
    
    return build_stacked_ngs(
        d_in=input_dim,
        d_out=output_dim,
        **model_config,
        **ablation_kwargs,
    )


def run_ablation(
    name: str,
    base_config: Dict[str, Any],
    ablation_config: Dict[str, Any],
    dataset: str = 'cifar10',
    epochs: int = 10,
    seed: int = 42,
) -> Dict:
    """Run single ablation experiment."""
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset with fast pre-loaded tensors
    train_loader, test_loader, input_dim, output_dim = get_fast_loaders(dataset)

    # Build model with ablation
    config = {**base_config, **ablation_config}
    model = build_model(config, input_dim, output_dim).to(device)

    # Train
    t0 = time.time()
    metrics = train_ngs_layer_model(
        model, train_loader, test_loader,
        epochs=epochs, lr=1e-3, weight_decay=1e-4,
        device=device, verbose=False,
    )
    wall_time = time.time() - t0

    return {
        'name': name,
        'config': config,
        'final_accuracy': metrics['final_accuracy'],
        'best_accuracy': metrics['best_accuracy'],
        'wall_time': wall_time,
        'params': metrics['total_params'],
    }


def main():
    """Run all ablations on CIFAR10 2-layer."""
    # Base config: CIFAR10 2-layer (best validated config)
    base_config = {
        'n_layers': 2,
        'd_latent': 128,
        'n_experts': 256,
        'n_heads': 1,
        'top_k': 8,
        'use_residual': True,
        'use_norm': True,
    }

    ablations = [
        ('baseline', {}),
        ('no_residual', {'use_residual': False}),
        ('no_layernorm', {'use_norm': False}),
        ('no_out_bias', {'_remove_out_bias': True}),  # handled in model build
        ('bad_mu_init', {'_router_mu_std': 1.0}),  # handled in model build
    ]

    # Multi-head ablation needs separate base
    mh_base = {**base_config, 'n_heads': 4}
    mh_ablations = [
        ('4h_baseline', {}),
        ('4h_no_multihead', {'n_heads': 1}),
    ]

    print("=" * 60)
    print("NGSLayer Ablation Study (CIFAR10)")
    print("=" * 60)

    results = {}

    # Standard ablations
    for name, abl_config in ablations:
        print(f"\n>>> {name}")
        result = run_ablation(name, base_config, abl_config, epochs=10)
        results[name] = result
        print(f"    Acc: {result['final_accuracy']:.4f}, Time: {result['wall_time']:.1f}s")

    # Multi-head ablations
    for name, abl_config in mh_ablations:
        print(f"\n>>> {name}")
        result = run_ablation(name, mh_base, abl_config, epochs=10)
        results[name] = result
        print(f"    Acc: {result['final_accuracy']:.4f}, Time: {result['wall_time']:.1f}s")

    # Summary table
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<25} {'Acc':>8} {'Δ vs baseline':>14} {'Time':>8}")
    print("-" * 60)
    baseline_acc = results['baseline']['final_accuracy']
    for name, r in results.items():
        if name.startswith('4h_'):
            ref = results['4h_baseline']['final_accuracy']
        else:
            ref = baseline_acc
        delta = r['final_accuracy'] - ref
        print(f"{name:<25} {r['final_accuracy']:.4f} {delta:+.4f} {r['wall_time']:.1f}s")

    return results


if __name__ == '__main__':
    main()
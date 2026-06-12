"""
Compute and memory profiling for LeanNGS.
"""
import torch
import torch.nn as nn
import time
import psutil
import os
from typing import Dict, List, Any
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np


@dataclass
class ProfileResult:
    """Results from profiling a model."""
    model_name: str
    forward_time_ms: float
    backward_time_ms: float
    peak_memory_mb: float
    param_count: int
    active_units: int
    flops_per_forward: float


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, input_shape: tuple) -> float:
    """Estimate FLOPs for a forward pass."""
    # Simple estimation - can be replaced with thop or fvcore
    total_flops = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # 2 * in_features * out_features for matmul + add
            total_flops += 2 * module.in_features * module.out_features
        elif isinstance(module, nn.Conv2d):
            total_flops += 2 * module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1]
    return total_flops


@contextmanager
def memory_tracker():
    """Context manager to track peak memory usage."""
    process = psutil.Process(os.getpid())
    peak_before = process.memory_info().rss / 1024 / 1024  # MB
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        peak_after = process.memory_info().rss / 1024 / 1024
        if torch.cuda.is_available():
            gpu_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"  CPU Peak Memory: {peak_after - peak_before:.1f} MB")
            print(f"  GPU Peak Memory: {gpu_peak:.1f} MB")
        else:
            print(f"  CPU Peak Memory: {peak_after - peak_before:.1f} MB")


def profile_model(
    model: nn.Module,
    input_shape: tuple,
    device: str = 'cuda',
    n_warmup: int = 10,
    n_iter: int = 100
) -> ProfileResult:
    """Profile a model's forward/backward time and memory."""
    model = model.to(device)
    model.train()
    
    # Create dummy input
    x = torch.randn(input_shape, device=device)
    # Get output dim from model
    out_dim = model.p_up.out_features if hasattr(model, 'p_up') else \
              model.mlp.net[-1].out_features if hasattr(model, 'mlp') else \
              model.net[-1].out_features if hasattr(model, 'net') else \
              model.head.out_features if hasattr(model, 'head') else 10
    y = torch.randint(0, out_dim, (input_shape[0],), device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Warmup
    for _ in range(n_warmup):
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
    
    # Profile forward
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.perf_counter()
    for _ in range(n_iter):
        logits = model(x)
    torch.cuda.synchronize() if device == 'cuda' else None
    forward_time = (time.perf_counter() - start) / n_iter * 1000  # ms
    
    # Profile backward
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.perf_counter()
    for _ in range(n_iter):
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize() if device == 'cuda' else None
    backward_time = (time.perf_counter() - start) / n_iter * 1000  # ms
    
    # Memory
    if device == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        import psutil
        process = psutil.Process(os.getpid())
        peak_mem = process.memory_info().rss / 1024 / 1024
    
    param_count = count_parameters(model)
    flops = estimate_flops(model, input_shape)
    
    active_units = 0
    if hasattr(model, 'K'):
        active_units = model.K
    elif hasattr(model, 'active_mask'):
        active_units = model.active_mask.sum().item()
    
    return ProfileResult(
        model_name=model.__class__.__name__,
        forward_time_ms=forward_time,
        backward_time_ms=backward_time,
        peak_memory_mb=peak_mem,
        param_count=param_count,
        active_units=active_units,
        flops_per_forward=flops
    )


def compare_models(
    models: Dict[str, nn.Module],
    input_shape: tuple,
    device: str = 'cuda'
) -> Dict[str, ProfileResult]:
    """Compare multiple models."""
    results = {}
    for name, model in models.items():
        print(f"\nProfiling {name}...")
        try:
            results[name] = profile_model(model, input_shape, device)
            r = results[name]
            print(f"  Forward: {r.forward_time_ms:.2f} ms")
            print(f"  Backward: {r.backward_time_ms:.2f} ms")
            print(f"  Peak Mem: {r.peak_memory_mb:.1f} MB")
            print(f"  Params: {r.param_count:,}")
            print(f"  Active Units: {r.active_units}")
        except Exception as e:
            print(f"  Error: {e}")
            results[name] = None
    return results


def print_comparison_table(results: Dict[str, ProfileResult]):
    """Print comparison table."""
    print("\n" + "="*100)
    print(f"{'Model':<20} {'Forward (ms)':>12} {'Backward (ms)':>14} {'Memory (MB)':>12} {'Params':>10} {'Active Units':>12} {'FLOPs':>15}")
    print("="*100)
    for name, r in results.items():
        if r is None:
            print(f"{name:<20} {'ERROR':>12}")
            continue
        print(f"{name:<20} {r.forward_time_ms:>12.2f} {r.backward_time_ms:>14.2f} {r.peak_memory_mb:>12.1f} {r.param_count:>10,} {r.active_units:>12} {r.flops_per_forward:>15,.0f}")


if __name__ == '__main__':
    # Quick test
    from experiments.config import EXPERIMENTS
    from experiments.baselines import create_baseline
    from experiments.lean_ngs_trainer import create_lean_ngs
    
    config = EXPERIMENTS['split_mnist']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_shape = (256, config.input_dim)
    
    models = {
        'MLP': create_baseline('mlp', config.input_dim, config.output_dim),
        'ER': create_baseline('er', config.input_dim, config.output_dim),
        'EWC': create_baseline('ewc', config.input_dim, config.output_dim),
        'LwF': create_baseline('lwf', config.input_dim, config.output_dim),
        'LoRA': create_baseline('lora', config.input_dim, config.output_dim),
        'LeanNGS': create_lean_ngs(config.input_dim, config.output_dim, **{
            'd_latent': 32, 'k_init': 128, 'max_k': 448, 'top_k': 8
        }),
    }
    
    results = compare_models(models, input_shape, device)
    print_comparison_table(results)
"""Performance profiling for NGS."""

import torch
import time
import numpy as np
from typing import Dict, List
from ngs import NGSConfig, build_ngs
from ngs.core.interfaces import RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement


def profile_forward(
    model: torch.nn.Module,
    input_shape: tuple,
    num_warmup: int = 10,
    num_iter: int = 100,
    device: str = "cuda",
) -> Dict[str, float]:
    """Profile forward pass latency."""
    model.eval()
    x = torch.randn(input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Timing
    times = []
    with torch.no_grad():
        for _ in range(num_iter):
            start = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    times = np.array(times) * 1000  # ms
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
    }


def profile_memory(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str = "cuda",
) -> Dict[str, float]:
    """Profile memory usage."""
    if device != "cuda":
        return {"allocated_mb": 0, "reserved_mb": 0}
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model.eval()
    x = torch.randn(input_shape, device=device)
    
    with torch.no_grad():
        _ = model(x)
    
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }


def profile_flops(model: torch.nn.Module, input_shape: tuple) -> Dict[str, float]:
    """Estimate FLOPs (rough approximation)."""
    # This is a rough estimate - for accurate FLOPs use torchprofile or fvcore
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Rough estimate: 2 FLOPs per multiply-add per parameter per sample
    # For transformer-like: ~6 * params * seq_len (very rough)
    flops_per_sample = total_params * 2
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "estimated_flops_per_sample": flops_per_sample,
        "estimated_gflops_per_sample": flops_per_sample / 1e9,
    }


def run_full_profile(
    configs: List[Dict],
    input_shape: tuple = (32, 784),
    num_iter: int = 100,
    device: str = "cuda",
) -> List[Dict]:
    """Run profiling on multiple configurations."""
    results = []
    
    for i, cfg_dict in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Profiling: {cfg_dict.get('name', 'unnamed')}")
        
        config = NGSConfig(**{k: v for k, v in cfg_dict.items() if k != 'name'})
        model = build_ngs(input_shape[-1], 10, config).to(device)
        
        # Forward latency
        latency = profile_forward(model, input_shape, num_iter=num_iter, device=device)
        
        # Memory
        memory = profile_memory(model, input_shape, device=device)
        
        # FLOPs
        flops = profile_flops(model, input_shape)
        
        # Model size
        size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        
        result = {
            "name": cfg_dict.get('name', f'config_{i}'),
            "config": {k: str(v) for k, v in config.__dict__.items()},
            "latency": latency,
            "memory": memory,
            "flops": flops,
            "model_size_mb": size_mb,
            "active_units": model.K,
        }
        
        results.append(result)
        
        print(f"  Latency: {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")
        print(f"  Memory: {memory.get('allocated_mb', 0):.1f} MB allocated")
        print(f"  Params: {flops['total_params']:,}")
        print(f"  Active K: {model.K}")
    
    return results


if __name__ == "__main__":
    import json
    from pathlib import Path
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    configs = [
        {
            "name": "monolithic_small",
            "max_k": 64,
            "k_init": 16,
            "latent_dim": 32,
            "routing": RoutingStrategy.MONOLITHIC_MAHALANOBIS,
            "parameter_storage": ParameterStorage.DIRECT_ADAPTER,
            "topology_control": TopologyControl.DISCRETE_HEURISTIC,
            "memory_management": MemoryManagement.PRE_ALLOCATED,
        },
        {
            "name": "factorized_medium",
            "max_k": 128,
            "k_init": 32,
            "latent_dim": 64,
            "routing": RoutingStrategy.FACTORIZED_SUBSPACE,
            "parameter_storage": ParameterStorage.HYPERNETWORK_GENERATED,
            "topology_control": TopologyControl.CONTINUOUS_DENSITY,
            "memory_management": MemoryManagement.DYNAMIC,
            "num_subspaces": 4,
        },
        {
            "name": "hierarchical_large",
            "max_k": 256,
            "k_init": 64,
            "latent_dim": 128,
            "routing": RoutingStrategy.HIERARCHICAL,
            "parameter_storage": ParameterStorage.LORA,
            "topology_control": TopologyControl.MERGE_AWARE,
            "memory_management": MemoryManagement.STRICT_CAPACITY,
        },
        {
            "name": "uncertainty_aware",
            "max_k": 128,
            "k_init": 32,
            "latent_dim": 64,
            "routing": RoutingStrategy.UNCERTAINTY_AWARE,
            "parameter_storage": ParameterStorage.DIRECT_ADAPTER,
            "topology_control": TopologyControl.CONTINUOUS_DENSITY,
            "memory_management": MemoryManagement.DYNAMIC,
        },
        {
            "name": "gaussian_attention",
            "max_k": 128,
            "k_init": 32,
            "latent_dim": 64,
            "routing": RoutingStrategy.GAUSSIAN_ATTENTION,
            "parameter_storage": ParameterStorage.HYPERNETWORK_GENERATED,
            "topology_control": TopologyControl.META_LEARNED,
            "memory_management": MemoryManagement.STRICT_CAPACITY,
        },
    ]
    
    results = run_full_profile(configs, input_shape=(32, 784), num_iter=50, device=device)
    
    # Save results
    Path("./profile_results").mkdir(exist_ok=True)
    with open("./profile_results/profile_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "="*100)
    print("PERFORMANCE SUMMARY")
    print("="*100)
    print(f"{'Config':<25} {'Latency (ms)':<15} {'Memory (MB)':<15} {'Params':<12} {'K':<6} {'Size (MB)':<10}")
    print("-"*100)
    for r in results:
        print(f"{r['name']:<25} {r['latency']['mean_ms']:<15.2f} "
              f"{r['memory'].get('allocated_mb', 0):<15.1f} "
              f"{r['flops']['total_params']:<12,} {r['active_units']:<6} {r['model_size_mb']:<10.1f}")
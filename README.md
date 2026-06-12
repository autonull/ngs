# MNGS — Modular Neural Gaussian System

A composable, configuration-driven framework for sparse neural networks with dynamic topology adaptation. MNGS decouples routing, parameter storage, and topology control into independent modules, enabling systematic ablation studies and efficient deployment.

## Quick Start

```bash
pip install -e .
python -m pytest tests/ -v
```

## Reproduce Experiments

```bash
# Baseline LeanNGS on Split-MNIST
python -m experiments.main --experiments split_mnist --models lean_ngs --seeds 42

# All MNGS profiles on all datasets
python -m experiments.main --experiments split_mnist split_fashion permuted_mnist rotated_mnist blurry_mnist noisy_mnist split_cifar10 split_cifar100 digits split_cifar100_20 --models mngs_baseline mngs_cfg_net mngs_ultra_edge mngs_abl_hyper --seeds 42 123 456
```

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         MNGS Model                  │
                    │  ┌──────────┐  ┌────────────────┐   │
Input ──► p_down ──►│  Router  │──►│ Parameter Store │──► p_up ──► Output
                    │ (strategy)│   │ (strategy)     │   │
                    └────┬─────┬──┘  └───────┬────────┘   │
                         │     │             │            │
                    ┌────┴─┐ ┌─┴────┐  ┌─────┴─────┐     │
                    │Topology│ │Memory│  │  Profile  │     │
                    │Manager │ │ Mgmt │  │  Config   │     │
                    └────────┘ └──────┘  └───────────┘     │
                    └─────────────────────────────────────┘
```

### Modular Components

| Component | Strategies |
|-----------|------------|
| **Routing** | `MONOLITHIC_MAHALANOBIS`, `FACTORIZED_SUBSPACE`, `LSH_APPROXIMATE` |
| **Parameter Storage** | `DIRECT_ADAPTER`, `HYPERNETWORK_GENERATED` |
| **Topology Control** | `DISCRETE_HEURISTIC`, `CONTINUOUS_DENSITY` |
| **Memory Management** | `DYNAMIC_GROWTH`, `PRE_ALLOCATED_MASKED`, `STRICT_CAPACITY` |

## Profiles

| Profile | Routing | Storage | Topology | Memory | Use Case |
|---------|---------|---------|----------|--------|----------|
| `Baseline_LeanNGS` | Monolithic | Direct | Discrete | Pre-allocated | Original LeanNGS control |
| `CFG_Net_Full` | Factorized | Hypernetwork | Continuous | Pre-allocated | Full CFG-Net upgrade |
| `Ultra_Edge_Sparse` | Factorized | Hypernetwork | Discrete | Strict | Microcontroller deployment |
| `Ablation_Hypernetwork_Only` | Monolithic | Hypernetwork | Discrete | Pre-allocated | Isolate hypernetwork value |

## Project Structure

```
mngs/
├── core/config.py           # Configuration schema (enums, MNGSConfig)
├── model.py                 # Main MNGS model
├── modules/
│   ├── routers.py           # Routing strategies
│   ├── parameter_stores.py  # Parameter storage strategies
│   └── topology_managers.py # Topology adaptation strategies
└── profiles.py              # Predefined profile configurations

experiments/
├── config.py                # Experiment configurations
├── runner.py                # Experiment runner
├── main.py                  # CLI entry point
├── mngs_trainer.py          # MNGS training loop
├── lean_ngs_trainer.py      # LeanNGS training loop
├── baselines.py             # Baseline models (MLP, ER, EWC, SI, LwF, LoRA)
├── datasets.py              # Dataset loaders
├── metrics.py               # Continual learning metrics
├── plotting.py              # Result visualization
├── comprehensive_eval.py    # Full evaluation suite
├── ablation.py              # Ablation study runner
├── hpo.py                   # Hyperparameter optimization
└── profiling.py             # Compute/memory profiling

tests/                       # Unit and end-to-end tests
```

## Development

```bash
# Run tests
pytest tests/ -v

# Run specific test file
pytest tests/test_end_to_end.py -v

# Run with coverage
pytest tests/ --cov=mngs --cov=experiments
```

## Configuration

All configurations use `MNGSConfig` dataclass with the following key fields:

```python
@dataclass
class MNGSConfig:
    # Core dimensions
    latent_dim: int = 32
    k_init: int = 128
    max_k: int = 512
    top_k: int = 8
    
    # Modular choices
    routing: RoutingStrategy = RoutingStrategy.MONOLITHIC_MAHALANOBIS
    parameter_storage: ParameterStorage = ParameterStorage.DIRECT_ADAPTER
    topology_control: TopologyControl = TopologyControl.DISCRETE_HEURISTIC
    memory_management: MemoryManagement = MemoryManagement.PRE_ALLOCATED_MASKED
    
    # Strategy-specific
    top_k_factorized: int = 2
    num_subspaces: int = 4
    hypernetwork_code_dim: int = 8
    hypernetwork_hidden_dim: int = 16
    split_threshold: float = 0.05
    prune_threshold: float = 0.01
    
    # Training
    lora_rank: int = 4
    tau: float = 1.0
    gamma_residual: float = 0.1
    ema_decay: float = 0.99
    diversity_weight: float = 0.01
    entropy_weight: float = 0.01
```

## Results

Results are saved as JSON files in `./results/` with the format:
```json
{
  "metrics": {"avg_final_accuracy": 0.75, "avg_forgetting": 0.12, ...},
  "accuracy_matrix": [[0.99, 0.0, ...], ...],
  "active_units": [128, 135, 142, ...],
  "config": "Split-MNIST",
  "model": "mngs_baseline",
  "seed": 42
}
```

## References

- [TODO.md](TODO.md) — Design narrative and development plan
- [TODO2.md](TODO2.md) — Self-contained execution guide with milestones
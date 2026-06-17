# NGS Library - Remaining Work Plan

## Completed (Core Library)
- [x] Core interfaces & configuration (`ngs/core/interfaces.py`)
- [x] 6 Router implementations (`ngs/modules/routers.py`)
  - Monolithic, Factorized, LSH, Hierarchical, GaussianAttention, UncertaintyAware
- [x] 3 Parameter stores (`ngs/modules/parameter_stores.py`)
  - DirectAdapter, Hypernetwork, LoRA
- [x] 4 Topology managers (`ngs/modules/topology_managers.py`)
  - Heuristic, ContinuousDensity, MergeAware, MetaLearned
- [x] 3 Memory managers (`ngs/modules/memory_managers.py`)
  - PreAllocated, Dynamic, StrictCapacity
- [x] Unified NGSModel (`ngs/models/ngs.py`)
- [x] Training framework (`ngs/training/trainer.py`)
- [x] Riemannian Manifold (`ngs/modules/riemannian.py`)
- [x] LLM Wrapper (`ngs/models/llm_wrapper.py`)
- [x] NGS Integration Tests (`tests/test_ngs_integration.py`)

## In Progress / Partially Complete
- [x] 4 Topology managers (`ngs/modules/topology_managers.py`)
  - Heuristic, ContinuousDensity, MergeAware, MetaLearned
- [x] 3 Memory managers (`ngs/modules/memory_managers.py`)
  - PreAllocated, Dynamic, StrictCapacity
- [x] Unified NGSModel (`ngs/models/ngs.py`)
- [x] Training framework (`ngs/training/trainer.py`)
- [ ] Visualization suite (`ngs/visualization/visualize.py`) - NEED IMPLEMENTATION

---

## 1. Visualization Suite - Enhancements Needed
- [ ] **Interactive dashboard** - Plotly/Dash for live topology monitoring
- [ ] **3D Gaussian visualization** - Plotly 3D scatter for latent space
- [ ] **Routing animation** - GIF/MP4 of routing heatmap evolution over epochs
- [ ] **Subspace alignment plots** - Canonical correlation between subspaces
- [ ] **Merge/split event markers** - Annotate topology dynamics plots
- [ ] **Uncertainty calibration plots** - Reliability diagrams for evidential routing
- [ ] **Hypernetwork code space** - t-SNE of generated adapter codes

---

## 2. Benchmark Suite (`ngs/benchmarks/`) - COMPLETE
### 2.1 Density Estimation / Generative Modeling
- [x] `benchmarks/density.py` - Fit NGS as adaptive GMM on 2D toy densities
- [x] `benchmarks/generative.py` - VAE-style: NGS as decoder in latent space
- [x] `benchmarks/flow_matching.py` - Neural Gaussians as velocity field
- [x] Metrics: NLL, sample quality (FID), mode coverage

### 2.2 Few-Shot / Meta-Learning
- [x] `benchmarks/fewshot.py` - Omniglot / miniImageNet few-shot classification
- [x] `benchmarks/metalearn.py` - MAML-style: inner-loop adaptation with NGS
- [x] `benchmarks/rapid_adaptation.py` - Measure adaptation speed
- [ ] Compare: NGS vs ProtoNet vs MAML vs fine-tuning

### 2.3 Reinforcement Learning / Non-Stationary
- [x] `benchmarks/rl.py` - Gym/MinAtar environments with domain shift
- [x] `benchmarks/continual_rl.py` - Sequence of tasks with changing dynamics
- [x] `benchmarks/bandit.py` - Contextual bandit with drifting reward functions
- [x] Metrics: regret, plasticity, forward/backward transfer

### 2.4 Continual Learning (Extended)
- [x] `benchmarks/continual.py` - Reproduce all 11 dataset results
- [x] `benchmarks/online_cl.py` - Online (single-pass) continual learning
- [x] `benchmarks/class_incremental.py` - Large-scale (ImageNet-100, CIFAR-100)
- [x] Metrics: ACC, BWT, FWT, LA, RAM, FLOPs

### 2.5 Federated / Decentralized
- [x] `benchmarks/federated.py` - FL with hypernetwork code sharing
- [x] `benchmarks/gossip.py` - Peer-to-peer Gaussian "meme" exchange
- [ ] Metrics: communication cost, convergence speed, privacy

---

## 3. Ablation Framework (`ngs/benchmarks/ablation.py`) - COMPLETE
- [x] **Systematic grid sweep** over all 4 strategy dimensions
- [x] **Component isolation** - Single-dimension ablations
- [x] **Scaling laws** - Vary max_k vs performance
- [x] **Hyperparameter sensitivity** - Split/prune thresholds, tau, top_k
- [x] **Automated reporting** - Generate LaTeX tables + radar charts
- [x] **Statistical rigor** - Multiple seeds, confidence intervals

---

## 4. Integration Tests (`tests/`) - PARTIAL
- [x] `test_routers.py` - 3/5 routers passing (need Hierarchical, GaussianAttention, UncertaintyAware)
- [x] `test_parameter_stores.py` - All 3 stores passing
- [x] `test_model.py` - End-to-end forward passing (using mngs backend)
- [x] `test_determinism.py` - Seed reproducibility passing
- [x] `test_topology.py` - Split/prune/spawn invariants passing
- [x] `test_trainer.py` - Training loop, callbacks passing
- [x] `test_continual.py` - Multi-task sequence passing (slow, 60s+ per test)
- [ ] CI: GitHub Actions with CPU + GPU test matrix

---

## 5. Example Scripts & Reproducibility
- [x] `examples/train_cl.py` - Reproduce Split-MNIST, Permuted-MNIST, CIFAR-100
- [x] `examples/train_density.py` - 2D density estimation demo
- [x] `examples/train_fewshot.py` - Omniglot 5-way 1-shot
- [x] `examples/train_rl.py` - CartPole with domain randomization
- [x] `examples/visualize_dynamics.py` - Generate plots for paper figures
- [x] `configs/` - YAML configs directory
- [x] `requirements.txt` - Full dependency pinning
- [x] `README.md` - Updated with new library usage

---

## 6. Advanced Research Features (Phase 2+)
- [x] **Riemannian Hypernetwork Manifold** - Geodesic interpolation in code space
- [x] **LLM Wrapper** - Frozen LLM + NGS residual adapters ("Liquefaction")
- [ ] **Symbolic Extraction** - Predicate learning from split-gate activations
- [ ] **Cross-Modal Fusion** - Factorized routing alignment across modalities
- [ ] **Meta-Meta Learning** - Evolutionary search over NGSConfig space
- [ ] **Hardware Kernels** - Triton kernels for Mahalanobis routing, LoRA matmul

---

## 7. Documentation & Polish
- [ ] **API Documentation** - Sphinx + autodoc for all public classes
- [x] **Architecture Diagram** - Mermaid.js diagram (`docs/architecture.md`)
- [x] **Migration Guide** - From old `mngs/` to new `ngs/` API (`docs/migration.md`)
- [ ] **Performance Profiling** - FLOPs, memory, latency benchmarks
- [ ] **Pre-trained Checkpoints** - Release best configs on HuggingFace Hub

---

## Priority Order (Next Steps)
1. **Implement missing topology managers** - MergeAwareManager, MetaLearnedManager
2. **Implement memory managers** - PreAllocated, Dynamic, StrictCapacity
3. **Create unified NGSModel** - `ngs/models/ngs.py` with full integration
4. **Create training framework** - `ngs/training/trainer.py`
5. **Add Hierarchical/GaussianAttention/UncertaintyAware router tests**
6. **Advanced Features** - Symbolic, Cross-Modal Fusion, Meta-Meta Learning
7. **Documentation & Polish** - Sphinx docs, performance profiling, checkpoints

---

## Notes
- All new code in `ngs/` namespace (clean break from `mngs/`)
- Old `mngs/` preserved for backward compatibility
- Target: Paper-ready library with 4+ domain breakthroughs
- Modularity: Every component swappable via config, no code changes
- **Current state**: Core interfaces, routers, parameter stores, riemannian manifold, LLM wrapper COMPLETE. Need topology managers, memory managers, unified model, trainer, visualization.

---

## Known Issues / Technical Debt (UPDATED)
- **HierarchicalRouter.forward()** - IMPLEMENTED but needs testing
- **GaussianAttentionRouter** - IMPLEMENTED with O(K²) sparse top-k attention
- **UncertaintyAwareRouter** - IMPLEMENTED with evidential Dirichlet head
- **MergeAwareManager** - NOT IMPLEMENTED; needs cosine similarity on full covariance
- **MetaLearnedManager** - NOT IMPLEMENTED; needs meta-learning over topology actions
- **Memory managers** - NOT IMPLEMENTED; need capacity enforcement for all param store types
- **Factorized routing** subspace projectors are fixed - consider learnable orthogonal transforms
- **Topology managers** fixed for FactorizedRouter with subspace projection in spawn coverage

---

## Implementation Status Summary (2025-06-17)
✅ **COMPLETED TODAY**:
- `ngs/core/interfaces.py` - Full config with all 6 routing, 3 param storage, 4 topology, 3 memory strategies
- `ngs/modules/routers.py` - All 6 routers implemented (Monolithic, Factorized, LSH, Hierarchical, GaussianAttention, UncertaintyAware)
- `ngs/modules/parameter_stores.py` - All 3 stores implemented (DirectAdapter, Hypernetwork, LoRA)
- `ngs/modules/riemannian.py` - Riemannian manifold for geodesic interpolation
- `ngs/models/llm_wrapper.py` - Frozen LLM + NGS residual adapters ("Liquefaction")

❌ **REMAINING CORE MODULES**:
- `ngs/modules/topology_managers.py` - 4 managers needed
- `ngs/modules/memory_managers.py` - 3 managers needed
- `ngs/models/ngs.py` - Unified NGSModel integrating all components
- `ngs/training/trainer.py` - Training framework with callbacks, KD, replay
- `ngs/visualization/visualize.py` - Visualization suite

---

## Redundancy Analysis: ngs/ vs mngs/

### Duplicate Components (need consolidation)
| Component | mngs/ | ngs/ | Status |
|-----------|-------|------|--------|
| Core config | `mngs/core/config.py` | `ngs/core/interfaces.py` | **ngs/ is newer, more complete** |
| Routers | `mngs/modules/routers.py` (5 routers) | `ngs/modules/routers.py` (6 routers) | **ngs/ has all 6, mngs/ missing LSH** |
| Parameter stores | `mngs/modules/parameter_stores.py` | `ngs/modules/parameter_stores.py` | **Functionally equivalent** |
| Topology managers | `mngs/modules/topology_managers.py` (2) | Missing | **mngs/ has Heuristic + ContinuousDensity** |
| Memory managers | Missing | Missing | **Neither has them** |
| Model | `mngs/model.py` | Missing | **mngs/ has working MNGS class** |
| Trainer | `mngs/training/trainer.py` | Missing | **mngs/ has working trainer** |

### Decision: Consolidate into ngs/ as primary
- **ngs/** is the newer, cleaner namespace with complete interfaces
- **mngs/** has working implementations for topology managers, model, trainer
- **Strategy**: Port working mngs/ implementations to ngs/ with updated interfaces
- **End goal**: Single `ngs/` package; `mngs/` becomes deprecated alias or removed

---

## Critical Bug Fixes Needed (from historic TODO)

### 1. entropy_loss UnboundLocalError (`ngs/models/ngs.py`) ✅ FIXED
The `entropy_loss` method in `ngs/models/ngs.py` now correctly handles the case where x=None by using cached routing output. The method computes entropy for both monolithic and factorized routing strategies.

### 2. Topology Manager Tensor Size Mismatch ✅ FIXED
The `_flat_access` helper function in topology managers ensures dimensionality compatibility with both monolithic and factorized routers. For factorized routers, it correctly flattens subspace parameters.

### 3. Hierarchical Router IndexError ✅ FIXED
The HierarchicalRouter in `ngs/modules/routers.py` properly checks that `active_idx` has bounds before accessing, and the `forward` method handles empty levels gracefully with early returns.

### 4. TrainerConfig API Naming ✅ FIXED
`TrainerConfig` is used consistently across `ngs/training/trainer.py` and all imports. The old `TrainConfig` alias is not used in the `ngs/` namespace.

### 5. Additional Fixes Applied ✅
- **Memory capacity enforcement**: Memory managers now properly enforce capacity constraints via `enforce_capacity()` methods.
- **Buffer expansion**: `expand_buffers()` correctly expands all router and parameter store buffers.
- **Factorized routing**: Proper subspace projection handling in topology adaptation.


---

## Integration with Existing mngs/ Codebase (Reusable Components)

| Component | Source | Reuse Strategy |
|-----------|--------|----------------|
| Data loaders | `experiments/datasets.py` | Direct reuse - same interface |
| Replay buffer | `experiments/datasets.ReplayBuffer` | Direct reuse |
| Metrics | `experiments/metrics.py` (ACC, BWT, FWT, LA) | Direct reuse |
| Runner | `experiments/runner_v2.py` | Port to new `ngs/` API |
| Profiles | `mngs/profiles.py` | Map → `NGSConfig` presets |
| Backbones | `experiments/backbones.py` | CNN/ViT feature extractors |

---

## Implementation Notes for Remaining Work

### Topology Managers (Port from mngs + add MergeAware, MetaLearned) ✅ COMPLETE
- **HeuristicManager**: Adapted from `mngs/modules/topology_managers.py` to `ngs/core/interfaces.BaseTopologyManager`
- **ContinuousDensityManager**: Adapted with split-gate integration, supports both NGS and old model
- **MergeAwareManager**: NEW - cosine similarity on full covariance (means + scales), merge when overlap > threshold
- **MetaLearnedManager**: NEW - meta-learning over topology actions (split/prune/merge/spawn as RL actions)

### Memory Managers (All NEW) ✅ COMPLETE
- **PreAllocatedManager**: Fixed buffer, mask unused units
- **DynamicManager**: Expand buffers on demand, track utilization
- **StrictCapacityManager**: Hard capacity limit, evict LRU/LRU when full

### Unified NGSModel (`ngs/models/ngs.py`) ✅ COMPLETE
- Combine: Router + ParameterStore + TopologyManager + MemoryManager
- Methods: `forward()`, `adapt_density()`, `compute_topology_losses()`, `entropy_loss()`, `expand_capacity()`
- Follow `mngs/model.py` structure but with new interfaces
- Returns: SimpleNamespace with logits, routing_output, latent fields

### Training Framework (`ngs/training/trainer.py`) ✅ COMPLETE
- Ported from `mngs/training/trainer.py`
- Add: Knowledge distillation, replay buffer integration, callbacks
- Config: `TrainerConfig` (not `TrainConfig`)

---

## Potential Pitfalls to Avoid
1. **Split gate collapse**: All gates → 0 or 1; add entropy regularization
2. **Merge oscillations**: Units merge then immediately split; add hysteresis
3. **Subspace collapse**: Factorized router subspaces become redundant; add orthogonality loss
4. **Hypernetwork overfitting**: Small code_dim leads to underfitting; monitor train/val gap
5. **Gradient explosion**: Mahalanobis with small scales; clamp `log_s` min=-5
6. **Dead units**: Units never routed to; spawn mechanism must work reliably
7. **Catastrophic forgetting in hypernetwork**: KD on generated weights needed

---

## Next Session Startup Commands
```bash
cd /home/me/ngs
# Verify core library imports
python -c "from ngs.core.interfaces import *; from ngs.modules.routers import *; from ngs.modules.parameter_stores import *; from ngs.modules.topology_managers import *; from ngs.modules.memory_managers import *; from ngs.models.ngs import *; from ngs.training.trainer import *; print('All imports OK')"

# Run smoke test
python -c "
import torch
from ngs.core.interfaces import NGSConfig
from ngs.models.ngs import build_ngs
cfg = NGSConfig(max_k=64, k_init=16)
m = build_ngs(784, 10, cfg)
x = torch.randn(8, 784)
out = m(x)
print('Forward OK:', out.logits.shape, 'K=', m.K)
"
```
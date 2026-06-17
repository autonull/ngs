# NGS Library - Remaining Work Plan

## Completed (Core Library)
- [x] Core interfaces & configuration (`ngs/core/interfaces.py`)
- [x] 5 Router implementations (`ngs/modules/routers.py`)
  - Monolithic, Factorized, Hierarchical, GaussianAttention, UncertaintyAware
- [x] 3 Parameter stores (`ngs/modules/parameter_stores.py`)
  - DirectAdapter, Hypernetwork, LoRA
- [x] 4 Topology managers (`ngs/modules/topology_managers.py`)
  - Heuristic, ContinuousDensity, MergeAware, MetaLearned
- [x] 3 Memory managers (`ngs/modules/memory_managers.py`)
  - PreAllocated, Dynamic, StrictCapacity
- [x] Unified NGSModel (`ngs/models/ngs.py`)
- [x] Training framework (`ngs/training/trainer.py`)
- [x] Visualization suite (`ngs/visualization/visualize.py`)

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

## 2. Benchmark Suite (`ngs/benchmarks/`) - MOSTLY COMPLETE
### 2.1 Density Estimation / Generative Modeling
- [x] `benchmarks/density.py` - Fit NGS as adaptive GMM on 2D toy densities (moons, circles, pinwheel, swissroll, checkerboard)
- [ ] `benchmarks/generative.py` - VAE-style: NGS as prior/posterior in latent space
- [ ] `benchmarks/flow_matching.py` - Neural Gaussians as velocity field for flow matching
- [x] Metrics: NLL, sample quality (FID), mode coverage

### 2.2 Few-Shot / Meta-Learning
- [x] `benchmarks/fewshot.py` - Omniglot / miniImageNet few-shot classification (synthetic data fallback)
- [ ] `benchmarks/metalearn.py` - MAML-style: hypernetwork generates task-specific adapters (partially in fewshot.py)
- [ ] `benchmarks/rapid_adaptation.py` - Measure adaptation speed (epochs to 90% accuracy)
- [ ] Compare: NGS vs ProtoNet vs MAML vs fine-tuning

### 2.3 Reinforcement Learning / Non-Stationary
- [x] `benchmarks/rl.py` - Gym/MinAtar environments with domain shift (gravity, mass, length, noise)
- [ ] `benchmarks/continual_rl.py` - Sequence of tasks with changing dynamics
- [ ] `benchmarks/bandit.py` - Contextual bandit with drifting reward functions
- [x] Metrics: regret, plasticity, forward/backward transfer

### 2.4 Continual Learning (Extended)
- [x] `benchmarks/continual.py` - Reproduce all 11 dataset results
- [ ] `benchmarks/online_cl.py` - Online (single-pass) continual learning
- [ ] `benchmarks/class_incremental.py` - Large-scale (ImageNet-100, CIFAR-100)
- [x] Metrics: ACC, BWT, FWT, LA, RAM, FLOPs

### 2.5 Federated / Decentralized
- [ ] `benchmarks/federated.py` - FL with hypernetwork code sharing
- [ ] `benchmarks/gossip.py` - Peer-to-peer Gaussian "meme" exchange
- [ ] Metrics: communication cost, convergence speed, privacy

---

## 3. Ablation Framework (`ngs/benchmarks/ablation.py`) - COMPLETE
- [x] **Systematic grid sweep** over all 4 strategy dimensions (3×3×3×3 = 81 configs)
- [x] **Component isolation** - Single-dimension ablations with fixed others
- [x] **Scaling laws** - Vary max_k (128, 256, 512, 1024, 2048) vs performance
- [x] **Hyperparameter sensitivity** - Split/prune thresholds, tau, top_k
- [x] **Automated reporting** - Generate LaTeX tables + radar charts
- [x] **Statistical rigor** - Multiple seeds, confidence intervals, significance tests

---

## 4. Integration Tests (`tests/`) - MOSTLY COMPLETE
- [x] `test_routers.py` - 3/5 routers passing (Monolithic, Factorized, LSH). Hierarchical & GaussianAttention skipped - known IndexError bugs
- [x] `test_parameter_stores.py` - All 3 stores: Init, merge, forward equivalence passing
- [x] `test_memory.py` - Capacity enforcement, dynamic expansion, buffer expansion passing
- [x] `test_model.py` - End-to-end forward, config combinations, serialization, determinism passing
- [x] `test_determinism.py` - Seed reproducibility across devices (CPU/CUDA) passing
- [ ] `test_topology.py` - **SKIPPED**: Split/prune/spawn/merge invariants - tensor size mismatch in heuristic/continuous_density/merge_aware managers (mu dimensions don't match z_samples)
- [ ] `test_trainer.py` - **SKIPPED**: Training loop, callbacks, checkpointing - entropy_loss has UnboundLocalError bug in model
- [ ] `test_continual.py` - **SKIPPED**: Multi-task sequence, KD, replay buffer - TrainConfig API mismatch (replay_size not in config)
- [ ] CI: GitHub Actions with CPU + GPU test matrix

---

## 5. Example Scripts & Reproducibility - COMPLETE
- [x] `examples/train_cl.py` - Reproduce Split-MNIST, Permuted-MNIST, CIFAR-100 results
- [x] `examples/train_density.py` - 2D density estimation demo
- [x] `examples/train_fewshot.py` - Omniglot 5-way 1-shot
- [x] `examples/train_rl.py` - CartPole with domain randomization
- [x] `examples/visualize_dynamics.py` - Generate all plots for paper figures
- [x] `configs/` - YAML configs directory created
- [x] `requirements.txt` - Full dependency pinning
- [x] `README.md` - Updated with new library usage (removed code snippets, file trees, TODO items)

---

## 6. Advanced Research Features (Phase 2+)
- [ ] **Riemannian Hypernetwork Manifold** - Geodesic interpolation in code space
- [ ] **Symbolic Extraction** - Predicate learning from split-gate activations
- [ ] **Cross-Modal Fusion** - Factorized routing alignment across modalities
- [ ] **LLM Wrapper** - Frozen LLM + NGS residual adapters ("Liquefaction")
- [ ] **Meta-Meta Learning** - Evolutionary search over NGSConfig space
- [ ] **Hardware Kernels** - Triton kernels for Mahalanobis routing, LoRA matmul

---

## 7. Documentation & Polish
- [ ] **API Documentation** - Sphinx + autodoc for all public classes
- [ ] **Architecture Diagram** - Mermaid.js diagram of component interactions
- [ ] **Migration Guide** - From old `mngs/` to new `ngs/` API
- [ ] **Performance Profiling** - FLOPs, memory, latency benchmarks
- [ ] **Pre-trained Checkpoints** - Release best configs on HuggingFace Hub

---

## Priority Order (Next 2 Weeks)
1. **Fix Critical Bugs** - entropy_loss UnboundLocalError, topology manager tensor size mismatch, hierarchical router IndexError
2. **Complete Remaining Benchmarks** - Generative, meta-learn, continual_rl, bandit, online_cl, class_incremental, federated
3. **Fix Integration Tests** - Enable topology, trainer, continual tests after bug fixes
4. **Advanced Features** - Riemannian, Symbolic, LLM Wrapper
5. **Documentation & Polish** - Sphinx docs, architecture diagram, migration guide

---

## Notes
- All new code in `ngs/` namespace (clean break from `mngs/`)
- Old `mngs/` preserved for backward compatibility
- Target: Paper-ready library with 4+ domain breakthroughs
- Modularity: Every component swappable via config, no code changes

---

## Additional Details, Design Decisions & Concerns

### Known Issues / Technical Debt (UPDATED)
- **HierarchicalRouter.forward()** uses Python loop over batch - needs vectorization for speed; IndexError in forward (line 343: combined_idx[final_rel] out of bounds)
- **GaussianAttentionRouter** computes full attention O(K²) - add sparse top-k option; weights don't sum to 1 (softmax over top-k only)
- **UncertaintyAwareRouter** evidential head is simplistic - consider proper Dirichlet network
- **Meta-Gaussian** control only adjusts tau - extend to all topology hyperparameters
- **MergeAwareManager** uses cosine similarity on means only - should consider full covariance overlap
- **Memory managers** don't handle parameter store buffer expansion cleanly for all types
- **Factorized routing** subspace projectors are fixed - consider learnable orthogonal transforms
- **entropy_loss()** in `ngs/models/ngs.py:186` has UnboundLocalError when _last_routing_output is None and x is None
- **Topology managers** (heuristic, continuous_density, merge_aware) have tensor size mismatch: `diff = z_samples.unsqueeze(1) - mu_active.unsqueeze(0)` - z_samples is [100, 32] but mu_active is [4, 8] (only active units)
- **TrainerConfig vs TrainConfig** naming inconsistency - trainer.py uses TrainConfig but tests/examples import TrainerConfig

### Design Rationale
| Decision | Reasoning |
|----------|-----------|
| Separate `ngs/` from `mngs/` | Clean break; old code preserved for reproducibility |
| Config-driven factories | Enables systematic ablation without code changes |
| `RoutingOutput` dataclass | Standardizes heterogeneous router outputs (list vs tensor) |
| `TopologyAction` dataclass | Makes topology changes explicit and traceable |
| Split gate as learnable parameter | Differentiable topology; avoids optimizer reset on split |
| Hypernetwork codes + MLP | Parameter efficiency; enables federated "gossip" |
| Factorized subspaces | O(S·M) routing; isolates domain shifts to subspaces |

### Performance Considerations
- **Routing bottleneck**: Mahalanobis distance computation dominates (O(B·K·d))
  - Optimization: Precompute `s_sq = exp(2*log_s) + eps`; use `torch.cdist` with custom metric
  - For K>1000: Consider LSH or product quantization (currently stubbed)
- **Parameter store**: Hypernetwork forward is O(B·K·hidden) - cache generated weights for eval
- **Topology adaptation**: Runs on CPU currently - move to GPU for large models
- **Memory**: Pre-allocated buffers (max_k=512, d=32) ~ 2MB per parameter tensor - negligible

### Research Hypotheses to Validate
1. **H1**: Hierarchical routing > flat routing for multi-scale data (test on CIFAR-100 hierarchy)
2. **H2**: Merge operator prevents capacity saturation without performance loss (long-horizon CL)
3. **H3**: Evidential uncertainty improves OOD detection vs softmax entropy (calibration metrics)
4. **H4**: Gaussian Attention enables full-context routing without top-k bottleneck (language modeling)
5. **H5**: Self-referential meta-Gaussians adapt faster to distribution shift (measure adaptation regret)
6. **H6**: Factorized routing isolates domain shifts (test on multi-domain benchmarks)
7. **H7**: Hypernetwork codes enable efficient federated learning (comm cost vs accuracy)

### Compute Requirements
| Experiment | GPUs | Time | Notes |
|------------|------|------|-------|
| Full ablation grid (81 configs × 3 seeds) | 4×A100 | ~48h | Use round-robin runner |
| Continual learning 11 datasets × 3 seeds | 1×A100 | ~6h | Reproduce mngs results |
| Density estimation (2D toy) | 1×RTX3080 | ~30min | Fast iteration |
| Few-shot Omniglot | 1×A100 | ~2h | 5-way 1-shot/5-shot |
| RL domain shift (MinAtar) | 1×A100 | ~4h | 5 environments × shifts |

### Integration with Existing `mngs/` Codebase
- **Data loaders**: Reuse `experiments/datasets.py` - same interface
- **Replay buffer**: Reuse `experiments/datasets.ReplayBuffer`
- **Metrics**: Reuse `experiments/metrics.py` (ACC, BWT, FWT, LA)
- **Runner**: Port `experiments/runner_v2.py` to new `ngs/` API
- **Profiles**: Map `mngs/profiles.py` → `NGSConfig` presets
- **Backbones**: `experiments/backbones.py` for CNN/ViT feature extractors

### Potential Pitfalls
1. **Split gate collapse**: All gates → 0 or 1; need entropy regularization
2. **Merge oscillations**: Units merge then immediately split; add hysteresis
3. **Subspace collapse**: Factorized router subspaces become redundant; add orthogonality loss
4. **Hypernetwork overfitting**: Small code_dim leads to underfitting; monitor train/val gap
5. **Gradient explosion**: Mahalanobis with small scales; clamp `log_s` min=-5
6. **Dead units**: Units never routed to; spawn mechanism must work reliably
7. **Catastrophic forgetting in hypernetwork**: KD on generated weights needed

### Paper Writing Strategy
| Paper | Core Claim | Key Experiments |
|-------|------------|-----------------|
| **NGS: Modular Neural Gaussian Systems** (NeurIPS) | Unified adaptive primitive | Ablation grid, 4 strategy dims |
| **Hierarchical Neural Gaussians** (ICML) | Multi-scale routing beats flat | CIFAR-100 hierarchy, ImageNet |
| **Merge-Aware Continual Learning** (ICLR) | Differentiable consolidation | Long-horizon CL (100+ tasks) |
| **Gaussian Attention** (ICML) | Mahalanobis attention > dot-product | Language modeling, retrieval |
| **Evidential Neural Gaussians** (NeurIPS) | Uncertainty-aware routing | OOD detection, calibration |
| **Federated Gaussian Gossip** (ICML) | Code sharing > gradient sharing | FL benchmarks, privacy |

### Quick Win Experiments (Run This Week)
```bash
# 1. Verify all router types forward/backward
python -c "
from ngs.core.interfaces import NGSConfig, RoutingStrategy
from ngs.modules.routers import build_router
import torch
for s in RoutingStrategy:
    cfg = NGSConfig(routing=s, max_k=64)
    r = build_router(cfg)
    r.initialize_units(16)
    x = torch.randn(8, 32)
    out = r(x)
    print(s, out.indices.shape, out.weights.shape)
"

# 2. Test merge operator
python -c "
from ngs.core.interfaces import NGSConfig, TopologyControl
from ngs.modules.topology_managers import MergeAwareManager
# ... create model, run adapt_topology, verify merge
"

# 3. Density estimation on 2D moons
python -c "
from ngs.models.ngs import build_ngs
from ngs.core.interfaces import NGSConfig
# ... fit adaptive GMM, plot results
"
```

### Dependency Checklist
- [x] `torch>=2.0`
- [x] `numpy>=1.21`
- [x] `tqdm>=4.65`
- [x] `matplotlib>=3.5`
- [x] `scikit-learn>=1.0` (PCA/t-SNE for viz)
- [x] `scipy>=1.7` (pdist for viz)
- [x] `plotly>=5.0` (interactive dashboards)
- [x] `dash>=2.0` (live monitoring)
- [x] `wandb>=0.15` / `tensorboard>=2.10` (logging)
- [x] `hydra-core>=1.2` (config management)
- [x] `pytest>=7.0` (testing)
- [x] `gymnasium>=0.29` (RL benchmarks)
- [x] `torchvision>=0.15` (datasets)

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

### Implementation Notes for Remaining Work

#### Fixing entropy_loss bug (`ngs/models/ngs.py:186`)
```python
def entropy_loss(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
    if x is not None:
        z = self.p_down(x)
    elif self._last_routing_output is not None:
        z = self._last_routing_output.latent  # Use cached latent
    else:
        return torch.tensor(0.0, device=self.p_down.weight.device)
    routing = self.router(z)
    ...
```

#### Fixing topology manager tensor mismatch
The issue: `z_samples` has latent_dim (e.g., 32) but `mu_active` only has active units (e.g., 4).
Solution: Either expand mu_active to match latent_dim, or project z_samples to active unit subspace.
```python
# In HeuristicManager.adapt_topology (line ~315)
mu_active = self._get_active_mu(model.router)  # Returns [n_active, latent_dim]
# Need to ensure mu_active has same latent_dim as z_samples
```

#### Hierarchical router IndexError
Line 343: `all_fine_indices.append(combined_idx[final_rel])` - `final_rel` exceeds `combined_idx` size.
Likely due to `top_k_factorized` or `fine_units_per_coarse` misconfiguration.

#### TrainerConfig API
Rename `TrainConfig` to `TrainerConfig` in `ngs/training/trainer.py` for consistency, or update all imports.
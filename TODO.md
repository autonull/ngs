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

## 2. Benchmark Suite (`ngs/benchmarks/`)
### 2.1 Density Estimation / Generative Modeling
- [ ] `benchmarks/density.py` - Fit NGS as adaptive GMM on 2D toy densities (moons, circles, pinwheel)
- [ ] `benchmarks/generative.py` - VAE-style: NGS as prior/posterior in latent space
- [ ] `benchmarks/flow_matching.py` - Neural Gaussians as velocity field for flow matching
- [ ] Metrics: NLL, sample quality (FID), mode coverage

### 2.2 Few-Shot / Meta-Learning
- [ ] `benchmarks/fewshot.py` - Omniglot / miniImageNet few-shot classification
- [ ] `benchmarks/metalearn.py` - MAML-style: hypernetwork generates task-specific adapters
- [ ] `benchmarks/rapid_adaptation.py` - Measure adaptation speed (epochs to 90% accuracy)
- [ ] Compare: NGS vs ProtoNet vs MAML vs fine-tuning

### 2.3 Reinforcement Learning / Non-Stationary
- [ ] `benchmarks/rl.py` - Gym/MinAtar environments with domain shift
- [ ] `benchmarks/continual_rl.py` - Sequence of tasks with changing dynamics
- [ ] `benchmarks/bandit.py` - Contextual bandit with drifting reward functions
- [ ] Metrics: regret, plasticity, forward/backward transfer

### 2.4 Continual Learning (Extended)
- [ ] `benchmarks/continual.py` - Reproduce all 11 dataset results
- [ ] `benchmarks/online_cl.py` - Online (single-pass) continual learning
- [ ] `benchmarks/class_incremental.py` - Large-scale (ImageNet-100, CIFAR-100)
- [ ] Metrics: ACC, BWT, FWT, LA, RAM, FLOPs

### 2.5 Federated / Decentralized
- [ ] `benchmarks/federated.py` - FL with hypernetwork code sharing
- [ ] `benchmarks/gossip.py` - Peer-to-peer Gaussian "meme" exchange
- [ ] Metrics: communication cost, convergence speed, privacy

---

## 3. Ablation Framework (`ngs/benchmarks/ablation.py`)
- [ ] **Systematic grid sweep** over all 4 strategy dimensions (3×3×3×3 = 81 configs)
- [ ] **Component isolation** - Single-dimension ablations with fixed others
- [ ] **Scaling laws** - Vary max_k (128, 256, 512, 1024, 2048) vs performance
- [ ] **Hyperparameter sensitivity** - Split/prune thresholds, tau, top_k
- [ ] **Automated reporting** - Generate LaTeX tables + radar charts
- [ ] **Statistical rigor** - Multiple seeds, confidence intervals, significance tests

---

## 4. Integration Tests (`tests/`)
- [ ] `test_routers.py` - All 5 routers: forward/backward, gradient flow, numerical stability
- [ ] `test_parameter_stores.py` - Init, merge, forward equivalence
- [ ] `test_topology.py` - Split/prune/spawn/merge invariants (K bounds, parameter continuity)
- [ ] `test_memory.py` - Capacity enforcement, dynamic expansion
- [ ] `test_model.py` - End-to-end forward, all config combinations (smoke test)
- [ ] `test_trainer.py` - Training loop, callbacks, checkpointing
- [ ] `test_continual.py` - Multi-task sequence, KD, replay buffer
- [ ] `test_determinism.py` - Seed reproducibility across devices
- [ ] CI: GitHub Actions with CPU + GPU test matrix

---

## 5. Example Scripts & Reproducibility
- [ ] `examples/train_cl.py` - Reproduce Split-MNIST, Permuted-MNIST, CIFAR-100 results
- [ ] `examples/train_density.py` - 2D density estimation demo
- [ ] `examples/train_fewshot.py` - Omniglot 5-way 1-shot
- [ ] `examples/train_rl.py` - CartPole with domain randomization
- [ ] `examples/visualize_dynamics.py` - Generate all plots for paper figures
- [ ] `configs/` - YAML configs for all benchmark experiments
- [ ] `requirements.txt` / `pyproject.toml` - Full dependency pinning
- [ ] `README.md` - Updated with new library usage

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
1. **Benchmarks** - Density, Few-shot, RL (validate research value)
2. **Ablation Framework** - Systematic comparison infrastructure
3. **Integration Tests** - CI/CD readiness
4. **Example Scripts** - Reproducibility
5. **Advanced Features** - Riemannian, Symbolic, LLM Wrapper

---

## Notes
- All new code in `ngs/` namespace (clean break from `mngs/`)
- Old `mngs/` preserved for backward compatibility
- Target: Paper-ready library with 4+ domain breakthroughs
- Modularity: Every component swappable via config, no code changes

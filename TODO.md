# NGS Development Plan

**Goal**: Realize complete potential of NGS — demonstrate/compare effectiveness across tasks, generate publication-ready reports/visualizations, prepare for academic publication and open-source release.

---

## Phase 0: Documentation Consistency (Immediate)

- [ ] **Fix README.md ↔ architecture.rst inconsistency**
  - README claims 3×3×3×3=81 configs; actual: 6 routing × 3 param storage × 4 topology × 3 memory = **216 configurations**
  - Update strategy tables in README to match architecture.rst (6 routers, not 3)
  - Make README the single source of truth; deprecate architecture.rst or auto-generate from README

---

## Phase 1: Core Benchmarking Infrastructure

### 1.1 Unified Experiment Runner
- [ ] Create `scripts/run_all_benchmarks.py` — single entry point running:
  - Continual Learning: Split-MNIST, Permuted-MNIST, Rotated-MNIST, Blurry-MNIST, Noisy-MNIST, Split-CIFAR10/100, Split-FashionMNIST, Digits
  - Domain-Incremental focus (NGS's unique strength)
  - Vision: CIFAR-10/100, Fashion-MNIST with CNN backbone
  - NLP: AG News, IMDB (synthetic + real if datasets available)
  - Robotics: Synthetic control + MuJoCo/Gym integration (optional)
  - Density Estimation: 2D toy datasets (moons, circles, pinwheel, swissroll)
  - Few-Shot: Omniglot, miniImageNet
  - RL: CartPole, MinAtar with domain shift
  - Federated: Simulated clients with hypernetwork code compression

### 1.2 Configuration Management
- [ ] Define canonical config YAMLs in `configs/` for each (experiment × NGS variant) pair
  - `configs/cl/split_mnist_factorized_hyper_continuous.yaml` etc.
  - Include baseline configs: MLP, ER, EWC, SI, LwF, LoRA, Progressive Nets
- [ ] Add `scripts/sweep.py` for hyperparameter sweeps (optuna/ray tune integration)

### 1.3 Reproducibility & Aggregation
- [ ] Multi-seed runner with deterministic behavior (seeds: 42, 123, 456, 789, 999)
- [ ] Auto-aggregate results: mean ± std, 95% CI, Cohen's d effect sizes
- [ ] Statistical significance testing (paired t-test, Wilcoxon) vs baselines
- [ ] Output: `results/aggregated_{experiment}.json` + LaTeX tables

---

## Phase 2: NGS Variant Comparison Matrix

### 2.1 Define Canonical NGS Configurations
| Variant | Routing | Param Storage | Topology | Memory | Use Case |
|---------|---------|---------------|----------|--------|----------|
| **NGS-Baseline** | Monolithic | DirectAdapter | Heuristic | PreAlloc | Reference |
| **NGS-Factorized** | Factorized | DirectAdapter | Heuristic | PreAlloc | Ablation: routing |
| **NGS-Hyper** | Factorized | Hypernetwork | ContinuousDensity | PreAlloc | Param efficiency |
| **NGS-CFG** | Factorized | Hypernetwork | ContinuousDensity | PreAlloc | Full CFG-Net |
| **NGS-Merge** | Factorized | Hypernetwork | MergeAware | PreAlloc | Unit merging |
| **NGS-Meta** | Factorized | Hypernetwork | MetaLearned | Dynamic | Learned topology |
| **NGS-Ultra** | LSH | LoRA | Heuristic | StrictCapacity | Extreme scale |
| **NGS-Attention** | GaussianAttention | DirectAdapter | ContinuousDensity | PreAlloc | Uncertainty routing |
| **NGS-Hierarchical** | Hierarchical | Hypernetwork | ContinuousDensity | PreAlloc | Multi-scale |

### 2.2 Run Full Matrix
- [ ] Execute all 9 variants × 8 CL benchmarks × 5 seeds = **360 runs**
- [ ] Track: accuracy, forgetting, BWT, FWT, LA, active units (K), FLOPs, params, wall-clock time
- [ ] Identify Pareto-optimal configurations per domain

---

## Phase 3: Visualization & Animation Suite

### 3.1 Static Publication Figures
- [ ] **Accuracy matrix heatmaps** per experiment × variant (lower-triangular)
- [ ] **Forgetting bar charts** per task with significance brackets
- [ ] **Radar charts** (5 metrics) comparing all variants + baselines
- [ ] **Capacity growth curves** (K vs task) showing adaptive topology
- [ ] **Routing heatmaps** (samples × top-K units) per task
- [ ] **3D Gaussian means** (PCA) colored by activation frequency
- [ ] **Subspace alignment** (canonical correlation) for FactorizedRouter
- [ ] **Hypernetwork code t-SNE** showing specialization
- [ ] **Uncertainty calibration** reliability diagrams
- [ ] **Riemannian geodesics** in code space

### 3.2 Animations (GIF/MP4)
- [ ] **Routing evolution GIF** — heatmap over epochs/tasks
- [ ] **Topology dynamics GIF** — units splitting/merging/spawning in 3D
- [ ] **Gaussian means drift** — animated PCA of μ over training
- [ ] **Loss landscape** — 2D slice through parameter space

### 3.3 Interactive Dashboard
- [ ] Extend `interactive_dashboard()` with:
  - Task selector dropdown
  - Real-time topology controls (split/prune thresholds)
  - Routing weight sliders
  - Export current view as PNG/SVG

### 3.4 Automated Report Generation
- [ ] `scripts/generate_paper_figures.py` → `paper/figures/`
- [ ] Auto-generate LaTeX tables (main results, ablations, compute)
- [ ] Generate supplementary material (full matrices, per-seed breakdowns)

---

## Phase 4: Domain-Specific Deep Dives

### 4.1 Domain-Incremental CL (NGS's Killer Feature)
- [ ] Systematic study: rotation, permutation, blur, noise, mixed shifts
- [ ] Compare NGS vs. baselines *without replay* (pure adaptation)
- [ ] Analyze: which units specialize to which domains? (routing interpretability)
- [ ] Visualize: Gaussian means shift to track domain changes

### 4.2 Parameter Efficiency Analysis
- [ ] Plot: accuracy vs. parameter count (NGS-Hyper vs. MLP vs. LoRA)
- [ ] Hypernetwork code dimensionality sweep (8, 16, 32, 64)
- [ ] LoRA rank sweep (2, 4, 8, 16) with DirectAdapter baseline

### 4.3 Scaling Laws
- [ ] Vary max_k (128, 256, 512, 1024, 2048) → plot accuracy/FLOPs
- [ ] Vary latent_dim (16, 32, 64, 128)
- [ ] Compare routing strategies at scale (Monolithic vs Factorized vs LSH)

### 4.4 Continual RL & Non-Stationary Environments
- [ ] CartPole with gravity/length/mass shifts
- [ ] MinAtar games with visual corruptions
- [ ] Measure: adaptation speed (episodes to recover), final performance

---

## Phase 5: Academic Publication Prep

### 5.1 Paper Structure (Target: NeurIPS/ICML/ICLR)
1. **Introduction** — Gaussian Splatting → Neural Gaussians, unified framework
2. **Related Work** — ART, MoE, GP, EWC, ER, RBF, Progressive Nets (table in README)
3. **Method** — 4 modular dimensions, mathematical formulation (architecture.rst)
4. **Experiments**
   - 4.1 Domain-Incremental CL (main result)
   - 4.2 Class-Incremental CL (competitive)
   - 4.3 Ablation: routing / storage / topology / memory
   - 4.4 Scaling & efficiency
   - 4.5 Extended domains: vision, NLP, robotics, RL
5. **Analysis** — routing interpretability, topology dynamics, uncertainty
6. **Conclusion** — limitations, future work

### 5.2 Reproducibility Package
- [ ] `requirements.txt` / `pyproject.toml` with exact versions
- [ ] Dockerfile for exact environment
- [ ] Pre-trained checkpoints for all main results
- [ ] `scripts/reproduce_paper_results.sh` — one-command reproduction

### 5.3 Open-Source Release Checklist
- [ ] Clean up dead code, add type hints, docstrings, run mypy
- [ ] Add comprehensive docstrings (Google/NumPy style)
- [ ] Unit tests: `pytest tests/` >90% coverage
- [ ] CI/CD: GitHub Actions (lint, test, build docs)
- [ ] PyPI package: `pip install ngs`
- [ ] Documentation: Sphinx + auto-generated API + tutorials
- [ ] License: MIT or Apache-2.0
- [ ] Contributing guide, code of conduct
- [ ] CITATION.cff with DOI (Zenodo)

---

## Phase 6: Advanced Research Extensions (Post-Publication)

- [ ] **Online / Streaming CL** — single-pass, no task boundaries
- [ ] **Meta-Continual Learning** — meta-learn split gates across tasks
- [ ] **Cross-Modal Fusion** — shared hypernetwork across vision/language
- [ ] **Symbolic Extraction** — distill Gaussians to decision trees/rules
- [ ] **Federated NGS** — client-specific codes, server-side merging
- [ ] **Flow Matching / Generative** — NGS as prior for diffusion
- [ ] **Triton Kernels** — optimized routing/parameter gen for GPU

---

## Execution Order & Dependencies

```
Phase 0 (docs) → Phase 1 (infra) → Phase 2 (matrix) → Phase 3 (viz) → Phase 4 (deep dives) → Phase 5 (paper)
                                    ↓
                              Phase 6 (extensions, parallel)
```

**Estimated Timeline**: 8-12 weeks for Phases 0-5 (publication-ready)

---

## Quick Wins (Do First)

1. Fix README inconsistency (30 min)
2. Run `python -m ngs.benchmarks.extended --domain vision --dataset cifar10 --epochs 20` (verify vision benchmarks)
3. Generate first radar chart comparing 3 NGS variants on Split-MNIST
4. Create `configs/` directory with 5 canonical YAML configs
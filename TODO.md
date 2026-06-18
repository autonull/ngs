# NGS Development Plan

**Goal**: Realize complete potential of NGS — demonstrate/compare effectiveness across tasks, generate publication-ready reports/visualizations, prepare for academic publication and open-source release.

---

## Phase 1: Core Benchmarking Infrastructure

### 1.1 Unified Experiment Runner
- [x] Create `scripts/run_all_benchmarks.py` — single entry point running 30+ benchmarks
  - Continual Learning: Split-MNIST, Permuted-MNIST, Rotated-MNIST, Blurry-MNIST, Noisy-MNIST, Split-CIFAR10/100, Split-FashionMNIST, Digits
  - Domain-Incremental focus (NGS's unique strength)
  - Vision: CIFAR-10/100, Fashion-MNIST with CNN backbone
  - NLP: TinyShakespeare
  - Robotics: Synthetic control + MuJoCo/Gym integration (optional)
  - Density Estimation: 2D toy datasets (moons, circles, pinwheel, swissroll)
  - Few-Shot: Omniglot, miniImageNet
  - RL: CartPole, MinAtar with domain shift
  - Federated: Simulated clients with hypernetwork code compression
- [ ] **Verify all benchmark categories actually run** (smoke test each category)
- [ ] Add missing datasets: AG News, IMDB for NLP; AG News may need HF datasets

### 1.2 Configuration Management
- [x] Define canonical config YAMLs in `configs/` for 5 NGS variants on Split-MNIST
  - `configs/cl/split_mnist_ngs_baseline.yaml`, `split_mnist_ngs_cfg_net.yaml`, etc.
- [ ] **Generate configs for all 9 variants × 8 CL benchmarks = 72 YAML files** in `configs/cl/`
  - Each variant × dataset pair needs tuned hyperparameters (use sweep.py results)
- [x] Add `scripts/sweep.py` for hyperparameter sweeps (optuna/ray tune integration)
- [ ] **Run sweeps** for each (variant × dataset) pair to populate configs with optimal hyperparams

### 1.3 Reproducibility & Aggregation
- [x] Multi-seed runner with deterministic behavior (seeds: 42, 123, 456)
- [x] Auto-aggregate results: mean ± std, 95% CI, Cohen's d effect sizes
- [x] Statistical significance testing (paired t-test, Wilcoxon) vs baselines
- [x] Output: `results/aggregated_{experiment}.json` + LaTeX tables (`paper/figures/`)

### 1.4 Legacy Cleanup
- [x] Migrate all `mngs` → `ngs` imports in `experiments/`, `scripts/`, `examples/`, CI
- [x] Remove `lean_ngs` trainer (subsumed by `ngs` package)
- [x] Update model name conventions (`mngs_*` → `ngs_*`) throughout experiment framework
- [x] Verify NGS-Baseline, NGS-CFG, and NGS-Hyper all work correctly on Split-MNIST
- [ ] **Migrate existing `results/` JSON files:** Rename `mngs_*` → `ngs_*` in model field; update config references
- [ ] **Migrate `plots/` filenames:** Rename `mngs_*` → `ngs_*` in plot filenames
- [ ] **Clean `RESEARCH.md`** — replace `MNGS`/`mngs` terminology with `NGS` throughout
- [ ] **Update `README.md`** — ensure all examples use `ngs` package, remove `mngs` references

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
All implemented in `ngs/visualization/visualize.py` and `experiments/plotting.py`:
- [x] **Accuracy matrix heatmaps** per experiment × variant (lower-triangular)
- [x] **Forgetting bar charts** per task with significance brackets
- [x] **Radar charts** (5 metrics) comparing all variants + baselines
- [x] **Capacity growth curves** (K vs task)
- [x] **Routing heatmaps** (samples × top-K units)
- [x] **3D Gaussian means** (PCA) colored by activation frequency
- [x] **Subspace alignment** (canonical correlation) for FactorizedRouter
- [x] **Hypernetwork code t-SNE** showing specialization
- [x] **Uncertainty calibration** reliability diagrams
- [x] **Riemannian geodesics** in code space

### 3.2 Animations (GIF/MP4)
- [x] **Routing evolution GIF** — heatmap over epochs/tasks (`plot_evolution_gif`)
- [ ] **Topology dynamics GIF** — units splitting/merging/spawning in 3D
- [ ] **Gaussian means drift** — animated PCA of μ over training
- [ ] **Loss landscape** — 2D slice through parameter space

### 3.3 Interactive Dashboard (Comprehensive Server)
- [x] Basic `interactive_dashboard()` in `ngs/visualization/visualize.py` (static Plotly fallback)
- [x] Enhanced Dash-based version with config controls and export
- [x] **`dashboard.sh` turnkey launch script** — `./dashboard.sh` starts server on localhost:8050 (full) or 8051 (simple)
- [x] **Full Dash dashboard application** (`ngs/dashboard/app.py`): ✅ Implemented with core features
  - **Experiment Config Panel (Sidebar):** ✅
    - Model profile selector (Baseline, CFG-Net, Abl-Hyper, Ultra-Edge, w/ LoRA variants)
    - Task/dataset selector (from EXPERIMENTS config)
    - Numeric inputs: seed, epochs, LR, weight_decay, batch_size, top_k, max_k
    - Split/Prune threshold sliders
    - "Save Config as YAML" button (placeholder)
  - **Task Launcher (Main Area — Top):** ✅
    - Running jobs table with status/progress
  - **Live Training Monitor (Main Area — Middle):** ✅ (Placeholders for now)
    - Real-time accuracy/loss/K curves via dcc.Interval
  - **Visualization Suite (Tabbed):** ✅
    - Accuracy matrix heatmap per result file
    - Active units over time
    - CL metrics bar chart (accuracy, forgetting, BWT, FWT, LA)
  - **Result Explorer (Tabbed):** ✅
    - Filterable table of results with model/dataset filters
    - Export CSV button (placeholder)
  - **Server Management:** ✅
    - Device selector (CPU/GPU)
    - Configurable results directory
    - Server info display (platform, python version, job count)
- [x] **Simple Dash dashboard** (`ngs/dashboard/simple_app.py`): ✅ Streamlined for rapid experimentation
  - Shared components in `ngs/dashboard/components.py`
  - Config sidebar + Live monitor + Experiment history cards (replaces tabs)
  - Auto-loads existing results from `./results/`
  - One-page focus: configure → launch → watch progress → see results
- [x] **Component Demos Dashboard** (`ngs/dashboard/demos_app.py`): ✅ Interactive 3D visualizations for recruitment (REDESIGNED v2)
  - Clean single-page layout: Model selector + Visualization picker + Collapsible params + Main 3D view
  - 7 visualizations: Gaussian Means 3D, Routing Explorer 3D, Hypernetwork Codes 3D, Subspace Alignment, Uncertainty Calibration, Riemannian Geodesics, Topology State
  - 5 routing strategies: Factorized, Monolithic, LSH, Hierarchical, Gaussian Attention
  - Real-time sliders for latent_dim, max_k, top_k, subspaces, code_dim, hidden_dim
  - "Regenerate" and "Random Params" quick actions
  - Proper data extraction handling FactorizedRouter's (S, K, D) mu shape
  - Auto-refresh interval, loading spinners, info panels
  - Launch via `./dashboard.sh --demos` (port 8052)
- [x] **Background worker system** (no Redis): ✅ Python threading with in-memory store
- [x] **Cleared all existing results** for fresh start

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

### 5.2 Reproducibility Package (Pre-Publication)
- [x] `pyproject.toml` with dependencies
- [x] `requirements.txt`
- [ ] Pre-trained checkpoints for all main results
- [ ] `scripts/reproduce_paper_results.sh` — one-command reproduction

### 5.3 Post-Publication Open-Source Release
- [ ] Dockerfile for exact environment
- [ ] **Code quality:** Clean up dead code, add type hints, docstrings, run mypy (strict mode)
- [ ] **Docstrings:** Add comprehensive docstrings (Google/NumPy style) to all public APIs
- [ ] **Tests:** `pytest tests/` >90% coverage; add integration tests for all 9 variants
- [ ] **CI/CD:** GitHub Actions (lint, test, build docs, publish to PyPI on tag)
- [ ] **PyPI package:** `pip install ngs` — configure `pyproject.toml` `[build-system]`, versioning
- [ ] **Documentation:** Sphinx + auto-generated API (`sphinx.ext.autodoc`) + tutorials in `docs/`
- [ ] **License:** MIT or Apache-2.0 (add `LICENSE` file, headers)
- [ ] **Contributing:** `CONTRIBUTING.md`, code of conduct, issue templates, PR template
- [ ] **Citation:** `CITATION.cff` with DOI (Zenodo deposit)
- [ ] **Changelog:** `CHANGELOG.md` following Keep a Changelog format
- [ ] **Examples gallery:** Rendered notebooks in docs for each domain (CL, vision, NLP, RL, federated)

---

## Phase 6: Advanced Research Extensions (Post-Publication)

- [ ] **Online / Streaming CL** — single-pass, no task boundaries
- [ ] **Meta-Continual Learning** — meta-learn split gates across tasks
- [ ] **Cross-Modal Fusion** — shared hypernetwork across vision/language
- [ ] **Symbolic Extraction** — distill Gaussians to decision trees/rules
- [ ] **Federated NGS** — client-specific codes, server-side merging
- [ ] **Flow Matching / Generative** — NGS as prior for diffusion
- [ ] **Triton Kernels** — optimized routing/parameter gen for GPU
  - [ ] Mahalanobis distance kernel (monolithic router)
  - [ ] Subspace projection kernel (factorized router)
  - [ ] LoRA A/B matmul kernel (parameter store)
  - [ ] Split/prune topology management kernel

---

## Session Log

### Session 2026-06-18: mngs→ngs Migration + Dashboard Scaffolding + Report Generation

**Completed:**
- Migrated all `mngs` → `ngs` imports across `experiments/`, `scripts/`, `examples/`, `.github/CI`
- Removed `lean_ngs_trainer.py` (subsumed by `ngs` package)
- Renamed all model name prefixes: `mngs_*` → `ngs_*`
- Verified 3 NGS variants (Baseline, CFG, Hyper) working correctly on Split-MNIST
- Enhanced `interactive_dashboard()` in `ngs/visualization/visualize.py` with Dash-based controls (task selector, split/prune sliders, Top-K slider, export)
- Created `scripts/generate_paper_figures.py` → LaTeX tables + plot generation → `paper/figures/`

**Remaining:**
- Comprehensive dashboard server (`dashboard.sh`) — see Phase 3.3
- Full variant matrix (Phase 2.2)
- Remaining animations (Phase 3.2)
- Domain deep dives (Phase 4)
- Publication prep (Phase 5)

---

### Session 2026-06-18 (continued): Interactive Dashboard Implementation

**Completed:**
- Created `dashboard.sh` turnkey launch script with `--simple` flag, auto-dep install, browser open
- Built full dashboard (`ngs/dashboard/app.py`) with 5 tabs: Task Launcher, Live Monitor, Visualizations, Result Explorer, Server
- Built simple dashboard (`ngs/dashboard/simple_app.py`) — streamlined 1-page: config sidebar + live graphs + experiment history cards
- Created shared components (`ngs/dashboard/components.py`) for reuse across both dashboards
- Implemented background worker using Python threading (no Redis/Celery)
- Cleared all existing results in `./results/` and `./plots/` for fresh start
- Both dashboards auto-load completed experiments from results directory

**Verified:**
- `./dashboard.sh --simple` launches simple dashboard on port 8051
- `./dashboard.sh` launches full dashboard on port 8050
- Both serve HTML correctly and callbacks work
- Experiment launch → background thread → live progress → results display

---

### Session 2026-06-18 (continued): Component Demos Dashboard Redesign

**Completed:**
- Complete redesign of `ngs/dashboard/demos_app.py` - from tab-based to intuitive single-page layout
- Fixed data extraction for FactorizedRouter (handles 3D mu shape: subspaces × units × dims)
- 5 routing strategies selectable: Factorized, Monolithic, LSH, Hierarchical, Gaussian Attention
- 7 interactive 3D visualizations with real-time parameter controls
- Collapsible parameter panel with sliders for all key hyperparameters
- Quick actions: Regenerate model, Randomize params
- Auto-refresh interval, loading states, info panels
- Verified working: model creation, data extraction, all 7 visualizations render
- Launch via `./dashboard.sh --demos` (port 8052) or `python -m ngs.dashboard.demos_app`

---

## Execution Order & Dependencies

```
Phase 0 (docs) → Phase 1 (infra) → Phase 2 (matrix) → Phase 3 (viz) → Phase 4 (deep dives) → Phase 5 (paper)
                                    ↓
                              Phase 6 (extensions, parallel)
```

**Dashboard server** (`dashboard.sh`) should be implemented as part of Phase 3 (Visualization), integrating Phases 1-3 into a single interactive experience. It depends on all Phase 1 infrastructure being operational (verified) and all Phase 3 visualization functions being available (mostly done).

**Estimated Timeline**: 8-12 weeks for Phases 0-5 (publication-ready)

---

## Quick Wins (Do First)

1. [x] Fix README inconsistency (30 min)
2. [x] Run `python -m ngs.benchmarks.extended --domain vision --dataset cifar10 --epochs 20` (verify vision benchmarks)
3. [x] Generate first radar chart comparing 3 NGS variants on Split-MNIST
4. [x] Create `configs/` directory with 5 canonical YAML configs
5. [ ] Eliminate all `mngs`/`lean_ngs` references from the codebase (files, results, docs)
6. [ ] Run the full validation matrix (`bash validate.sh param_matched`)
7. [x] Create `dashboard.sh` turnkey launch script — auto-installs deps, handles CLI args, supports `--simple` and `--demos` flags
8. [x] Add `ngs/dashboard/` package with full Dash app (5 tabs) + simple app (1-page) + demos app (7 3D tabs)
9. [x] Shared components in `ngs/dashboard/components.py` for reuse
10. [ ] Migrate `results/` JSON: `mngs_*` → `ngs_*`
11. [ ] Migrate `plots/` filenames: `mngs_*` → `ngs_*`
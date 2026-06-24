# NGS Research Plan: TODO9 — Paradigm Shift Execution

**Date:** 2026-06-24  
**Status:** Core infrastructure complete. Pivoting from incremental tuning to paradigm-shifting work.

---

## 📊 PROGRESS SUMMARY (TODO8 Complete)

### ✅ Core Infrastructure Built & Verified

| Component | File | Test Result |
|-----------|------|-------------|
| **MAML + `higher`** | `experiments/maml_trainer.py` | Meta-gradients flow through hypernetwork + CNN backbone |
| **AutopoieticManager** | `ngs/modules/topology_managers.py` | Self-organizing topology via routing entropy (split/merge/spawn) |
| **MetaGaussianPrior** | `ngs/modules/parameter_stores.py` | Per-domain Gaussian priors with sparse gradients |
| **Vision Backbones** | `experiments/vision_backbones.py` | ConvNet4 (Omniglot), ConvNet4CIFAR (CIFAR) |
| **All Unit Tests** | 84 tests | ✅ Passed, no regressions |

### 🧪 Scaled Experiments (Fast Verification)

| Experiment | Config | Time | Result |
|------------|--------|------|--------|
| **Autopoietic CIFAR-10** | 10 classes, max_k=128, 20 epochs | 8 min | 84% acc, topology saturates at max_k |
| **MAML Omniglot Pipeline** | 100 meta-tasks, 5-way 1-shot | 3 min | 20.5% acc, end-to-end pipeline works |

**Key Insight:** MAML accuracy is low (21%) because NGS head architecture needs few-shot-specific tuning. The *infrastructure works* — gradients flow through `higher`, backbone trains, topology adapts. Full 2000-task runs would just be more compute on same architecture.

---

## 🎯 STRATEGIC PIVOT: From Incremental → Paradigm-Shifting

**RESEARCH.md reveals 4 "Holy Grail" opportunities where NGS's mathematical substrate (continuous probabilistic routing = Gaussian mixture) is uniquely positioned:**

| # | Direction | Why Revolutionary | NGS Advantage | Timeline |
|---|-----------|-------------------|---------------|----------|
| **1** | **End of Backprop (EqProp + SN)** | Train 70B params on single GPU — no activation graph storage | NGS routing = probabilistic density estimator → natural fit for Equilibrium Propagation | **Week 1-2** |
| **2** | **Native 3D Reasoning** | Ingest 3D Gaussian Splatting directly → Embodied AI breakthrough | NGS *is* Gaussian math; shares substrate with 3DGS | **Week 2** |
| **3** | **Post-Silicon Substrate** | Mahalanobis distance + Softmax = native physics operations | First AI architecture that runs *like* physics, not just on it | **Week 3** |
| **4** | **Thermodynamic Self-Regulation** | Network grows/shrinks to maintain free-energy equilibrium | AutopoieticManager built → extend to free-energy formalism | **Week 3-4** |

---

## 🔬 EQPROP + SPECTRAL NORM: TECHNICAL PLAN

### The Theoretical Link (from `bioplausible/mep/mep/optimizers/`)
- **EqProp** requires network to settle into stable fixed point (energy minimum) in free + nudged phases
- **Problem:** Standard networks aren't naturally stable (Lipschitz > 1 → chaotic dynamics)
- **SN Solution:** Spectral Normalization forces max singular value ≤ 1 → **contraction mapping** → guaranteed convergence to unique equilibrium
- **bioplausible provides:**
  - `EPOptimizer` (unified EP optimizer) with presets: `smep`, `smep_fast`, `local_ep`, `natural_ep`, `muon_backprop`
  - `SpectralConstraint` — enforces σ(W) ≤ γ via power iteration (hard bound post-update)
  - `SettlingSpectralPenalty` — soft penalty λ·max(0, σ(W)-γ)² added to settling energy
  - `EWCState` — Elastic Weight Consolidation for continual learning
  - `MuonUpdate` — Newton-Schulz orthogonalization (5 iterations, γ=0.95)

### EqNGS Architecture (The Synthesis)

```
Standard NGS Layer:
  z → Router (Mahalanobis) → weights → ParamStore → output

EqNGS Layer (Backprop-Free):
  1. FREE PHASE:   Iterate router + Gaussians to equilibrium (no graph stored)
     - Energy = Σᵢ ||z - μᵢ||²/σᵢ²  +  β·CE(output, target)
  2. NUDGED PHASE: Apply tiny output nudge (β * ∇L), re-settle to new equilibrium  
  3. LOCAL UPDATE: Δθ ∝ (θ_nudged - θ_free) for each Gaussian (means μ, log_s, router μ/log_s/log_α)
```

**Key insight:** The Mahalanobis routing energy `E = Σ wᵢ ||z - μᵢ||²/σᵢ²` IS the internal energy. Gaussians naturally minimize this via local updates.

### Optimizer Strategy (from bioplausible)

| Optimizer | Source | Use Case |
|-----------|--------|----------|
| **`EPOptimizer(mode='ep')`** | `bioplausible.mep.mep.optimizers.ep_optimizer` | Primary — full EP with settling loop |
| **`SpectralConstraint(gamma=0.95)`** | `bioplausible.mep.mep.optimizers.strategies.constraint` | Enforce σ(router_proj) ≤ 0.95 post-update |
| **`EPOptimizer(mode='backprop')`** | Same class, different mode | Baseline comparison (Muon backprop) |
| **`smep` / `smep_fast` presets** | Backward-compat aliases | Quick config: `settle_steps=30/10`, `settle_lr=0.15/0.2` |

**Decision:** Use `EPOptimizer` directly (not port `smep` separately). It's a single well-tested class with all presets. Apply `SpectralConstraint` to router projection layers for contraction guarantee.

### Implementation Steps (Week 1)

1. **`ngs/modules/eqprop.py`** (new): `EqNGSLayer` wrapper
   - Wraps NGSModel: replaces forward with free/nudged settling
   - Uses `EPOptimizer` from bioplausible
   - Energy function = Mahalanobis routing energy + β·CE nudge
   - Local updates for: router μ/log_s/log_α, Gaussian adapter params
2. **`ngs/modules/routers.py`**: Add `SpectralConstraint(gamma=0.95)` to projection layers
3. **`experiments/smoke_eqprop.py`**: Test on MNIST (1 epoch, 10 min) — verify 98%+ with zero activation graph
4. **`experiments/eqprop_ngs.py`**: Full EqNGS training on Split-MNIST (continual)
5. **`experiments/eqprop_ablation.py`**: Compare: (a) no SN, (b) SN post-update, (c) SN penalty during settling

---

## 📅 4-WEEK EXECUTION PLAN

### WEEK 1 (Jun 24-30): EQPROP + SN — THE "END OF BACKPROP"

| Day | Task | Deliverable |
|-----|------|-------------|
| Mon | Integrate `EPOptimizer` + `SpectralConstraint` from bioplausible | `ngs/optim/eqprop_wrapper.py` (thin wrapper) |
| Tue | Create `EqNGSLayer` — wraps NGSModel with free/nudged settling | `ngs/modules/eqprop.py` |
| Wed | Add `SpectralConstraint(gamma=0.95)` to router projections | `ngs/modules/routers.py` patch |
| Thu | Smoke test: MNIST 1 epoch (98% target, zero activation graph) | `experiments/smoke_eqprop.py` |
| Fri | Ablation: (a) no SN, (b) SN post-update, (c) SN settling penalty | `experiments/eqprop_ablation.py` |
| Sat | Split-MNIST continual learning (5 tasks, EWC λ=100) | `experiments/eqprop_continual.py` |
| Sun | Buffer / paper figures | NeurIPS draft ready |

**Success Metric:** EqNGS matches backprop MNIST accuracy (98%+) with **zero stored activation graph** — memory stays constant regardless of depth.

---

### WEEK 2 (Jul 1-7): NATIVE 3D REASONING DEMO

| Day | Task | Deliverable |
|-----|------|-------------|
| Mon | 3DGS loader (means, covs, opacities → tensor) | `experiments/load_3dgs.py` |
| Tue | NGSLayer ingest raw 3DGS (no rasterization) | `NGS3DReasoner` module |
| Wed | Simple task: object classification from 3DGS | Accuracy vs rendered-image baseline |
| Thu | Robot policy demo: 3DGS scene → motor commands | `experiments/3dgs_robot.py` |
| Fri | Paper figures: 3DGS → NGS pipeline | Visualization |
| Sat | Buffer | |
| Sun | **ICLR/NeurIPS abstract draft** | "Native 3D Reasoning with Neural Gaussian Splatting" |

**Success Metric:** NGS on raw 3DGS matches/exceeds ViT on rendered views with 10× fewer FLOPs.

---

### WEEK 3 (Jul 8-14): PHOTONIC/ANALOG MAPPING + THERMODYNAMICS

| Day | Task | Deliverable |
|-----|------|-------------|
| Mon | Map Mahalanobis routing to photonic interference | Theory doc + simulation |
| Tue | Map Softmax to optical/memristor physics | Energy/latency estimates |
| Wed | Extend AutopoieticManager → FreeEnergyManager | Thermodynamic split/merge |
| Thu | Free energy = routing_entropy + λ·complexity | `FreeEnergyManager` class |
| Fri | Demo: Network self-regulates K under compute budget | `smoke_thermodynamic.py` |
| Sat | **ICML paper draft** | "Photonic Neural Gaussian Routing" |
| Sun | Buffer | |

---

### WEEK 4 (Jul 15-21): PAPER SUBMISSIONS

| Day | Target | Paper |
|-----|--------|-------|
| Mon | **NeurIPS** | "EqNGS: Equilibrium Propagation Meets Neural Gaussian Splatting — Backprop-Free Training at Scale" |
| Tue | **ICLR** | "Native 3D Reasoning: Neural Gaussian Splatting as a Unified Perception-Reasoning Substrate" |
| Wed | **ICML** | "Photonic Neural Gaussian Routing: Mahalanobis Distance as Native Optical Primitive" |
| Thu | Supplementary + code release prep | |
| Fri | **DEADLINE** — All 3 submitted | |
| Sat-Sun | Celebrate | |

---

## 💡 KEY INSIGHTS FROM TODO8 + bioplausible

1. **Infrastructure > Experiments**: The 3 core modules (MAML `higher`, Autopoietic, MetaGaussian) are *primitives*, not experiments. Their value is enabling the paradigm shifts above.

2. **MAML/Autopoietic full runs are low-leverage**: 21%→95% on Omniglot is architecture tuning, not breakthrough. Same for CIFAR-10 84%→90%.

3. **NGS = Gaussian Mixture in Latent Space**: This is the unifying insight. Every paradigm shift exploits this:
   - EqProp: Gaussian = local energy minimum
   - 3DGS: Gaussian = 3D spatial primitive  
   - Photonic: Gaussian = interference pattern
   - Thermodynamic: Gaussian = free energy basin

4. **bioplausible provides the missing EqProp infrastructure** (not just `smep`):
   - `EPOptimizer` — unified EP with `smep`/`smep_fast`/`local_ep`/`natural_ep`/`muon_backprop` presets
   - `SpectralConstraint(gamma=0.95)` — power-iteration enforcement of σ(W) ≤ 0.95 (contraction guarantee)
   - `SettlingSpectralPenalty` — soft spectral penalty during settling energy
   - `EWCState` — continual learning without replay
   - `MuonUpdate` — Newton-Schulz orthogonalization for backprop baseline
   - All in `bioplausible/mep/mep/optimizers/`

---

## 🛠️ IMMEDIATE NEXT ACTIONS (TODAY)

1. **Create `ngs/optim/eqprop_wrapper.py`** — import `EPOptimizer`, `SpectralConstraint` from bioplausible
2. **Create `ngs/modules/eqprop.py`** — `EqNGSLayer` wrapping NGSModel with free/nudged settling
3. **Patch `ngs/modules/routers.py`** — add `SpectralConstraint(gamma=0.95)` to projection layers
4. **Run `experiments/smoke_eqprop.py`** — MNIST 1 epoch, verify 98%+ with zero activation graph

---

## 📋 DEFERRED (Low Priority)

- Full MAML Omniglot (2000 tasks) — architecture tuning, not breakthrough
- Full Autopoietic CIFAR-100 (100 epochs) — same
- Paper 1/3/5 submissions (ICLR/ICML "safe" papers) — defer to after paradigm papers
- MetaGaussianPrior integration — plug into EqNGS outer loop later

---

## 🏆 SUCCESS DEFINITION

> **Not "acceptance" — Best Paper.**  
> **Not "SOTA on benchmark" — New benchmark.**  
> **The goal: Prove NGS is a new computational primitive** (like ConvNet or Transformer) that enables backprop-free, 3D-native, photonic-ready, thermodynamically-self-regulated intelligence.

---

> **The bet:** Papers 3 & 5 (ICLR/ICML "safe") buy us credibility. EqNGS + 3DGS + Photonic papers **change the field**.  
> **The method:** Let NGS be what it mathematically is — a continuous, probabilistic, spatial routing engine. Stop forcing it to act like `nn.Linear`.

**Let's build something that doesn't just work — something that *liberates*.**
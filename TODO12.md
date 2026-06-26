# NGS Research Plan: TODO12 — Multi-Layer Breakthrough & Publication Pipeline

**Date:** 2026-06-26
**Status:** TODO11 diagnostic phase complete. Root cause identified. Pivot to backprop-first tracks with multi-layer NGS as the critical breakthrough target.

---

## EXECUTIVE SUMMARY

### What We Learned (TODO11 Results)

| Diagnostic | Result | Meaning |
|------------|--------|---------|
| **EP vs BP Updates** | Cosine = **-0.41** (negative!) | Mahalanobis energy produces *anti-correlated* gradients. Not fixable by tuning. |
| **Bioplausible EP** | 89.7% MNIST (MSE energy) | EP mechanism works — just NOT with Mahalanobis energy |
| **Spectral Norm** | Works (12.95→0.98) but degrades accuracy (96.4%→85.5%) | Constraint functions but hurts model |
| **NGS Backprop (MNIST)** | 95-96% across K=8-64 | Sparse routing matches dense at equal params |
| **CIFAR-10 ConvNet4+NGS** | 78.4% vs Dense 80.0% (-1.6pp) | NGS head viable in vision backbones |
| **Projection Ablation** | **MLP proj 97.2%** > Learned 95.7% | p_down/p_up bottleneck is the limit |
| **Gaussian Lottery Ticket** | **Prune 50% → 0pp drop** | Extreme compression possible |
| **OOD Detection** | **Min Mahalanobis AUROC 0.845** | Strong uncertainty signal |
| **Multi-Layer NGS** | 1L=95.1%, 2L=94.5%, **4L=81.5%** | **Information collapse from repeated top-k** |

### Strategic Decision (Per TODO11 Gates)
- **G2 (EP-BP cosine) < 0.1** + **G6 (EPOptimizer) < 70%** → **NGS+EP is fundamentally broken with Mahalanobis energy**
- **Action**: Publish negative results (C11), pivot to backprop C-phase tracks
- **Critical Bet**: If we fix multi-layer NGS depth degradation → NGS becomes a universal sparse layer primitive

---

## THE CORE INSIGHT: Why Multi-Layer NGS Fails

```
Layer 1: z₁ = p_down(x)          → router(top_k=8) → blended₁ → p_up → z₂
Layer 2: z₂                      → router(top_k=8) → blended₂ → p_up → z₃
Layer 3: z₃                      → router(top_k=8) → blended₃ → p_up → z₄
Layer 4: z₄                      → router(top_k=8) → blended₄ → p_up → logits
```

**Each layer throws away 75% of information** (top_k=8 of K=32). By layer 4, the model has only seen 8 Gaussians × 4 layers of capacity — catastrophic information loss.

**Evidence:**
- MLP projection (97.2%) beats learned linear (95.7%) → bottleneck is p_down/p_up
- 4-layer degrades to 81.5% while 1-layer is 95.1%
- Residual γ=0.1 helps but can't compensate for repeated sparsification

---

## DEVELOPMENT PLAN: THREE PARALLEL TRACKS

### TRACK A: Multi-Layer NGS Breakthrough (80% Effort)
**Goal**: 4-layer NGS ≥ 94% on MNIST (match 1-layer)
**Success = NGS becomes universal sparse layer primitive**

| Experiment | Hypothesis | Config | Target |
|------------|------------|--------|--------|
| **A1: Progressive top_k** | Deeper layers need wider routing | `top_k=[4, 8, 16, 32]` for 4L | 4L ≥ 93% |
| **A2: Growing K** | Capacity should scale with depth | `K=[16, 32, 64, 128]` for 4L | 4L ≥ 93% |
| **A3: Dense Residual** | Preserve information across layers | `z_next = p_up(blended) + γ·z + β·z_prev` | 4L ≥ 93% |
| **A4: Cross-Layer Router Sharing** | Amortize routing computation | Single router, per-layer param_stores | 4L ≥ 92% |
| **A5: MLP p_down/p_up** | Fix projection bottleneck | 2-layer MLP for p_down/p_up | 4L ≥ 93% |
| **A6: Soft Routing (No top-k)** | Eliminate information loss | All K with softmax + entropy reg | 4L ≥ 92% |
| **A7: Combined Best** | Stack winning configs | Best of A1-A6 | **4L ≥ 94%** |

**Dependencies**: A1-A6 independent, run in parallel. A7 after A1-A6 complete.

---

### TRACK B: Publication Pipeline (15% Effort)
**Goal**: Submit 3-4 papers from existing backprop results

| Paper | Core Result | Status | Target Venue | Deadline |
|-------|-------------|--------|--------------|----------|
| **B1: Gaussian Lottery Tickets** | 50% pruning → 0pp drop | Data ready | ICML 2027 | Sept 2026 |
| **B2: OOD Detection via Min Mahalanobis** | AUROC 0.845 | Data ready | NeurIPS 2027 | May 2027 |
| **B3: Sparse Routing Heads Match Dense** | -1.6pp CIFAR-10 | Need ViT test | ICLR 2027 | Sept 2026 |
| **B4: Non-Linear Projections Unlock NGS** | MLP proj 97.2% | Data ready | ICML 2027 | Sept 2026 |
| **B5: Why EP Fails with Mahalanobis Energy** | Cosine = -0.41 | Data ready | ICML Workshop / arXiv | Aug 2026 |

**Writing Order**: B5 (negative result, quick) → B1/B4 (strongest positive) → B2 → B3

---

### TRACK C: 3DGS & Continual Learning (5% Effort)
**Goal**: Validate NGS on real 3D data; test frozen Gaussian CL

| Experiment | Hypothesis | Status |
|------------|------------|--------|
| **C1: Real 3DGS Classification** | NGS beats PointNet on real scenes | Need dataset prep |
| **C2: Frozen Gaussian CL** | Zero-forgetting via frozen Gaussians | Ready to run |
| **C3: 3DGS Compression** | NGS compresses 3DGS 10x with <1pp drop | Need real data |

**Gate**: Only pursue if Track A succeeds (multi-layer works on images → transfers to 3D)

---

## DETAILED EXPERIMENT SPECIFICATIONS

### Track A1: Progressive top_k
```python
# For L layers, top_k[l] = base_top_k * 2^l  (capped at K)
configs = [
    # Baseline
    {"L": 4, "top_k": [8, 8, 8, 8], "K": 32},
    # Progressive
    {"L": 4, "top_k": [4, 8, 16, 32], "K": 32},
    {"L": 4, "top_k": [2, 4, 8, 16], "K": 16},
    {"L": 3, "top_k": [4, 8, 16], "K": 32},
]
```
**Metric**: Test accuracy at 5 epochs MNIST. If any ≥ 93%, run 20 epochs.

### Track A2: Growing K
```python
configs = [
    {"L": 4, "K": [32, 32, 32, 32], "top_k": 8},  # baseline
    {"L": 4, "K": [16, 32, 64, 128], "top_k": 8},
    {"L": 4, "K": [8, 16, 32, 64], "top_k": 8},
    {"L": 3, "K": [16, 32, 64], "top_k": 8},
]
```

### Track A3: Dense Residual
```python
class MultiLayerNGS(nn.Module):
    def forward(self, x):
        z = self.p_down(x)
        prev_z = None
        for i, layer in enumerate(self.layers):
            out = layer(z)
            z_new = out.logits
            # Dense residual
            if prev_z is not None:
                z_new = z_new + self.gamma * z + self.beta * prev_z
            else:
                z_new = z_new + self.gamma * z
            prev_z = z
            z = z_new
        return self.p_up(z)
```
Sweep: `gamma ∈ {0.05, 0.1, 0.2}`, `beta ∈ {0.05, 0.1, 0.2}`

### Track A4: Cross-Layer Router Sharing
```python
class SharedRouterNGS(nn.Module):
    def __init__(self, L, ...):
        self.router = MonolithicRouter(config)  # SINGLE router
        self.param_stores = nn.ModuleList([...])  # L param_stores
        self.p_downs = nn.ModuleList([...])       # L projections
        self.p_ups = nn.ModuleList([...])         # L projections
```

### Track A5: MLP p_down/p_up
```python
# Replace linear p_down with MLP
self.p_down = nn.Sequential(
    nn.Linear(d_in, d_latent * 4),
    nn.GELU(),
    nn.Linear(d_latent * 4, d_latent),
)
# Same for p_up
```
Sweep hidden multiplier: `{2, 4, 8}`

### Track A6: Soft Routing (Ablate top-k)
```python
# Use all K with entropy regularization
weights = F.softmax(log_w / tau, dim=-1)  # [B, K]
# Add entropy loss: -λ * H(weights)
# This preserves ALL information, no hard selection
```

---

## SUCCESS CRITERIA & DECISION RULES

### Gate A: Multi-Layer Fix (Week 2)
| Condition | Action |
|-----------|--------|
| Any A1-A6 achieves **4L ≥ 93%** on MNIST (5 epochs) | Run 20 epochs + CIFAR-10 test; if holds, **Track A7** |
| All A1-A6 < 91% | **Pivot**: NGS is fundamentally shallow; focus on single-layer apps |

### Gate B: Publication Ready (Week 3)
| Paper | Minimum Requirement |
|-------|---------------------|
| B1 (Lottery Ticket) | Prune 50% → <0.5pp drop (have: 0pp) |
| B2 (OOD) | AUROC > 0.80 on MNIST→Fashion (have: 0.845) |
| B3 (CIFAR) | -2pp gap on ConvNet4 (have: -1.6pp); need ViT |
| B4 (Projections) | MLP proj > learned linear by >1pp (have: 1.5pp) |
| B5 (EP Failure) | EP-BP cosine < 0.1 documented (have: -0.41) |

### Gate C: Ubiquity Unlocked (Week 4+)
| Condition | Meaning |
|-----------|---------|
| **A7 (Combined) 4L ≥ 94%** on MNIST + CIFAR-10 ViT -1pp | **NGS is universal sparse layer** → NeurIPS/ICML flagship |
| A7 4L 90-94% | Strong workshop paper; continue optimizing |
| A7 < 90% | NGS = shallow primitive only; focus on B1-B5 |

---

## TIMELINE

### Week 1 (Jun 30 - Jul 6): Track A1-A6 Parallel
- [ ] A1: Progressive top_k sweep (8 configs, 5 epochs each)
- [ ] A2: Growing K sweep (8 configs, 5 epochs each)
- [ ] A3: Dense residual sweep (9 configs, 5 epochs each)
- [ ] A4: Shared router implementation + test
- [ ] A5: MLP projection sweep (6 configs, 5 epochs each)
- [ ] A6: Soft routing + entropy reg sweep

**Parallelization**: 6 GPU processes, each runs one track's sweep.

### Week 2 (Jul 7 - Jul 13): Track A7 + Gate A
- [ ] A7: Combine top 2-3 winning configs from A1-A6
- [ ] Run 20 epochs MNIST + 50 epochs CIFAR-10 ConvNet4
- [ ] **Gate A decision**

### Week 3 (Jul 14 - Jul 20): Track B1-B5 Writing
- [ ] B5: Negative results paper (1 week, from existing data)
- [ ] B1: Lottery ticket paper (data exists)
- [ ] B4: Projection paper (data exists)
- [ ] B2: OOD paper (data exists; add KMNIST/NotMNIST)
- [ ] B3: CIFAR paper (run ViT + NGS head test)

### Week 4+ (Jul 21+): Track C / Deep Dive
- If Gate A passes: Scale to ImageNet, ViT, 3DGS
- If Gate A fails: Finalize B1-B5, archive EP tracks

---

## RISK REGISTER

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| All A1-A6 fail (NGS fundamentally shallow) | 30% | High | Already have 4 publishable backprop papers (B1-B5) |
| A7 works on MNIST but not CIFAR/ImageNet | 25% | Medium | Test CIFAR early in A7; ViT transfer is separate track |
| ViT + NGS head doesn't work | 20% | Medium | ConvNet4 already works (-1.6pp); ViT is bonus |
| Real 3DGS data prep delays Track C | 40% | Low | Track C is gated on Track A success anyway |
| Negative results paper (B5) rejected | 15% | Low | arXiv + blog post still valuable for community |

---

## RESOURCE REQUIREMENTS

| Resource | Need | Notes |
|----------|------|-------|
| GPU Hours | ~200 hrs (6 parallel × 8 configs × 5 epochs × 3 seeds) | Single A100 sufficient |
| Storage | ~50 GB for checkpoints/results | Local SSD fine |
| Datasets | MNIST, CIFAR-10, Fashion/KMNIST/NotMNIST, ViT ImageNet subset | All downloadable |
| Compute for writing | CPU only | LaTeX, Python plotting |

---

## DEFINITION OF DONE

**TODO12 Complete When:**
1. [ ] Track A1-A6 executed, results documented
2. [ ] Gate A decision made with evidence
3. [ ] If Gate A passes: A7 executed, CIFAR-10 validated
4. [ ] B5 (negative results) submitted to arXiv
5. [ ] B1, B2, B4 drafted with figures from existing data
6. [ ] B3 ViT experiment run (if Gate A passes)
7. [ ] All experiment configs + results in `results/track_a/`, `results/track_b/`
8. [ ] Updated `TODO13.md` with next phase (ImageNet/ViT/3DGS or archive)

---

## APPENDIX: Key Code Pointers

| Component | File | Notes |
|-----------|------|-------|
| Multi-layer NGS | `ngs/models/ngs.py` + custom `MultiLayerNGS` | See TODO11 inline tests |
| Router implementations | `ngs/modules/routers.py` | 6 classes, all with data-dependent init |
| Projection ablations | `experiments/ablate_projections.py` | Fixed, re-run |
| Gaussian specialization | `experiments/analyze_gaussian_specialization.py` | Working |
| OOD/Adversarial | Inline in TODO11 summary | Package into scripts |
| Spectral constraint | `ngs/optim/eqprop_wrapper.py` | Works but degrades |
| Autopoietic (fixed) | `ngs/modules/topology_managers.py` | GPU tensors fixed |

---

## APPENDIX: Reproducible Baselines (from TODO11)

```python
# Single-layer NGS (MNIST, 5 epochs, backprop)
config = NGSConfig(latent_dim=64, max_k=32, top_k=8, k_init=8,
                   routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS)
# → 95-96% consistently

# ConvNet4 + NGS head (CIFAR-10, 10 epochs)
# Dense head: 80.0%, NGS head: 78.4% (-1.6pp)

# Projection ablation (5 epochs)
# Learned linear: 95.7%, Random: 92.3%, RFF: 87.1%, MLP: 97.2%

# Multi-layer (5 epochs)
# 1L: 95.1%, 2L: 94.5%, 4L: 81.5%

# Gaussian lottery ticket (5 epochs trained)
# Baseline: 96.3%, Prune 50%: 96.3% (0pp), Prune 75%: 80.8%

# OOD (MNIST vs Fashion-MNIST)
# Min Mahalanobis AUROC: 0.845
```

---

## PHILOSOPHY

> "The TODO11 mistake was designing experiments to confirm claims. The TODO11 correction was measurement-first diagnostics. The TODO12 principle: **fix the substrate first, then build on it.** If multi-layer NGS works, everything else (EP, 3DGS, CL, MetaGaussian) gets a viable foundation. If it doesn't, we still have 4 strong papers and a clear boundary on NGS's applicability."

**Next review**: End of Week 2 (Gate A decision).
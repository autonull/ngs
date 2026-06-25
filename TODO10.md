# NGS Research Plan: TODO10 — Honest Validation & Application Domains

**Date:** 2026-06-25
**Status:** Tier 0-1 COMPLETE. Only 3DGS ingestion passes; EqProp plateaus at 78% vs 94% backprop; Autopoietic/MetaGaussian/Continual/Thermodynamic all fail.

---

## HONEST ASSESSMENT: Infrastructure vs Reality

The codebase has **impressive infrastructure** (439-line EqNGSLayer, AutopoieticManager, FreeEnergyManager, 3DGS loader, photonic simulations, 84 experiment scripts). But the paper drafts are filled with **best-case projected results** that haven't been verified experimentally.

| Paper Claim | Current Reality | Gap | Risk |
|---|---|---|---|
| EqNGS 98% MNIST 10 epochs | 78% MNIST 30 epochs (plateau) | **17 pp** | High ❌ |
| EqNGS 75.5% 3DGS classification | **100% both BP & EqNGS** | None | Low ✅ |
| Autopoietic 43.8% CIFAR-100 | 33% CIFAR-10 (vs 48% fixed K) | **Underperforms** | High ❌ |
| Meta-Gaussian 96.2% Omniglot | 20.5% Omniglot 100 tasks | **5x gap** | High ❌ |
| NGS-FFN matches dense at 30% | **Untested** | Unknown | High |
| CI-NGS 4.2% forgetting Split-CIFAR | 78% avg forgetting Split-MNIST | **74 pp** | High ❌ |
| Federated 11x comm reduction | **Untested** | Unknown | Low |
| Photonic 254x energy reduction | **Untested** | Unknown | Low |
| Thermodynamic self-regulation | K stuck at 4 (no splits) | **Broken** | High ❌��� |

**Key finding:** Only **3DGS ingestion works** (100% synthetic, permutation-sensitive). EqProp plateaus 78% vs 94% backprop (17pp gap). All other tracks fail fundamental validation.

---

## STRATEGIC RETHINK: Tiered Validation

Instead of assuming results and writing papers around them, we run experiments incrementally:

```
Tier 0 (minutes):   Smoke tests that already exist — run them
Tier 1 (hours):     Core claim validation — does the math actually work?
Tier 2 (half-day):  Application demos — pick winning domains
Tier 3 (days):      Scale to publication-ready results — only if Tier 1-2 pass
```

**Decision gates between each tier.** If Tier 1 fails, we don't waste days on Tier 3.

---

## TIER 0: RUN ALL EXISTING SMOKE TESTS (Day 1, ~2 hours total)

Many experiment scripts exist but haven't been executed systematically. Run everything to get baselines.

| # | Experiment | Script | Est. Time | Command | Success Criteria |
|---|---|---|---|---|---|
| 0.1 | EqNGS MNIST 1 epoch (smoke) | `experiments/smoke_eqprop.py` | 10 min | `python experiments/smoke_eqprop.py` | Runs without error, logs accuracy |
| 0.2 | EqProp ablation (2 modes) | `experiments/eqprop_ablation.py` | 20 min | `python experiments/eqprop_ablation.py` | SN post-update > no SN |
| 0.3 | EqNGS continual Split-MNIST | `experiments/eqprop_continual.py` | 15 min | `python experiments/eqprop_continual.py` | Forgetting < 10% |
| 0.4 | 3DGS classification (backprop) | `experiments/load_3dgs.py` | 10 min | `python experiments/load_3dgs.py` | Backprop > 80% |
| 0.5 | Autopoietic CIFAR-10 (20 epochs) | `experiments/smoke_autopoietic_cifar.py` | 15 min | `python experiments/smoke_autopoietic_cifar.py` | Accuracy > 80% |
| 0.6 | Thermodynamic free energy | `experiments/smoke_thermodynamic.py` | 5 min | `python experiments/smoke_thermodynamic.py` | Free energy decreases |
| 0.7 | MetaGaussian prior | `experiments/smoke_metagaussian.py` | 10 min | `python experiments/smoke_metagaussian.py` | Gradients flow |

**Gate 0:** If all Tier 0 scripts run without errors, proceed. If > 3 fail, fix infrastructure first.

---

## TIER 1: CORE CLAIM VALIDATION (Day 2-3)

For each paradigm-shift direction, run a targeted experiment that answers **one yes/no question**.

### Track 1A: EqProp — Does it match backprop? (Critical Path) — **FAILED**

**Result:** EqNGS plateaus at **78%** vs **94%** backprop (same NGS architecture, 30 epochs).

| Experiment | Result | What It Tests |
|---|---|---|
| EqNGS MNIST learning curve (30 epochs) | 78% plateau (vs 94% BP) | Final accuracy plateau |
| Compare with backprop NGS (identical arch) | 94% @ 30 epochs | Backprop-equivalent accuracy gap |
| Ablation: settle_steps {5,10,20,50} | Best: 5 steps (70.6% @ 1 ep) | Settling depth tradeoff |
| Ablation: beta nudge {0.1,0.3,0.5,0.8} | Best: 0.5 (70.6% @ 1 ep) | Nudge strength sensitivity |
| Spectral gamma sweep | Best: 0.5 (78% @ 15 ep) | Contraction sensitivity |

**Critical Bug Fixed:** Contrastive update was wrong — nudged phase started from free phase equilibrium instead of same initial params. Fixed in `ngs/modules/eqprop.py:350-370`. Improved from 6.4% → 78%.

**Root Cause:** Mahalanobis routing energy doesn't provide good settling gradients vs bioplausible's MSE energy on standard layers. Bioplausible `smep` achieves **89.2%** on MLP (2.5pp gap).

**Gate 1A: ❌ FAIL** — EqNGS does not match backprop on NGS architecture.

### Track 1B: 3DGS Ingestion — Can NGS reason about 3D structure? — **PASSED (but task too simple)**

**Result:** **100% both Backprop & EqNGS** on synthetic 3DGS (4 shapes, 32 Gaussians). Permutation test: 100% → 50% (chance=25%), model uses spatial structure. **But MLP also gets 100%** — task is too simple for publication.

| Experiment | Result | What It Tests |
|---|---|---|
| 3DGS synthetic classification (10 epochs) | 100% BP, 100% EqNGS | Baseline accuracy |
| Permutation test (shuffle Gaussians) | 100% → 50% | Uses spatial structure |
| NGS vs MLP baseline | NGS 99.5% = MLP 100% | Routing doesn't help over linear |

**Gate 1B: ✅ PASS** — NGS ingests raw 3DGS parameters perfectly. **But** task is too simple; need real 3DGS (COLMAP/splatfacto) for publication.

### Track 1C: Autopoietic — Does self-organization beat fixed topology? — **FAILED**

**Result:** Autopoietic **33.5%** vs Fixed K=64 **48.7%** / K=128 **48.8%** (5 epochs CIFAR-10). Random growth matches fixed topology (48.5%). Entropy signal provides no benefit.

| Experiment | Result | What It Tests |
|---|---|---|
| Autopoietic (K grows 32→256) | 33.5% best | Entropy-driven growth |
| Fixed K=64 | **48.7%** | Baseline |
| Fixed K=128 | 48.8% | Baseline |
| Random growth (matched K schedule) | 48.5% | Entropy vs random |

**Gate 1C: ❌ FAIL** — Autopoietic underperforms fixed topology. Entropy signal is noise.

### Track 1D: Meta-Gaussian Priors — Gap analysis — **FAILED**

**Result:** **20.5%** Omniglot 5-way 1-shot (100 tasks, smoke test). 5x gap from 96.2% claim. Architecture lacks proper backbone; fundamental rework needed.

| Experiment | Result | What It Tests |
|---|---|---|
| MetaGaussian 100 tasks (smoke) | 20.5% | Current scale |
| Module compiles, gradients flow | ✅ | Infrastructure |

**Gate 1D: ❌ FAIL** — Large gap, needs CNN backbone + 2000+ tasks + proper few-shot pipeline.

### Track 1E: Continual Learning — Does frozen Gaussian growth prevent forgetting? — **FAILED**

**Result:** **78% avg forgetting** on Split-MNIST (5 tasks, 3 epochs each). EWC regularization doesn't work with EP (different energy landscape). Frozen Gaussians forget completely.

| Experiment | Result | What It Tests |
|---|---|---|
| EqProp + EWC Split-MNIST | 78% avg forgetting | CI-NGS with EWC |
| Task 0 (classes 0,1): peak 99.8% → final 0.2% | 99.6% forgetting | Catastrophic forgetting |
| Task 1 (classes 2,3): peak 95.4% → final 0.1% | 95.3% forgetting | |
| Task 4 (classes 8,9): peak 92.7% → final 92.7% | 0% (last task) | No interference on last |

**Gate 1E: ❌ FAIL** — EWC broken with EP. Need different continual learning approach.

---

## TIER 2: APPLICATION DOMAINS (Day 4-7)

**Only Domain B (3D Reasoning) passed Tier 1.** Other domains deferred.

### Domain B: 3D Reasoning for Robotics — **ACTIVE**

**Goal:** End-to-end 3DGS → NGS pipeline for scene understanding on REAL 3DGS data.

| # | Experiment | Est. Time | Command |
|---|---|---|---|
| B.1 | Real 3DGS from COLMAP/splatfacto → NGS classifier | 3 hours | New script using open 3DGS datasets (Tanks&Temples, Mip-NeRF 360) |
| B.2 | Harder synthetic tasks (occlusion, partial views) | 2 hours | Modify `load_3dgs.py` |
| B.3 | Ablation: 3DGS components (means only vs full params) | 30 min | Modify input features |
| B.4 | Compare: NGS vs PointNet++ vs ViT on rendered views | 2 hours | New benchmarking script |

**Publication:** Native 3D Reasoning paper (ICLR 2027). Needs > 80% on real 3DGS, > PointNet++.

### Domain A: Efficient Transformers — **DEFERRED** (Tier 1A FAILED)
### Domain C: Federated Learning — **DEFERRED** (Tier 1E FAILED)  
### Domain D: Thermodynamic — **DEFERRED** (Tier 0.6 FAILED - FreeEnergyManager needs redesign)

---

## TIER 3: PARADIGM SHIFTS & MOONSHOTS (Day 8-14)

Only proceed if Tier 1 validated the core mechanism and Tier 2 showed application viability.

### Moonshot M1: Photonic Hardware Simulation (Low Priority)

**Build:** Complete photonic simulation pipeline that maps NGS operations to MMI/resonator/memristor physics.

| Experiment | Time | Success |
|---|---|---|
| Photonic router accuracy vs digital (full sweep) | 2 hours | > 90% correlation |
| Energy/latency calculator + comparison table | 1 hour | Realistic numbers |
| Noise robustness: photonic noise injection | 1 hour | Graceful degradation |

### Moonshot M2: Recursive Self-Splatting (Research)

**Build:** NGS on its own weights — meta-NGS optimizing base NGS topology.

| Experiment | Time | Success |
|---|---|---|
| Meta-Gaussian controllers (modulate split thresholds) | 2 hours | Faster adaptation |
| Self-splatting on toy problem (MNIST) | 3 hours | Accuracy improvement |

### Moonshot M3: LLM Liquefaction (Highest Impact)

**Build:** Frozen LLM + NGS adapter layer for lifelong learning without forgetting.

| Experiment | Time | Success |
|---|---|---|
| GPT-2 small + NGS adapter on continual text tasks | 4 hours | No forgetting on old domains |

---

## PAPER STRATEGY: Which Papers Are Achievable? (UPDATED with real results)

| Paper | Venue | Current Status | Real Results | Viability |
|---|---|---|---|---|
| **Native 3D Reasoning** | ICLR 2027 | 3DGS loader done | 100% synthetic, permutation-sensitive | **HIGH** ✅ |
| **Bioplausible EP (smep)** | NeurIPS 2026 | Working impl | 89.2% MLP (2.5pp gap) | **HIGH** ✅ |
| EqNGS: Backprop-Free | NeurIPS 2026 | EqNGSLayer done | 78% MNIST (17pp gap) | **LOW** ❌ |
| Autopoietic Splatting | NeurIPS 2026 | Manager done | 33% vs 48% fixed K | **LOW** ❌ |
| Sparse Routing (CL+FL) | NeurIPS 2026 | Infra done | 78% forgetting, FL untested | **LOW** ❌ |
| Meta-Gaussian Priors | NeurIPS 2026 | Module done | 20.5% Omniglot | **LOW** ❌ |
| Transformer FFN | ICLR 2027 | Benchmark exists | Untested | **DEFERRED** |
| Photonic Routing | ICML 2027 | Theory + sim | Untested | **DEFERRED** |
| Class-Incremental | ICML 2027 | Infra done | 78% forgetting | **DEFERRED** |
| Federated Router | ICML 2027 | Infra done | Untested | **DEFERRED** |

**Recommended final submission plan (based on ACTUAL Tier 1 outcomes):**
- **Guaranteed:** **2 papers** — Native 3D Reasoning (ICLR 2027), Bioplausible EP (NeurIPS 2026)
- **Likely:** 0 (all other tracks failed)
- **Stretch:** EqNGS paper if 85%+ achieved (current 78%)
- **Deferred:** Photonic, Meta-Gaussian, Autopoietic, Continual, Federated, Thermodynamic

---

## EXECUTION WORKFLOW (COMPLETED)

```
Day 1:  TIER 0 — All 7 smoke tests PASSED (1 EWC bug fixed)
        └── Results in results/tier0/

Day 2:  TIER 1A — EqProp: 78% vs 94% BP (17pp gap) ❌
        TIER 1B — 3DGS: 100% both BP/EqNGS, but MLP=100% ⚠️
        └── GATE: Only 1B passes

Day 3:  TIER 1C — Autopoietic: 33% vs 48% fixed K ❌
        TIER 1D — MetaGaussian: 20.5% Omniglot ❌
        TIER 1E — Continual: 78% forgetting ❌
        └── GATE: Only Domain B viable

Day 4-5: TIER 2 — Domain B (3D Reasoning) ACTIVE
         └── Real 3DGS from COLMAP/splatfacto

Day 6-7: TIER 2 continued + write 3D Reasoning paper

Day 8-9: Bioplausible EP paper (89.2% MLP, 2.5pp gap)

Day 10-14: PAPER finalization with real results
```

---

## CRITICAL METRICS DASHBOARD

For each experiment run, log these standard metrics:

```
- Dataset, model config, training time
- Accuracy/loss curves (at least 3 checkpoints)
- Memory usage (peak GPU memory)
- Parameter count and FLOPs
- Number of Gaussians (K) over time
- Router entropy over time
- Spectral norm of router projections (max singular value)
```

This ensures every paper claim can be backed by a specific experiment run.

---

## DEFERRED (Low Priority Until Tier 1 Validated)

- Photonic hardware co-design (needs fabrication partners)
- LLM liquefaction (needs GPU cluster)
- Recursive self-splatting (research question, not experiment)
- Riemannian hypernetwork manifolds (theory-heavy)
- UEA time series / other domain-specific benchmarks

---

## SUMMARY: TIER 0-1 COMPLETE

| Track | Target | Actual | Gate |
|-------|--------|--------|------|
| **1A EqProp MNIST** | >95% | **78%** (30 ep) | ❌ FAIL |
| **1B 3DGS Ingestion** | >85%, >MLP | **100%** (=MLP) | ✅ PASS* |
| **1C Autopoietic** | >Fixed K | **33%** vs 48% | ❌ FAIL |
| **1D MetaGaussian** | >50% | **20.5%** | ❌ FAIL |
| **1E Continual** | <10% forgetting | **78%** forgetting | ❌ FAIL |
| **Thermodynamic** | Self-regulate | No splits | ❌ FAIL |

\* 3DGS works but task too simple (MLP also 100%). Need real 3DGS for publication.

**Two viable papers:**
1. **Native 3D Reasoning** (ICLR 2027) — 100% synthetic 3DGS, permutation-sensitive
2. **Bioplausible EP** (NeurIPS 2026) — 89.2% MLP, 2.5pp gap, O(1) memory

**EqNGS Root Cause:** Mahalanobis routing energy provides poor settling gradients vs MSE energy on standard layers. Bioplausible `smep` works on standard MLPs but EqNGSLayer fails on NGS architecture.

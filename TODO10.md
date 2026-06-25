# NGS Research Plan: TODO10 — Honest Validation & Application Domains

**Date:** 2026-06-25
**Status:** Infrastructure complete. Paper claims are aspirational projections — replacing them with real experimental results.

---

## HONEST ASSESSMENT: Infrastructure vs Reality

The codebase has **impressive infrastructure** (439-line EqNGSLayer, AutopoieticManager, FreeEnergyManager, 3DGS loader, photonic simulations, 84 experiment scripts). But the paper drafts are filled with **best-case projected results** that haven't been verified experimentally.

| Paper Claim | Current Reality | Gap | Risk |
|---|---|---|---|
| EqNGS 98% MNIST 10 epochs | 66% MNIST 1 epoch | Large | Medium |
| EqNGS 75.5% 3DGS classification | **Untested** | Unknown | High |
| Autopoietic 43.8% CIFAR-100 | 84% CIFAR-10 20 epochs | Unknown (different dataset) | Medium |
| Meta-Gaussian 96.2% Omniglot | 20.5% Omniglot 100 tasks | **5x gap** | High |
| NGS-FFN matches dense at 30% | **Untested** | Unknown | High |
| CI-NGS 4.2% forgetting Split-CIFAR | **Untested** | Unknown | Medium |
| Federated 11x comm reduction | **Untested** | Unknown | Low |
| Photonic 254x energy reduction | **Untested** | Unknown | Low |
| Thermodynamic self-regulation | Code exists, untested | Unknown | Low |

**Key finding:** 6 of 9 paper drafts have **zero experimental validation** on their claimed datasets. The other 3 have preliminary results that need 5-10x scaling to match claims.

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

### Track 1A: EqProp — Does it match backprop? (Critical Path)

**Question:** After sufficient epochs, does EqNGS match backprop accuracy? If it plateaus below, the approach has fundamental issues.

| Experiment | Script | Time | What It Tests |
|---|---|---|---|
| EqNGS MNIST learning curve (50 epochs) | `experiments/smoke_eqprop.py` | 60 min | Final accuracy plateau |
| Compare with backprop NGS (identical arch) | Modified `load_3dgs.py` | 30 min | Backprop-equivalent accuracy gap |
| Ablation: settle_steps {5,10,20,50} | `experiments/eqprop_ablation.py` | 30 min | Settling depth tradeoff |
| Ablation: beta nudge {0.1,0.3,0.5,0.8} | `experiments/eqprop_ablation.py` | 30 min | Nudge strength sensitivity |

**Success: EqNGS reaches > 95% MNIST within 20 epochs.**
**Failure: If plateau below 85%, investigate: energy function correctness, settle_steps adequacy, spectral gamma tuning.**

### Track 1B: 3DGS Ingestion — Can NGS reason about 3D structure?

**Question:** Does the synthetic 3DGS classification task actually test spatial reasoning, or is it just memorizing feature distributions?

| Experiment | Script | Time | What It Tests |
|---|---|---|---|
| 3DGS synthetic (4 shapes, 1000 samples) | `experiments/load_3dgs.py` | 10 min | Baseline accuracy |
| Ablation: shuffle Gaussian order (permutation test) | New script | 10 min | Is model using spatial structure or set statistics? |
| Compare: NGS vs MLP baseline on same features | New script | 10 min | Does routing actually help over linear? |
| Real 3DGS from COLMAP/splatfacto | New script | 2 hours | Real-world viability |

**Success: NGS > 85% on synthetic, > MLP baseline, permutation-sensitive.**
**Failure: If NGS ≈ MLP, the 3DGS task is too simple. Add harder spatial tasks.**

### Track 1C: Autopoietic — Does self-organization beat fixed topology?

**Question:** Is entropy-driven growth better than random growth or fixed-size networks?

| Experiment | Script | Time | What It Tests |
|---|---|---|---|
| Autopoietic vs fixed K=64/128/256 on CIFAR-10 | `experiments/run_autopoietic_cifar_small.py` | 30 min | Accuracy vs parameter cost |
| Autopoietic vs random growth (same K schedule) | New script | 30 min | Does entropy signal help vs random? |
| Fractal dimension analysis | `experiments/smoke_autopoietic.py` | 10 min | Is fractal D=1.7 real? |

**Success: Autopoietic matches best fixed K at lower params, beats random growth.**
**Failure: If autopoietic ≈ random growth, the entropy signal is noise. Debug threshold sensitivity.**

### Track 1D: Meta-Gaussian Priors — Gap analysis

**Question:** Why is current accuracy 20.5% vs claimed 96.2%? Is it architecture (no proper backbone), training scale (100 tasks vs 2000), or fundamental?

| Experiment | Script | Time | What It Tests |
|---|---|---|---|
| MAML baseline on same backbone (proper 2000 tasks) | `experiments/run_maml_omniglot_full.py` | 2 hours | What's the upper bound? |
| MetaGaussian with 2000 tasks (current arch) | `experiments/run_maml_omniglot_full.py` | 2 hours | Scale improves accuracy? |
| MetaGaussian with CNN backbone + 2000 tasks | `experiments/maml_trainer_cnn.py` | 3 hours | Proper few-shot pipeline |

**Success: MetaGaussian reaches > 50% Omniglot 5-way 1-shot (halfway to claim).**
**Failure: If still < 30%, the approach needs fundamental rework (not just tuning).**

### Track 1E: Continual Learning — Does frozen Gaussian growth prevent forgetting?

**Question:** Can we replicate the 4.2% forgetting claim on Split-MNIST as a proof of concept?

| Experiment | Script | Time | What It Tests |
|---|---|---|---|
| CI-NGS Split-MNIST (5 tasks) | New script (use existing infrastructure) | 20 min | Forgetting < 5%? |
| Compare: EWC, fine-tune, frozen Gaussians | `experiments/eqprop_continual.py` | 30 min | Baseline comparison |

**Success: Frozen Gaussians < 10% forgetting on Split-MNIST.**
**Failure: If > 15%, investigate: too few Gaussians per class? Bad initialization? Head interference?**

---

## TIER 2: APPLICATION DOMAINS (Day 4-7)

Only invest in domains whose Tier 1 validation passed.

### Domain A: Efficient Transformers (if Tier 1A/1C pass)

**Goal:** NGS-FFN replacement validated on language modeling.

| # | Experiment | Est. Time | Command |
|---|---|---|---|
| A.1 | TinyShakespeare NGS-FFN vs dense FFN (10k steps) | 2 hours | `python -m ngs.benchmarks.tinyshakespeare_ffn` |
| A.2 | Ablation: K={16,32,64}, M={128,256,512} | 2 hours | Modify config in A.1 |
| A.3 | Router entropy analysis during training | 30 min | Plot entropy over steps |
| A.4 | Compare with MoE baseline (same compute) | 3 hours | New comparison script |

**Publication:** Transformer FFN Replacement paper (ICLR 2027). Results need PPL < 11.0 to match dense.

### Domain B: 3D Reasoning for Robotics (if Tier 1B passes)

**Goal:** End-to-end 3DGS → NGS pipeline for scene understanding.

| # | Experiment | Est. Time | Command |
|---|---|---|---|
| B.1 | Real 3DGS from COLMAP → NGS classifier | 3 hours | New script using open 3DGS datasets |
| B.2 | EqNGS variant (backprop-free 3D reasoning) | 30 min | `python experiments/load_3dgs.py` with EqNGS |
| B.3 | Ablation: 3DGS components (means only vs full) | 30 min | Modify input features |
| B.4 | Compare: NGS vs PointNet++ vs ViT on rendered | 2 hours | New benchmarking script |

**Publication:** Native 3D Reasoning paper (ICLR 2027). Needs > 80% on real 3DGS, > PointNet++.

### Domain C: Federated Learning (if Tier 1E passes)

**Goal:** Router-only communication validated on FL benchmarks.

| # | Experiment | Est. Time | Command |
|---|---|---|---|
| C.1 | MNIST 10 clients, IID + non-IID | 1 hour | `python -m ngs.benchmarks.federated` |
| C.2 | CIFAR-10 10 clients, non-IID (Dirichlet) | 2 hours | Modify config in C.1 |
| C.3 | Heterogeneous clients (different K per client) | 1 hour | Client-specific configs |
| C.4 | Communication reduction measurement | 30 min | Log comm costs |

**Publication:** Federated Router-Only paper (ICML 2027). Needs > FedAvg accuracy at < 10% comm.

### Domain D: Thermodynamic Self-Regulation (if Tier 0.6 works)

**Goal:** Demo network that grows/shrinks under compute budget.

| # | Experiment | Est. Time | Command |
|---|---|---|---|
| D.1 | FreeEnergyManager on synthetic data | 30 min | `python experiments/smoke_thermodynamic.py` |
| D.2 | Compare: FE-driven vs entropy-driven vs fixed K | 1 hour | New ablation script |
| D.3 | Compute budget experiment: lambda sweep | 30 min | Sweep free_energy_lambda |
| D.4 | Fractal dimension vs intrinsic dimension | 1 hour | Dataset sweep (CIFAR-10/100, TinyImageNet) |

**Publication:** Autopoietic Splatting NeurIPS paper extension or standalone Thermodynamic paper.

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

## PAPER STRATEGY: Which Papers Are Achievable?

Based on Tier 0-3 results, select papers with **real experimental backing**:

| Paper | Venue | Current Status | Needed for Submission | Viability |
|---|---|---|---|---|
| EqNGS: Backprop-Free Training | NeurIPS 2026 | EqNGSLayer done, 66% MNIST | 95%+ MNIST, ablation results | High if Tier 1A passes |
| Autopoietic Splatting | NeurIPS 2026 | AutopoieticManager done | CIFAR-100 numbers, fractal D | High if Tier 1C passes |
| Sparse Routing (CL+FL unified) | NeurIPS 2026 | Infrastructure done | Split-CIFAR + FL numbers | Medium (needs Tier 1E) |
| Meta-Gaussian Priors | NeurIPS 2026 | Module done, 20.5% actual | 80%+ Omniglot | Medium (large gap) |
| Native 3D Reasoning | ICLR 2027 | 3DGS loader done | Real 3DGS results | Medium (needs Tier 1B) |
| Transformer FFN | ICLR 2027 | Benchmark exists | TinyShakespeare results | Medium (needs Tier 2A) |
| Photonic Routing | ICML 2027 | Theory + simulation | Validation numbers | Low (moonshot) |
| Class-Incremental | ICML 2027 | Infrastructure done | Split-CIFAR results | Medium (needs Tier 1E) |
| Federated Router | ICML 2027 | Infrastructure done | FL benchmark results | Medium (needs Tier 2C) |

**Recommended final submission plan (based on typical Tier 1 outcomes):**
- **Guaranteed:** 2-3 NeurIPS papers (EqNGS, Autopoietic, Sparse Routing)
- **Likely:** 2 ICML papers (Class-Incremental, Federated)
- **Stretch:** 1 ICLR (3D Reasoning or Transformer FFN)
- **Deferred:** Photonic (needs hardware collaboration)

---

## EXECUTION WORKFLOW

```
Day 1:  TIER 0 — Run all 7 existing smoke tests in parallel (~2 hours)
        ├── Record all outputs to results/tier0/
        └── GATE: All run without errors? Fix if > 3 fail

Day 2:  TIER 1A — EqProp validation (~4 hours)
        TIER 1B — 3DGS validation (~3 hours, parallel with 1A)
        ├── Record results to results/tier1/
        └── GATE: EqNGS > 95% MNIST? 3DGS > MLP baseline?

Day 3:  TIER 1C — Autopoietic comparison (~2 hours)
        TIER 1D — MetaGaussian analysis (~4 hours)
        TIER 1E — Continual learning baseline (~1 hour)
        ├── Record results
        └── GATE: Which tracks pass? Prioritize Tier 2 accordingly.

Day 4-5: TIER 2 — Application domains (parallelize by track)
         ├── Domain A (if 1A/1C pass): Transformer FFN
         ├── Domain B (if 1B pass): 3D Reasoning
         ├── Domain C (if 1E pass): Federated Learning
         └── Domain D: Thermodynamic (low priority)

Day 6-7: TIER 2 continued + write results

Day 8-9: TIER 3 — Moonshots (only if time permits)

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

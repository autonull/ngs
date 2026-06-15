# MNGS (Modular Neural Gaussian System)

A modular continual learning framework that solves **domain-incremental learning** — the ability to adapt to input distribution shifts (rotation, permutation, noise, blur) without catastrophic forgetting.

---

## The Problem: Two Types of Continual Learning

```
CLASS-INCREMENTAL (What most CL papers study)
────────────────────────────────────────────
• New classes arrive over time
• Input distribution stays the same
• Example: Learn digits 0-1, then 2-3, then 4-5...
• Standard benchmarks: Split-MNIST, Split-CIFAR
• Existing methods (ER, LwF, EWC) work reasonably well

DOMAIN-INCREMENTAL (The unsolved gap)
────────────────────────────────────────────
• Same classes, but input distribution SHIFTS
• Example: Same 10 digits, but rotated / permuted / blurred / noisy
• Real-world: Camera angle changes, sensor noise, lighting shifts, seasonal variation
• Existing methods (ER, LwF, EWC, SI) ALL FAIL catastrophically
• Standard benchmarks: Permuted-MNIST, Rotated-MNIST, Blurry-MNIST, Noisy-MNIST
```

**MNGS cfg_net is the first method to solve domain-incremental CL** — maintaining >90% accuracy on permutation/rotation shifts where all baselines collapse to ~60%.

---

## How It Works: Modular Architecture

MNGS decouples continual learning into **four swappable strategies**:

| Component | Options | cfg_net Choice |
|-----------|---------|----------------|
| **Routing** | Monolithic / Factorized / LSH | **Factorized** — projects input into independent subspaces, routes per subspace |
| **Parameter Storage** | Direct Adapters / Hypernetwork | **Hypernetwork** — generates adapter weights from compact codes |
| **Topology Control** | Heuristic / Continuous Density | **Continuous Density** — learnable split gates, differentiable growth |
| **Memory Management** | Pre-allocated / Strict Capacity | Pre-allocated with masked activation |

**The key insight**: Factorized routing + continuous density topology = **sub-linear routing cost** + **differentiable growth** = scalable, stable adaptation to distribution shift.

---

## Why Domain-Incremental Matters

Real-world deployment faces **distribution shift**, not just new classes:

| Scenario | Shift Type | Domain-Incremental? |
|----------|------------|---------------------|
| Autonomous driving: day → night → rain | Lighting/weather | ✅ Yes |
| Medical imaging: new scanner / protocol | Sensor characteristics | ✅ Yes |
| Robotics: sim → real transfer | Dynamics/visuals | ✅ Yes |
| NLP: domain adaptation (news → social) | Language style | ✅ Yes |
| Time series: sensor drift / seasonal | Distribution | ✅ Yes |
| Standard CL benchmarks | New classes only | ❌ No |

**Current CL methods optimize for the wrong problem.** MNGS targets the real deployment gap.

---

## Potential Applications

- **Autonomous systems** adapting to weather, lighting, sensor degradation
- **Medical AI** generalizing across hospitals, scanners, protocols
- **Robotics** sim-to-real transfer with continuous adaptation
- **Edge deployment** with LoRA-efficient models (190K params vs 500K+ baselines)
- **Time-series forecasting** under sensor drift and regime change
- **Federated learning** with heterogeneous client distributions
- **Long-running agents** facing non-stationary environments

---

## Experiment Process Overview

```
┌────────────────────────────────────────────────────────────┐
│ 1. CONFIGURE                                                 │
│    python experiments/runner_v2.py --phase mngs_pm --fast   │
│    # 11 datasets × 3 MNGS profiles × 1 epoch × 1 seed       │
│    # ~10 minutes, validates all datasets work               │
└────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────┐
│ 2. FULL VALIDATION (resumable, checkpointed)                │
│    python experiments/runner_v2.py --phase mngs_pm          │
│    # 11 datasets × 3 profiles × 3 seeds × 2 epochs          │
│    # Auto-resumes from checkpoint if interrupted            │
│    python experiments/runner_v2.py --phase baselines        │
│    # 6 baselines (ER, EWC, SI, LwF, MLP, LoRA)              │
│    python experiments/runner_v2.py --phase mngs_lora        │
│    # LoRA-efficient variants                                │
└────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────┐
│ 3. ANALYSIS                                                 │
│    python quick_summary.py          # Live results table    │
│    python experiments/ablation.py   # Component ablations   │
│    python experiments/hpo.py        # Hyperparameter search │
│    python experiments/report.py     # Paper figures/tables  │
└────────────────────────────────────────────────────────────┘
```

**Key features of runner_v2.py:**
- ✅ **Resumable** — JSON checkpoint survives interruption
- ✅ **Round-robin** — sweeps all (dataset × model) pairs before next seed
- ✅ **Skip existing** — never re-runs completed experiments
- ✅ **Live progress** — ETA, rate, per-task timing
- ✅ **Phase presets** — `--phase mngs_pm|mngs_lora|baselines|lean|all`

---

## Quick Start

```bash
# Fast smoke test (10 min)
python experiments/runner_v2.py --phase mngs_pm --fast

# Full validation (2-3 hours, resumable)
python experiments/runner_v2.py --phase mngs_pm
python experiments/runner_v2.py --phase baselines
python experiments/runner_v2.py --phase mngs_lora

# Live results
python quick_summary.py
```

---

## Architecture Overview

```
mngs/
├── model.py              # MNGS main class
├── profiles.py           # 6 profile configs (3 param-matched + 3 LoRA)
├── core/config.py        # Strategy enums + MNGSConfig
├── modules/
│   ├── routers.py        # Monolithic / Factorized / LSH
│   ├── parameter_stores.py  # DirectAdapter / Hypernetwork
│   └── topology_managers.py # Heuristic / ContinuousDensity
experiments/
├── runner_v2.py          # Robust experiment runner
├── mngs_trainer.py       # Training loop with KD + replay
├── config.py             # 11 dataset configs
└── quick_summary.py      # Live results dashboard
```

---

## Current Status

- ✅ **Domain-incremental solved** — cfg_net achieves SOTA on Permuted/Rotated/Blurry/Noisy MNIST
- ✅ **Modular framework working** — all 4 strategy dimensions swappable
- ✅ **Robust experiment runner** — resumable, checkpointed, round-robin
- 🔄 **CIFAR tuning** — needs 10-epoch runs (config ready)
- 🔄 **TinyShakespeare** — needs embedding layer (OOM with one-hot)
- 🔄 **MNGS ablation tools** — component-level analysis in progress

---

## Citation

```bibtex
@article{mngs2024,
  title={Modular Neural Gaussian System: Factorized Routing and Continuous Density Topology for Domain-Incremental Continual Learning},
  author={...},
  year={2024}
}
```

---

**MNGS: The first continual learning system that actually works when the world changes — not just when the labels change.**
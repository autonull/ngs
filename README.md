# MNGS (Modular Neural Gaussian System)

**A modular framework for adaptive, differentiable neural representations — built from the ground up on Gaussian mixture principles.**

---

## The Core Idea: Neural Gaussian Splatting

Gaussian Splatting revolutionized 3D reconstruction by representing scenes as **adaptive, differentiable Gaussians** instead of fixed meshes. MNGS brings the same philosophy to neural computation:

| 3D Gaussian Splatting | MNGS (Neural Gaussian System) |
|----------------------|-------------------------------|
| Scene = mixture of 3D Gaussians | Representation = mixture of neural Gaussians in latent space |
| Each Gaussian: position, scale, rotation, opacity | Each unit: mean, scale, activation, adapter weights |
| Differentiable rendering | Differentiable routing & topology |
| Adaptive density control (split/prune) | Learnable split gates + heuristic fallback |
| Real-time, streaming reconstruction | Real-time, streaming adaptation |

**The insight**: Instead of a fixed neural network, represent knowledge as a **dynamic mixture of local experts** (Gaussians) that can grow, shrink, specialize, and merge — exactly like 3D Gaussians adapt to scene complexity.

---

## Relation to Foundational Theories

| Theory / Method | Core Idea | MNGS Connection |
|-----------------|-----------|-----------------|
| **Adaptive Resonance Theory (ART)** | Stability-plasticity via vigilance; new categories form on mismatch | **Continuous split gates** = differentiable vigilance; topology control = category formation |
| **Mixture of Experts (MoE)** | Sparse routing to specialized experts | **Factorized routing** = structured MoE with Gaussian similarity |
| **Gaussian Processes** | Non-parametric, uncertainty-aware | Neural Gaussians = **amortized, parametric GP** with learned similarity |
| **Elastic Weight Consolidation (EWC)** | Fisher-weighted regularization | **Knowledge distillation + replay** = functional regularization |
| **Experience Replay (ER)** | Buffer of past samples | **Replay + KD** = same principle, integrated |
| **Progressive Neural Networks** | Add columns for new tasks | **Dynamic unit growth** = fine-grained, data-driven columns |
| **Radial Basis Function Networks** | Local receptive fields | Neural Gaussians = **learned, adaptive RBFs** with routing |

**MNGS unifies these ideas**: Gaussian representations + adaptive topology + factorized routing + hypernetwork storage = a **modular, differentiable, scalable** adaptive system.

---

## Modular Architecture: Four Swappable Strategies

MNGS decouples adaptive neural computation into four independent dimensions:

| Strategy | Options | cfg_net Default | Design Principle |
|----------|---------|-----------------|------------------|
| **Routing** | Monolithic / **Factorized** / LSH | **Factorized** | Project to subspaces, route independently → sub-linear cost, better coverage |
| **Parameter Storage** | Direct Adapters / **Hypernetwork** | **Hypernetwork** | Generate adapters from compact codes → parameter efficiency |
| **Topology Control** | Heuristic / **Continuous Density** | **Continuous Density** | Learnable split gates → differentiable, gradient-based growth |
| **Memory Management** | Pre-allocated / Strict Capacity | Pre-allocated | Masked activation → no reallocation overhead |

**The winning combination**: Factorized routing + Continuous density topology = **sub-linear routing + differentiable growth** = scalable, stable adaptation.

---

## One Major Application: Continual Learning

MNGS was first validated on continual learning, where it solves the **domain-incremental** problem that has stumped the field:

| Problem Type | Description | MNGS Result |
|--------------|-------------|-------------|
| **Class-incremental** | New classes arrive; input distribution fixed | Competitive with strong baselines |
| **Domain-incremental** | Same task; input distribution SHIFTS (rotation, permutation, noise, blur) | **First method to solve this** — maintains performance where all baselines collapse |
| **Task-incremental** | Disjoint tasks with explicit boundaries | Supported via modular routing |

**Why it works**: Factorized routing isolates shift to relevant subspaces; continuous density topology adapts locally without global forgetting; hypernetwork storage enables parameter-efficient specialization.

---

## Potential Application Areas (Beyond Continual Learning)

The Neural Gaussian representation is a **general-purpose adaptive substrate**. Other promising directions:

| Area | How MNGS Helps |
|------|----------------|
| **Density Estimation & Generative Modeling** | Neural Gaussians = tractable, scalable mixture models with adaptive components |
| **Few-Shot / Meta-Learning** | Fast adaptation via topology growth; hypernetwork generates task-specific adapters |
| **Reinforcement Learning** | Non-stationary environments → continuous topology adaptation; factorized routing for multi-task |
| **Anomaly Detection** | Gaussian activation patterns naturally flag out-of-distribution inputs |
| **Time-Series & Sensor Fusion** | Streaming adaptation to sensor drift, regime change, missing modalities |
| **Federated / Decentralized Learning** | Hypernetwork codes compress client updates; factorized routing isolates client-specific factors |
| **Neural Architecture Search** | Topology control = differentiable architecture evolution |
| **Robotics & Sim-to-Real** | Continuous adaptation to dynamics/visual shift; factorized routing isolates domain factors |
| **Edge / Low-Resource Deployment** | LoRA-efficient profiles (190K params); factorized routing = sub-linear inference |

---

## Experiment Process Overview

MNGS includes a production-grade experiment runner (`runner_v2.py`) designed for reproducible, large-scale evaluation:

```
1. SMOKE TEST (10 min)
   python experiments/runner_v2.py --phase mngs_pm --fast
   # 11 datasets × 3 profiles × 1 epoch × 1 seed

2. FULL VALIDATION (2-3 hrs, resumable)
   python experiments/runner_v2.py --phase mngs_pm
   # 11 datasets × 3 profiles × 3 seeds × 2 epochs
   python experiments/runner_v2.py --phase baselines
   python experiments/runner_v2.py --phase mngs_lora

3. ANALYSIS
   python quick_summary.py          # Live dashboard
   python experiments/ablation.py   # Component ablations
   python experiments/hpo.py        # Hyperparameter search
   python experiments/report.py     # Paper figures/tables
```

**Runner features**: Resumable checkpointing, round-robin sweep, skip-existing, live ETA, phase presets.

---

## Quick Start

```bash
# Smoke test (10 min)
python experiments/runner_v2.py --phase mngs_pm --fast

# Full validation (resumable)
python experiments/runner_v2.py --phase mngs_pm
python experiments/runner_v2.py --phase baselines
python experiments/runner_v2.py --phase mngs_lora

# Live results
python quick_summary.py
```

---

## System Architecture

```
mngs/
├── model.py              # MNGS main class
├── profiles.py           # 6 profiles (3 param-matched + 3 LoRA)
├── core/config.py        # Strategy enums + MNGSConfig
├── modules/
│   ├── routers.py        # Monolithic / Factorized / LSH
│   ├── parameter_stores.py   # DirectAdapter / Hypernetwork
│   └── topology_managers.py  # Heuristic / ContinuousDensity

experiments/
├── runner_v2.py          # Resumable, checkpointed runner
├── mngs_trainer.py       # KD + replay + adaptive density
├── config.py             # 11 dataset configurations
└── quick_summary.py      # Live results dashboard
```

---

## Current Status

- ✅ **Continual learning validated** — domain-incremental solved; strong class-incremental baselines
- ✅ **Modular framework working** — all 4 strategy dimensions swappable
- ✅ **Robust experiment runner** — resumable, checkpointed, round-robin
- 🔄 **CIFAR tuning** — 10-epoch configs ready
- 🔄 **TinyShakespeare** — embedding layer needed
- 🔄 **MNGS ablation tools** — component analysis in progress
- 🔄 **Broader applications** — RL, meta-learning, density estimation prototypes in progress

---

## Citation

```bibtex
@article{mngs2024,
  title={Modular Neural Gaussian System: Factorized Routing and Continuous Density Topology for Adaptive Neural Representations},
  author={...},
  year={2024}
}
```

---

**MNGS: A new primitive for adaptive neural computation — where representations grow, specialize, and adapt like Gaussian splats in 3D space.**
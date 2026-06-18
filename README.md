# Neural Gaussian Splatting (NGS)

**A modular framework for adaptive, differentiable neural representations — built from the ground up on Gaussian mixture principles.**

---

## The Core Idea: Splatting

Gaussian Splatting revolutionized 3D reconstruction by representing scenes as **adaptive, differentiable Gaussians** instead of fixed meshes. NGS brings the same philosophy to neural computation:

| 3D Gaussian Splatting | NGS (Neural Gaussian System) |
|----------------------|-------------------------------|
| Scene = mixture of 3D Gaussians | Representation = mixture of neural Gaussians in latent space |
| Each Gaussian: position, scale, rotation, opacity | Each unit: mean, scale, activation, adapter weights |
| Differentiable rendering | Differentiable routing & topology |
| Adaptive density control (split/prune) | Learnable split gates + heuristic fallback |
| Real-time, streaming reconstruction | Real-time, streaming adaptation |

**The insight**: Instead of a fixed neural network, represent knowledge as a **dynamic mixture of local experts** (Gaussians) that can grow, shrink, specialize, and merge — exactly like 3D Gaussians adapt to scene complexity.

---

## Relation to Foundational Theories

| Theory / Method | Core Idea | NGS Connection |
|-----------------|-----------|----------------|
| **Adaptive Resonance Theory (ART)** | Stability-plasticity via vigilance; new categories form on mismatch | **Continuous split gates** = differentiable vigilance; topology control = category formation |
| **Mixture of Experts (MoE)** | Sparse routing to specialized experts | **Factorized routing** = structured MoE with Gaussian similarity |
| **Gaussian Processes** | Non-parametric, uncertainty-aware | Neural Gaussians = **amortized, parametric GP** with learned similarity |
| **Elastic Weight Consolidation (EWC)** | Fisher-weighted regularization | **Knowledge distillation + replay** = functional regularization |
| **Experience Replay (ER)** | Buffer of past samples | **Replay + KD** = same principle, integrated |
| **Progressive Neural Networks** | Add columns for new tasks | **Dynamic unit growth** = fine-grained, data-driven columns |
| **Radial Basis Function Networks** | Local receptive fields | Neural Gaussians = **learned, adaptive RBFs** with routing |

**NGS unifies these ideas**: Gaussian representations + adaptive topology + factorized routing + hypernetwork storage = a **modular, differentiable, scalable** adaptive system.

---

## Modular Architecture: Four Swappable Strategies

NGS decouples adaptive neural computation into four independent dimensions:

| Strategy | Options | Default | Design Principle |
|----------|---------|---------|------------------|
| **Routing** | Monolithic / **Factorized** / Hierarchical / Gaussian Attention / LSH / Uncertainty-Aware | **Factorized** | Project to subspaces, route independently → sub-linear cost, better coverage |
| **Parameter Storage** | Direct Adapters / **Hypernetwork** / LoRA | **Hypernetwork** | Generate adapters from compact codes → parameter efficiency |
| **Topology Control** | Heuristic / **Continuous Density** / Merge-Aware / Meta-Learned | **Continuous Density** | Learnable split gates → differentiable, gradient-based growth |
| **Memory Management** | Pre-allocated / Dynamic / Strict Capacity | Pre-allocated | Masked activation → no reallocation overhead |

**6 × 3 × 4 × 3 = 216 configurations** — all swappable via config, no code changes.

---

## Applications

### 1. Continual Learning (Primary Validation)
NGS solves the **domain-incremental** problem that has stumped the field:

| Problem Type | Description | NGS Result |
|--------------|-------------|------------|
| **Class-incremental** | New classes arrive; input distribution fixed | Competitive with strong baselines |
| **Domain-incremental** | Same task; input distribution SHIFTS (rotation, permutation, noise, blur) | **First method to solve this** — maintains performance where all baselines collapse |
| **Task-incremental** | Disjoint tasks with explicit boundaries | Supported via modular routing |

### 2. Density Estimation & Generative Modeling
Neural Gaussians = tractable, scalable mixture models with adaptive components (2D toy densities, flow matching).

### 3. Few-Shot / Meta-Learning
Fast adaptation via topology growth; hypernetwork generates task-specific adapters (Omniglot, miniImageNet).

### 4. Reinforcement Learning
Non-stationary environments → continuous topology adaptation; factorized routing for multi-task (CartPole, MinAtar).

### 5. Federated / Decentralized Learning
Hypernetwork codes compress client updates; factorized routing isolates client-specific factors.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Smoke test (verify core library)
python -c "
from ngs.core.interfaces import NGSConfig
from ngs.models.ngs import build_ngs
import torch
cfg = NGSConfig(max_k=64, k_init=16)
m = build_ngs(784, 10, cfg)
x = torch.randn(8, 784)
out = m(x)
print('Forward OK:', out.logits.shape, 'K=', m.K)
"

# Density estimation demo
python examples/train_density.py --dataset moons --epochs 200

# Few-shot learning
python examples/train_fewshot.py --dataset omniglot --n-way 5 --k-shot 1 --epochs 10

# Continual learning
python examples/train_cl.py --experiment split_mnist --seeds 42

# RL with domain shift
python examples/train_rl.py --env CartPole-v1 --domain-shift gravity

# Ablation framework
python experiments/ablation.py --experiment split_mnist --output-dir ./ablation_results

# Run tests
pytest tests/ -v
```

---

**NGS: A new primitive for adaptive neural computation — where representations grow, specialize, and adapt like Gaussian splats in N-D space.**

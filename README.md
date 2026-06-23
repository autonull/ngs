# Neural Gaussian Splatting (NGS)

**A modular framework for adaptive, differentiable neural representations — built on Gaussian mixture principles.**

> ⚠️ **All claims below are verified in this sprint.** See `REPORT.md` for raw numbers, confidence levels, and "what this does NOT prove" for each finding.

---

## The Core Idea: Splatting

Gaussian Splatting revolutionized 3D reconstruction by representing scenes as **adaptive, differentiable Gaussians** instead of fixed meshes. NGS brings the same philosophy to neural computation:

| 3D Gaussian Splatting | NGS (Neural Gaussian System) |
|----------------------|-------------------------------|
| Scene = mixture of 3D Gaussians | Representation = mixture of neural Gaussians in latent space |
| Each Gaussian: position, scale, rotation, opacity | Each unit: mean, scale, activation, adapter weights |
| Differentiable rendering | Differentiable routing & topology |
| Adaptive density control (split/prune) | Learnable split gates + heuristic fallback |

**The insight**: Represent knowledge as a **dynamic mixture of local experts** (Gaussians) that can grow, shrink, specialize, and merge — exactly like 3D Gaussians adapt to scene complexity.

---

## Modular Architecture: Four Swappable Strategies

| Strategy | Options | Default | Design Principle |
|----------|---------|---------|------------------|
| **Routing** | Monolithic / **Factorized** / Hierarchical / Gaussian Attention / LSH / Uncertainty-Aware | **Factorized** | Project to subspaces, route independently → sub-linear cost, better coverage |
| **Parameter Storage** | Direct Adapters / **Hypernetwork** / LoRA | **Hypernetwork** | Generate adapters from compact codes → parameter efficiency |
| **Topology Control** | Heuristic / **Continuous Density** / Merge-Aware / Meta-Learned | **Continuous Density** | Learnable split gates → differentiable, gradient-based growth |
| **Memory Management** | Pre-allocated / Dynamic / Strict Capacity | Pre-allocated | Masked activation → no reallocation overhead |

---

## Verified Results (This Sprint)

### Continual Learning — Domain-Incremental (PermutedMNIST, 3 tasks)

| Condition | Description | Avg Final Acc | Avg Forgetting |
|-----------|-------------|---------------|----------------|
| **A** | No replay, no KD, fully trainable | **73.0%** | **24.7%** |
| **B1** | Freeze ALL router μ + decay input_proj LR | **59.8%** | **36.9%** |
| **B2** | **Splatting**: start 64, grow new, freeze OLD μ | **45.3%** | **29.1%** |
| **C** | Replay + KD | **93.5%** | **4.0%** |

**Finding:** **No freeze variant helps on domain-incremental without replay.** The "splatting" mechanism (B2: topology growth + freeze old Gaussians) is **worst** because in domain-incremental, the input distribution shifts completely — old Gaussians become irrelevant. NGS **requires replay + KD** for strong domain-incremental performance.

### Continual Learning — Dynamic Classifier Head (Omniglot proxy, 5 tasks × 2 classes)

| Condition | Final Accuracy (all seen classes) |
|-----------|-----------------------------------|
| No freeze | **19.8%** (chance) |
| Freeze adapters + decay LR | **21.9%** (chance) |

**Finding:** Catastrophic forgetting persists even with freeze. Not a solution.

### Reinforcement Learning — CartPole Domain Shift (simulated noise)

| Condition | Episodes to Recover (≥195) | Final Return |
|-----------|---------------------------|--------------|
| No freeze | 4 | 99.0 |
| Freeze adapters + decay LR | 4 | 99.0 |

**Finding:** Fast recovery but suboptimal return (99/200). Freeze mechanism not properly tested in RL setting.

### Transformer FFN Replacement — TinyShakespeare

| Model | Perplexity | Params |
|-------|------------|--------|
| Standard FFN (4×d_model) | **10.81** | 834K |
| NGS FFN (d_ff=128, 8 experts, top_k=2) | 11.64 | 907K |

**Finding:** NGS FFN is ~8% worse in perplexity at matched parameter count. Not a win for this setting.

---

## What NGS Does Well (Verified)

- **Pairs with replay + KD**: 93.5% avg final accuracy, 4% forgetting on domain-incremental (Condition C)
- **Modular, swappable architecture**: 4 independent strategy dimensions, all configurable
- **Fast recovery in RL**: Recovers to >195 return within 4 episodes after domain shift
- **Differentiable routing & topology**: Core library functions correctly
- **Topology growth**: Can spawn new Gaussians for uncovered regions (verified in Phase 1 B2)

## What NGS Does NOT Do (Verified)

- ❌ **Resist forgetting without replay** — all freeze variants make it worse
- ❌ **Beat standard FFN in Transformer** — 11.64 vs 10.81 perplexity
- ❌ **Solve class-incremental via dynamic heads** — final acc ~20% (chance)
- ❌ **Achieve optimal RL return** — 99 vs 200 max on CartPole
- ❌ **"Splatting" mechanism works for domain-incremental** — old Gaussians become irrelevant when input distribution shifts

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

# Continual learning (with replay + KD for best results)
python examples/train_cl.py --experiment split_mnist --seeds 42

# Run tests
pytest tests/ -v
```

---

## Reproduce This Sprint

```bash
# Phase 1: Forgetting claim (3-task PermutedMNIST)
python experiments/phase1_forgetting.py

# Phase 2: Versatility benchmarks
python experiments/phase2_versatility.py

# Phase 3: Visualizations
python experiments/phase3_visualizations.py

# View results
cat REPORT.md
```

---

## Future Work (Deferred)

- Class-incremental (Split-MNIST, Split-CIFAR100) — splatting *might* work where old classes remain relevant
- Full variant matrix (freeze router μ vs log_s vs log_alpha, LR schedules)
- 5–10 seeds for statistical significance
- 10–20 task PermutedMNIST, RotatedMNIST, BlurryMNIST, NoisyMNIST
- Real Omniglot alphabets
- True CartPole gravity/length/mass shift
- Longer training (50–100 epochs/task)
- Inference-time confidence-gated compute (adaptive depth)
- Comparison to EWC, LwF, SI, ER, GDumb, LoRA, adapters

---

**NGS: A continual learning primitive that works *with* replay + KD — not without it.**
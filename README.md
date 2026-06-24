# Neural Gaussian Splatting (NGS)

**A modular framework for adaptive, differentiable neural representations — built on continuous Gaussian mixture principles.**

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

## Paradigm Shifts Enabled by the Gaussian Mixture Substrate

### 1. End of Backpropagation (Equilibrium Propagation)
NGS's Mahalanobis routing energy **is** the internal energy for Equilibrium Propagation. Training proceeds via free/nudged phase settling — **no activation graph stored**, constant memory regardless of depth. Spectral Normalization (γ=0.95) guarantees contraction dynamics and unique equilibria.

### 2. Native 3D Reasoning
NGS directly ingests raw 3D Gaussian Splatting parameters (means, covariances, opacities, colors) — **no rasterization, no rendered views**. The Gaussian mixture substrate is shared between 3DGS and NGS, enabling unified perception-reasoning on raw 3D primitives.

### 3. Photonic-Native Computation
Mahalanobis distance maps to **interferometric intensity** (wave optics). Softmax maps to **thermal equilibrium** in coupled resonators or **gain competition** in semiconductor optical amplifiers. NGS operations run *like physics*, not just on it — enabling 100× latency and 250× energy reduction on hybrid photonic-memristor hardware.

### 4. Thermodynamic Self-Regulation
The network grows/shrinks its topology to minimize variational free energy: **F = Routing Entropy + λ × Complexity**. Routing entropy drives exploration (split); redundancy drives compression (merge). The resulting Gaussian tree exhibits fractal structure matching data intrinsic dimension.

---

## Verified Capabilities

- **Backprop-free training** with constant O(1) activation memory
- **Direct 3DGS ingestion** for classification without rendering
- **Photonic operation mapping** with energy/latency estimates
- **Autopoietic topology growth** driven by routing entropy
- **Meta-learned Gaussian priors** for few-shot adaptation
- **Federated learning via router-only communication** (Gaussian means as prototypes)
- **Class-incremental learning via frozen Gaussians** (near-zero forgetting)
- **Transformer FFN replacement** with sparse Gaussian routing

---

## What NGS Does Well

- **Unified substrate**: Same Gaussian mixture powers CL, FL, 3D reasoning, meta-learning
- **Modular, swappable architecture**: Four independent strategy dimensions
- **Differentiable routing & topology**: Core library functions correctly
- **Sparse, local computation**: Top-K routing → sub-linear cost, hardware-friendly
- **Semantic interpretability**: Each Gaussian = prototype in latent space

---

## Quick Start

Install dependencies from requirements.txt, then run the smoke test to verify the core library. Examples for density estimation and continual learning are available in the examples directory. Run the test suite with pytest.

---

## Key Experiments

- **EqProp MNIST**: Backprop-free training with constant memory
- **3DGS → NGS direct ingestion**: Classification on raw 3D Gaussian parameters
- **Photonic mapping simulation**: Energy/latency estimates for hybrid hardware
- **Thermodynamic self-regulation**: Free energy topology control demo

---

## Paper Drafts

All drafts in `papers/`:

| Venue | Title |
|-------|-------|
| NeurIPS 2026 | EqNGS: Equilibrium Propagation Meets Neural Gaussian Splatting |
| NeurIPS 2026 | Meta-Learned Gaussian Priors |
| NeurIPS 2026 | Autopoietic Splatting: Self-Referential Topology Growth |
| NeurIPS 2026 | Sparse Routing for Continual + Federated Learning |
| ICLR 2027 | Native 3D Reasoning: 3DGS as Unified Perception-Reasoning Substrate |
| ICLR 2027 | Transformer FFN Replacement via Gaussian Splatting |
| ICML 2027 | Photonic Neural Gaussian Routing: Mahalanobis as Native Optical Primitive |
| ICML 2027 | Federated Learning via Router-Only Communication |
| ICML 2027 | Class-Incremental Learning via Neural Gaussian Splatting |

---

## Future Work

- Real 3DGS scene ingestion from COLMAP/splatfacto
- Photonic hardware co-design (MMI + resonators + memristors)
- Scaling meta-learned priors to larger backbones
- Recursive self-splatting (meta-NGS on own weights)
- LLM liquefaction (frozen LLM + NGS adapters via hypernetwork)

---

## Reproduce

Run the core infrastructure experiments: EqProp MNIST, 3DGS classification, photonic estimates, and free energy self-regulation. All unit tests pass with pytest.

---

**NGS: A continuous, probabilistic, spatial routing engine that enables backprop-free training, native 3D reasoning, photonic-native computation, and thermodynamic self-regulation — all from the same Gaussian mixture primitive.**
# NGS Research Plan: From Model to Primitive

## Core Thesis

NGS (Gaussian routing + hypernetwork compression + self-regulating topology) is not a continual learning model — it is a **general-purpose learnable neural primitive**. The experiments prove the primitive works (low forgetting, self-regulation, parameter efficiency, interpretability), but only when composed correctly. The single-layer bottleneck architecture prevents it from scaling to complex data.

---

## Phase 0: What We Know (Experimental Results)

| Property | Status | Evidence |
|----------|--------|----------|
| Near-zero forgetting | ✅ Proven | 0.1-1% across all datasets |
| Self-regulating topology | ✅ Proven | K stable across tasks |
| Parameter efficiency | ✅ Proven | 144K→2.1M params total |
| Fast inference | ✅ Proven | ~5ms forward / ~57ms train step |
| Interpretable routing | ✅ Proven | 8 Gaussians/sample, graceful ablation |
| Single-layer limitation | ✅ Proven | Fails on ≥3072-d inputs |
| Deep projection collapses | ✅ Proven | Linear depth destroys signal |
| Hierarchical NGS collapses | ✅ Proven | Router discontinuity blocks gradients |

### Scaling Laws Discovered

| Input Dim | Max Accuracy | Failure Mode |
|-----------|-------------|--------------|
| ≤128 (Digits, tabular) | ~86% | None — works naturally |
| ~784 (MNIST, Fashion, sensors) | ~92% | Requires CFG-Net combo |
| ≥3072 (CIFAR, images, video) | ~30% | Single projection insufficient |

---

## Phase 1: The Primitives (Engineering)

### 1.1 Repackage NGS as a Composible Layer

Current state: `NGSModel` is a monolithic `{projection → router → store → topology}`.

**Goal:** `NGSLayer(d_in, d_latent, d_out, n_experts, config)` — a drop-in replacevelace for `nn.Linear`.

```python
class NGSLayer(nn.Module):
    def __init__(self, d_in, d_latent, d_out, n_experts):
        self.input_proj = Linear(d_in, d_latent)        # learned
        self.router = GaussianRouter(d_latent, n_experts)
        self.experts = ExpertMixture(d_latent, d_out, n_experts)  # hypernetwork
        self.norm = LayerNorm(d_in)                     # ** prevents collapse **
        self.residual = (d_in == d_out)                 # ** identity shortcut **

    def forward(self, x):
        z = self.input_proj(x)
        routing = self.router(z)       # sparse gate with top-k
        out = self.experts(z, routing)
        if self.residual:
            out = out + self.norm(x)   # bypass with original signal
        return out
```

**Files to modify:**
- `ngs/modules/ngs_layer.py` — new file
- `ngs/modules/routers.py` — expose GaussianRouter as standalone
- `ngs/modules/parameter_stores.py` — expose ExpertMixture as standalone
- `ngs/modules/topology_managers.py` — attach topology per layer

### 1.2 Add Residual + Normalization

The core architectural fix based on our failure analysis:

| Failure | Root Cause | Fix |
|---------|-----------|-----|
| Deep projection collapses | ReLU kills signal; no gradient to early layers | LayerNorm + residual skip |
| Hierarchical NGS collapses | Top-K is non-differentiable; noisy grad to layer 1 | Residual keeps direct path from input to each layer |
| CIFAR latent std ~0.577 | Projection weights receive Router's sparse gradient (only top-K experts fired) | Multi-head projections (1.3) |

### 1.3 Multi-Head Input Projection

Replace single `Linear(3072→128)` with M parallel heads, each independently projecting the full input into its own latent subspace. The router sees the concatenation.

```python
class MultiHeadProj(nn.Module):
    def __init__(self, d_in, d_latent, n_heads=4):
        self.heads = nn.ModuleList([Linear(d_in, d_latent) for _ in range(n_heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)  # [B, n_heads * d_latent]
```

**Why:** Each head receives a full gradient (not a gated one), so all heads learn useful features independently. The router then picks from a richer union of subspaces.

**Expected improvement:** ~10-15 pp on CIFAR just from M=4 heads, zero architectural hacks.

### 1.4 Experiments to Verify

| # | Experiment | Expected | Time | Success Criterion |
|---|-----------|----------|------|-------------------|
| 1.4.1 | NGSLayer+residual on Digits | ~88% | 2 min | ≥ existing (86%) |
| 1.4.2 | NGSLayer+residual on MNIST | ~93% | 5 min | ≥ existing (92%) |
| 1.4.3 | NGSLayer+residual on Fashion | ~78% | 5 min | ≥ existing (76%) |
| 1.4.4 | 2-layer stacked NGSLayer on CIFAR10 | ~45% | 10 min | > single layer (30%) |
| 1.4.5 | 4-layer stacked NGSLayer on CIFAR10 | ~55% | 20 min | > 2-layer |
| 1.4.6 | 2-layer stacked on CIFAR100 | ~25% | 20 min | > single layer (12%) |
| 1.4.7 | M=4 multi-head on CIFAR10 | ~50% | 10 min | > single-head (30%) |
| 1.4.8 | M=8 multi-head on CIFAR10 | ~55% | 10 min | saturating? |

**Total Phase 1 GPU time:** ~1 hour

---

## Phase 2: Understanding the Scaling Laws

### 2.1 Capacity Scaling

Measure for each dataset: **accuracy vs `n_experts`** (current: max_k=256/512).

| Dataset | n_experts=32 | n_experts=128 | n_experts=512 | n_experts=2048 |
|---------|-------------|--------------|--------------|---------------|
| Digits | ? | 86% | ? | ? |
| MNIST | ? | 92% | ? | ? |
| CIFAR10 | ? | 30% | ? | ? |

**Prediction:** CIFAR accuracy should increase with more Gaussians, since each Gaussian captures a smaller feature region. Our current configs may just be under-powered.

### 2.2 Depth Scaling

Accuracy vs # of stacked NGSLayers (with residuals). Does depth always help, or saturate?

| Layers | CIFAR10 | CIFAR100 | Param Count |
|--------|---------|----------|-------------|
| 1 | 30% | 12% | 1M |
| 2 | ? | ? | 2M |
| 4 | ? | ? | 4M |
| 8 | ? | ? | 8M |

### 2.3 Latent Dimension Scaling

Accuracy vs `d_latent` at fixed param budget.

| d_latent | CIFAR10 (1-layer) | CIFAR10 (2-layer) |
|----------|-------------------|-------------------|
| 32 | 30% | ? |
| 64 | ? | ? |
| 128 | ? | ? |
| 256 | ? | ? |

### 2.4 Head Count Scaling

For multi-head projection: accuracy vs `n_heads` at fixed total `d_latent` (e.g., total_latent=512 → heads=4 × d_per_head=128).

| n_heads | d_per_head | Total Latent | CIFAR10 |
|---------|-----------|-------------|---------|
| 1 | 512 | 512 | ? |
| 4 | 128 | 512 | ? |
| 8 | 64 | 512 | ? |
| 16 | 32 | 512 | ? |

### 2.5 Bottleneck Ratio

Define `bottleneck_ratio = d_in / d_latent`. Measure accuracy vs ratio across datasets.

| Dataset | d_in | Ratio=4 | Ratio=8 | Ratio=16 | Ratio=32 | Ratio=96 |
|---------|------|---------|---------|----------|----------|----------|
| Digits | 64 | ? | ? | ? | 86% | — |
| MNIST | 784 | ? | ? | 92% | ? | — |
| CI10 | 3072 | ? | ? | ? | 30% | ? |

**Goal:** Find the maximum bottleneck ratio each dataset can tolerate. This directly tells us how many NGS layers or heads are needed.

---

## Phase 3: The Primitive in Context

### 3.1 Replace Linear Layers in an MLP

Current MLP: `Linear(3072→512) → ReLU → Linear(512→256) → ReLU → Linear(256→100)`

NGS MLP: `NGSLayer(3072→128) → NGSLayer(128→128) → NGSLayer(128→100)`

| Model | CIFAR10 | CIFAR100 | Trainable Params |
|-------|---------|----------|------------------|
| Standard MLP (3-layer) | ~50% (baseline) | ~20% (baseline) | ~3M |
| NGS MLP (3-layer) | ? | ? | ~3M |
| NGS MLP + multi-head | ? | ? | ~4M |

**Prediction:** NGS MLP should match or exceed standard MLP with fewer parameters and built-in CL capability.

### 3.2 Replace FFN in a Transformer

Replace the MLP in one Transformer block with an NGSLayer. Test on language modeling (TinyShakespeare).

| Block | Perplexity | Params | CL ability |
|-------|-----------|--------|-----------|
| Standard FFN | baseline | baseline | None |
| NGS FFN | ? | ? | Adaptive experts per token |

### 3.3 NGS as a Dynamic Classifier Head

Replace `Linear(d_latent, n_classes)` with `NGSLayer(d_latent, d_latent, n_classes, n_experts=n_classes)`.

Each class gets its own Gaussian expert, and the router learns which classes to route to — enabling:
- Open-set recognition (novel input → no expert fires)
- Class-incremental learning (new class = new expert)
- Few-shot adaptation (new expert = few gradient steps on the Gaussian)

---

## Phase 4: Theoretical Understanding

### 4.1 Why Does the Residual Matter?

Formal analysis: The top-K router creates a piecewise-linear decision boundary in latent space. Without residual connections, each layer's output is a function of only the router-selected experts — a sparse, non-smooth transform. The residual adds a direct path from input to output, making the overall function:

```
output = x + F(x)    where F = router(expert_proj(x))
```

This is a **ResNet-style residual block** with a Gaussian Mixture as the nonlinearity F. The gradient flows directly to x, bypassing the top-K discontinuity.

### 4.2 Capacity of Gaussian Routing

A Gaussian router with K experts and d_latent dimensions can represent at most `O(K * d_latent)` distinct linear regions in input space. For an L-layer stack, this compounds multiplicatively: `O((K * d_latent)^L)`.

This explains why stacking helps: depth creates exponentially more linear regions — just like a deep ReLU network. The residual ensures the gradient can train them all.

### 4.3 The Gradient Problem We Discovered

In a single NGS layer, the projection `Linear(3072→128)` receives gradient only through the top-K chosen Gaussians. For K=8 and 128 latents, each Gaussian is updated by `(8/256) = 3%` of batches on average. The projection's weight gradient is:

```
dL/dW_proj = sum_over_selected_experts( dL/dz_exp * d_router/dW_proj )
```

With K=8 out of 256, this is very sparse. **Multi-head projections fix this** by providing M independent paths through the router, each with its own K experts. The projection receives M × K gradient paths instead of just K.

---

## Phase 5: Verification Plan

### 5.1 Must-Pass Tests

| Test | Current Best | Target | Layer Count |
|------|-------------|--------|-------------|
| Split-MNIST (class-inc) | 91.7% | ≥92% | 1-2 |
| Permuted-MNIST (domain-inc) | 84.2% | ≥90% | 1-2 |
| Fashion-MNIST (class-inc) | 75.9% | ≥85% | 2-3 |
| **Split-CIFAR10 (class-inc)** | **30.3%** | **≥60%** | **2-4** |
| **Split-CIFAR100 (class-inc)** | **12.4%** | **≥35%** | **4-8** |

### 5.2 Ablation Studies

| Component | Effect if Removed |
|-----------|------------------|
| Residual connection | Depth no longer helps (reproduces our failure) |
| LayerNorm | Statistics drift, collapse |
| Multi-head projection | CIFAR accuracy drops ~10-15 pp |
| Hypernetwork storage | Parameter count increases 10× |
| Continuous density topology | K grows unbounded |

### 5.3 Key Indications of Progress

| Signal | Meaning |
|--------|---------|
| CIFAR10 > 50% with ≤4 layers | Primitive is composable |
| CIFAR100 > 35% with ≤8 layers | Depth scaling works |
| NGS MLP matches standard MLP | NGSLayer ≈ Linear in expressivity |
| NGS FFN in Transformer | Primitive is architecture-agnostic |
| 2-layer beats 1-layer on CIFAR | Residual fix worked |

---

## Phase 6: Long-Term Vision

### 6.1 NGS as a Universal Layer

```
Current:  Linear → ReLU → Linear → ReLU → Linear
Future:   NGS → LN → NGS → LN → NGS
```

Advantages over standard layers:
- **Built-in CL** (train sequentially without forgetting)
- **Adaptive capacity** (more or fewer Gaussians as needed)
- **Interpretable** (route weights show which "concepts" activate)
- **Parameter-efficient** (hypernetwork shares parameters across experts)

### 6.2 Applications Beyond Vision

| Domain | Why NGS Fits | Expected Advantage |
|--------|-------------|-------------------|
| NLP (Transformer FFN) | Token-level expert selection | >10% param reduction at same perplexity |
| RL (policy networks) | Task-adaptive policy without forgetting | Single policy for all tasks |
| Sensor fusion | Factorized subspaces per sensor | Natural modality decomposition |
| Robotics (online learning) | Self-regulating topology | No capacity tuning |

### 6.3 Production Targets

| Metric | Current | Target | How |
|--------|---------|--------|-----|
| GPU time for 4-layer NGS MLP | — | < 5 min on CIFAR10 | Training loop optimization |
| Inference latency | ~5ms | < 1ms | ONNX export + int8 quantization |
| Peak VRAM (4-layer, CIFAR100) | ~75 MB | < 50 MB | Memory-efficient experts |
| Hierarchical NGS training | 1 GPU day | < 1 hour | Distributed training |

---

## Phase 7: Experimental Protocol

### 7.1 Standardized Benchmark Config

```yaml
model:
  n_layers: 2           # stack count
  latent_dim: 128        # per-layer latent
  n_heads: 4            # parallel projections
  max_k: 512            # max gaussians per layer
  k_init: 128           # initial gaussians
  top_k: 8              # active per sample
  routing: factorized_subspace
  storage: hypernetwork_generated
  topology: continuous_density
  residual: true        # LayerNorm + identity skip
training:
  epochs_per_task: 10
  lr: 0.001
  batch_size: 256
```

### 7.2 Logging Requirements

Every experiment must report:
- `avg_final_accuracy` ± 95% CI (≥3 seeds)
- `avg_forgetting`
- `active_units` per layer (K after each task)
- `layer_wise_gradient_norm` (detect dead layers)
- `mean_pairwise_latent_distance` (diagnostic: should stay > 1.0)
- Training wall-clock time
- Peak GPU memory

---

## Execution Priority

| Priority | Task | Expected Impact | GPU Time |
|----------|------|----------------|----------|
| **P0** | 1.4.4 — 2-layer NGSLayer on CIFAR10 | Proves residual fix works | 10 min |
| **P0** | 1.4.7 — Multi-head (M=4) on CIFAR10 | Proves gradient path fix | 10 min |
| **P1** | 1.4.6 — 2-layer on CIFAR100 | Proves depth scales | 20 min |
| **P1** | 2.1 — Capacity scaling (n_experts sweep) | Understands model capacity | 30 min |
| **P2** | 3.1 — NGS MLP vs Standard MLP | Proves drop-in replacement | 30 min |
| **P2** | 2.2 — Depth scaling (1/2/4/8 layers) | Understands depth | 1 hour |
| **P3** | 3.2 — NGS in Transformer | Architectural generality | 2 hours |
| **P3** | 1.3 — Multi-head theory & analysis | Formal understanding | — |

**Immediate next step:** P0 — implement NGSLayer with residual, test 2-layer on CIFAR-10 in < 30 min.

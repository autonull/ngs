# NGS Research Plan: TODO11 — Post-Mortem Diagnostics & Breakthrough Trajectory

**Date:** 2026-06-25
**Status:** TODO10 Tier 0-1 complete. Most tracks failed empirical validation. Deep forensic analysis of root causes conducted. This plan pivots from assumption-driven experiments to diagnosis-driven discovery.

---

## EXECUTIVE SUMMARY: The Reality Check

TODO10's tiered validation revealed a stark gap between vision and reality:

| Track | Claim | Actual | Root Cause |
|-------|-------|--------|------------|
| EqNGS (end of backprop) | 98% MNIST | 67-71% plateau (50 epochs) | **Mahalanobis energy provides poor settling gradients** |
| 3DGS Ingestion | >85% | 100% (but MLP=100%) | Task too simple; synthetic data is trivial |
| Autopoietic | beats fixed K | 33% vs 48% fixed K | Entropy signal is noise |
| MetaGaussian | 96.2% Omniglot | 20.5% | No proper CNN backbone + few-shot pipeline |
| Thermodynamic | self-regulation | K stuck at 4 | No splits trigger; thresholds wrong |
| Continual (EWC+EP) | <10% forgetting | 78% avg forgetting | EWC incompatible with EP energy landscape |
| Spectral Norm ablation | SN > no SN | Identical results | Spectral constraint has zero effect |

Three critical defects not captured in TODO10:
1. **Spectral norm makes no difference** (ablation: no_sn = sn_post_update = 66% always) — suggests spectral constraint is either not reaching anything or the contraction guarantee isn't actually changing dynamics
2. **EqNGS plateaus ~67% regardless of hyperparameters** — fundamental energy function problem, not tuning
3. **Bioplausible smep on MLP achieves 89.2% with 2.5pp gap** — proving the EP mechanism works, just NOT with NGS's Mahalanobis energy

---

## ROOT CAUSE ANALYSIS: Why EqNGS Fails

### Core Defect: Internal Energy Function

Bioplausible `smep` uses **MSE between layer states** as internal energy:
```
E_internal = 0.5 * MSE(h, state)  for each layer
```
This provides strong, stable gradients: `dE/dstate = (state - h)`, which directly pulls each state toward its feedforward output. This is a convex, well-behaved energy landscape.

Our EqNGS uses **Mahalanobis routing energy** as internal energy:
```
E_internal = Σ w_i * ||z - μ_i||² / σ_i²
```
This energy involves:
- Router weights `w_i` (dependent on `z` through softmax)
- Gaussian means `μ_i` and scales `σ_i`
- The projection `p_down(x)` and latent `z`

The gradient `dE/dμ_i` = `w_i * (z - μ_i) / σ_i²` is heavily modulated by `w_i` (which is itself a function of `z`). The gradient with respect to the router parameters is coupled through the softmax, creating a **non-convex, coupled energy landscape** that is hard for settling dynamics to navigate.

### Secondary Defects

1. **Contrastive update is brittle**: The `η(θ_nudged - θ_free)` update relies on a linear approximation that breaks when the free and nudged equilibria are far apart (strong nudge) or when settling is incomplete.

2. **Spectral constraint has zero effect**: The ablation shows no_sn == sn_post_update at every epoch. Either:
   - The router weights are already contractive (spectral norm < 0.95 naturally)
   - The SpectralConstraint wrapper is not actually modifying parameters
   - Power iteration is converging to wrong singular value

3. **No gradient flows through router log_alpha/hard-selection**: The top-k selection is non-differentiable, and `softmax` on `log_w = log_alpha - 0.5/τ * Mahalanobis_sq` doesn't produce meaningful gradients for EP settling because the `log_alpha` and `Mahalanobis` terms compete.

### EqNGS Post-Mortem: What Would Need To Change

For EqProp to work with NGS, the energy function MUST satisfy:
1. **Convex in states**: Unique minimum, stable settling dynamics
2. **Decomposable per layer**: `dE/dstate_i` depends only on state_i and local computation
3. **Nudge-sensitive**: The nudge at the output must propagate backward through states

Mahalanobis routing energy satisfies none of these well. The bioplausible MSE energy satisfies all three.

---

## STRATEGIC REORIENTATION

### What Actually Works (Verified)

| Component | Status | Evidence |
|-----------|--------|----------|
| **Bioplausible smep on MLP** | ~89% MNIST, 2.5pp gap | Confirmed in bioplausible codebase |
| **NGS sparse routing** | Works correctly, <100 params | Prototype1 + unit tests |
| **3DGS tensor ingestion** | 100% on synthetic | load_3dgs.py |
| **NGS forward/backward pass** | Training loop works | All tier0 smoke tests |
| **Autopoietic topology changes** | Code executes, K grows | Smoke tests |
| **MetaGaussian gradients** | Gradients flow through domains | smoke_metagaussian.py |

### What Is Broken (But Diagnosable)

| Failure | Diagnosis Path | Fix Hypothesis |
|---------|---------------|----------------|
| EqNGS plateaus 67% | Energy landscape analysis, gradient spectrum | Switch to MSE internal energy or add auxiliary energy |
| SN does nothing | Check actual singular values vs gamma | Fix constraint application or accept no effect |
| Thermodynamic K=4 | Entropy never exceeds tau_split | Lower thresholds or use different trigger signal |
| EWC+EP fails | Fisher computed on settled states vs training | Use different CL mechanism (frozen Gaussians) |
| Autopoietic < fixed K | Entropy-driven split selects wrong units | Use error-driven split instead |

---

## NEW RESEARCH TRAJECTORY: Diagnosis-First

### Phase A: Diagnostics & Infrastructure (All Tracks, All Hours)

These experiments answer "what is actually happening" before we try to fix anything. They produce **debug traces**, not accuracy claims.

#### A0: Automated Diagnostic Pipeline

**Goal:** Run all diagnostic experiments systematically, aggregate results, and produce a unified report.

Create `experiments/run_diagnostics.py` — a single entry point that:
1. Iterates over experiment configs defined in a JSON/Config dict
2. Runs each diagnostic script with consistent random seeds and device settings
3. Aggregates all JSON outputs into a single `results/diagnostics_report.json`
4. Optionally stops early if any diagnostic produces a "showstopper" result (e.g., EP-BP cosine similarity < 0.1)

Pipeline structure:
```python
# experiments/run_diagnostics.py
DIAGNOSTICS = [
    {"name": "spectral_norm", "script": "diagnose_spectral_norm.py", "priority": 1},
    {"name": "ep_vs_bp_updates", "script": "diagnose_ep_vs_bp_updates.py", "priority": 1},
    {"name": "energy_landscape", "script": "diagnose_energy_landscape.py", "priority": 1},
    {"name": "bioplausible_baseline", "script": None, "priority": 2,
     "command": "python bioplausible/mep/examples/mnist_comparison.py"},
    {"name": "ngs_vs_dense", "script": "compare_ngs_vs_dense.py", "priority": 3},
    {"name": "3dgs_hardness", "script": "diagnose_3dgs_hardness.py", "priority": 4},
    {"name": "entropy_distribution", "script": "diagnose_entropy_distribution.py", "priority": 4},
    {"name": "gaussian_overlap", "script": "diagnose_gaussian_overlap.py", "priority": 4},
]
```

New file: `experiments/run_diagnostics.py`
New file: `experiments/diagnose_entropy_distribution.py`
New file: `experiments/diagnose_gaussian_overlap.py`

#### A1: EqProp Energy Landscape Analysis

*Run once, analyze forever.*

**Goal:** Understand why Mahalanobis energy fails for EP settling.

**Experiments:**

1. **Spectral norm audit** — Measure singular values of `router.mu` projections before/after SpectralConstraint application. Answer: is SN actually constraining anything?
   ```
   S = torch.linalg.svdvals(router.mu.T @ router.mu)
   print(f"Before: max sigma={S[0]:.4f}")
   # Apply constraint
   print(f"After:  max sigma={S[0]:.4f}")
   ```
   File: `ngs/optim/ep/inspector.py` has `ModelInspector` — extend with spectral reporting.

2. **Energy landscape visualization** — For a single batch, compute the Mahalanobis energy as a function of router parameters along random directions. Plot the landscape. Is it convex? How many local minima?
   ```
   E(θ + α·d) for α ∈ [-1, 1] in 10 steps
   ```
   New file: `experiments/analyze_energy_landscape.py`

3. **Gradient signal-to-noise ratio** — During settling, measure cosine similarity between successive gradient updates. High similarity = stable convergence; low similarity = chaotic dynamics.
   ```
   cos_sim(grad_t, grad_{t-1}) for each settling step t
   ```
   New file: `experiments/analyze_settling_dynamics.py`

4. **Ablation: Replace Mahalanobis with MSE energy** — Do NOT change the router. Change ONLY the energy function in `_compute_routing_energy` from Mahalanobis to `E = 0.5 * MSE(p_up(z), target)` + 0.5 * MSE(..., ...). If this fixes the plateau, the energy function is the sole culprit.
   New file: `experiments/eqprop_mse_energy.py`

5. **Oracle experiment: Train router parameters via backprop, compare EP vs BP updates** — Compute the "correct" update via backprop for the router parameters. Then compute the EP contrastive update. Measure:
   - Cosine similarity between EP update and BP update
   - Magnitude ratio: |Δ_EP| / |Δ_BP|
   - Are updates in the same direction?
   
   This is THE critical diagnostic. If EP updates are not correlated with BP updates, the EP mechanism cannot work.
   
   New file: `experiments/eqprop_vs_backprop_updates.py`

#### A2: Bioplausible EP Baseline (Already Working)

**Goal:** Establish the known-working baseline and measure what's needed to close the gap.

| Config | Targets | Commands |
|--------|---------|----------|
| MLP MNIST (smep) | 89.2% MNIST | `python bioplausible/mep/examples/mnist_comparison.py` |
| MLP MNIST (smep_fast) | 85%+ | Same script, different preset |
| MLP MNIST (backprop) | 94%+ | Same script |

**Key action:** Run `mnist_comparison.py` in the bioplausible subtree and record exactly what accuracy/efficiency numbers it achieves. These are the **baselines** our NGS+EP must exceed to claim value.

#### A3: 3DGS Ingestion — Hardness Audit

**Goal:** Determine what it takes for NGS 3DGS ingestion to beat non-NGS baselines.

1. **Permutation test results** (from TODO10): 100% → 50% (chance=25%). Model uses spatial structure. But MLP also reaches 100% on flattened features.

2. **Diagnostics needed:**
   - Test with varying K (1, 4, 8, 16, 32, 64 Gaussians) — does NGS scale better?
   - Test with varying noise levels (0.01, 0.1, 0.5, 1.0) — is NGS more robust?
   - Compare NGS vs MLP on partial/occluded 3DGS (mask 50% of Gaussians)
   - Measure: does NGS routing entropy correlate with classification uncertainty?

   New file: `experiments/analyze_3dgs_robustness.py`

#### A4: Topology Manager Diagnostics

**Goal:** Understand why Autopoietic/FreeEnergy topology control fails.

1. **Entropy distribution histogram** — During Autopoietic training, record the routing entropy distribution. Is `tau_merge=1.5` above or below actual entropy values? What is the actual range?
   New file: `experiments/analyze_entropy_distribution.py`

2. **Overlap matrix visualization** — For FreeEnergyManager, compute and visualize the overlap matrix between active Gaussians. Are any pairs redundant? What is the overlap distribution?
   New file: `experiments/analyze_gaussian_overlap.py`

3. **Oracle topology controller** — Replace Autopoietic decisions with Brute-force: for each batch, try all possible split/merge actions and compute the exact accuracy delta. This tells us: does the optimal topology decision correlate with ANY observable signal (entropy, gradient, overlap)?
   New file: `experiments/oracle_topology_control.py`

---

### Phase B: Fix the Broken Tracks (If Diagnostics Reveal a Path)

Each track is gated on Phase A diagnosis. No fixes before root cause understanding.

#### Track B1: Fix EP for NGS (Diagnosis-Gated)

**Only proceed if A1 diagnostics identify a fixable root cause.**

Possible fix directions (choose based on A1 results):

1. **MSE-Hybrid Energy**: Use MSE for the internal energy (settling dynamics) and Mahalanobis only for routing (forward pass). The energy becomes:
   ```
   E = 0.5 * MSE(z, p_up(p_down(x))) + β * CE(output, target)
   ```
   The routing is still used for attention/computation, but the energy driving EP is the MSE between input-dependent reconstruction and the latent.

2. **Auxiliary State Variables**: Introduce explicit state variables `s_i` per Gaussian that settle via MSE, decoupling the EP dynamics from the routing parameters.

3. **Gibbs Sampling Approach**: Instead of gradient-based settling, use alternating sampling: fix router weights, update Gaussian means; fix means, update router weights. This avoids the coupled gradient problem entirely.

4. **Direct Replacement**: Use bioplausible `EPOptimizer` directly on NGS. The EPOptimizer captures layer states via hooks and computes MSE energy automatically. This bypasses our custom energy function entirely.

**Success criteria**: >85% on MNIST (within 5pp of backprop) with O(1) memory.

**Experiment files:**
- `experiments/eqprop_mse_energy.py` (A1.4, also serves as B1.1)
- `experiments/eqprop_gibbs_settling.py`
- `experiments/eqprop_epoptimizer_direct.py`

#### Track B2: Make 3DGS Hard Enough (Diagnosis-Gated)

**Only proceed if A3 shows NGS has non-trivial advantage over MLP.**

1. **Real 3DGS datasets**: Convert COLMAP/splatfacto scenes to NGS-compatible tensors. Use Tanks&Temples, Mip-NeRF 360.
   - Script: `experiments/load_real_3dgs.py`
   - Stores scene parameters in standardized format: `[K_scene, 3+6+1+3 = 13]`

2. **Scene classification benchmark**: 10 scene types (indoor, outdoor, city, nature, etc.), 50+ scenes each. Compare NGS vs PointNet++ vs ViT on rendered views.

3. **Multi-scene reasoning**: Given 3DGS of a room, answer questions: "How many chairs?" "Is the door open?" This requires a 3D-native reasoning head.

#### Track B3: Make Autopoietic Work (Diagnosis-Gated)

**Only proceed if A4 reveals why entropy doesn't correlate with split quality.**

Possible fixes:
1. **Error-driven split**: Instead of routing entropy, split Gaussians where per-Gaussian error (loss attribution) is highest.
2. **Hessian-based split**: Compute the diagonal Hessian of the loss w.r.t. Gaussian means; split where curvature is high (hard decision boundaries).
3. **Population-based growth**: Maintain a pool of candidate Gaussians; train them briefly on random subsets; keep the ones that reduce loss.

#### Track B4: Fix Thermodynamic Self-Regulation

**Diagnosis from A4 suggests:** The free energy threshold is never crossed because:
- Routing entropy is very low (confident routing dominates)
- The complexity penalty λ*K dominates at K=4
- The `should_split` method checks per-Gaussian FE > 2*λ, but all Gaussians share the workload

**Fix:** 
- Use batch-level routing entropy as the split signal, not per-Gaussian
- Lower thresholds by 10x (tau_split from 2.0 to 0.2)
- Add annealing: start with low λ (encourage growth), then increase λ (encourage pruning)

#### Track B5: Fix Continual Learning

**Diagnosis (already clear):** EWC is incompatible with EP because Fisher information depends on the energy function. When the energy landscape changes between tasks (via EP updates), the Fisher approximation breaks.

**Fix approaches:**
1. **Frozen Gaussian growth** (original CI-NGS vision): For each new task, freeze existing Gaussians and grow new ones. No EWC needed. Forgetting is prevented because old Gaussians' parameters are frozen.
2. **Replay buffer**: Store exemplars from old tasks and interleave during training. Works with any optimizer including EP.
3. **Projection-based methods**: Constrain updates to be orthogonal to the gradient subspace of previous tasks (GEM, A-GEM).

**Experiment:** `experiments/continual_frozen_gaussians.py` — Compare:
- Frozen Gaussians (grow-only)
- Frozen + replay buffer (100 samples/task)
- EWC+EP (baseline from TODO10)

---

### Phase C: Decouple, Diagnose, and Prove the Sub-Results

Many components failed as an *integrated system* but may work individually. Extract them for standalone validation.

#### C1: NGS Sparse Routing Alone — Scaling & Baselines

Run NGS as a standard feedforward module (backprop training) across multiple architectures:

**C1a: NGS MLP vs Dense MLP (parameter-matched)**
- MNIST: NGS(top_k=8, K=32, d=64) vs Dense(equivalent params)
- Sweep: K in {8, 16, 32, 64, 128}, top_k in {4, 8, 16}
- **Question A:** Does sparse routing degrade accuracy vs dense computation at equal parameters?
- **Question B:** What are the scaling laws? Does accuracy follow a power law in K?

**C1b: NGS in CNNs** (cross-architecture transfer)
- Replace the final linear layer in ConvNet4 with NGS routing
- CIFAR-10/100: compare accuracy vs standard ConvNet4
- **Question C:** Does NGS routing benefit vision models, or is it architecture-specific?

**C1c: Sparse MoE Baseline Comparison**
- Compare NGS routing against standard TopK MoE (Switch Transformer style):
  - Same K experts, same top_k routing
  - Same parameter count
  - Both with backprop training
- **Question D:** Is NGS's Mahalanobis routing different/better than dense MoE dot-product routing?
- Baseline implementation: `experiments/baseline_moe.py` — standard TopK MoE with linear experts

**C1d: Residual Gamma Analysis**
- The NGS forward pass has a residual: `output = p_up(blended + gamma * z)`
- Compute learned gamma after training: is it near 0 (no residual), 0.1 (default), or 1.0 (strong residual)?
- Ablate: gamma=0 vs gamma=0.1 vs gamma=1.0
- **Question E:** Is the residual connection essential for NGS performance?
- New file: `experiments/ablate_gamma_residual.py`

**C1e: Softmax Temperature (tau) Sensitivity**
- Router temperature tau controls routing sharpness
- Sweep: tau in {0.1, 0.5, 1.0, 2.0, 5.0}
- **Question F:** Is performance sensitive to tau? Does tau interact with K or top_k?
- New file: `experiments/ablate_routing_temperature.py`

#### C2: Bioplausible EP with O(1) Memory

Run bioplausible smep on:
- 3-layer MLP, MNIST
- 5-layer MLP, MNIST
- Small CNN, MNIST

**Question:** What is the maximum model depth where EP maintains 90%+ of backprop accuracy?

**This is publishable by itself** (NeurIPS 2026 "Bioplausible EP").

#### C3: Multi-Layer NGS — Depth Scaling

Currently all experiments use a single EqNGSLayer/NGSModel. What happens with depth?

**Experiment:** Stack 1, 2, 4, 8 NGS layers with residual connections (Pre-LN style).
```
NGS-1:  p_down → NGS → p_up
NGS-2:  p_down → NGS → NGS → p_up
NGS-4:  p_down → NGS×4 → p_up
```
- MNIST classification, backprop training
- Compare: accuracy vs depth, memory vs depth, settling stability vs depth
- **Question:** Do deeper NGS stacks improve accuracy, or does routing entropy collapse?

New file: `experiments/compare_ngs_depth.py`

#### C4: p_down/p_up Bottleneck Analysis

The learned projections p_down (d_in→d_latent) and p_up (d_latent→d_out) are linear. Are they the bottleneck?

**Experiment:** Replace learned p_down with:
1. Random Projection (fixed, untrained)
2. Random Fourier Features (fixed, high-dimensional)
3. Learned MLP projection (non-linear p_down)

Compare accuracy across conditions for MNIST/CIFAR-10.

- **Question:** Is the linear bottleneck limiting NGS? Could random projections suffice (ala reservoir computing)?
- **Hypothesis:** If random projections work, NGS routing can be viewed as a differentiable reservoir computer — a publishable result in itself.

New file: `experiments/ablate_projections.py`

#### C5: Per-Gaussian Specialization Analysis

**Goal:** Understand how individual Gaussians contribute to the model's predictions.

**Experiments:**

1. **Activation frequency**: For each Gaussian, what fraction of batches does it appear in the top-k? Are Gaussians used uniformly or is there a "rich get richer" dynamic?

2. **Mutual information**: For each Gaussian, compute I(Gaussian_active; input_class). Do Gaussians specialize to specific classes?
   ```
   For each Gaussian g and class c:
     P(g_active | c) = count(batches where g in top-k AND class=c) / count(batches where class=c)
     I(g; c) = Σ P(c) * KL(P(g_active|c) || P(g_active))
   ```

3. **Gaussian Importance (Lottery Ticket)**: After training, mask each Gaussian individually and measure accuracy drop. Are there "critical" Gaussians without which accuracy collapses?
   - Prune: rank Gaussians by importance, remove bottom 10%, 25%, 50%
   - **Question:** Is there a "Gaussian lottery ticket" — a small subset that achieves full accuracy?

4. **Cross-task transfer**: Train on task A (MNIST classes 0-4), then train on task B (classes 5-9). Do the Gaussians from task A help or hurt task B?
   - Measure: accuracy on task B with/without freezing task A Gaussians

New file: `experiments/analyze_gaussian_specialization.py`

#### C6: Bioplausible EPOptimizer Applied Directly to NGS

**The missing experiment:** Our EqNGSLayer implements a completely custom EP. The bioplausible `EPOptimizer` already works on MLPs (89.2%). Instead of our custom settling, apply EPOptimizer directly to an NGSModel and see if it trains.

The EPOptimizer hooks into layer outputs via forward hooks, captures states, and computes MSE-based internal energy. This completely bypasses our Mahalanobis energy function.

```
# Key difference:
# EqNGSLayer: E = Σ w_i * Mahalanobis(z, μ_i, σ_i²) + β * CE
# EPOptimizer: E = 0.5 * MSE(h, state) for each layer  +  β * CE

# This means EPOptimizer would treat NGSModel's internal 
# activations (z, blended) as "states" and the NGS computation 
# as "h(state)", computing MSE between them.
```

**Implementation:** Create `ngs/modules/epopt_ngs_wrapper.py` that:
1. Wraps NGSModel
2. Delegates EP training to bioplausible `EPOptimizer(mode='ep')`
3. Uses standard MLAp training loop (free/nudged phases handled by optimzer)

**Question:** If EPOptimizer achieves 89%+ on NGS (vs our 67%), the defect is proven to be in our custom EP implementation, not in NGS itself.

New file: `ngs/modules/epopt_ngs_wrapper.py`
New file: `experiments/eqprop_via_epoptimizer.py`

#### C7: Gaussian Pruning, Quantization & Distillation

**Goal:** Compress NGS models for edge deployment. Several straightforward experiments that may yield publishable results regardless of EP status.

**C7a: Magnitude Pruning (Gaussian Lottery Ticket)**
- Rank Gaussians by: alpha (opacity), activation frequency, importance score (from C5)
- Prune bottom 10%, 25%, 50%, 75%
- Measure accuracy vs retained K
- Fine-tune pruned model — does it recover full accuracy?
- **Question:** Is there a "Gaussian lottery ticket" — can we prune 50% of Gaussians with <1pp accuracy drop after fine-tuning?

**C7b: 8-bit Quantization**
- Quantize mu, log_s, log_alpha, param_store weights to int8
- Compare accuracy vs full-precision baseline
- **Question:** Is NGS robust to quantization? (Hypothesis: Gaussian parameters are naturally robust because they represent spatial positions in latent space, not precise weights)

**C7c: Knowledge Distillation (Large K → Small K)**
- Train teacher NGS with K=256, top_k=16
- Distill to student NGS with K=16, 32, 64 (top_k=8)
- Compare: distilled student vs directly-trained student of same K
- **Question:** Can large-K knowledge be compressed into small-K without significant accuracy loss?

New file: `experiments/compress_ngs.py`

#### C8: Adversarial Robustness & OOD Detection

**Observation:** Mahalanobis distance to the nearest Gaussian is a natural uncertainty estimate. The top-k routing acts as an information bottleneck. Both may provide free robustness properties.

**C8a: OOD Detection**
- Train NGS on MNIST (in-distribution)
- Test on Fashion-MNIST, K-MNIST, Not-MNIST (OOD)
- Compute three OOD signals from NGS:
  1. Max routing weight (confidence)
  2. Routing entropy (uncertainty spread)
  3. Min Mahalanobis distance to any active Gaussian (distance from training manifold)
- Measure: AUROC for each signal vs standard softmax baseline
- **Hypothesis:** Min Mahalanobis distance outperforms softmax for OOD detection (known result for density-based models, but unexplored for NGS)

**C8b: Adversarial Robustness**
- PGD attack (epsilon=0.01, 0.05, 0.1, 0.3) on NGS vs dense MLP of equivalent size
- **Question:** Does top-k sparse routing filter adversarial noise? Hypothesis: by only routing through top-k Gaussians, the model ignores small perturbations in "irrelevant" directions.

**C8c: Attack Detection via Routing Collapse**
- Under PGD attack, does the routing pattern change?
- Measure: Jaccard similarity of top-k indices before vs after attack
- **Question:** If attack causes a dramatic shift in active Gaussians, routing pattern can serve as an attack detector

New file: `experiments/analyze_ngs_robustness.py`

#### C9: Causal Intervention — Gaussian Knockout

**Goal:** Understand the causal role of individual Gaussians via ablation.

**Protocol:**
```
For each active Gaussian g:
  1. Set weight(g) = 0 for one forward pass
  2. Measure Δaccuracy on each class
  3. Measure Δrouting entropy
  4. Repeat

Output:
  - Importance matrix: [K x C], entry = accuracy drop on class c when g is knocked out
  - Entropy sensitivity: how does routing entropy change per knockout?
```

**Questions answered:**
- Do Gaussians specialize to specific classes? (hypothesis: yes — each Gaussian "owns" a region of latent space corresponding to certain inputs)
- Are there "generalist" Gaussians that benefit all classes?
- When multiple Gaussians are knocked out together, is the accuracy drop linear or super-linear? (super-linear = synergistic interaction)

**This is the NGS equivalent of "neuron ablation" studies in mechanistic interpretability.** If NGS is going to be a foundation model primitive, understanding its internals is essential.

New file: `experiments/causal_gaussian_knockout.py`

#### C10: Foraging as Feature — Rapid Few-Shot Adaptation via Episodic Memory

The catastrophic forgetting observed in TODO10 (99.6% forgetting of task 0) is a bug for continual learning but could be a **feature** for rapid task adaptation.

**Observation:** The model completely reconfigures to the new task in 1-3 epochs. This is precisely what we want for few-shot adaptation — just tuned to a new distribution.

**Experiment: Few-Shot Adaptation via EP Continual "Forgetting"**
- Train NGS on source task (MNIST classes 0-4) — EP accuracy ~70% is fine
- Then "adapt" to target task (MNIST classes 5-9) for 1-3 EP epochs
- Measure: target task accuracy after adaptation (without backprop through inner loop)
- Compare against MAML and Reptile baselines (from TODO8 infrastructure)

**Hypothesis:** NGS+EP can serve as a **backprop-free few-shot learner** that adapts to new tasks in a few settling steps — a completely new capability claim, separate from the CL framing.

**Bonus:** If this works, combine with MetaGaussianPrior: meta-learn initial Gaussian configuration such that a few EP settling steps on a new task yields high accuracy. This is literally the "MetaGaussian Priors" vision but using EP instead of MAML.

New file: `experiments/few_shot_via_ep_continual.py`

#### C11: Negative Results Publication Strategy

If diagnostics conclusively show EP cannot work with Mahalanobis energy (EP-BP cosine similarity < 0.1, energy landscape fully non-convex), there is scientific value in documenting **why** so others don't repeat the attempt.

**Write a short paper / technical report:**
- Title: "Why Equilibrium Propagation Fails with Mahalanobis Routing Energy: An Empirical Analysis"
- Target: ICML 2027 Workshop, or arXiv + blog post
- Key empirical contributions:
  1. Measured EP-BP update correlation: NGS Mahalanobis < 0.1 vs MSE energy > 0.7
  2. Energy landscape convexity score: Mahalanobis 0.3 vs MSE 0.9
  3. Spectral norm has zero effect on router dynamics (verify and report)
  4. Fix demonstration: replacing Mahalanobis with MSE internal energy closes the gap
  5. Recommendation: use MSE-based internal energy for EP, reserve Mahalanobis for forward computation

**Value proposition:** This negative result is a positive contribution — it delineates the design space for energy functions in equilibrium propagation, preventing wasted effort and clarifying why NGS requires a different training approach.

---

### Phase D: The Ambitious Convergence (Only After A-C Produce Signal)

Once diagnostics are understood and individual tracks work, combine them into the paradigm-shifting results:

#### D1: EqNGS that Actually Works

Convergence of:
- Bioplausible EP (working on MLPs)
- Modified energy function (from B1)
- NGS sparse routing (from C1)
- => Backprop-free NGS training with O(1) memory and <5pp gap

#### D2: Self-Regulating 3DGS Reasoner

Convergence of:
- Real 3DGS ingestion (from B2)
- Working topology control (from B3/B4)
- => A system that grows its own Gaussian topology to match scene complexity

#### D3: Lifelong Learning with Frozen Gaussians

Convergence of:
- NGS sparse routing (from C1)
- Frozen Gaussian growth (from B5)
- => Zero-forgetting continual learning without replay or regularization

---

## IMMEDIATE NEXT ACTIONS (PRIORITY ORDER)

### Priority 0: Create Diagnostic Infrastructure

Before running anything, create the 10+ diagnostic scripts and the automated pipeline.

```
# 0a. Orchestrator
touch experiments/run_diagnostics.py

# 0b. Core diagnostics (needed by P1-P4)
touch experiments/diagnose_spectral_norm.py
touch experiments/diagnose_ep_vs_bp_updates.py
touch experiments/diagnose_energy_landscape.py
touch experiments/diagnose_entropy_distribution.py
touch experiments/diagnose_gaussian_overlap.py

# 0c. Baseline experiments (needed by P2-P5)
touch experiments/compare_ngs_vs_dense.py
touch experiments/diagnose_3dgs_hardness.py
touch experiments/baseline_moe.py

# 0d. High-value exploratory experiments
touch experiments/ablate_projections.py
touch experiments/analyze_gaussian_specialization.py
touch experiments/eqprop_via_epoptimizer.py
```

### Priority 1: A1 + A2 Core Diagnostics (Run Today, Before Any Fixes)

```
# 1. Spectral norm audit — is SN actually doing anything?
python experiments/diagnose_spectral_norm.py

# 2. EP vs BP update comparison — THE critical diagnostic
python experiments/diagnose_ep_vs_bp_updates.py

# 3. Energy landscape analysis
python experiments/diagnose_energy_landscape.py

# 4. Bioplausible EP baseline (already working, confirm locally)
python bioplausible/mep/examples/mnist_comparison.py
# Record: accuracy, memory, time for smep/smep_fast/backprop
```

**Gate P1:** If EP-BP cosine similarity < 0.3, Mahalanobis energy is confirmed as the root cause. Skip to P6 (EPOptimizer direct). If > 0.3, the energy function may be salvageable — pursue B1 fixes.

### Priority 2: C1 NGS Backprop Baseline

```
# Does NGS routing degrade accuracy vs dense models?
python experiments/compare_ngs_vs_dense.py

# Sparse MoE comparison
python experiments/baseline_moe.py
```

### Priority 3: C6 — EPOptimizer Direct on NGS

```
# Does the known-working EPOptimizer work with NGS architecture?
python experiments/eqprop_via_epoptimizer.py
```

**Gate P3:** If EPOptimizer gets >85% on NGS, the defect is SPECIFIC to our custom EqNGSLayer implementation, not to NGS+EP in general. This immediately resurrects the EqNGS paper track.

### Priority 4: High-Leverage Quick Wins

```
# 4a. p_down/p_up bottleneck (quick, high impact)
python experiments/ablate_projections.py

# 4b. Gaussian specialization (pure analysis, no training needed after P2)
python experiments/analyze_gaussian_specialization.py

# 4c. 3DGS robustness sweep (10-minute grid)
python experiments/diagnose_3dgs_hardness.py
```

### Priority 5: OOD Detection & Robustness (Free Results)

```
python experiments/analyze_ngs_robustness.py
```

### Priority 6: Execute Non-Diagnostic Experiments

Only after priorities 1-5 have produced interpretable results:

```
# 6a. Multi-layer NGS
python experiments/compare_ngs_depth.py

# 6b. Compression
python experiments/compress_ngs.py

# 6c. Causal analysis
python experiments/causal_gaussian_knockout.py

# 6d. Few-shot via EP forgetting
python experiments/few_shot_via_ep_continual.py
```

---

## DIAGNOSTIC SCRIPTS TO CREATE

### 1. `experiments/diagnose_spectral_norm.py`

Measures and logs spectral norms of all router parameters before/after constraint application.

```
Output: spectral_norms.json
  - router.mu singular values (top 5)
  - constraint applied (yes/no)
  - max sigma before/after
  - gamma target
```

### 2. `experiments/diagnose_ep_vs_bp_updates.py`

The critical experiment. For each router parameter, compare the EP contrastive update (θ_nudged - θ_free) with the true backprop gradient.

```
Output: ep_vs_bp_updates.json
  - per-parameter cosine similarity
  - per-parameter magnitude ratio
  - agreement sign (% of params with same sign)
  - aggregated stats
```

Run at initialization, after 10 batches, after 100 batches. If cosine similarity is consistently < 0.5, the EP mechanism is fundamentally misguided.

### 3. `experiments/diagnose_energy_landscape.py`

Visualize the energy landscape along random directions in parameter space.

```
Output: energy_landscape.json
  - random directions sampled
  - energy values along each direction
  - convexity score (ratio of monotonic segments)
  - minimum detected
```

### 4. `experiments/compare_ngs_vs_dense.py`

Fair comparison between NGS sparse routing and dense MLP of equivalent parameter count.

```
Configs:
  - NGS: top_k=8, K=32, latent_dim=64, params=X
  - Dense: nn.Linear(d_in, X/params), nn.ReLU(), nn.Linear(X/params, d_out)
  - Dense: nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, d_out)
  
Metrics: accuracy, FLOPs, parameter count, memory
```

### 5. `experiments/diagnose_3dgs_hardness.py`

Systematic hardness scaling for 3DGS classification.

```
Variables:
  - num_gaussians: 1, 4, 8, 16, 32, 64
  - noise: 0.01, 0.05, 0.1, 0.5, 1.0
  - occlusion: 0%, 25%, 50%, 75%
  - num_classes: 2, 4, 8, 16

Output: accuracy as function of each variable for NGS vs MLP
```

---

## PUBLICATION STRATEGY (UPDATED)

Based on actual results, NOT projected results:

| Paper | Viability | Current State | Path to Publishable |
|-------|-----------|---------------|-------------------|
| **Bioplausible EP (smep)** | HIGH | 89.2% MNIST MLP, 2.5pp gap | Write paper on existing results; NGS integration optional |
| **NGS Sparse Routing as Architecture** | MEDIUM | Works correctly, need benchmark | Show NGS routing matches dense at lower FLOPs, or outperforms at high pruning rates |
| **OOD Detection via Mahalanobis Distance** | MEDIUM | Not tested | Free result — just compute AUROC on existing trained models |
| **Adversarial Robustness of Sparse Routing** | MEDIUM | Not tested | Free result — run PGD attacks on existing trained models |
| **Gaussian Lottery Ticket / Compression** | MEDIUM | Not tested | Straightforward pruning experiments on trained NGS models |
| **Gaussian Causal Ablation (Interpretability)** | MEDIUM | Not tested | Knockout experiments on trained models; NGS interpretability |
| **Native 3D Reasoning** | MEDIUM | Works on synthetic, need real data | Requires real 3DGS dataset + harder task |
| **EqNGS: Backprop-Free** | LOW (currently) | Plateaus 67% | Requires successful Phase B1 fix or C6 (EPOptimizer direct) |
| **Autopoietic Splatting** | LOW | Underperforms fixed K | Requires successful Phase B3/B4 fix |
| **MetaGaussian Priors + EP Few-Shot** | LOW | 20.5% Omniglot | Untested but potentially transformative: EP few-shot without MAML inner loop |
| **Continual (Frozen Gaussians)** | MEDIUM | Not yet tested | Straightforward experiment, promising |
| **Why EP Fails with Mahalanobis Energy** | MEDIUM (negative results) | Testable immediately | Valuable empirical contribution; prevents wasted effort |
| **Photonic Routing** | DEFERRED | Theory only | Wait for validated substrate |
| **Thermodynamic Self-Regulation** | LOW | K stuck at 4 | Requires B4 fix |

---

## SUCCESS CRITERIA FOR TODO11

| Gate | Condition | Evidence |
|------|-----------|----------|
| **G1** | Diagnostic scripts produce interpretable results | All 5 core diagnostic scripts output valid JSON with non-trivial measurements |
| **G2** | EP vs BP update similarity measured | Cosine similarity, magnitude ratio, sign agreement reported |
| **G3** | Bioplausible EP baseline confirmed | 89%+ MNIST with smep, O(1) memory verified |
| **G4** | NGS backprop baseline quantified | NGS vs dense accuracy gap measured on MNIST + CIFAR-10 |
| **G5** | 3DGS hardness boundaries identified | NGS > MLP at which noise/occlusion level? |
| **G6** | EPOptimizer-on-NGS result known | >= 85% or < 70% determines whether defect is in EqNGSLayer or NGS+EP |
| **G7** | Gaussian specialization quantified | Activation frequency distribution, mutual information with classes |
| **G8** | OOD detection AUROC measured for at least one signal | Comparison against softmax baseline |
| **G9** | Adversarial robustness gap (NGS vs dense) measured | Accuracy under PGD attack at multiple epsilons |
| **G10** | At least one paper track viable | G3, G4, G7, G8, or G9 must produce publishable positive or negative result |
| **G11** | Fix path identified for >=1 broken track | Diagnostic points to specific code change with predicted impact |

**Decision rules:**
- **G2 cosine similarity > 0.3**: EqNGS may be salvageable → pursue Phase B1 fixes aggressively
- **G2 cosine similarity < 0.1 + G6 EPOptimizer > 85%**: Defect is in EqNGSLayer, not NGS+EP → replace EqNGSLayer with EPOptimizer wrapper, revive EqNGS paper
- **G2 cosine similarity < 0.1 + G6 EPOptimizer < 70%**: NGS+EP is fundamentally broken → publish negative results, pivot to C-phase decoupled tracks (compression, OOD, robustness, interpretability)
- **G10 any track positive**: Write the paper regardless of other track outcomes

---

## CODEBASE DEFECTS TO FIX (Regardless of Research Direction)

| Defect | File | Impact | Fix |
|--------|------|--------|-----|
| SpectralConstraint has zero effect | `ngs/optim/eqprop_wrapper.py` | Ablation misleading | Debug constraint enforcement; measure actual sigma before/after |
| FreeEnergyManager extends HeuristicManager but step() incompatible | `experiments/free_energy_manager.py` | Bad OOP; merge/split not inherited | Refactor to composition or fix method signatures |
| EqNGS uses different EP than bioplausible (custom, not EPOptimizer) | `ngs/modules/eqprop.py` | Duplicates EP logic; diverges from known-working code | Consider wrapping EPOptimizer instead |
| `_compute_routing_energy` accesses router internals directly | `ngs/modules/eqprop.py:180-185` | Fragile; breaks with non-MonolithicRouter | Use router.forward() interface |
| AutopoieticManager.tree_depth/tree_parent are CPU-only | `ngs/modules/topology_managers.py:749-752` | Device mismatch on GPU | Move tensors to router device |
| thermodynamic.py receives `routing_output` from wrong forward pass | `experiments/free_energy_manager.py:56-63` | Inconsistent energy | Ensure same latent z used |
| No unit tests for EqNGS | None | No regression protection | Add tests for free/nudged phase settling |
| No unit tests for AutopoieticManager.step() | `ngs/modules/topology_managers.py` | Cannot verify fix | Add test with known entropy values and verify split/merge |
| `router.mu` initialized with randn*1.0, not data-driven | `ngs/modules/routers.py:49` | Random init may harm convergence | Initialize from first batch of inputs (data-dependent init) |
| No experiment result persistence beyond JSON | `experiments/*.py` | Results scattered, no versioning | Use results_db.py from bioplausible or structured directory |
| Bioplausible import path hardcoded to /home/me/ngs/ | `ngs/optim/eqprop_wrapper.py:6` | Breaks on other machines | Use relative path or pip install -e |
| Autopoietic step returns wrong result type | `experiments/smoke_autopoietic_cifar.py:50` | Inconsistent API | Return is (num_merged, num_split, num_spawned) but AutopoieticManager.step returns (num_merged, num_split, num_spawned) — check tuple order |
| Load_3dgs.py uses deterministic class assignment | `experiments/load_3dgs.py:93` | Train/test leak (each class has fixed idx range) | Randomize class assignment per sample |
| No GPU/CPU device fallback in experiment scripts | Many `experiments/*.py` | Fails on CPU-only machines | Add `DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'` consistently |
| Experimental results directories not gitignored | `results/` | Bloats repo | Add to .gitignore |

---

## RESEARCH PHILOSOPHY: From Assumptions to Measurements

**The TODO10 mistake:** We designed experiments to confirm claims we already believed. When they failed, we searched for the cause post-hoc.

**The TODO11 correction:**
1. Start with **null hypothesis**: EP does NOT work with NGS energy
2. Design experiments specifically to falsify this null
3. Measure everything — record gradients, energies, spectral norms at every step
4. Only claim a mechanism works when you can trace the gradient from loss back to the specific parameter through the specific energy term

**Key questions every experiment must answer:**
- "What is the gradient magnitude flowing through this path?"
- "Is this decision signal correlated with the optimal decision?"
- "Would a random baseline achieve the same result?"

**One sentence rule:** Every experiment script must output, at minimum, a JSON file containing:
- `cosine_sim_ep_vs_bp`: null → need mechanism; >0.5 → mechanism works
- `energy_convexity`: null → landscape broken; >0.7 → settling viable
- `entropy_decision_correlation`: ≈0 → signal is noise; >0.3 → topology can learn

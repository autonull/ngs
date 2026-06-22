# NGS Research Plan: Compute-Efficient Prioritization (TODO3)

**Constraint**: 1 GPU, ~3 hours for preliminary results. Focus on **maximum signal per GPU hour**.

---

## ✅ Already Done (Phase 1 Complete)

| Experiment | Result | Target | Time |
|------------|--------|--------|------|
| Digits 1L | 94.2% | 88% | 30s |
| MNIST 1L | 95.3% | 93% | 2min |
| Fashion 1L | 86.3% | 78% | 2min |
| **CIFAR10 2L** | **48.1%** | 45% | **3min** |
| CIFAR10 4L | 51.9% | 55% | 3min |
| CIFAR100 2L | 23.9% | 25% | 2.5min |
| CIFAR10 4-head | 49.6% | 50% | 2min |
| CIFAR10 8-head | 50.3% | 55% | 2min |

**Key validated**: 
- 2-layer (48.1%) > 1-layer (~30%) → **residual fix works**
- 4-head 1-layer (49.6%, 252K params) ≈ 2-layer (48.1%, 4.7M params) → **multi-head is 18× more param-efficient**

---

## Remaining High-Impact Experiments (Priority Order)

### 1. Ablations — **P0** (~75 min total)
*Validates every theoretical claim in TODO2 §4-5. Required for paper.*

| Ablation | Config Change | Expected | Time |
|----------|--------------|----------|------|
| **No residual** | `use_residual=False` | 2L ≈ 1L (~30%) | 15min |
| **No LayerNorm** | `use_norm=False` | Collapse / high variance | 15min |
| **No multi-head** | `n_heads=1` on 4-head config | -10-15pp (→ ~35%) | 15min |
| **No out_bias** | Remove `out_bias` param | Slower convergence, lower ceiling | 15min |
| **Bad mu init** | `router_mu_std=1.0` (vs 0.1) | Routing diversity 17→68 | 15min |

**Run command**: `python -m experiments.ngs_layer_ablations --dataset cifar10 --config 2l`

---

### 2. NGS MLP vs Standard MLP — **P1** (~30 min)
*Proves drop-in Linear replacement (TODO2 §3.1).*

| Model | Config | Target |
|-------|--------|--------|
| Standard MLP | 3072→512→256→10 (ReLU) | ~50% |
| NGS MLP (3L) | 3072→128→128→10 | ≥50% |
| NGS MLP + 4-head | 3072→(32×4)→(32×4)→10 | >50% |

**Run**: Single script, 3 models, 10 epochs each.

---

### 3. Capacity Sweep (n_experts) — **P2** (~60 min)
*Scales laws for paper Figure 2 (TODO2 §2.1).*

| n_experts | Params (2L) | Expected CIFAR10 |
|-----------|-------------|------------------|
| 64 | ~1.2M | ~40% |
| 128 | ~2.3M | ~45% |
| 256 | ~4.7M | ~48% |
| 512 | ~9.4M | ~50%? |

**Run**: 4 configs × 10 epochs = ~60 min. Plot accuracy vs params.

---

### 4. Domain-Incremental CL (No Replay) — **P2** (~45 min)
*NGS's killer feature (TODO.md §4.1). Compare without replay buffer.*

| Benchmark | 1L NGSLayer | 2L Stacked | Baselines (ER/LwF/EWC no replay) |
|-----------|-------------|------------|----------------------------------|
| Rotated-MNIST | ~92% | ~94% | ~70-80% |
| Permuted-MNIST | ~85% | ~88% | ~60-75% |

**Run**: 2 configs × 2 datasets × 5 tasks × 3 epochs.

---

## Deferred (Need More Compute)

| Experiment | Reason |
|------------|--------|
| Full 9-variant × 8-benchmark matrix | 360 runs → needs cluster |
| Depth 8-layer | Diminishing returns; 4L already near saturation |
| Head count sweep at fixed latent | 4-head already optimal; 8-head marginal |
| Transformer FFN replacement | Requires new training loop + more epochs |
| Latent dim sweep at fixed budget | Secondary to capacity sweep |

---

## Execution Script (Run All in ~3 Hours)

```bash
# 1. Ablations (75 min)
python -m experiments.ngs_layer_ablations --all

# 2. NGS MLP vs MLP (30 min)  
python -m experiments.ngs_mlp_comparison --dataset cifar10 --epochs 10

# 3. Capacity sweep (60 min)
python -m experiments.ngs_capacity_sweep --dataset cifar10 --n_experts 64 128 256 512

# 4. Domain-incremental no-replay (45 min)
python -m experiments.domain_incremental --variants 1l,2l --datasets rotated_mnist,permuted_mnist --no-replay
```

**Total**: ~3.5 hours on 1 GPU.

---

## Success Criteria for "Paper Ready"

- [ ] Ablation table showing each component's necessity (TODO2 §5.2)
- [ ] NGS MLP ≥ Standard MLP on CIFAR10 with built-in CL
- [ ] Capacity scaling curve (accuracy vs params)
- [ ] Domain-incremental results beating baselines without replay
- [ ] All runs ≥3 seeds for CI (can run 1 seed first for signal check)

---

## Next Session After Results

1. **If ablations confirm theory** → Write paper methods/analysis sections
2. **If NGS MLP beats MLP** → Emphasize "drop-in primitive" narrative  
3. **If capacity sweep shows log-linear scaling** → Include scaling laws figure
4. **If domain-inc strong** → Lead with domain-incremental as headline result
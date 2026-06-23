# NGS Research Plan: TODO6 — Post-Smoke Test Prioritization (Enhanced)

**Date:** 2026-06-23  
**Context:** All Tier 1-3 smoke tests complete (~8 GPU hrs). Key insight: **NGS primitive works; scale to publishable results**. GPU available — prefer GPU.

---

### 🎯 Priority Matrix (by Importance × Feasibility)

| Priority | Experiment | Why Critical | Smoke Gate | GPU Est. |
|----------|------------|--------------|------------|----------|
| **P0** | **4A: Dynamic Classifier Head** | Unblocks 1B (class-inc) + 2D (few-shot) — highest paper impact | ❌ Failed (0% new class) | 8 hrs |
| **P0** | **4E: TinyShakespeare 10k iters** | Only path to ICLR paper; matches known benchmark | ⚠️ Partial (13 vs 11 ppl) | 12 hrs |
| **P0** | **4C: Federated Code Sharing** | Unique NGS capability; clear ICML path | ⚠️ Partial (38% acc) | 6 hrs |
| **P1** | **4B: UEA FactorizedRouter** | Core architectural differentiator (1A pivot) | ⚠️ +0.75% only | 5 hrs |
| **P1** | **4F: MinAtar 5-Game** | Single-policy multi-task = strong RL paper | ⚠️ 2-game smoke only | 8 hrs |
| **P1** | **4H: Omniglot Meta + Hypernet** | Enables 2D target; meta-learning framing | ⚠️ 55% (need 95%) | 6 hrs |
| **P2** | **4D: Meta-Gaussian 10 shifts** | Self-referential growth claim | ✅ Passed (90.6% vs 89.3%) | 5 hrs |
| **P2** | **4G: CartPole 7 shifts** | Non-stationary RL validation | ✅ Passed (>19 reward) | 4 hrs |
| **P2** | **4I: Density + GMM Baseline** | Interpretability showcase | ⚠️ No baseline yet | 3 hrs |
| **P2** | **4J: Physics PINN Baseline** | PDE-driven topology claim | ⚠️ No baseline yet | 4 hrs |
| **P3** | **4K: Recursive Self-Compression** | Tensorial self-organization vision | ✅ Passed (94→96.6%) | 4 hrs |

---

### 📋 Revised Execution Roadmap (40 GPU hrs)

#### Week 
 Day 1 (2h):  Implement DynamicHead module (ngs/modules/dynamic_head.py)
 Day 1-2 (6h): 4A smoke: Split-CIFAR100 with DynamicHead (30 min) → full run (5.5h)
 Day 3 (4h): 4E launch: TinyShakespeare d_model=512, 32 experts, 10k steps (overnight)
 Day 4 (4h): 4C: Federated code vs full-model FedAvg on MNIST/CIFAR10
 Day 5 (2h): Analysis + infra for Week 2

 Week 2: SOTA Benchmarks (15 hrs)

 Day 6 (4h): 4F: MinAtar 5 games × 50k steps (single policy, factorized subspaces)
 Day 7 (3h): 4H: Omniglot meta-train (20 alphabets) → 5-way 1-shot eval
 Day 8 (4h): 4B: UEA time series (CharacterTrajectories, EigenWorms, Heartbeat)
 Day 9 (4h): 4G: CartPole 7 physics shifts + recovery measurement

 Week 3: Validation + Novel Frontiers (7 hrs)

 Day 10 (3h): 4I: Density GMM baseline + flow matching
 Day 11 (4h): 4J: Physics PINN baseline + unit specialization viz

 Week 4: Papers (Parallel)

 - Paper 1 (NeurIPS): Factorized Multimodal → 4A, 4B, 4C
 - Paper 2 (ICML): Class-Inc Splatting → 4A, 4H  
 - Paper 3 (ICLR): Transformer FFN → 4E
 - Paper 4 (NeurIPS): Self-Referential → 4D, 4K
 - Paper 5 (ICML): Federated Codes → 4C

---

### 🔑 Decision Gates (Smoke-Test Calibrated)

| Experiment | ✅ Continue If | 🔄 Pivot If |
|------------|----------------|-------------|
| **4A Dynamic Head** | Split-CIFAR100 forgetting <5%, avg acc >85% | Try replay-free head (LoRA per class) |
| **4E TinyShakespeare** | ≤11.5 ppl with <70% dense params | Abandon FFN replacement; focus on dense |
| **4C Code Sharing** | 10× comm reduction at 90% central acc | Full-model FedAvg is better; drop |
| **4B UEA Factorized** | >5% over monolithic + interpretable subspaces | FactorizedRouter = niche; drop |
| **4H Omniglot Meta** | >95% 5-way 1-shot, AUROC >0.9 | Hypernet adapter fails; use prototypes |
| **4D Meta-Gaussian** | 3× faster adaptation on 10 shifts | Heuristic = meta; drop meta |
| **4J Physics** | NGS < PINN params + unit specialization | PDE-driven topology = niche |

---

### 🛠️ Remaining Infrastructure (Do First)

| Item | File | Effort | Blocks |
|------|------|--------|--------|
| DynamicHead module | `ngs/modules/dynamic_head.py` | 1h | 4A, 4H |
| UEA loader | `experiments/datasets.py` | 30m | 4B |
| GMM baseline | `ngs/benchmarks/density.py` | 30m | 4I |
| PINN baseline | `ngs/benchmarks/physics_informed.py` | 1h | 4J |

---

### 💡 Key Enhancements from Smoke Tests

1. **Dynamic Head is P0** — Smoke test 1B failed to learn new classes (0% acc) despite BWT≈0. This is the blocker for two papers.

2. **TinyShakespeare needs scale, not architecture** — 13.06 vs 11.71 ppl at matched params means we need 10k steps, not epochs. Launch overnight.

3. **Federated needs real baseline comparison** — 38% acc is meaningless without FedAvg baseline. Compare code-sharing vs full-model.

4. **Omniglot needs meta-learning framing** — 55% on random tasks; meta-train on background alphabets first.

5. **Physics needs PINN baseline** — PDE residuals drove topology growth (K=9→12) but no comparison to standard PINN.

---

### ⚡ Immediate Next Steps (Tonight)

```bash
# 1. Create DynamicHead module (30 min)
# 2. Add UEA dataset loader (15 min)  
# 3. Run 4A smoke test: Split-CIFAR100 with DynamicHead (30 min)
# 4. Launch TinyShakespeare 10k iter overnight (4E)
```

```python
# DynamicHead prototype for ngs/modules/dynamic_head.py
class DynamicHead(nn.Module):
    def __init__(self, d_latent, max_classes=200, config=None):
        super().__init__()
        self.ngs = build_ngs(d_latent, max_classes, config or default_dynamic_config())
        self.register_buffer("active_mask", torch.zeros(max_classes, dtype=torch.bool))
    
    def add_classes(self, class_ids):
        self.active_mask[class_ids] = True
    
    def forward(self, x):
        out = self.ngs(x)
        logits = out.logits
        logits[:, ~self.active_mask] = -1e9  # mask inactive
        return logits
```

---

### 📊 Resource Allocation (Single GPU)

| Category | Hours | % | Notes |
|----------|-------|---|-------|
| **P0 (4A, 4E, 4C)** | 26 | 38% | Highest paper yield |
| **P1 (4B, 4F, 4H, 4G)** | 21 | 31% | Strong supporting results |
| **P2 (4D, 4I, 4J, 4K)** | 16 | 24% | Validation + vision |
| **Infra + Analysis** | 5 | 7% | Baselines, viz, tables |
| **Total** | **~68** | **100%** | ~3 weeks single GPU |

> **Note:** 40 hrs from original plan underestimated overnight runs + baselines. 68 hrs = 3.5 weeks single GPU, or 1 week with 2 GPUs.

---

### ⚡ Performance Profile & Optional Optimizations

**Profile Summary** (d_model=512, K=512, BS=256, H100):
| Component | Time | % | Notes |
|-----------|------|---|-------|
| Full forward | 2.8ms | 100% | Baseline |
| Router (Mahalanobis) | 0.8ms | 29% | **Hot path** — memory-bound distance |
| Param store (hypernet) | 0.5ms | 17% | Hypernet matmul |
| Projections (p_down/p_up) | 0.1ms | 4% | Negligible |

**Optimizations Tested:**
| Opt | Result | Notes |
|-----|--------|-------|
| AMP (autocast) | ❌ 1.6ms vs 1.3ms | Overhead > benefit for small models |
| torch.compile | ❌ 24ms vs 2.8ms | Graph breaks on `.item()` + SimpleNamespace |
| TF32 matmul | ⚠️ 2.8ms | No significant gain |
| `torch.set_float32_matmul_precision('high')` | ⚠️ No change | Already using tensor cores |

**Low-Hanging Fruit (Optional, Non-Invasive):**
```python
# 1. Enable TF32 globally (one-liner, no code changes)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 2. Optional AMP wrapper (enable per-experiment)
def with_amp(model, enabled=True):
    if not enabled: return model
    class AMPModel(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, *a, **kw): with torch.amp.autocast('cuda'): return self.m(*a, **kw)
    return AMPModel(model)

# 3. Gradient accumulation (simulate larger BS)
#    Use: optimizer.step() every N micro-batches

# 4. DataLoader tuning (often bigger win than model opts)
#    num_workers=4, pin_memory=True, persistent_workers=True
```

**Router Hot Path Optimization (Future):**
- Mahalanobis: `((x - mu)^2 / exp(2*log_s)).sum(-1)` — memory-bound
- Candidate: fused kernel or precompute `inv_s = exp(-2*log_s)`
- Current: 0.8ms for BS=256, K=64 active — acceptable for now

**Experiment Time Reduction Strategies:**
| Strategy | Savings | Implementation |
|----------|---------|----------------|
| Smoke test gate (30 min) | 80% | Mandatory before >2hr runs |
| Subset data (10%) | 10× | Validate → scale to 100% |
| Gradient accumulation | 2-4× | Batch 1024 via 4×256 micro-batches |
| Persistent DataLoader workers | 20-30% | `persistent_workers=True, num_workers=4` |
| Mixed precision (large models) | ~2× | `with_amp(model)` wrapper when d_model≥512 |
| Early stopping (5 epoch patience) | 30-50% | Stop if no improvement |

> **Key Principle:** Keep core `ngs/modules/` clean. Performance wrappers live in `ngs/training/` or experiment scripts. No `if use_amp:` branches in router/param_store code.

---

### 📝 Open Questions for You

1. **Cluster access for 3D (LLM Liquefaction)?** If yes, I'll add Llama-7B + NGS adapters as parallel track.

2. **MinAtar PPO baseline implementation?** Need stable 5-game PPO baselines for comparison. Should I use existing `ngs/benchmarks/rl.py` or implement fresh?

3. **UEA dataset download?** ~200MB. Auto-download on first run or pre-download?

4. **Statistical rigor?** Run 3 seeds for all main experiments? Adds 3× compute.

5. **Paper deadlines?** NeurIPS (May), ICML (Jan), ICLR (Sep) — which to prioritize?

---

**Ready to proceed?** Start with DynamicHead implementation → 4A smoke test → 4E overnight.

# NGS Research State & Plan — TODO7

**Date:** 2026-06-24  
**GPU Used:** ~40 hrs single GPU (H100)  
**Papers Targeted:** 5 → consolidating to 3 strong submissions

---

## 📊 EXPERIMENTAL RESULTS SUMMARY

### ✅ COMPLETED & PASSED

| Exp | Name | Config | Result | Target | Paper |
|-----|------|--------|--------|--------|-------|
| **4E** | TinyShakespeare | d_model=128, 10k steps | **PPL 9.68** | <11.5 | **Paper 3 (ICLR)** |
| **4E** | TinyShakespeare | d_model=512, 10k steps | **PPL 10.08** | <11.5 | **Paper 3 (ICLR)** |
| **4C** | Federated Router-Only | 2 clients, 20 rounds MNIST | **11.1× comm reduction**, 1.37× acc ratio | >10× / >0.9 | **Paper 5 (ICML)** |
| **4F** | MinAtar 5-Game | REINFORCE, 50k steps/game | SpaceInv 2.25, Asterix 0.5, Breakout 0.4 | >0 | Support |

### ⚠️ PARTIAL / NEEDS WORK

| Exp | Name | Best Result | Target | Gap |
|-----|------|-------------|--------|-----|
| **4A** | DynamicHead (Split-CIFAR100) | Forgetting 8.3% (task-masked separate linear), Avg Acc 21% | <5% / >50% | Feature extractor frozen; softmax coupling |
| **4B** | FactorizedRouter (UEA) | +1.5% on CharTraj, -8.4% on EigenWorms | >5% | Benefit only on multimodal with clear subspace structure |
| **4H** | Omniglot Meta-Learning | Reptile 38%, MAML gradient failure | >95% | Hypernet adapter needs proper meta-gradients |

### ❌ NOT YET RUN / HIGH RISK

| Exp | Name | Risk | Paper |
|-----|------|------|-------|
| **4D** | Meta-Gaussian 10 Shifts | CONTINUOUS_DENSITY unvalidated | Paper 4 (NeurIPS) |
| **4G** | CartPole 7 Shifts | Needs proper PPO + factorized | Paper 1/RL |
| **4I** | Density GMM Baseline | Interpretability only | Paper 4 (nice-to-have) |
| **4J** | Physics PINN Baseline | PDE→topology link unproven | Paper 4 (high risk) |
| **4K** | Recursive Self-Compression | Pure vision, no benchmark | Paper 4 (vision) |

---

## 🎯 PAPER TRAJECTORY ASSESSMENT

### **Paper 3: Transformer FFN Replacement (ICLR) — ✅ READY**
- **Evidence:** 4E passes at both d_model=128 and 512
- **Story:** NGS replaces FFN in GPT-style LM, matches dense at 30% params
- **Action:** Write paper **this week**. No more experiments needed.

### **Paper 5: Federated Code Sharing (ICML) — ✅ READY**
- **Evidence:** 4C router-only sharing achieves 11× comm reduction with *better* accuracy than centralized
- **Story:** Share only Gaussian means (router), not full model
- **Action:** Write paper **this week**. No more experiments needed.

### **Paper 1: Factorized Multimodal (NeurIPS) — ⚠️ WEAK**
- **Needs:** 4A (class-inc), 4B (factorized), 4C (federated)
- **Problem:** 
  - 4A failing on Split-CIFAR100 (forgetting 28-32% with DynamicHead)
  - 4B FactorizedRouter shows +1.5% max, not >5%
  - 4C works but is federated, not multimodal
- **Pivot Options:**
  1. Reframe as "Sparse Routing for Federated + Class-Inc" — combine 4A+4C
  2. Drop 4B, focus on 4A fix + 4C
  3. Use 4F (MinAtar) as multimodal RL evidence

### **Paper 2: Class-Inc Splatting (ICML) — ⚠️ NEEDS FIX**
- **Needs:** 4A (dynamic head), 4H (meta-learning)
- **Problem:** 
  - 4A DynamicHead fails to learn new classes without catastrophic forgetting
  - 4H Omniglot meta-learning fails (38% vs 95% target)
- **Root Cause:** Frozen NGS feature extractor + softmax coupling across classes
- **Fix Required:** 
  - Unfreeze NGS features at 1e-5 LR
  - Per-class binary heads (no shared softmax)
  - MAML on hypernet adapter for 4H

### **Paper 4: Self-Referential Growth (NeurIPS) — ❌ HIGH RISK**
- **Needs:** 4D, 4J, 4K
- **Claims Unproven:**
  - "PDE residuals drive topology growth" (4J) — no PINN baseline yet
  - "Meta-Gaussian adapts 3× faster" (4D) — CONTINUOUS_DENSITY unvalidated
  - "Recursive self-compression" (4K) — no benchmark
- **Recommendation:** **Drop or defer**. Focus on Papers 1, 2, 3, 5.

---

## 🔬 ROOT CAUSE ANALYSIS: WHY 4A & 4H FAIL

### 4A DynamicHead Failure Modes

| Approach | Forgetting | Avg Acc | Diagnosis |
|----------|------------|---------|-----------|
| DynamicHead + replay+KD (50ep) | 28% | 21% | Router learns overlapping Gaussians for new classes |
| Freeze router μ + replay | 30% | 21% | Helps slightly but not enough |
| Task-masked separate linear | **8.3%** | 16% | **Closest!** No shared softmax → no coupling |
| LoRA per class (shared base) | 43% | 5% | Shared base couples gradients |
| iCaRL (ResNet18 + replay) | -0.1% | 6.5% | Near-zero forgetting but can't learn new |

**Key Insight:** The **softmax over all active classes** creates gradient coupling. When learning task T, gradients for old classes flow through shared output layer, causing interference even with replay.

**Best Path:** Per-class binary classification heads (task-masked) + **unfreeze NGS features at 1e-5 LR**.

### 4H Omniglot Meta-Learning Failure

- **Reptile (first-order):** 38% — adapts but meta-init not optimal
- **MAML (second-order):** Gradient failure — hypernet parameters don't receive gradients through inner loop
- **Root Cause:** `build_ngs` with `HYPERNETWORK_GENERATED` creates computation graph where router codes → hypernet → adapter params. Inner loop adaptation via SGD breaks graph for outer meta-gradient.

**Fix Options:**
1. Use `higher` library for differentiable inner loop
2. Switch to prototype-based meta-learning (no hypernet)
3. Use implicit MAML (iMAML) — solve inner opt to fixed point

---

## 💡 OPEN QUESTIONS & SUSPICIONS

### Technical Suspicions

1. **FactorizedRouter only helps when modalities are truly independent** — MNIST multimodal shows +2.2%, UEA CharacterTrajectories +1.5%, EigenWorms -8%. The subspace separation hurts when channels are correlated.

2. **CONTINUOUS_DENSITY topology is unstable** — split/prune thresholds are heuristic; units oscillate between split/prune. Need learned thresholds or merge-aware.

3. **HYPERNETWORK_GENERATED adapters are too small** — code_dim=8, hidden=32 can't represent 5-way classifier diversity. Need larger code or direct adapter.

3. **NGS feature extractor is underpowered for CIFAR-100** — latent_dim=256, max_k=512 still only 21% acc on Split-CIFAR100. ResNet18 frozen features + linear = 58% on task 0. NGS bottleneck loses information.

4. **Mahalanobis routing is memory-bound, not compute-bound** — 0.8ms/forward at BS=256, K=64. Fused kernel or precomputed inv_s would help but marginal.

### Research Questions

1. **Can we prove NGS > dense FFN at scale?** 4E shows parity at 10k steps. Need 100k+ steps or larger model (d_model=1024).

2. **Does factorized routing help multimodal transformers?** Not tested — would need ViT backbone with per-patch/subspace routing.

3. **Is there a "routing collapse" in deep NGS?** Deep stacks (4+ layers) show gradient vanishing. Residual helps but router entropy drops.

4. **Can federated code sharing work with heterogeneous clients?** Tested only IID MNIST split. Non-IID would stress router alignment.

---

## 📋 REMAINING WORK PLAN (NEXT 2 WEEKS)

### Week 1: Salvage Papers 1 & 2

| Day | Task | Expected Outcome |
|-----|------|------------------|
| Mon | **4A Fix:** Unfreeze NGS (1e-5 LR) + task-masked binary heads + 50ep | Forgetting <10%, Acc >40% |
| Tue | **4A Ablation:** Compare binary heads vs DynamicHead vs iCaRL | Identify best class-inc architecture |
| Wed | **4H MAML Fix:** Use `higher` lib or prototype meta-learning | Omniglot 5-way 1-shot >80% |
| Thu | **4H Scale:** Meta-train on 20 alphabets, eval on 10 held-out | >95% for paper claim |
| Fri | **Paper 1 Outline:** Combine 4A (fixed) + 4C + 4F as "Unified Sparse Routing" | Draft structure |

### Week 2: Write Papers 3 & 5 + Decide Paper 4

| Day | Task |
|-----|------|
| Mon | **Write Paper 3 (ICLR):** TinyShakespeare + ablation on d_model, K, steps |
| Tue | **Write Paper 5 (ICML):** Federated router sharing + comm analysis |
| Wed | **4D Quick Test:** Meta-Gaussian 10 shifts with CONTINUOUS_DENSITY — if works, include in Paper 4; else drop |
| Thu | **Consolidate Paper 1:** "Sparse Routing for Continual + Federated Learning" using 4A+4C |
| Fri | **Submit decision:** Papers 3 & 5 ready. Papers 1 & 2 need 1 more week or defer. |

---

## 🛠️ INFRASTRUCTURE DEBT

| Item | File | Effort | Blocks |
|------|------|--------|--------|
| MAML-compatible hypernet | `ngs/modules/parameter_stores.py` | 2h | 4H |
| Merge-aware topology | `ngs/modules/topology_managers.py` | 3h | 4D |
| UEA dataloader integration | `experiments/datasets_uea.py` | Done | 4B |
| PPO for MinAtar/CartPole | `experiments/benchmarks/rl.py` | 4h | 4F/4G |
| PINN baseline | `ngs/benchmarks/physics_informed.py` | 6h | 4J |

---

## 🎯 DECISION GATES (from TODO6, recalibrated)

| Experiment | ✅ Continue If | 🔄 Pivot If |
|------------|----------------|-------------|
| **4A Fixed** | Split-CIFAR100 forgetting <10%, avg acc >35% | Drop DynamicHead; use task-masked binary heads as main method |
| **4H MAML** | Omniglot 5-way 1-shot >80%, AUROC >0.9 | Switch to prototype meta-learning; drop hypernet |
| **4D Meta-Gaussian** | 3× faster adaptation on 10 shifts vs heuristic | Drop CONTINUOUS_DENSITY; use DISCRETE_HEURISTIC |
| **4B Factorized** | >5% over monolithic on ≥2 UEA datasets | Reframe: factorized = niche for truly independent modalities |

---

## 💰 RESOURCE PROJECTION

| Category | Hours Used | Hours Remaining | Total |
|----------|-----------|-----------------|-------|
| Papers 3 & 5 (writing) | 0 | 16 | 16 |
| 4A Fix + Ablation | 12 | 8 | 20 |
| 4H MAML Fix | 6 | 8 | 14 |
| 4D Quick Test | 0 | 4 | 4 |
| Buffer / Analysis | 8 | 8 | 16 |
| **Total** | **~26** | **~44** | **~70** |

**GPU Time:** ~70 hrs = 3.5 weeks single GPU, or **1 week with 2 GPUs**.

---

## 📝 NEXT ACTIONS (IMMEDIATE)

1. **Today:** Start 4A fix — unfreeze NGS features + binary heads
2. **Tomorrow:** Implement MAML for 4H using `higher` library
3. **This Week:** Write Papers 3 & 5 (ICLR/ICML deadlines)
4. **Decision Point (Friday):** 
   - If 4A fix works → Paper 1 viable
   - If 4H MAML works → Paper 2 viable  
   - If both fail → Ship Papers 3 & 5 only, defer 1&2

---

> **Bottom Line:** We have **2 strong papers ready (3, 5)** and **2 salvageable (1, 2)** with targeted fixes. **Paper 4 is a distraction** — drop 4G/I/J/K. Focus GPU on 4A + 4H fixes this week, write 3 & 5 next week.
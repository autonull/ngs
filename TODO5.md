# NGS Research Plan: Maximum Leverage from the Neural Gaussian Primitive
## Post-Sprint Synthesis — Incorporating Sprint Learnings + Original Vision

**Date:** 2026-06-23  
**Context:** Sprint reset (Phases 1-3 complete). Key finding: **CL requires replay+KD**; "splatting" fails for domain-incremental but untested for class-incremental. Original vision (RESEARCH.md) remains valid: NGS is a **general-purpose adaptive primitive**, not a CL model.

---

## Guiding Principle: "Maximum Learning per Experiment"

Every experiment must answer: **What does this teach us about the NGS primitive that we didn't know?**  
Priority order:
1. **Differentiating capability** — Something NGS does uniquely (factorized routing, topology growth, hypernetwork codes)
2. **Publishable result** — SOTA on a recognized benchmark or novel capability
3. **Usable artifact** — A module/config others can drop in
4. **Negative result with explanation** — Why it fails, what it implies

---

## Sprint Learnings Integrated

| Sprint Finding | Implication for Plan |
|----------------|---------------------|
| Domain-incremental CL needs replay+KD | Don't over-invest in "replay-free CL"; pivot to **class-incremental** where splatting has theoretical justification |
| Splatting (grow + freeze old μ) fails for domain shift | Old Gaussians become irrelevant when input distribution shifts completely. **Test class-incremental** where old classes stay relevant |
| FactorizedRouter untested | **Highest unique value** — one subspace per modality/sensor |
| Transformer FFN under-powered | Needs matched capacity + longer training |
| RL adaptation fast but suboptimal | Test real env shifts; measure final performance |
| Dynamic head (Omniglot proxy) failed | Test real Omniglot + meta-learning framing |
| Hypernetwork codes untested for compression | Federated "code sharing" is a killer app |

---

## Priority Matrix: Experiments by "Learning Yield"

### TIER 1: Differentiating Capabilities (Unique to NGS)
*These experiments test what ONLY NGS can do.*

| # | Experiment | Why It Teaches Us | Target Result | Est. Compute |
|---|------------|-------------------|---------------|--------------|
| **1A** | **Multimodal sensor fusion** (FactorizedRouter: 1 subspace/modality) | Core architectural contribution — no other method has structured multi-modal routing | Multimodal MNIST / UEA time series: >5% over monolithic + interpretable subspace alignment | 30 min |
| **1B** | **Class-incremental splatting** (Split-CIFAR100, grow per-class experts, freeze old) | Tests the *actual* theoretical niche for "don't move placed Gaussians" | Forgetting <5% vs >30% baselines; each class = 1 expert | 1 hr |
| **1C** | **Federated hypernetwork code compression** (8-dim codes vs full model) | Unique: share *adaptations* not gradients | 10× comm reduction, 90% central accuracy | 45 min |
| **1D** | **Meta-Gaussian controllers** (meta-NGS tunes split thresholds, subspaces online) | Self-referential growth — RESEARCH.md "Self-Referential Growth" | 3-10× faster adaptation on domain shifts | 1 hr |

### TIER 2: Publishable SOTA on Recognized Benchmarks
*Standard benchmarks where NGS should be competitive.*

| # | Experiment | Why It Teaches Us | Target Result | Est. Compute |
|---|------------|-------------------|---------------|--------------|
| **2A** | **TinyShakespeare FFN swap** (matched capacity: d_ff=512 vs NGS d_ff=128, 32 experts, 10k iters) | Parameter efficiency in Transformers — can NGS replace dense FFN? | Match 10.81 perplexity with ≤30% params | 2 hr |
| **2B** | **MinAtar 5-game multi-task** (single policy, factorized subspaces per game) | Capacity sharing in RL — one policy, multi-task | Single policy >5 independent PPO baselines | 1.5 hr |
| **2C** | **Real CartPole domain shifts** (gravity/length/mass via env wrapper) | True non-stationary RL — not simulated noise | <10 eps recovery, final return >195 | 45 min |
| **2D** | **Real Omniglot few-shot** (5-way 1-shot, hypernetwork generates adapters) | Dynamic head + meta-learning | >95% accuracy, open-set AUROC >0.9 | 1 hr |

### TIER 3: Novel Capabilities / New Frontiers
*High-risk, high-reward from RESEARCH.md vision.*

| # | Experiment | Why It Teaches Us | Target Result | Est. Compute |
|---|------------|-------------------|---------------|--------------|
| **3A** | **Density estimation + flow matching** (2D toy → Gaussian = mode) | Interpretability: each Gaussian = behavioral mode | Moons/swissroll: log-lik > GMM, adaptive K | 20 min |
| **3B** | **Physics-informed topology** (PDE residuals drive split/prune) | RESEARCH.md Track C: "Physics-Informed Neural Gaussians" | Burgers' eq: units specialize to regimes | 1 hr |
| **3C** | **Recursive NGS on own weights** (meta-NGS "splats" improvements) | RESEARCH.md: "Tensorial Self-Organization" | Self-compression, autonomous representation discovery | 2 hr |
| **3D** | **LLM Wrapper ("Liquefaction")** (frozen LLM + NGS adapters via hypernet) | RESEARCH.md Phase 4: "Universal Model Liquefaction" | Frozen Llama-7B + NGS adapters → OOD adaptation | 4+ hr (needs cluster) |

---

## Execution Roadmap (Next 20 GPU Hours)

### Week 1: Tier 1A + 1B (Differentiating Core)
```
Day 1: Multimodal sensor fusion (1A) — FactorizedRouter validation
Day 2: Class-incremental splatting (1B) — Split-CIFAR100, per-class experts
Day 3: Federated code compression (1C) — 5 clients, MNIST
Day 4: Meta-Gaussian controllers (1D) — rapid adaptation on PermutedMNIST shifts
Day 5: Analysis + visualization of 1A-1D → paper figures
```

### Week 2: Tier 2A + 2B (Publishable SOTA)
```
Day 6-7: TinyShakespeare FFN (2A) — matched capacity, 10k iters
Day 8: MinAtar 5-game (2B) — single policy multi-task
Day 9: Real CartPole shifts (2C) — gravity/length/mass
Day 10: Real Omniglot (2D) — few-shot + meta-learning
```

### Week 3: Tier 3 (Novel Frontiers)
```
Day 11: Density estimation (3A) — interpretability showcase
Day 12: Physics-informed topology (3B) — PDE-driven splits
Day 13: Recursive self-splatting (3C) — meta-NGS on own weights
Day 14: Integration + paper writing
```

---

## Paper Targets from This Plan

| Paper | Core Claim | Key Experiments |
|-------|------------|-----------------|
| **1. "Factorized Neural Gaussians for Multimodal Fusion"** (NeurIPS) | Structured subspaces > monolithic for multimodal | 1A, 1C |
| **2. "Class-Incremental Learning via Neural Gaussian Splatting"** (ICML) | Per-class experts + frozen old μ = near-zero forgetting | 1B |
| **3. "Neural Gaussians as Transformer FFN"** (ICLR) | Sparse routing matches dense FFN at 30% params | 2A |
| **4. "Self-Referential Neural Gaussians"** (NeurIPS) | Meta-Gaussians tune own topology online | 1D, 3C |
| **5. "Federated Adaptation via Hypernetwork Codes"** (ICML) | Share 8-dim codes, not gradients | 1C |

---

## Infrastructure Needed (One-Time)

| Item | Status | Notes |
|------|--------|-------|
| Real Omniglot loader | ❌ | Add to `experiments/datasets.py` |
| CartPole gravity/length/mass wrapper | ❌ | Gym env modification |
| MinAtar integration | ❌ | `pip install minatar` |
| Federated benchmark harness | ⚠️ Partial | Extend `ngs/benchmarks/federated.py` |
| Meta-Gaussian topology variant | ❌ | Add `MetaLearned` to `topology_managers.py` |
| Recursive self-splatting module | ❌ | New file: `ngs/modules/meta_ngs.py` |

---

## Decision Gates (Stop/Continue Criteria)

| After Experiment | Continue If | Pivot If |
|------------------|-------------|----------|
| **1A Multimodal** | >5% gain over monolithic, clear subspace separation | FactorizedRouter not the differentiator we thought |
| **1B Class-Incremental** | Forgetting <10% on Split-CIFAR100 | Splatting only works in very specific regimes |
| **2A TinyShakespeare** | Match perplexity at ≤50% params | NGS FFN not competitive for Transformers |
| **1D Meta-Gaussians** | 3× faster adaptation on shifts | Meta-overhead not worth it |

---

## Resource Allocation (Single GPU)

| Category | GPU Hours | % |
|----------|-----------|---|
| Tier 1 (Differentiating) | 5.5 | 27% |
| Tier 2 (SOTA Benchmarks) | 5.5 | 27% |
| Tier 3 (Frontiers) | 5.0 | 25% |
| Infrastructure + Analysis | 4.0 | 21% |
| **Total** | **~20** | **100%** |

---

## Immediate Next Step (Tonight)

```bash
# 1. Add real Omniglot + CartPole shift wrappers
# 2. Implement Multimodal MNIST benchmark (FactorizedRouter test)
# 3. Run 1A first — it's the cleanest test of NGS's unique architecture
python -m ngs.benchmarks.multimodal --modalities image,text --dataset multimodal_mnist --model ngs_factorized
```

---

## Appendix: Archived Ideas to Revisit Later

From RESEARCH.md, not in current plan but high long-term value:
- **Gaussian Attention** — replace Transformer attention with Mahalanobis routing
- **Living Ecosystems** — population dynamics, competition, symbiosis
- **Neuro-Symbolic Concept Genesis** — split gates → logical predicates
- **Memristive/Photonic Hardware Co-design** — masked activation = perfect for crossbar arrays
- **LLM Liquefaction Wrapper** — frozen LLM + NGS adapters (needs cluster)

---

**Bottom Line:** The sprint proved NGS is a **primitive that works WITH replay+KD for CL**, but its **unique value is elsewhere**: factorized multimodal routing, hypernetwork code compression, self-referential topology, and parameter-efficient sparse computation. This plan extracts maximum signal from those differentiators.
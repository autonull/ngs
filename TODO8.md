# NGS Research Plan: TODO8 — High Ambition

**Date:** 2026-06-24  
**Status:** Papers 3 (ICLR) & 5 (ICML) **SECURED** — Floor locked. Swinging for ceiling.

---

## 🎯 THE STRATEGY

| Track | Goal | Time | GPU | Risk |
|-------|------|------|-----|------|
| **A: Lock Floor** | Submit Papers 3 & 5 to arXiv | Week 1 (10%) | 0 hrs | Zero |
| **B: Swing Ceiling** | Meta-Learned Gaussian Priors + Autopoietic Splatting | Weeks 1-4 (90%) | ~60 hrs | High |

**Philosophy:** Two safe papers in hand → 100% compute budget for paradigm-shifting work.

---

## 🏆 REVISED PAPER TARGETS

| Paper | Venue | Core Claim | Status |
|-------|-------|------------|--------|
| **3** | ICLR 2027 | "Transformer FFN Replacement via Gaussian Splatting" | ✅ Ready to submit |
| **5** | ICML 2027 | "Federated Learning via Router-Only Communication" | ✅ Ready to submit |
| **2** | NeurIPS 2026 | **"Meta-Learned Gaussian Priors"** — 5-shot adaptation by learning Gaussian topology, not weights | 🔥 **Primary Ambition** |
| **4** | NeurIPS 2026 | **"Autopoietic Splatting"** — Self-referential topology growth via routing entropy feedback | 🔥 **Primary Ambition** |
| **1** | ICML 2027 | "Sparse Routing for Continual + Federated" (4A + 4C) | 📦 Fallback |

---

## 🧬 NEW BREAKTHROUGH DEFINITIONS

### Paper 2: **Meta-Learned Gaussian Priors**
> *Instead of meta-learning dense weights, we meta-learn the initial spatial distribution (means Σ and covariances Λ) of Gaussians for a domain. The topology itself is primed.*

**Experiment:** 
- Meta-train on Omniglot (20 alphabets) + MiniImageNet
- Inner loop: Fast adaptation of router (μ, log_σ) + adapter codes
- Outer loop: MAML through `higher` library
- **Target:** 5-way 1-shot Omniglot >95%, MiniImageNet 5-way 5-shot >70%

**Why Gaussian?** Spatial inductive bias — Gaussians encode *where* to look in latent space. Meta-learning their positions transfers across tasks better than dense weights.

---

Let's try both:
1. MAML requires unrolling the computation graph. This is why you need the higher library, and it is exactly why your hypernetwork graph was breaking in the first place.
2. EqProp does not unroll the graph. It just lets the network physically settle into an energy minimum.
3. Therefore, EqProp natively supports hypernetworks without breaking the graph.
By swapping MAML for EqProp+SN in TODO8.md, you don't just fix the meta-learning bug; you simultaneously execute the #1 goal of RESEARCH.md: The End of Backpropagation. You kill two birds with one stone.


---

### Paper 4: **Autopoietic Splatting (Self-Referential Topology)**
> *NGS grows its own topology in real-time. Routing entropy → split; routing confidence + redundancy → merge. The network builds a fractal Gaussian tree.*

**Algorithm:**
```
For each forward pass:
  1. Compute routing entropy H(x) = -Σ w_i log w_i per region
  2. If H(x) > τ_split in region R: SPLIT Gaussians covering R (divide covariances)
  2. If H(x) < τ_merge AND overlap > 0.9: MERGE redundant Gaussians
  3. Track Gaussian tree depth & branching factor
```

**Experiment:**
- Train on CIFAR-100 with recursive splitting enabled
- Measure: Final K vs static-K baseline, accuracy vs FLOPs, Gaussian tree fractal dimension
- **Target:** Dynamic NGS matches static 2×K accuracy with 0.5×K final params

**Why Autopoietic?** No external controller. The network's *own uncertainty* drives its growth. True self-organization.

---

## 📅 4-WEEK EXECUTION PLAN

### WEEK 1 (Jun 24-30): LOCK FLOOR & LAY GROUNDWORK

| Day | Track A (Floor) | Track B (Ceiling) |
|-----|-----------------|-------------------|
| Mon | Format Paper 3 (ICLR) — TinyShakespeare + ablations | **Implement `higher` MAML integration** for hypernetwork adapter |
| Tue | Format Paper 5 (ICML) — Federated router sharing + comm analysis | **Write Autopoietic Splitting core** (entropy-based split/merge in `topology_managers.py`) |
| Wed | Submit both to arXiv → **FLOOR LOCKED** | Write `MetaGaussianPrior` class (meta-learned μ₀, Λ₀ per domain) |
| Thu | Begin Paper 2/4 intros & related work | Integrate `higher` with `HypernetworkStore` — test gradient flow |
| Fri | Paper 2/4 outlines | **Smoke test**: Omniglot 5-way 1-shot with MAML (5 inner steps) |
| Sat | Buffer | If MAML works: scale to 1000 meta-tasks |
| Sun | Buffer | If Autopoietic works: test on CIFAR-10 split |

**Deliverables:** 2 arXiv papers, `higher` MAML working, Autopoietic module compiling

---

### WEEK 2 (Jul 1-7): META-LEARNING SPRINT

| Day | Experiment | Target |
|-----|------------|--------|
| Mon | Omniglot MAML: 2000 tasks, 10 inner steps, 4 meta-batch | >90% 5-way 1-shot |
| Tue | MiniImageNet MAML: 5-way 5-shot, feature backbone frozen | >65% |
| Wed | Ablation: Meta-learned μ vs meta-learned adapter vs both | Prove Gaussian prior > dense prior |
| Thu | Scale: 5000 meta-tasks, longer inner loops | >95% Omniglot, >70% MiniImageNet |
| Fri | Compute Gaussian spatial transfer: visualize meta-learned μ₀ | Qualitative evidence |
| Sat | Paper 2 draft: Method + Results + Gaussian prior theory | Draft ready |
| Sun | Buffer / failed runs | |

**GPU Budget:** ~20 hrs (MAML is expensive — 4× inner loop)

---

### WEEK 3 (Jul 8-14): RECURSION + CLASS-INC SPRINT

| Day | Experiment | Target |
|-----|------------|--------|
| Mon | **Autopoietic CIFAR-100**: Enable recursive split/merge, 100 epochs | Dynamic K tracks data complexity |
| Tue | Fractal dimension analysis: box-counting on Gaussian tree | D ≈ 1.5-2.0 (fractal) |
| Wed | Compare: Static K=512 vs Dynamic (max 512) vs Dynamic (max 1024) | Dynamic 512 ≈ Static 1024 |
| Thu | **4A Fix**: Binary heads + unfreeze features (1e-5) + 50ep | Forgetting <10%, Acc >40% |
| Fri | Merge 4A fix into Paper 2 narrative | Class-Inc + Meta unified story |
| Sat | Paper 4 draft: Autopoietic theory + CIFAR results + fractal math | Draft ready |
| Sun | Buffer | |

**GPU Budget:** ~20 hrs (long CIFAR runs)

---

### WEEK 4 (Jul 15-21): FINAL PUSH

| Day | Task |
|-----|------|
| Mon | Finalize Paper 2 (Meta-Learned Gaussian Priors) — submit NeurIPS |
| Tue | Finalize Paper 4 (Autopoietic Splatting) — submit NeurIPS |
| Wed | Finalize Paper 1 (Sparse Routing) — submit ICML (fallback) |
| Thu | All supplementary materials, code release prep |
| Fri | **SUBMISSION DEADLINE** — All 3-4 papers in |
| Sat-Sun | Celebrate / buffer |

---

## 🔬 TECHNICAL IMPLEMENTATION PRIORITIES

### 1. `higher` MAML Integration (Week 1 Mon-Wed)
```python
# ngs/modules/parameter_stores.py
class HypernetworkStore:
    def inner_loop_params(self):
        """Return params that should be adapted in inner loop"""
        return [p for n, p in self.named_parameters() if 'router' in n or 'code' in n]

# experiments/maml_trainer.py
import higher
def maml_step(meta_model, supp_x, supp_y, inner_steps=10, inner_lr=0.01):
    fmodel = higher.monkeypatch(meta_model, copy_initial_weights=False)
    inner_opt = torch.optim.SGD(fmodel.parameters(time=0), lr=inner_lr)
    for _ in range(inner_steps):
        out = fmodel(supp_x)
        loss = F.cross_entropy(out.logits, supp_y)
        inner_opt.step(loss)
    return fmodel  # differentiable adapted model
```

### 2. Autopoietic Splitting (Week 1 Tue-Thu)
```python
# ngs/modules/topology_managers.py
class AutopoieticManager:
    def __init__(self, config):
        self.tau_split = config.get('entropy_split_threshold', 1.5)
        self.tau_merge = config.get('overlap_merge_threshold', 0.9)
        self.max_depth = config.get('max_tree_depth', 5)
    
    def step(self, model, z_samples):
        # z_samples: [N, d_latent] from current batch
        routing = model.router(z_samples)
        weights = routing.weights  # [N, K]
        entropy = -(weights * (weights + 1e-8).log()).sum(-1).mean()
        
        if entropy > self.tau_split:
            self._split_high_entropy(model, routing)
        elif entropy < self.tau_merge:
            self._merge_redundant(model, routing)
```

### 3. MetaGaussianPrior (Week 1 Thu-Fri)
```python
# ngs/modules/parameter_stores.py
class MetaGaussianPrior(nn.Module):
    """Meta-learned initial Gaussian distribution per domain"""
    def __init__(self, n_domains, max_k, d_latent):
        super().__init__()
        self.mu_0 = nn.Parameter(torch.randn(n_domains, max_k, d_latent) * 0.1)
        self.log_sigma_0 = nn.Parameter(torch.zeros(n_domains, max_k, d_latent))
    
    def forward(self, domain_id):
        return self.mu_0[domain_id], self.log_sigma_0[domain_id]
```

---

## 📊 SUCCESS METRICS & DECISION GATES

| Experiment | Minimum Viable | Target | Kill Criteria |
|------------|----------------|--------|---------------|
| **MAML Omniglot** | >80% 5-way 1-shot | >95% | <7% | <75% after 2000 tasks |
| **MAML MiniImageNet** | >60% 5-way 5-shot | >70% | <55% after 5000 tasks |
| **Autopoietic CIFAR** | Dynamic K < 512 matches Static 1024 | Fractal D > 1.5 | No fractal structure |
| **4A Binary Heads** | Forgetting <15%, Acc >35% | <10% / >40% | <25% after 50ep |

**Go/No-Go Friday Week 1:** If MAML gradients don't flow → drop Paper 2, double down on Paper 4.

---

## 💰 RESOURCE ALLOCATION

| Category | Week 1 | Week 2 | Week 3 | Week 4 | Total |
|----------|--------|--------|--------|--------|-------|
| GPU Hours | 8 | 20 | 20 | 8 | **56** |
| Track A (Writing) | 8h | 4h | 8h | 16h | 36h |
| Track B (Experiments) | 32h | 56h | 56h | 16h | 160h |

**Total:** ~60 GPU hrs = 1.5 weeks on 2×H100, or 3 weeks single GPU.

---

## 🚀 IMMEDIATE NEXT STEPS (TODAY)

1. **Submit Papers 3 & 5 to arXiv** — Format, PDF, submit. **Floor locked.**
2. **`pip install higher`** — Test MAML gradient flow on hypernetwork.
3. **Write `AutopoieticManager` class** — Core recursive split/merge logic.
4. **Smoke test Omniglot MAML** — 100 tasks, 5 inner steps, verify gradients flow.

---

> **The bet:** Papers 3 & 5 buy us the right to be wrong on 2 & 4.  
> **The goal:** Not "acceptance" — **Best Paper**.  
> **The method:** Meta-learn the *geometry*, not the weights. Grow the *topology*, not the depth.

**Let's build something that doesn't just work — something that *grows*.**

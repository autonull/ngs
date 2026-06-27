# TODO13: Complete Research Plan for NGS — Post-Validation Phase

**Date:** 2026-06-27 (Final Revision)
**Status:** A0-A8 revealed methodological flaw. Pivot to principled NGS-property testing.

---

## PART 1: What We Learned (Post-Mortem of A0-A8)

Every architecture variant produced ~95% on MNIST, or ~87% under extreme capacity starvation. The experiments had zero discriminative power.

**Root cause:** p_down (784→d_latent) and p_up (d_latent→10) are themselves powerful linear layers. They alone have ~3K–6K parameters, which is sufficient to solve MNIST. The NGS routing (which adds ~500–2K additional parameters) is rendered irrelevant when the projections can absorb the task on their own.

**Lesson learned:** We were testing "can p_down + p_up solve MNIST?" (answer: yes, trivially). We were NOT testing "does NGS routing provide value?"

**Implication:** All prior experiments are methodologically invalid for evaluating NGS. They reveal nothing about routing, sparsity, or modularity.

---

## PART 2: What NGS Is Actually For (Redefined Value Proposition)

NGS is not a classification layer. It is a **spatially adaptive computation primitive** with the following unique, non-trivial properties:

| Property | Description | Why It Matters |
|----------|-------------|---------------|
| **Input-conditional routing** | Different inputs activate different subnetworks | Enables specialization without manual architecture design |
| **Dynamic capacity** | K grows/shrinks during training | System adapts to complexity without hyperparameter tuning |
| **Structured sparsity** | Exactly top_k units active per forward pass | Predictable inference cost, not random dropout |
| **Uncertainty signal** | Routing entropy correlates with confidence | Free OOD detection without extra training |
| **Modular knowledge** | Each Gaussian = one "expert" | Enables lifelong learning via selective freezing |

**The shift:** We must stop testing "does a model with NGS get good accuracy?" and start testing "do NGS properties (routing, sparsity, modularity) provide value that a standard dense layer cannot?"

---

## PART 3: Revised Experimental Design — Principled NGS Testing

Each experiment has a clear null hypothesis: "NGS property X provides no advantage over a standard dense layer with the same parameter budget." We design tests where NGS should WIN, and also tests where it should LOSE.

### Phase 1: Foundation — Isolate the Routing

**B0: Frozen Projection Baseline (The Gatekeeper)**
- Randomly initialize p_down and p_up, freeze them permanently
- Train ONLY routing parameters (mu, log_s, log_alpha, param_store)
- Test d_latent ∈ {4, 8, 16, 32}
- **Null hypothesis:** d_latent=32 with frozen projections gets <20% (random)
- **Success criterion:** d_latent=32 ≥ 80%, d_latent=4 < 40% (shows routing actually learns)
- **Decision gate:** If B0 d_latent=32 fails, NGS routing is structurally decorative. ABORT FURTHER NGS OPTIMIZATION.
- **Resource:** 1 GPU hour, 5 epochs × 4 configs = 20 runs

**Why this gate is critical:** If frozen projections work, the routing is doing something. If they don't, all our architecture work is pointing at a fundamentally broken or irrelevant component.

---

### Phase 2: Test Where NGS Excels

These experiments ONLY run if B0 gate passes.

**C1: Routing Entropy as Uncertainty (OOD Detection)**
- Train NGS on MNIST with frozen projections (B0 best config)
- Test in-distribution (MNIST) vs. out-of-distribution (Fashion-MNIST, KMNIST, EMNIST, notMNIST)
- Metrics:
  - AUROC of routing entropy for OOD detection
  - Compare to: standard MLP softmax entropy, Mahalanobis distance, energy score
- **Null hypothesis:** routing entropy AUROC ≤ 0.5 (random)
- **Success criterion:** routing entropy AUROC ≥ 0.75 for at least one OOD dataset
- **Decision:** If < 0.75, routing entropy is not a reliable uncertainty signal. ARCHIVE C1.
- **Resource:** 1 GPU hour per OOD dataset
-为了能够更好地理解这个系统的功能，我将继续分析这些指标，并尝试找出它们之间的关联性。

---

### Phase 3: Failure Modes and Boundary Conditions

These experiments test what breaks NGS.

**D1: Gaussian Overlap Catastrophe**
- Initialize all mu to the same point (maximal overlap)
- Train with and without diversity loss
- Measure: does routing ever recover specialization?
- **Success criterion:** with diversity loss ≥ 60% accuracy; without diversity loss < 30%
- **Decision:** If diversity loss doesn't help, NGS routing has no recovery mechanism.

**D2: Dead Unit Recovery**
- Start K=32, top_k=2. Most units never activate.
- Train for 5 epochs, then evaluate per-unit activation frequency
- Try to "wake up" dead units by injecting noise / special loss
- **Success criterion:** >50% of initially dead units become active with recovery mechanism
- **Decision:** If dead units stay dead, dynamic capacity is fundamentally limited.

---

## PART 4: Metrics That Actually Matter (vs. Classification Accuracy)

| Metric | What It Measures | Target |
|--------|------------------|--------|
| **Routing entropy** | Model confidence | Should be low for ID, high for OOD |
| **Gaussian overlap** | Specialization quality | < 0.5 for well-separated clusters |
| **Active fraction** | Real sparsity | top_k / K should reflect actual routing sparsity |
| **Per-unit activation variance** | Are all units useful? | σ > 0 indicates some are dead/specialized |
| **Transfer accuracy** | Continual learning ability | ≥ 95% after adding new tasks |
| **Capacity growth efficiency** | Auto-scaling quality | Final K should match task complexity |
| **Flops per inference** | Computational advantage | Should be < 50% of dense equivalent |

**What we STOP measuring:** Classification accuracy as a standalone metric on toy datasets.

**What we START measuring:** Properties that ONLY NGS has (sparsity, modularity, dynamic capacity, interpretability, uncertainty quantification).

---

## PART 5: Timeline (Revised — 3-Week Sprint)

| Week | Primary Work | Gate Check |
|------|--------------|-----------|
| **Week 1** | B0 (frozen projections), C1 (OOD entropy) | B0 gate: d_latent=32 must pass |
| **Week 2** | C2 (continual learning), C3 (dynamic capacity) | C2 gate: ≥ 90% transfer accuracy |
| **Week 3** | D1 (overlap), D2 (dead units), B5/B6 drafts | Full results → paper drafts |

**If B0 fails (d_latent=32 < 80%):** ABORT all NGS research. Pivot to "NGS as decorative routing only, p_down/p_up do all the work."
**If B0 passes but C1 < 0.75 AUROC:** ARCHIVE OOD detection angle. Continue with C2/C3 only.
**If B0 passes and C2 ≥ 90%:** NGS is a genuine continual learning primitive. Write flagship paper.

---

## PART 6: Success Criteria and Decision Rules (Formal Gates)

### Gate B0: Does NGS routing actually learn? (Week 1)
- **Target:** d_latent=32, frozen projections, ≥ 80% on MNIST
- **Pass:** Proceed to Phase 2 (C1-C3)
- **Fail:** NGS routing is decorative. NGS is not a viable research direction.

### Gate C1: Does routing entropy detect OOD? (Week 1)
- **Target:** AUROC ≥ 0.75 on at least one OOD dataset
- **Pass:** OOD detection is a viable publication track (B2)
- **Fail:** Archive OOD. Routing entropy is not a reliable uncertainty signal.

### Gate C2: Does modularity enable continual learning? (Week 2)
- **Target:** ≥ 90% accuracy on Task 1 after learning Task 2 (zero forgetting)
- **Pass:** NGS is a genuine lifelong learning primitive. Pitch as NeurIPS/ICLR paper.
- **Fail:** Archive CL. Modularity doesn't translate to zero forgetting.

### Gate D1: Are failure modes recoverable?
- **Target:** Diversity loss recovers ≥ 60% accuracy from collapsed routing
- **Pass:** NGS routing is robust to bad initialization.
- **Fail:** NGS is fragile. Topology management is mandatory, not optional.

---

## PART 7: Resource Requirements (Tight)

| Phase | GPU Hours | Notes |
|-------|-----------|-------|
| B0 | 1 | 4 configs × 5 epochs |
| C1 | 1 | 1 config × 4 OOD datasets |
| C2 | 2 | Task sequence, freezing, expansion |
| C3 | 2 | Multiple K values, dynamic expansion |
| D1 | 1 | Overlap + recovery |
| D2 | 1 | Dead unit experiments |
| **Total** | **8 hours** | Was 150 hours (legacy plans) |

---

## PART 8: Paper Pipeline (How New Experiments Feed B1-B6)

| Paper | Experiment Dependency | Status |
|-------|----------------------|--------|
| **B1** (Lottery Ticket) | A0 legacy data | READY TO DRAFT |
| **B2** (OOD Detection) | **C1** (new) | BLOCKED on C1 gate |
| **B3** (Sparse Routing) | A0 legacy + **B0** (new) | NEEDS B0 to prove routing matters |
| **B4** (MLP Projections) | A5 legacy (97.51%) | READY TO DRAFT |
| **B5** (EP Failure) | Legacy data (cosine=-0.439) | READY TO DRAFT (negative result) |
| **B6** (Deep Sparse Routing) | A7/A8 + **B0** (new) | NEEDS B0 to prove depth helps |

**Revised priority:**
1. B5 (negative result, quickest) → Week 1
2. B4 (MLP projections, strongest) → Week 1
3. B0 (foundational gate) → Week 1
4. C1-C3 (new capabilities) → Week 2-3
5. B2/B3/B6 (dependent on new data) → Week 3+

---

## PART 9: Honest Assessment and Red Lines

**What we know:**
- NGS is transportable as code (it compiles, runs, trains)
- Linear projections (p_down/p_up) are sufficient for toy classification
- We have NO evidence that routing does anything beyond what a dense layer would

**What we need to prove:**
- That routing sparsity provides computational or representational value
- That modularity enables zero-forgetting continual learning
- That routing entropy is a meaningful uncertainty signal
- That dynamic capacity scales better than fixed capacity

**Red lines (if crossed, we must radically change direction):**
- B0 fails (routing is decorative): ABORT NGS
- C2 fails (no zero forgetting): NGS is not a lifelong learning solution
- D1 fails (no recovery from overlap): NGS is fragile, not robust

**The ideal trajectory:**
- If B0 passes → C1 succeeds (OOD paper) → C2 succeeds (lifelong learning paper) → D1 passes (robustness paper)
- This gives us 3 strong papers (B2, B3/B6 reframe, plus B4/B5)
- If any gate fails, we pivot immediately to what's left standing

---

## APPENDIX: Key Code Pointers

| Component | File | Purpose |
|-----------|------|---------|
| MultiLayerNGS | `ngs/models/ngs.py` | Main multi-layer class |
| Frozen projection test | `experiments/b0_frozen_projection.py` (TO BE CREATED) | Gate B0 |
| OOD entropy test | `experiments/c1_ood_entropy.py` (TO BE CREATED) | Gate C1 |
| Continual learning | `experiments/c2_continual.py` (TO BE CREATED) | Gate C2 |
| Dynamic capacity | `experiments/c3_dynamic_k.py` (TO BE CREATED) | Gate C3 |
| Diversity recovery | `experiments/d1_overlap_recovery.py` (TO BE CREATED) | Gate D1 |

---

**Next action:** Run B0 (frozen projection baseline). If it passes, the rest follows. If it fails, we know definitively that NGS routing is not the right substrate to build on.

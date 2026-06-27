# NGS: Final Technical Report

## Executive Summary

**NGS routing provides ZERO distinguishing value.** The architecture is equivalent to training a standard MLP on frozen random features. All claimed differentiating properties (continual learning, OOD detection, dynamic capacity, modular generalization) fail empirically. Papers 3 (Transformer FFN replacement) and 5 (Federated communication) have valid results and should proceed to publication.

## Experimental Validation

### Gate B0: Frozen Projection Baseline
- Setup: Freeze p_down and p_up after random initialization, train only mu/log_s/log_alpha/adapters
- d_latent sweep: {4, 8, 16, 32}
- Result: 83.46% accuracy at d_latent=32
- Criterion (>=80%): PASSED

### Control: MLP on Frozen p_down Features
- Setup: Freeze p_down, train 2-layer MLP with hidden dim = latent_dim * factor
- Configurations tested:
  - factor=0.5: 698 params -> 80.54%
  - factor=1.0: 1,386 params -> 81.94%
  - factor=2.0: 2,762 params -> 84.73% <- BEATS NGS B0
  - factor=4.0: 5,514 params -> 86.55%
  - factor=8.0: 11,018 params -> 88.57%
- Conclusion: MLP with 10x fewer parameters matches NGS performance

## Failed Gates Summary

| Gate | Claim | Result | Criterion | Status |
|------|-------|--------|-----------|--------|
| C1 | OOD detection via routing entropy | AUROC 0.33 | >=0.75 | FAILED |
| C2 | Continual learning via modularity | 56% forgetting | <10pp | FAILED |
| C3 | Dynamic capacity scaling | K stuck at 16 | Growth observed | FAILED |
| D1 | Overlap recovery via diversity loss | Both >93% | Diversity required | FAILED |

## Parameter Analysis (d_latent=32, K=32)

### NGS B0 Trainable Parameters
- DirectAdapterStore.W: K * d * d = 32 * 32 * 32 = 32,768
- Router.mu: K * d = 1,024
- Router.log_s: K * d = 1,024
- Router.log_alpha: K = 32
- Total: ~34,848 trainable parameters

### MLP Control Parameters
- factor=2.0: 32 * 64 + 64 * 10 + 64 + 10 = 2,762 parameters
- factor=4.0: 32 * 128 + 128 * 10 + 128 + 10 = 5,514 parameters

Efficiency ratio: MLP achieves same accuracy with ~10x fewer parameters.

## Historical Trajectory

### TODO 3 (Compute-Efficient Prioritization) - COMPLETED
- Multi-layer > single-layer
- Multi-head projection: 3% gain at 18x param efficiency

### TODO 10 (Honest Validation) - MOST FAILED
- EqProp + Mahalanobis: 66% MNIST vs 94% backprop (28pp gap)
- Autopoietic: Underperforms fixed K (33% vs 48%)
- 3DGS ingestion: Only passed track (100% success)

### TODO 11 (Post-Mortem Diagnostics)
- EP cosine correlation: -0.439 (anti-correlated, irreparable)
- Spectral norm: No effect (66% always)
- Multi-layer fix discovered: depth=4 achieves 95.83% MNIST

### TODO 12 (Multi-Layer Validation) - PARTIAL SUCCESS
- Depth=4 NGS: 95.83% MNIST (within 1pp of single-layer optimum)
- Gaussian lottery ticket: prune 75% -> 0.45pp drop (valid)
- p_down/p_up MLP projections fix saturation

### TODO 13 (Post-Validation Phase) - NEGATIVE RESULTS
- B0 passed but provided no discriminating insight
- C1-C3-D1 all failed definitively
- Methodological flaw revealed: unfrozen projections mask true capability

## Root Cause Diagnosis

1. Routing is Decorative: Gaussian centroids + top-k selection provides no algorithmic advantage over standard MLP layers.

2. Performance from Parameter Count: What appeared as "routing learning" was simply ~35K adapter parameters having sufficient capacity to fit MNIST.

3. Failed Differentiators:
   - OOD detection: Routing entropy AUROC=0.33 (random chance)
   - Continual learning: Catastrophic forgetting (56pp drop)
   - Dynamic capacity: No growth mechanism works
   - Overlap recovery: Diversity loss unnecessary

4. No Algorithmic Advantage: Multi-layer works but any MLP architecture works equally well. NGS adds complexity without benefit.

## Remaining Publishable Results

### Paper 3 (Transformer FFN Replacement) - ICLR 2027
- Data: MLP replacement shows compute-efficient gains
- Status: Strong empirical results, ready for draft

### Paper 5 (Federated Router Communication) - ICML 2027
- Data: Router-only parameter exchange achieves communication compression
- Status: Valid results, ready for draft

## Recommendation

Archive NGS as abandoned. Negative control (MLP outperforms NGS with 10x fewer params) definitively proves sparse routing adds no value. Publish Papers 3 and 5 with appropriate framing of lessons learned.
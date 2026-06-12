# MNGS / LeanNGS Development Plan (TODO2.md)

Single-source-of-truth plan consolidating all incomplete items from TODO.md, prototype1
ideas to absorb, and new validation targets.

**Legend**: (N) = new item, (I) = inherited incomplete from TODO.md, (A) = absorb from
prototype1, (V) = validation/dataset addition.

---

## Phase 0: Gaps in Existing Modular Framework

### 0.1 Optimizer State Interpolation (I)
`mngs/modules/topology_managers.py:173` — `ContinuousDensityManager.adapt_topology()` only
prunes today.  It must be extended to implement the full differentiable split-gate (gamma)
mechanism described in TODO.md Section 4 point 3: when a unit's gamma triggers a split,
Adam `exp_avg` / `exp_avg_sq` should be smoothly interpolated between parent and child
rather than zeroed.  This is the single most impactful missing piece for training stability.

### 0.2 MemoryManagement — Dead Enum (I)
`mngs/core/config.py:25-29` defines `MemoryManagement` with three strategies, but the
value is **stored and never read** anywhere:
- No code branches on `config.memory_management`.
- `STRICT_CAPACITY` profile exists (`profiles.py:69`) but nothing enforces a capacity cap.
- `DYNAMIC_GROWTH` has no implementation in the modular framework.
- `PRE_ALLOCATED_MASKED` is the de facto (and only) working mode.

**Fix**: wire `MemoryManagement` into `MNGS.__init__` so it controls tensor allocation,
masking, and pruning/spawning logic.  The `DYNAMIC_GROWTH` strategy can be modeled on
prototype1's `_adapt_density_dynamic` (see 0.3).

### 0.3 Absorb prototype1 Ideas (A)

| Idea | prototype1 Location | Action |
|------|---------------------|--------|
| Hook-based grad EMA (`register_hook`) | `model.py:38` | Replace manual `update_grad_ema()` call pattern — hook fires automatically on backward |
| Dynamic tensor resizing (real `DYNAMIC_GROWTH`) | `model.py:203-305` (`_adapt_density_dynamic`) | Port as a new `DynamicGrowthManager` that calls `_resize_params()` and patches Adam state |
| Optimizer state patching for changed shapes | `model.py:281-303` | Extract into a shared utility `pytorch_utils.py` |
| `FixedLeanNGS` as simpler baseline | `baselines.py:19` | Add a `FixedCapacityMNGS(n_units)` convenience wrapper for ablation |
| Catastrophic-forgetting-over-time plot | `experiment.py:54-73` | Port to `experiments/plotting.py` |

### 0.4 LSH Router Stub (I)
`mngs/modules/routers.py:177-223` — `LSHRouter.forward()` returns random pseudo-distances
and never actually performs LSH.  Either implement real bucket-based LSH (hash tables or
multi-probe), or replace with a note that this is deferred.

### 0.5 FactorizedRouter + ParameterStore Mismatch (I)
`mngs/model.py:174-200` — `_forward_factorized` converts subspace-local indices to global
indices via `indices + s * units_per_space`.  This assumes a contiguous, static allocation
per subspace, which breaks under pre-allocated masked memory when units are at arbitrary
positions.  **Fix**: either (a) give `FactorizedRouter` its own `parameter_store` per
subspace, or (b) teach `MNGS` to use per-subspace parameter stores that the router owns.

### 0.6 ContinuousDensityManager — Gate Initialization Issue
`topology_managers.py:193-197` — `initialize_gates()` creates `nn.Parameter` tensors as
plain attributes, not registered sub-modules.  These won't be picked up by optimizer
`.parameters()` calls.  Move to `nn.Parameter` registration or add to `MNGS` as a
top-level buffer.

### 0.7 Silent-Drop Bug: `epochs_per_task` vs `epochs` Mismatch (N·BUG)
`experiments/runner.py:51` — `train_kwargs = asdict(config.train)` produces
`epochs_per_task=N`, but every trainer function has parameter `epochs=5`.  The trainer
receives `epochs_per_task` in `**kwargs` where it's silently discarded, so **every
experiment trains for exactly 5 epochs regardless of `TrainConfig.epochs_per_task`**.

Same bug in `ablation.py:96` and `quick_runner.py:32-103`.  **Fix**: in `runner.py` and
`ablation.py`, rename `epochs_per_task -> epochs` before calling the trainer, or use a
unified adapter.

### 0.8 Double-Softmax in `entropy_loss` — Monolithic Path (N·BUG)
`mngs/model.py:216-219` — `MonolithicRouter.forward()` already returns softmax-normalized
weights (`F.softmax(topk_vals, dim=-1)`).  Then `entropy_loss` applies
`p = F.softmax(weights, dim=-1)` a second time.  The factorized branch correctly uses
`p = weights` (line 212) without a second softmax.

**Fix**: align the monolithic branch with the factorized branch: remove the second
softmax.

### 0.9 FWT Metric Uses Wrong Random Baseline (N·BUG)
`experiments/metrics.py:64` — `random_baseline` defaults to `0.1` (10-class random
guess).  For Split-MNIST (2 classes) the correct baseline is `0.5`; for
Split-CIFAR100-20 (5 classes) it is `0.2`.  Forward Transfer is computed against a wrong
value for most experiments.

**Fix**: accept `random_baseline` as a parameter to `compute_metrics` and have callers
compute `1.0 / output_dim`.

### 0.10 `ContinuousDensityManager` Params Outside `nn.Module` (N·BUG)
`mngs/modules/topology_managers.py:8-23` — `BaseTopologyManager(ABC)` is not an
`nn.Module`.  When `ContinuousDensityManager.initialize_gates()` (line 191) creates
`self.split_gate = nn.Parameter(...)`, the parameter is **not registered** with any
module.  It will not appear in `model.parameters()`, will not be moved by `.to()`, and
will not receive gradients.

**Fix**: either make `BaseTopologyManager` inherit from `nn.Module`, or store gates in
the `MNGS` module itself.

### 0.11 `hpo.py` Sets `config.train.top_k` — Not a Dataclass Field (N·BUG)
`experiments/hpo.py:34` — `config.train.top_k = trial.suggest_categorical(...)` sets an
attribute `top_k` on the `TrainConfig` instance, but `TrainConfig` does not have a `top_k`
field.  `asdict(config.train)` only includes actual dataclass fields, so **`top_k` is
never serialized or passed to the model**.  The HPO is tuning a dead parameter.

**Fix**: move `top_k` to `ModelConfig` where it belongs, or add it to `TrainConfig`.

### 0.12 `__import__` Hack in `quick_runner.py` (N·BUG)
`experiments/quick_runner.py:259,270,283` — `__import__('experiments.config').config.TrainConfig`
behaves differently for namespace packages (no `__init__.py` in `experiments/`).  Python's
`__import__` returns the **top-level** package for dotted names, so
`__import__('experiments.config')` returns `experiments`, and `.config.TrainConfig`
accesses it through the namespace.  This is fragile and implementation-dependent.

**Fix**: use `from experiments.config import ...` at the top of the file.

### 0.13 `tau` Changed from Learnable to Fixed (N·Design)
`lean_ngs.py:18` — `tau` was `nn.Parameter(torch.tensor(1.0))` (learnable).  In the
modular code, `mngs/modules/routers.py:39` stores `self.tau = tau` as a plain float.  The
routing temperature can no longer be optimized.

**Fix**: restore `nn.Parameter` or keep as config-only (document the change).

### 0.14 `diversity_weight` / `diversity_loss` Never Called (N·Design)
`mngs/core/config.py:61` defines `diversity_weight: float = 0.01`.  `mngs/model.py:246-257`
defines `diversity_loss()`.  `experiments/mngs_trainer.py:24` accepts `diversity_weight`
as a param.  But the trainer **never calls `model.diversity_loss()`** in the loss
computation (line 72).  The diversity loss is dead code.

**Fix**: add `diversity_weight * model.diversity_loss()` to the total loss in
`train_mngs`.

### 0.15 `config.output_dim` Never Read by Model (N·Design)
`mngs/core/config.py:37` defines `output_dim: int = 64`.  `MNGS.__init__(d_out)` uses
`d_out` directly; `config.output_dim` is never accessed.

**Fix**: either remove from config or use it during construction.

### 0.16 `ModelConfig` Dead Fields (N·Design)
`experiments/config.py:16-19` — `gamma_init`, `tau_init`, `mu_init_std`, `w_init_std` are
never extracted by `create_lean_ngs` (which only reads `d_latent, k_init, max_k, top_k,
lora_rank`).  These are silently dropped in all experiments.

**Fix**: remove unused fields or wire them into `LeanNGS.__init__`.

### 0.17 Enum String Comparisons vs Member Identity (N·Design)
`mngs/model.py:54,61,70,84,91,106,112` — uses `routing.name == "MONOLITHIC_MAHALANOBIS"`
instead of `routing == RoutingStrategy.MONOLITHIC_MAHALANOBIS`.  Works but fragile if
enum names are ever renamed.

**Fix**: convert to enum member comparison throughout `_build_*` methods.

### 0.18 Integer Division Losing Capacity (N·Design)
`mngs/model.py:62` — `units_per_space = k_init // num_subspaces`.  If `k_init=10,
num_subspaces=4`, total capacity = 8 instead of 10 (20% loss).  Same for
`d_latent // num_subspaces` in `routers.py:124`, which can also produce zero-dim
projectors when `num_subspaces > d_latent` (latent crash).

**Fix**: use floor division but warn if remainder > 0; guard against zero dimensions.

### 0.19 `log_alpha` Halving Is Mathematically Wrong (N·Design)
`lean_ngs.py:133-134,145` — `self.log_alpha.data[...] += torch.log(torch.tensor(0.5))`
intends to halve the opacity, but `sigmoid(x + log(0.5)) ≠ 0.5 * sigmoid(x)`.  The
halving intent is not achieved in probability space.

Same pattern in `topology_managers.py:130-132`.

**Fix**: halve in probability space: `p = sigmoid(log_alpha); new_p = p * 0.5; new_log_alpha =
logit(new_p)`.

### 0.20 Missing Seed Setting (N·Design)
- `train_split_mnist.py` — no `torch.manual_seed` or `random.seed` call at all (9.1).
- `experiments/hpo.py:38-40` — all trials use the same `seed=42`, making HPO comparisons
  less statistically sound (9.2).
- `tests/` — no seed setting, non-deterministic but functionally correct (11.1).

### 0.21 Missing Tests (I) — Updated
| Test | File |
|------|------|
| `ContinuousDensityManager` with actual gradient flow through gamma | `tests/test_topology_and_training.py` |
| End-to-end multi-task training loop (5-task Split-MNIST) with metric verification | new file `tests/test_end_to_end.py` |
| `MemoryManagement` wiring (all 3 strategies) | `tests/test_memory_management.py` |
| Factorized router + hypernetwork store gradient flow | `tests/test_cfg_net_profile.py` |
| Profile comparison: same input yields same-dim output for all 4 profiles | augment `test_topology_and_training.py::test_all_profiles_smoke` |

---

## Phase 1: Validation Targets

### 1.1 Existing Datasets Needing Full Runs (V)
These are already configured in `experiments/config.py` but have **no results** (only
`Split-MNIST` × 1 seed exists in `results/test/`):

| Dataset | Config Key | Tasks | Notes |
|---------|-----------|-------|-------|
| Split-FashionMNIST | `split_fashion` | 5 | Same structure as MNIST |
| Split-CIFAR10 | `split_cifar10` | 5 | Needs backbone or larger latent |
| Split-CIFAR100 | `split_cifar100` | 10 | Same |
| Permuted-MNIST | `permuted_mnist` | 10 | 10 permutations, domain-incremental |
| Rotated-MNIST | `rotated_mnist` | 10 | Angle-based domain shift |
| Blurry-MNIST | `blurry_mnist` | 5 | Gaussian blur domain shift |
| Noisy-MNIST | `noisy_mnist` | 5 | Noise domain shift |
| SVHN | `svhn` | 5 | Street-view house numbers |
| Digits (MNIST subset) | `digits` | 5 | 2-class splits from MNIST |
| Split-CIFAR100-20 | `split_cifar100_20` | 20 | 5-class splits, 20 tasks |

Each should be run with seeds [42, 123, 456] across all 4 MNGS profiles + 5 baselines
(MLP, ER, EWC, SI, LwF) to fill the comparison matrix.

### 1.2 New Datasets / Tasks to Add (V)

#### A. Full MNIST (Non-Split)
For capacity scaling analysis: does LeanNGS/MNGS still show zero-forgetting properties
when all 10 digits are learned simultaneously?  Useful as a "saturation test" showing
whether dynamic capacity adds value over a fixed-capacity model.

**How**: Add `full_mnist` to `experiments/config.py` with `n_tasks=1`,
`classes_per_task=10`, `output_dim=10`.

#### B. TinyShakespeare (Language Domain)
Introduces a **different modality** (character-level LM) to test whether the Gaussian
routing mechanism generalizes beyond vision.  This is the strongest "out-of-domain"
validation.

**How**: New file `experiments/datasets_tinyshakespeare.py` — download the raw text,
create fixed-length chunk sequences, split into "tasks" by chapter or by character n-gram
distribution shift.  Requires `output_dim` = vocab size (~65 chars).

**Models**: Only need `lean_ngs` / `mngs_baseline` vs `mlp` / `lwf` — cross-entropy on
next-char prediction.

#### C. CartPole / RL Validation (Lower Priority)
`experiments/backbones.py` already supports image backbones.  For RL, we'd need to wrap
the NGS head as a policy network for OpenAI Gym.  This is deferred until the supervised
CL benchmarks are complete — noted here for awareness only.

#### D. Synthetic Blob / MoG Dataset
A controlled 2D synthetic dataset where ground-truth clusters are known.  Useful for
visualizing Gaussian unit placement and split events.  Not a "popular" dataset but high
diagnostic value.

**How**: `datasets.py` — `make_blobs`-style Gaussian mixture, assign each cluster as a
"task" in class-incremental format.

---

## Phase 2: Systematic Ablation & Analysis

### 2.1 Run Phase-1 of TODO.md Strategy (I)
TODO.md §3 prescribes a structured 3-phase ablation.  Only Phase 1 (baseline anchor) has
been partially run.  Execute:

| Phase | Configs | What it tells us |
|-------|---------|------------------|
| 1 | `Baseline_LeanNGS` | Lock accuracy/forgetting/memory/flops baselines |
| 2a | Swap routing → `FACTORIZED_SUBSPACE` | Routing FLOPs change without accuracy drop? |
| 2b | Swap storage → `HYPERNETWORK_GENERATED` | Memory savings without accuracy loss? |
| 2c | Swap topology → `CONTINUOUS_DENSITY` | Smoother loss curves? (requires 0.1 fix) |
| 3 | Winners from 2a+2b+2c combined | Synergistic gains? |

### 2.2 Per-Dataset Hyperparameter Tuning
The `BEST_CONFIGS` dict in `comprehensive_eval.py:18-79` has hard-coded values from
preliminary runs.  For CIFAR-100, Permuted-MNIST, and SVHN, we need dedicated HPO runs
(`experiments/hpo.py`).

### 2.3 Parameter-Matched Comparison
The current `max_k=448` was chosen to match MLP's ~534K parameters on MNIST.  For each
new dataset, compute the parameter-matched `max_k` for a fair comparison.  Add a helper
in `experiments/config.py`.

---

## Phase 3: Infrastructure

### 3.1 CI / Reproducibility
- Add `pytest` to CI (GitHub Actions `.github/workflows/test.yml`).
- Pin dependency versions in `pyproject.toml` (currently `torch>=2.0`, `numpy`).
- Add `seed == 42` reproducible defaults to all experiment configs.
- After fixing 0.6, ensure `torch.save` / `torch.load` round-trips work for MNGS models.

### 3.2 Documentation
- Fill `README.md` with: architecture diagram, running instructions, reproduction
  commands for each profile, expected results table.
- Add docstrings to all exported functions (already good, but some like `mngs_trainer.py`
  functions are missing param docs).

### 3.3 Extensibility Hooks
TODO.md §4 describes:
- ✅ Factory pattern (`build_mngs`)
- ✅ Unified forward signature (indices + weights)
- ❌ Custom optimizer wrapper for `ContinuousDensityManager` — see 0.1

Add `build_mngs` to `mngs/__init__.py` docs.

---

## Execution Priority

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P0 | 0.1 ContinuousDensityManager differentiable split gate | 3d | Enables CFG-Net topology hypothesis |
| P0 | 0.2 Wire MemoryManagement into model | 2d | Unblocks STRICT_CAPACITY + DYNAMIC_GROWTH |
| P0 | **0.7 epochs_per_task silent-drop bug** | **0.5d** | **All experiments train for wrong duration** |
| P0 | **0.8 Double-softmax in entropy_loss** | **0.25d** | **Monolithic routing entropy regularization wrong** |
| P0 | **0.9 FWT wrong random baseline** | **0.25d** | **All FWT metrics invalid** |
| P0 | **0.10 nn.Parameter outside nn.Module** | **0.5d** | **ContinuousDensityManager has no learnable params** |
| P0 | **0.11 HPO tunes dead top_k param** | **0.25d** | **HPO silently ineffective** |
| P0 | 1.1 Run all existing datasets × models × 3 seeds | 5d | Fills the evidence matrix |
| P1 | 0.3 Absorb prototype1 ideas | 1d | Cleaner code, dynamic growth working |
| P1 | 0.5 FactorizedRouter ↔ ParameterStore integration | 2d | Factorized + pre-allocated compat |
| P1 | 0.12 __import__ hack in quick_runner.py | 0.25d | Import reliability |
| P1 | 0.14 diversity_loss never called | 0.25d | Dead feature reactivation |
| P1 | 0.17 Enum strings vs member identity | 0.5d | Robustness |
| P1 | 0.18 Integer division losing capacity | 0.5d | Correctness at edge configs |
| P1 | 0.19 log_alpha halving math error | 0.5d | Topology math correctness |
| P1 | 1.2 New datasets (Full MNIST, TinyShakespeare) | 2d | Modality diversity |
| P1 | 2.1 Structured ablation (3 phases) | 3d | Core research deliverable |
| P2 | 0.13 tau learnable → fixed regression | 0.5d | Optimization fidelity |
| P2 | 0.15 config.output_dim never read | 0.25d | Config hygiene |
| P2 | 0.16 ModelConfig dead fields | 0.25d | Config hygiene |
| P2 | 0.20 Missing seed setting | 0.5d | Reproducibility |
| P2 | 0.21 Missing tests | 2d | Regression safety |
| P2 | 2.2 Per-dataset HPO | 2d | Best-practice tuning |
| P2 | 3.1 CI / seed pinning | 0.5d | Reproducibility |
| P3 | 0.4 LSH Router full impl | 3d | Research-scale routing |
| P3 | 0.6 Gate init fix (subsumed by 0.10) | — | Merged into 0.10 |
| P3 | 3.2 README / docs | 1d | Usability |

---

## Key

| Tag | Meaning |
|-----|---------|
| (I) | Inherited incomplete from original TODO.md |
| (A) | Absorb from prototype1/ |
| (V) | Validation / dataset addition |
| (N·BUG) | New bug found during audit |
| (N·Design) | New design flaw found during audit |
| P0-P3 | Execution priority |

**Total P0 items**: 8 — 6 newly discovered bugs + 2 architecture gaps + fill results matrix.
**Total P1 items**: 9 — absorb prototype1, fix import/design issues, new datasets, ablation.
**Total P2+P3 items**: 10 — polish, docs, edge features, tests.
**Grand total**: 27 tracked items (was 14 before audit).

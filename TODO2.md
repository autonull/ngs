# MNGS — Development Plan

Self-contained execution guide.  Work through milestones in order.  Every task
has an exact file path, a concrete change, and a verification step.  Do not
skip tasks within a milestone.

**Legend**: `[BUG]` = produces silently wrong results today.  `[DESIGN]` =
incomplete or fragile but not producing wrong results.

---

## M1: FIX — Correctness Sprint

*Goal: all bugs that silently skew experimental results are eliminated.*
**Status: ✅ COMPLETE (7/7)**

### M1.1 Training duration bug `[FIXED]`
- **Area**: `experiments/runner.py:51`, `experiments/ablation.py:96`,
  `experiments/quick_runner.py:144`
- **Problem**: `asdict(config.train)` produces `epochs_per_task=N`, but all
  trainer functions accept `epochs` (default 5).  `epochs_per_task` lands in
  `**kwargs` and is **silently dropped**.  Every experiment trains for 5 epochs
  regardless of config.
- **Fix**: Create a shared adapter in `experiments/config.py`:
  ```python
  def as_train_kwargs(cfg: TrainConfig) -> dict:
      kw = asdict(cfg)
      kw['epochs'] = kw.pop('epochs_per_task', 5)
      return kw
  ```
  Replace every `asdict(config.train)` with `as_train_kwargs(config.train)` in
  `runner.py`, `ablation.py`, `quick_runner.py`, and `comprehensive_eval.py`.
  Also fix the `train_kwargs = asdict(config.train)` in `mngs_trainer.py:61`
  and `lean_ngs_trainer.py:51` if they shadow the runner's dict.
- **Also fix** same pattern in `train_backbone_ngs` (`quick_runner.py:137`): its
  `train_kwargs` dict is built manually, so just rename the key from
  `'epochs'` to match the config field name, or use the same adapter.
- **Verification**: After fix, add a test: configure `TrainConfig(epochs_per_task=1)`,
  run training, assert that the trainer's `range(epochs)` loop runs exactly 1
  iteration per epoch (not 5).  Or simply: `pytest` passes on new
  `test_end_to_end.py` (#M4.1).

### M1.2 Double-softmax in entropy_loss `[FIXED]`
- **Area**: `mngs/model.py:216-219`
- **Problem**: `MonolithicRouter.forward()` already returns
  `F.softmax(topk_vals, dim=-1)` as its second output.  Then `entropy_loss`
  (line 218) applies `F.softmax(weights, dim=-1)` again, double-normalizing.
  The factorized branch (line 212) correctly uses raw weights.
- **Fix**: Change the monolithic branch to match the factorized branch:
  ```python
  # p = F.softmax(weights, dim=-1)  ← DELETE
  p = weights                        ← ADD, same as factorized path
  ```
- **Verification**: `pytest tests/test_topology_and_training.py::test_entropy_loss`
  passes.  Also add an explicit check: entropy of uniform weights should be
  `log(K)`.  With double-softmax it is not.

### M1.3 FWT random baseline is always 0.1 `[FIXED]`
- **Area**: `experiments/metrics.py:64,91`
- **Problem**: `compute_metrics()` hard-codes `random_baseline=0.1` (correct only
  for 10-class tasks).  Split-MNIST has 2 classes (random=0.5),
  Split-CIFAR100-20 has 5 classes (random=0.2).  FWT is computed against a wrong
  value for most experiments.
- **Fix**:
  1. Remove the default argument: `def compute_metrics(accuracy_matrix, random_baseline):`
  2. In `CLMetrics`, store `random_baseline` (no default).
  3. Update every call site to pass `1.0 / output_dim`:
     - `experiments/runner.py:153` — has access to `config.output_dim`
     - `experiments/ablation.py:134` — has access to the config
     - `experiments/quick_runner.py:216` — needs `output_dim` passed through
     - `experiments/online_eval.py` — uses `compute_metrics` too
     - `experiments/plotting.py:241` — calls `aggregate_results` not `compute_metrics`
       directly (check).
  4. Update `CLMetrics.to_dict()` and `CLMetrics.save()` to include the corrected
     values.
- **Verification**: `pytest` passes.  Manual: `compute_metrics(acc_matrix_2class, 0.5)`
  returns different FWT from `compute_metrics(acc_matrix_2class, 0.1)`.

### M1.4 HPO tunes a dead `top_k` field `[FIXED]`
- **Area**: `experiments/hpo.py:34`
- **Problem**: `config.train.top_k = trial.suggest_categorical(...)` sets an
  attribute on `TrainConfig`, but `TrainConfig` has no `top_k` dataclass field.
  `asdict(config.train)` does not include it.  The hyperparameter is optimized
  but never passed to the model.
- **Fix**: `top_k` already exists in `ModelConfig` (line 14).  Change HPO to
  write to `config.model.top_k` instead of `config.train.top_k`.
- **Verification**: After fix, run `hpo.py` with 1 trial and print the config.
  `config.model.top_k` must be the suggested value.

### M1.5 `nn.Parameter` registered outside `nn.Module` `[FIXED]`
- **Area**: `mngs/modules/topology_managers.py:191-198`
- **Problem**: `ContinuousDensityManager` inherits from `BaseTopologyManager(ABC)`,
  not `nn.Module`.  `nn.Parameter(...)` creates an unregistered tensor that
  will not appear in `model.parameters()`, will not be moved by `.to()`, and
  will not receive gradients.
- **Fix**: Move the split gate into the `MNGS` model itself.
  ```python
  # In MNGS.__init__ (mngs/model.py):
  # After line 43, add:
  self.split_gate = nn.Parameter(torch.full((config.max_k,), 0.0))
  self.activation_density = torch.zeros(config.max_k)
  self.error_density = torch.zeros(config.max_k)
  ```
  Then in `ContinuousDensityManager`, accept the gate from `model` instead of
  creating it.  Remove `initialize_gates()` entirely.
- **Verification**: After fix, `model.parameters()` includes the gate.
  `model.to(device)` moves it.  `out.sum().backward()` shows non-None
  `model.split_gate.grad`.

### M1.6 `__import__` hack in quick_runner `[FIXED]`
- **Area**: `experiments/quick_runner.py:259,270,283`
- **Problem**: `__import__('experiments.config').config.TrainConfig(...)` is
  fragile with namespace packages (no `__init__.py` in `experiments/`).
- **Fix**: Add `from experiments.config import TrainConfig, ModelConfig` at the
  top of the file and replace all three `__import__` calls with direct
  references.
- **Verification**: `python -m experiments.quick_runner --help` runs without
  import errors.

### M1.7 Missing random seeds `[FIXED]`
- **Area**: `train_split_mnist.py`, `tests/*.py`
- **Problem**: `train_split_mnist.py` sets no seeds at all.  Tests have no seed
  setting.
- **Fix**:
  1. `train_split_mnist.py`: add `torch.manual_seed(42); np.random.seed(42); random.seed(42)`
     at the top of `main()`.
  2. Each test file: add a `def setup_method(self):` or module-level
     `torch.manual_seed(42)` in the test fixtures.  Or add a conftest.py.
  3. Create `tests/conftest.py`:
     ```python
     import pytest, torch, numpy as np, random
     @pytest.fixture(autouse=True)
     def seed_all():
         torch.manual_seed(42)
         np.random.seed(42)
         random.seed(42)
     ```
- **Verification**: Running the same test twice produces identical outputs
  (pick a test that samples random initialization).

---

## M2: BUILD — Complete the Framework

*Goal: all modular abstractions are fully wired, no stubs, no dead code.*
**Status: ◐ 9/10 complete — M2.3 not done (non-blocking for existing profiles)**

### M2.1 Wire `config.memory_management` into the model `[FIXED]`
- **Area**: `mngs/model.py`, `mngs/core/config.py`
- **Depends on**: M1.5 (param registration)
- **Problem**: `MemoryManagement` enum is defined, stored in config, set by
  every profile, but never read.  Three strategies exist only as labels:
  - `DYNAMIC_GROWTH`: no implementation in modular code.
  - `PRE_ALLOCATED_MASKED`: currently the de facto default (hard-coded).
  - `STRICT_CAPACITY`: profile exists, but nothing enforces a capacity cap.
- **Fix**:
  1. In `MNGS.__init__`, read `config.memory_management` and branch:
     - `PRE_ALLOCATED_MASKED`: current behavior (pre-allocate `max_k` tensors, use `active_mask`).
     - `DYNAMIC_GROWTH`: start with `k_init` units, grow via `torch.cat` when
       spawning/splitting — implement using prototype1's `_adapt_density_dynamic`
       (see `prototype1/lean_ngs/model.py:203-305`).
     - `STRICT_CAPACITY`: same as PRE_ALLOCATED but prune to stay under `max_k`
       before spawning — add a budget check in `adapt_density()`.
  2. Remove the early-return heuristic at `topology_managers.py:68` that gates
     on `hasattr(model.router, 'active_mask')` — let the strategy decide.
  3. `FactorizedRouter` also needs a `DYNAMIC_GROWTH` path if used in that
     mode (per-subspace tensor resizing).
- **Verification**: Each strategy does what the name says:
  - `PRE_ALLOCATED_MASKED`: `len(model.router.mu) == max_k` always.
  - `DYNAMIC_GROWTH`: `len(model.router.mu)` grows by splitting.
  - `STRICT_CAPACITY`: `model.K <= max_k` after every `adapt_density` call.

### M2.2 ContinuousDensityManager: implement differentiable split gate `[FIXED]`

- **Area**: `mngs/modules/topology_managers.py:173-222`
- **Depends on**: M1.5, M2.1
- **Fix implemented**: `MNGS.forward()` updates density EMA. `split_gate_loss()` method added. `ContinuousDensityManager.adapt_topology()` executes splits when `sigmoid(gamma) > 0.5 & error_density > 1e-3`.
- **Verification**: `pytest` passes.
- **Area**: `mngs/modules/topology_managers.py:173-222`
- **Depends on**: M1.5 (gate registration), M2.1 (MemoryManagement wiring)
- **Problem**: `adapt_topology()` only prunes (lines 210-221).  It always
  returns `(num_pruned, 0, 0)`.  `split_gate`, `activation_density`, and
  `error_density` are never updated.
- **Fix—three sub-steps**:  
  **M2.2a — Density tracking**: Update `activation_density` and `error_density`
  EMAs after every forward pass.  This requires integration in `MNGS.forward()`:
  ```python
  # After the weighted combination (model.py:169), update densities:
  if isinstance(self.topology_manager, ContinuousDensityManager):
      # Increase activation density for selected units
      # Increase error density proportional to loss contribution
      activation_density = ...
      error_density = ...
      self.topology_manager.activation_density = (
          self.topology_manager.density_decay * self.topology_manager.activation_density
          + (1 - self.topology_manager.density_decay) * activation_density
      )
      # same for error_density
  ```
  **M2.2b — Split gate loss**: Add a regularization term that pushes gamma to
  {0, 1} based on local error density.  Add to the model's forward pass:
  ```python
  gamma_reg = gamma * (1 - gamma) * error_density  # push to extremes where error is high
  ```
  **M2.2c — Split execution**: When gamma > 0.5 and error_density > threshold,
  execute a split by creating a child unit with perturbed code / mu, reset
  gamma to 0 for both parent and child.  Interpolate Adam states.
- **Verification**: `model.adapt_density()` returns `(0, N>0, 0)` after enough
  forward+backward steps on data with high local error.  `model.K` increases.

### M2.3 FactorizedRouter + ParameterStore integration `[NOT DONE]`

`[DESIGN]` = incomplete but not producing wrong results. Only exercised when `DYNAMIC_GROWTH` paired with `FACTORIZED_SUBSPACE` — no current profile does this. No blocking issues.
- **Area**: `mngs/model.py:174-200`
- **Depends on**: M2.1
- **Problem**: `_forward_factorized` converts subspace-local indices to global
  indices via `indices + s * units_per_space`.  This assumes contiguous,
  static allocation per subspace.  With pre-allocated masked memory or dynamic
  growth, active units can be at arbitrary positions, making this conversion
  invalid.
- **Fix**: Two options — pick one:
  - **(a) Per-subspace parameter store**: Give `FactorizedRouter` its own
    `ParameterStore` per subspace (a `ModuleList` of `DirectAdapterStore` or
    `HypernetworkStore`).  `FactorizedRouter.forward()` returns indices that
    are local per subspace; `_forward_factorized` uses `router.param_stores[s]`.
  - **(b) Flat index conversion**: Maintain a mapping tensor `subspace_to_flat`
    inside `FactorizedRouter` that maps `(s, local_idx) -> global_flat_idx`.
    Update this mapping after every topology change.  Simpler but requires
    `FactorizedRouter` to participate in memory management.
- **Verification**: Same forward/backward results whether using `MONOLITHIC`
  vs `FACTORIZED` with `num_subspaces=1` (should produce identical global
  behavior).  Cross-reference with `_forward_standard`.

### M2.4 Wire `diversity_loss` into training `[FIXED]`
- **Area**: `experiments/mngs_trainer.py:72`
- **Problem**: `diversity_weight` is accepted as a param but `diversity_loss()`
  is never called in the loss sum.
- **Fix**: Add diversity loss to the total:
  ```python
  total_loss = ce_loss + kd_weight * kd_loss + entropy_weight * entropy_loss
  if diversity_weight > 0:
      total_loss += diversity_weight * model.diversity_loss()
  ```
- **Verification**: When `diversity_weight > 0`, loss value differs from
  when it is 0.  Test with `model.diversity_loss().backward()` — gradients
  flow into `model.router.mu`.

### M2.5 Absorb prototype1 hook-based grad EMA `[FIXED]`
- **Area**: `mngs/model.py:221-232` (current `update_grad_ema`), `mngs/modules/routers.py`
- **Problem**: Current code calls `model.update_grad_ema()` manually in the
  training loop (`mngs_trainer.py:76`).  If the caller forgets, the EMA never
  updates and topology stalls.
- **Fix**: Replace manual call with `torch.Tensor.register_hook` inside
  `MonolithicRouter.__init__()`:
  ```python
  self.mu.register_hook(self._update_mu_grad_ema)
  ```
  Move the EMA tracking into the router (it knows its own `mu`).  Keep
  `update_grad_ema()` as a public method for backward compat but deprecate it.
- **Verification**: Remove `model.update_grad_ema()` from `mngs_trainer.py:76`.
  Run training — `model.grad_mu_ema` still updates (non-zero after backward).

### M2.6 Fix enum string comparisons `[FIXED]`
- **Area**: `mngs/model.py:54,61,70,84,91,106,112`
- **Problem**: Uses `routing.name == "MONOLITHIC_MAHALANOBIS"` instead of
  `routing == RoutingStrategy.MONOLITHIC_MAHALANOBIS`.  Fragile under rename.
- **Fix**: Replace all `.name == "STR"` with direct enum member comparison
  `== EnumType.MEMBER`.
- **Verification**: All `_build_*` branches still execute correctly.  No
  runtime error.  Test: `model = build_mngs(..., config=CFG_Net_Full())`
  builds a `FactorizedRouter` (not `Monolithic`).

### M2.7 Fix integer division capacity loss `[FIXED]`
- **Area**: `mngs/model.py:62`, `mngs/modules/routers.py:124`
- **Fix**: `k_init // num_subspaces` → `-(-a // b)` ceiling div. `d_latent // num_subspaces` → `max(d // n, 1)`.
- **Verification**: `FactorizedRouter(d_latent=32, num_subspaces=3, units_per_space=10)`
  has total units >= `k_init`.  `FactorizedRouter(d_latent=3, num_subspaces=10)`
  raises a warning but does not crash.

### M2.8 Fix `log_alpha` halving in probability space `[FIXED]`

### M2.9 Clean up dead config fields `[FIXED]`

### M2.10 Restore `tau` as learnable parameter `[FIXED]`

---

## M3: RUN — Execute the Validation Matrix

*Goal: every dataset × model × seed produces a result file.*

### M3.1 Test infrastructure
- **Depends on**: M1 complete (all bugs fixed), ideally M2 complete
- **Action**: Create `tests/test_end_to_end.py` with:
  1. Smoke test: run 1 epoch of Split-MNIST for each profile, assert no crash.
  2. Baseline repro: run `Baseline_LeanNGS` on Split-MNIST with accepted
     hyperparams (from `comprehensive_eval.py:BEST_CONFIGS['split_mnist']`),
     assert `avg_final_accuracy > 0.70` and `avg_forgetting < 0.10`.
  3. Exact-shape test for all 4 profiles on 3 different input dims
     (784, 3072, 64).
  4. Factorized vs monolithic equivalence check with `num_subspaces=1`.
  5. MemoryManagement smoke test: all 3 strategies do not crash.
- **Verification**: `python -m pytest tests/ -v` shows all green.

### M3.2 Run baseline LeanNGS on all configured datasets
- **Depends on**: M1 complete
- **Action**: For each config in `experiments/config.py:EXPERIMENTS`, run
  `python -m experiments.main --experiments <name> --models lean_ngs --seeds 42 123 456`.
  Save results to `./results/<name>/`.
- **Datasets**: `split_mnist`, `split_fashion`, `permuted_mnist`, `rotated_mnist`,
  `blurry_mnist`, `noisy_mnist`, `split_cifar10`, `split_cifar100`, `digits`,
  `svhn`, `split_cifar100_20`.  Use `BEST_CONFIGS` from `comprehensive_eval.py`
  for hyperparams (with M1.1 + M1.3 fix applied, these now work correctly).
- **Verification**: `ls results/` shows one JSON per (dataset, model, seed)
  combination.  Each JSON has a valid `accuracy_matrix` with shape
  `(n_tasks, n_tasks)`.

### M3.3 Run all MNGS profiles on all datasets
- **Depends on**: M2 complete
- **Action**: Same as M3.2, but include `--models mngs_baseline mngs_cfg_net
  mngs_ultra_edge mngs_abl_hyper`.
- **Verification**: 4 new result files per dataset per seed.

### M3.4 Run all baselines on all datasets
- **Depends on**: M1 complete
- **Action**: Same as M3.2, but include `--models mlp er ewc si lwf lora`.
- **Verification**: 6 new result files per dataset per seed.

### M3.5 Structural ablation (3 phases from TODO.md §3)
- **Depends on**: M3.3 results exist
- **Action**: For each dataset, compare:
  - **Phase 1**: `Baseline_LeanNGS` vs all baselines (locked baselines).
  - **Phase 2a**: `Baseline_LeanNGS` vs routing-only swap to `FACTORIZED`.
  - **Phase 2b**: `Baseline_LeanNGS` vs storage-only swap to `HYPERNETWORK`.
  - **Phase 2c**: `Baseline_LeanNGS` vs topology-only swap to `CONTINUOUS_DENSITY`.
  - **Phase 3**: `CFG_Net_Full` (all three swaps combined).
- **Output**: Generate `results/ablation_phase1.json`, `phase2a.json`, etc.
  containing `{dataset: {profile: metrics}}`.
- **Verification**: The ablation JSON is loadable by `experiments/ablation.py`.
  Bar plots in `plots/` show Δ for each swap.

### M3.6 HPO per dataset
- **Depends on**: M3.2 results
- **Action**: For datasets where `BEST_CONFIGS` doesn't have tuned parameters
  (check: `split_cifar100`, `split_cifar100_20`, `svhn`, `permuted_mnist`),
  run `python -m experiments.hpo --experiment <name> --trials 50`.
  Store best params in `BEST_CONFIGS`.
- **Verification**: `hpo_results/<name>/best_params.json` exists with tuned
  values.  Apply them and verify accuracy improves over the default.

### M3.7 Generate paper artifacts
- **Depends on**: M3.2 through M3.6 complete
- **Action**:
  1. `python -m experiments.comprehensive_eval --plot-only` generates all plots.
  2. `python -m experiments.report` generates `plots/validation_report.md`.
  3. `python -m experiments.profiling` generates the compute comparison table.
- **Verification**: `plots/` contains: radar chart, comparison bars, heatmaps,
  forgetting plots.  `validation_report.md` exists.

---

## M4: PACKAGE — Ship Ready

*Goal: reproducible, documented, CI-gated.*

### M4.1 Reader docs
- **Area**: `README.md`
- **Action**: Write README with:
  - What MNGS is (2-sentence elevator pitch).
  - Quick start: `pip install -e . && python -m tests` (or equivalent).
  - Reproduce command: `python -m experiments.main --experiments split_mnist --models lean_ngs`.
  - Architecture diagram (ASCII or mermaid).
  - Profiles table.
  - Reference to TODO.md for design narrative.
- **Verification**: README is readable end-to-end in 3 minutes.

### M4.2 CI
- **Area**: `.github/workflows/test.yml` (new file)
- **Action**: Create GitHub Actions workflow:
  ```yaml
  name: Test
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - uses: actions/setup-python@v4
          with: { python-version: '3.10' }
        - run: pip install -e ".[dev]"
        - run: pytest tests/ -v
  ```
- **Verification**: Push a branch; CI turns green.

### M4.3 Dependency pinning
- **Area**: `pyproject.toml`
- **Action**: Replace loose pins with tested versions.  Minimum:
  ```toml
  dependencies = [
      "torch>=2.0.0,<2.2",
      "numpy>=1.21,<1.25",
  ]
  [project.optional-dependencies]
  dev = ["pytest>=7.0,<8.0"]
  ```
- **Verification**: `pip install -e .` on a clean venv succeeds.

### M4.4 MNGS model serialization round-trip
- **Area**: `mngs/model.py` + test
- **Depends on**: M1.5 (param registration fix), M2.1, M2.2
- **Problem**: `torch.save` / `torch.load` must work for any MNGS
  configuration.
- **Action**: Write a test:
  ```python
  def test_model_serialization():
      for profile_fn in [Baseline_LeanNGS, CFG_Net_Full, Ultra_Edge_Sparse, Ablation_Hypernetwork_Only]:
          config = profile_fn()
          model = build_mngs(784, 10, config)
          x = torch.randn(4, 784)
          y1 = model(x)
          torch.save(model.state_dict(), '/tmp/mngs.pt')
          model2 = build_mngs(784, 10, config)
          model2.load_state_dict(torch.load('/tmp/mngs.pt'))
          y2 = model2(x)
          assert torch.allclose(y1, y2, atol=1e-6)
  ```
- **Verification**: The test passes for all 4 profiles.

### M4.5 New dataset: Full MNIST (non-split)
- **Area**: `experiments/config.py`
- **Action**: Add:
  ```python
  'full_mnist': ExperimentConfig(
      name='Full-MNIST',
      dataset='mnist',
      scenario='class_incremental',
      n_tasks=1,
      classes_per_task=10,
      input_dim=784,
      output_dim=10,
  ),
  ```
- **Run**: `python -m experiments.main --experiments full_mnist --models lean_ngs mlp`.
- **Verification**: Result file shows accuracy on 10-class MNIST (should be
  > 90% for MLP, competitive for LeanNGS).

### M4.6 New dataset: TinyShakespeare
- **Area**: new file `experiments/datasets_tinyshakespeare.py` + config
- **Action**:
  1. Create `datasets_tinyshakespeare.py`:
     - Download from `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
     - Build vocab of unique chars (~65).
     - Chunk into sequences of length 64.
     - Split into 5 "tasks" by chunks per act (or 5 contiguous splits).
     - Return `train_loader, test_loader` per task.
  2. Add config:
     ```python
     'tinyshakespeare': ExperimentConfig(
         name='TinyShakespeare',
         dataset='tinyshakespeare',
         scenario='class_incremental',
         n_tasks=5,
         classes_per_task=65,  # full vocab each task
         input_dim=64,  # sequence length
         output_dim=65,  # vocab size
     ),
     ```
  3. The model treats this as next-char prediction (cross-entropy).
     `input_dim = sequence_length` since characters are one-hot or embedding.
     Adjust `p_down` / `p_up` if needed.
- **Verification**: Model achieves > 20% accuracy on held-out chars (random is
  ~1.5%).

---

## Execution Order

```
M1.1 ──→ M1.2 ──→ M1.3 ──→ M1.4 ──→ M1.5 ──→ M1.6 ──→ M1.7
     (M1.1—M1.4 independent of each other; M1.5 independent of M1.1—M1.4)

M1 complete ──→ M2.1 ──→ M2.2 ──→ M2.3[TODO] ──→ M2.4 ──→ M2.5
                     │         │         │
                     └──── M2.6 ──→ M2.7 ──→ M2.8 ──→ M2.9 ──→ M2.10
                          (all complete except M2.3)

M1+M2(mostly) ──→ M3.1 ──→ M3.2 ──→ M3.3 ──→ M3.4
                               │         │
                               └──── M3.5 ──→ M3.6 ──→ M3.7
                                    (analysis, not training)

M1+M2+M3 complete ──→ M4.1 ──→ M4.2 ──→ M4.3 ──→ M4.4 ──→ M4.5 ──→ M4.6
```

Blocking chains:
- M1.5 → M2.1 → M2.2 → M2.3 (gate registration → MemoryManagement → split gate)
- M2.1 → M2.3 (FactorizedRouter needs MemoryManagement)
- M2.1 → M2.5 (dynamic growth needs MemoryManagement)
- M3.2 requires M1.1, M1.3 (training duration + metrics correctness)
- M3.3 requires M2 (framework complete, M2.3 is non-blocking)
- M3.4 requires M1 (no framework changes needed for baselines)

Everything else can be run in parallel within a milestone.

---

## Task Count

| Milestone | Tasks | Complete | Remaining | Training runs required |
|-----------|-------|----------|-----------|----------------------|
| M1 Fix | 7 | 7 | 0 | 0 |
| M2 Build | 10 | 9 | 1 (M2.3) | 0 |
| M3 Run | 7 | 0 | 7 | ~382 |
| M4 Package | 6 | 0 | 6 | 2 |
| **Total** | **30** | **16** | **14** | **~382** |

# NGS Broad Evaluation Plan (TODO4) — Multi-Domain Validation

**Goal**: Demonstrate NGSLayer generalizes beyond vision CL. One GPU, ~3 hours.

---

## Domain Matrix (Prioritized by Signal/Compute)

| Domain | Benchmark | NGS Advantage | Time | Priority |
|--------|-----------|---------------|------|----------|
| **Vision CL** | Split-CIFAR10/100 | Zero-forgetting + replay | 45 min | ✅ Done |
| **RL** | CartPole + domain shifts | Adaptive policy, no catastrophic forgetting | 30 min | **P0** |
| **RL** | MinAtar (5 games) | Multi-task single policy | 45 min | **P0** |
| **NLP** | TinyShakespeare (FFN replacement) | Token-level experts, dynamic capacity | 30 min | **P1** |
| **Time Series** | Synthetic control / UEA | Factorized subspaces per sensor | 20 min | **P1** |
| **Few-shot** | Omniglot 5-way 1-shot | Dynamic classifier head (§3.3) | 20 min | **P1** |
| **Density** | 2D toy (moons, swissroll) | Interpretable Gaussians | 10 min | **P2** |
| **Federated** | Federated MNIST (5 clients) | Hypernetwork code compression | 30 min | **P2** |

---

## What We've Overlooked (Critical Gaps)

### 1. **Dynamic Classifier Head** (TODO2 §3.3) — *High Impact, Low Effort*
```
Replace Linear(d_latent, n_classes) with 
NGSLayer(d_latent, d_latent, n_classes, n_experts=n_classes)
```
- Each class = one expert
- Open-set: novel input → no expert fires (low max routing weight)
- Class-incremental: new class = add expert
- Few-shot: new expert = few gradient steps on Gaussian

### 2. **RL Non-Stationarity** — *NGS's Killer App*
- CartPole: gravity/length/mass shifts every N episodes
- Measure: adaptation speed (episodes to recover), final performance
- Compare: NGS vs PPO + EWC vs PPO + replay

### 3. **Multi-Head = Sensor Fusion** (TODO2 §6.2)
- FactorizedRouter: one subspace per sensor modality
- Natural modality decomposition without hand-design

### 4. **Transformer FFN Replacement** (TODO2 §3.2)
- Single NGSLayer in one GPT block → token-level expert selection
- Dynamic capacity per token (some tokens need more experts)

### 5. **Hypernetwork Code Compression** (Federated)
- Client-specific codes (8-dim) instead of full model
- Server merges codes → global model

---

## Execution Order (3 Hours)

### Hour 1: RL + Dynamic Head (Highest Novelty)
```bash
# 1. CartPole domain shifts (30 min)
python -m ngs.benchmarks.rl --env CartPole-v1 \
    --domain-shift gravity --shift-every 50 --timesteps 50000 \
    --model ngs_layer_policy

# 2. Dynamic classifier head on Omniglot (20 min)
python -m experiments.dynamic_head --dataset omniglot --n_way 20 --k_shot 5

# 3. MinAtar 5 games multi-task (45 min) 
python -m ngs.benchmarks.rl --env MinAtar --multi-task \
    --games asterix,breakout,freeway,seaquest,space_invaders
```

### Hour 2: NLP + Time Series (Differentiation)
```bash
# 4. TinyShakespeare FFN replacement (30 min)
python -m ngs.benchmarks.nlp --dataset tinyshakespeare \
    --model ngs_transformer --replace-ffn-layer 3

# 5. UEA time series / synthetic control (20 min)
python -m ngs.benchmarks.time_series --dataset synthetic_control \
    --model ngs_factorized --n_subspaces 4
```

### Hour 3: Density + Federated + Ablations (Validation)
```bash
# 6. 2D density (moons, swissroll) - interpretability viz (10 min)
python -m ngs.benchmarks.density --dataset moons --visualize-gaussians

# 7. Federated MNIST 5 clients (30 min)
python -m ngs.benchmarks.federated --n-clients 5 --rounds 20 \
    --local-epochs 1 --model ngs_hyper

# 8. Critical ablations on best config (3 seeds) (40 min)
python -m experiments.ngs_layer_ablations --config best --seeds 42,123,456
```

---

## Success Criteria (Paper-Ready)

| Domain | Target | Why It Matters |
|--------|--------|----------------|
| **RL CartPole shifts** | Recover in <10 episodes after shift | Proves fast adaptation |
| **MinAtar multi-task** | Single policy >5-game PPO baselines | Proves capacity sharing |
| **TinyShakespeare** | Match perplexity with 30% fewer params | Proves Transformer generality |
| **Dynamic head Omniglot** | >95% 5-way 1-shot, open-set detection | Proves §3.3 vision |
| **Federated** | 90% central accuracy with 10× comm reduction | Proves code compression |

---

## Theoretical Claims to Validate

| Claim (from TODO2) | Experiment | Validation Metric |
|-------------------|------------|-------------------|
| "Zero forgetting" | Split-CIFAR + RL shifts | Forgetting <1% |
| "Self-regulating topology" | All domains | K stabilizes per task |
| "Parameter efficiency" | Hyper vs DirectAdapter | 10× params saved |
| "Interpretable routing" | Density + RL | Gaussian = behavior mode |
| "Multi-head = gradient fix" | Ablation table | 3% drop when removed |
| "Dynamic head = open-set" | Omniglot + novel classes | AUROC >0.9 |

---

## Compute Budget

| Phase | Experiments | Est. Time |
|-------|-------------|-----------|
| RL + Dynamic Head | 3 | 1.5 hr |
| NLP + Time Series | 2 | 0.75 hr |
| Density + Federated + Ablations | 3 | 0.75 hr |
| **Total** | **8** | **~3 hr** |

---

## Decision Points After Hour 1

- **If RL adaptation works** → Double down on RL + Robotics (TODO2 §6.2)
- **If TinyShakespeare matches** → Push Transformer FFN paper
- **If dynamic head succeeds** → Open-set/few-shot becomes headline
- **If all modest** → Focus on CL + Federated as core contributions

---

## Files to Create

1. `experiments/rl_ngs_policy.py` — NGSLayer as policy network
2. `experiments/dynamic_head.py` — Classifier head replacement
3. `experiments/transformer_ffn.py` — GPT block integration
4. `experiments/federated_hyper.py` — Code compression benchmark
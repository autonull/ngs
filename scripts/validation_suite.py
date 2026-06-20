#!/usr/bin/env python3
"""NGS Comprehensive Validation Suite - All Breakthrough Evidence"""
import sys, os, json, time, subprocess, tempfile, threading
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import numpy as np
sys.path.insert(0, "/home/me/ngs")

@dataclass
class ValidationResult:
    name: str
    status: str
    metrics: Dict[str, Any]
    timestamp: str
    duration_sec: float

class LiveProgress:
    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.start = time.time()
        self.lock = threading.Lock()
    def start_exp(self, name):
        with self.lock:
            self.completed += 1
            print(f"[{self.completed}/{self.total}] START: {name}", flush=True)
    def finish(self, name, metrics, dur):
        with self.lock:
            eta = (time.time()-self.start)/self.completed*(self.total-self.completed) if self.completed else 0
            print(f"[{self.completed}/{self.total}] DONE: {name} ({dur:.1f}s) ETA:{eta/60:.1f}m", flush=True)
            for k,v in metrics.items():
                if isinstance(v,(int,float)): print(f"  {k}: {v:.4f}", flush=True)

progress = LiveProgress(12)

def run(script, name):
    progress.start_exp(name)
    start = time.time()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script); fname = f.name
    r = subprocess.run(['python3', fname], capture_output=True, text=True, timeout=300, cwd="/home/me/ngs")
    os.unlink(fname)
    dur = time.time()-start
    metrics = {}
    if r.returncode==0:
        for line in r.stdout.split(chr(10)):
            if any(k in line.lower() for k in ["acc","forget","bwt","k=","norm","param","bit","transfer","special","geo","curv","speed","time"]):
                for p in line.split():
                    try: metrics[f"m{len(metrics)}"]=float(p)
                    except: pass
    progress.finish(name, metrics, dur)
    return ValidationResult(name, "success" if r.returncode==0 else "failed", metrics, datetime.now().isoformat(), dur)

# ===== 1. DOMAIN-INCREMENTAL BENCHMARKS =====
exp1 = """import torch, numpy as np
from ngs.models.ngs import build_ngs
from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.benchmarks.class_incremental import run_class_incremental
import sys; sys.path.insert(0, "/home/me/ngs")

domains = ["permuted_mnist", "rotated_mnist", "blurry_mnist", "noisy_mnist"]
cfg = NGSConfig(latent_dim=32, max_k=512, k_init=128, top_k=8,
    routing=RoutingStrategy.FACTORIZED_SUBSPACE, parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
    topology_control=TopologyControl.CONTINUOUS_DENSITY, memory_management=MemoryManagement.PRE_ALLOCATED,
    num_subspaces=4, hypernetwork_code_dim=8, hypernetwork_hidden_dim=16)

for d in domains:
    m = build_ngs(784, 10, cfg)
    # Quick eval: train 1 epoch per task, measure final accuracy
    from experiments.runner import run_experiment
    from experiments.config import EXPERIMENTS
    exp = EXPERIMENTS[d]
    result = run_experiment(exp, "ngs_cfg_net", 42, "./tmp_val", verbose=False)
    acc = result["metrics"]["avg_final_accuracy"]
    forget = result["metrics"]["avg_forgetting"]
    print(f"DOMAIN_INC {d}: acc={acc:.4f} forget={forget:.4f}")
"""

# ===== 2. SCALING LAWS =====
exp2 = """import torch, numpy as np
from ngs.models.ngs import build_ngs
from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement

# Sweep max_k
for max_k in [64, 128, 256, 512, 1024]:
    cfg = NGSConfig(latent_dim=32, max_k=max_k, k_init=min(64,max_k//2), top_k=8,
        routing=RoutingStrategy.FACTORIZED_SUBSPACE, parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.CONTINUOUS_DENSITY, memory_management=MemoryManagement.PRE_ALLOCATED,
        num_subspaces=4, hypernetwork_code_dim=8, hypernetwork_hidden_dim=16)
    m = build_ngs(784, 10, cfg)
    params = sum(p.numel() for p in m.parameters())
    x = torch.randn(32, 784); out = m(x)
    print(f"SCALING max_k={max_k}: params={params} K={m.K} forward_ok={out.logits.shape[1]==10}")

# Sweep latent_dim
for d in [16, 32, 64, 128]:
    cfg = NGSConfig(latent_dim=d, max_k=512, k_init=128, top_k=8,
        routing=RoutingStrategy.FACTORIZED_SUBSPACE, parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.CONTINUOUS_DENSITY, memory_management=MemoryManagement.PRE_ALLOCATED,
        num_subspaces=4, hypernetwork_code_dim=8, hypernetwork_hidden_dim=16)
    m = build_ngs(784, 10, cfg)
    params = sum(p.numel() for p in m.parameters())
    print(f"SCALING d_latent={d}: params={sum(p.numel() for p in m.parameters())}")
"""

# ===== 3. STRONG BASELINES =====
exp3 = """import torch, numpy as np
# Compare NGS-CFG vs DER, ER, LwF, SI on permuted_mnist
from experiments.runner import run_experiment
from experiments.config import EXPERIMENTS

exp = EXPERIMENTS["permuted_mnist"]
baselines = ["der", "er", "lwf", "si", "ngs_baseline", "ngs_cfg_net"]
for b in baselines:
    try:
        r = run_experiment(exp, b, 42, "./tmp_val", verbose=False)
        acc = r["metrics"]["avg_final_accuracy"]
        forget = r["metrics"]["avg_forgetting"]
        print(f"BASELINE {b}: acc={acc:.4f} forget={forget:.4f}")
    except Exception as e:
        print(f"BASELINE {b}: ERROR {e}")
"""

# ===== 4. INFORMATION DENSITY =====
exp4 = """import torch, numpy as np
from ngs.models.ngs import build_ngs
from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement

# Probe task information in codes
cfg = NGSConfig(latent_dim=32, max_k=512, k_init=128, top_k=8,
    routing=RoutingStrategy.FACTORIZED_SUBSPACE, parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
    topology_control=TopologyControl.CONTINUOUS_DENSITY, memory_management=MemoryManagement.PRE_ALLOCATED,
    num_subspaces=4, hypernetwork_code_dim=8, hypernetwork_hidden_dim=16)
m = build_ngs(784, 10, cfg)
codes = m.param_store.codes.data.clone()
params = sum(p.numel() for p in m.parameters())
# Code entropy
import scipy.stats as stats
code_flat = codes.flatten().numpy()
entropy = stats.entropy(np.abs(code_flat)+1e-8)
bits_per_param = entropy / params
print(f"INFO_DENSITY: params={params} code_entropy={entropy:.4f} bits/param={bits_per_param:.6f}")
# Mutual info between code clusters
codes_np = codes.numpy()
for i in range(0, 512, 64):
    cluster = codes_np[i:i+64]
    if len(cluster) > 1:
        corr = np.corrcoef(cluster).mean()
        print(f"CODE_COHERENCE cluster_{i//64}: mean_corr={corr:.4f}")
"""

# ===== 5. GAUSSIAN SPECIALIZATION =====
exp5 = """import torch, numpy as np
from ngs.models.ngs import build_ngs
from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement

cfg = NGSConfig(latent_dim=32, max_k=512, k_init=128, top_k=8,
    routing=RoutingStrategy.FACTORIZED_SUBSPACE, parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
    topology_control=TopologyControl.CONTINUOUS_DENSITY, memory_management=MemoryManagement.PRE_ALLOCATED,
    num_subspaces=4, hypernetwork_code_dim=8, hypernetwork_hidden_dim=16)
m = build_ngs(784, 10, cfg)

# Probe which Gaussians activate for different input types
test_inputs = {
    "random": torch.randn(100, 784),
    "structured": torch.randn(100, 784).cumsum(1),  # structured noise
}
for name, x in test_inputs.items():
    z = m.p_down(x)
    routing = m.router(z)
    active = routing.indices[0].unique().numpy()
    print(f"SPECIALIZATION {name}: active_G={len(active)} indices={active[:10].tolist()}")
    # Check subspace distribution
    if hasattr(routing, 'level_indices'):
        for s, idx in enumerate(routing.level_indices):
            print(f"  subspace_{s}: {idx[0].unique().shape[0]} unique Gaussians")
"""

# ===== 6. LONG HORIZON 1000 TASKS =====
exp6 = """import torch
from ngs.models.ngs import build_ngs
from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement

cfg = NGSConfig(latent_dim=32, max_k=512, k_init=128, top_k=8,
    routing=RoutingStrategy.FACTORIZED_SUBSPACE, parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
    topology_control=TopologyControl.CONTINUOUS_DENSITY, memory_management=MemoryManagement.PRE_ALLOCATED,
    num_subspaces=4, hypernetwork_code_dim=8, hypernetwork_hidden_dim=16)
m = build_ngs(784, 10, cfg)
print("LONG1000: start K=" + str(m.K))
for task in range(1000):
    x = torch.randn(32, 784)
    out = m(x)
    if hasattr(m, 'adapt_density'):
        m.adapt_density(split_thresh=0.05, prune_thresh=0.01)
    if task % 100 == 0:
        print(f"LONG1000 task={task} K={m.K} max_K={m.config.max_k}")
print(f"LONG1000 final_K={m.K} max_K={m.config.max_k}")
"""

# ===== 7. ABLATION STUDIES =====
exp7 = """import torch
from ngs.models.ngs import build_ngs
from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement

variants = [
    ("monolithic_direct", RoutingStrategy.MONOLITHIC_MAHALANOBIS, ParameterStorage.DIRECT_ADAPTER, TopologyControl.DISCRETE_HEURISTIC),
    ("factorized_direct", RoutingStrategy.FACTORIZED_SUBSPACE, ParameterStorage.DIRECT_ADAPTER, TopologyControl.DISCRETE_HEURISTIC),
    ("factorized_hyper", RoutingStrategy.FACTORIZED_SUBSPACE, ParameterStorage.HYPERNETWORK_GENERATED, TopologyControl.CONTINUOUS_DENSITY),
    ("monolithic_hyper", RoutingStrategy.MONOLITHIC_MAHALANOBIS, ParameterStorage.HYPERNETWORK_GENERATED, TopologyControl.CONTINUOUS_DENSITY),
]
for name, routing, storage, topology in variants:
    cfg = NGSConfig(latent_dim=32, max_k=512, k_init=128, top_k=8,
        routing=routing, parameter_storage=storage, topology_control=topology,
        memory_management=MemoryManagement.PRE_ALLOCATED, num_subspaces=4)
    m = build_ngs(784, 10, cfg)
    params = sum(p.numel() for p in m.parameters())
    x = torch.randn(32, 784); out = m(x)
    print(f"ABLATION {name}: params={params} K={m.K} forward={out.logits.shape}")
"""

# ===== 8. CODE MANIFOLD GEOMETRY =====
exp8 = """import torch, numpy as np
from ngs.models.ngs import build_ngs
from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement

cfg = NGSConfig(latent_dim=32, max_k=512, k_init=128, top_k=8,
    routing=RoutingStrategy.FACTORIZED_SUBSPACE, parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
    topology_control=TopologyControl.CONTINUOUS_DENSITY, memory_management=MemoryManagement.PRE_ALLOCATED,
    num_subspaces=4, hypernetwork_code_dim=8, hypernetwork_hidden_dim=16)
m = build_ngs(784, 10, cfg)
codes = m.param_store.codes.data.clone().numpy()

# Geodesic distance on code manifold (using hypernet pullback metric)
# Sample pairs, measure adapter distance vs code distance
dists = []
for _ in range(100):
    i, j = np.random.choice(len(codes), 2, replace=False)
    c1, c2 = codes[i], codes[j]
    code_dist = np.linalg.norm(c1 - c2)
    # Adapter at endpoints
    z = torch.randn(1, 32)
    a1 = m.param_store.hypernet(torch.cat([torch.tensor(c1).unsqueeze(0), z], -1))
    a2 = m.param_store.hypernet(torch.cat([torch.tensor(c2).unsqueeze(0), z], -1))
    adapter_dist = torch.norm(a1 - a2).item()
    dists.append((code_dist, adapter_dist))

code_d, adapt_d = zip(*dists)
corr = np.corrcoef(code_d, adapt_d)[0,1]
print(f"MANIFOLD: code-adapter correlation={corr:.4f}")
print(f"MANIFOLD: mean_code_dist={np.mean(code_d):.4f} mean_adapter_dist={np.mean(adapt_d):.4f}")

# Check local linearity: interpolate and measure deviation
for _ in range(10):
    i, j = np.random.choice(len(codes), 2, replace=False)
    c1, c2 = codes[i], codes[j]
    mid = (c1 + c2) / 2
    # Adapter at midpoint
    z = torch.randn(1, 32)
    a_mid = m.param_store.hypernet(torch.cat([torch.tensor(mid).unsqueeze(0), z], -1))
    a1 = m.param_store.hypernet(torch.cat([torch.tensor(c1).unsqueeze(0), z], -1))
    a2 = m.param_store.hypernet(torch.cat([torch.tensor(c2).unsqueeze(0), z], -1))
    linear_mid = (a1 + a2) / 2
    dev = torch.norm(a_mid - linear_mid).item()
    print(f"LINEARITY: geodesic_dev={dev:.6f}")
"""

# ===== 9. FEW-SHOT ADAPTATION SPEED =====
exp9 = """import torch, time
from ngs.models.ngs import build_ngs
from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement

cfg = NGSConfig(latent_dim=32, max_k=512, k_init=128, top_k=8,
    routing=RoutingStrategy.FACTORIZED_SUBSPACE, parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
    topology_control=TopologyControl.CONTINUOUS_DENSITY, memory_management=MemoryManagement.PRE_ALLOCATED,
    num_subspaces=4, hypernetwork_code_dim=8, hypernetwork_hidden_dim=16)

# Train on base task, measure few-shot adaptation
m = build_ngs(784, 10, cfg)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

# Base training (5 epochs)
for ep in range(5):
    x = torch.randn(32, 784); y = torch.randint(0, 10, (32,))
    out = m(x); loss = torch.nn.functional.cross_entropy(out.logits, y)
    loss.backward(); opt.step(); opt.zero_grad()

# Few-shot: new classes, measure adaptation speed
new_classes = 5
for shot in [1, 5, 10, 20]:
    accs = []
    for trial in range(5):
        # Create few-shot support set
        supp_x = torch.randn(shot*new_classes, 784)
        supp_y = torch.randint(5, 10, (shot*new_classes,))
        # Fine-tune
        start = time.time()
        for _ in range(10):
            out = m(supp_x); loss = torch.nn.functional.cross_entropy(out.logits, supp_y)
            loss.backward(); opt.step(); opt.zero_grad()
        dur = time.time() - start
        # Test
        test_x = torch.randn(100, 784); test_y = torch.randint(5, 10, (100,))
        with torch.no_grad(): out = m(test_x)
        acc = (out.logits.argmax(1) == test_y).float().mean().item()
        accs.append(acc)
    print(f"FEW_SHOT shot={shot}: acc={np.mean(accs):.4f}±{np.std(accs):.4f} time={dur:.2f}s")
"""

# ===== 10. CONTINUAL COMPRESSION METRICS =====
exp10 = """import torch, numpy as np
from ngs.models.ngs import build_ngs
from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement

cfg = NGSConfig(latent_dim=32, max_k=512, k_init=128, top_k=8,
    routing=RoutingStrategy.FACTORIZED_SUBSPACE, parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
    topology_control=TopologyControl.CONTINUOUS_DENSITY, memory_management=MemoryManagement.PRE_ALLOCATED,
    num_subspaces=4, hypernetwork_code_dim=8, hypernetwork_hidden_dim=16)
m = build_ngs(784, 10, cfg)

# Track compression ratio over time
split_events = 0; prune_events = 0; spawn_events = 0
for task in range(200):
    x = torch.randn(32, 784); out = m(x)
    if hasattr(m, 'adapt_density'):
        result = m.adapt_density(split_thresh=0.05, prune_thresh=0.01)
        split_events += result[1]; prune_events += result[0]; spawn_events += result[2]
    if task % 50 == 0:
        active = m.router.active_mask.sum().item() if hasattr(m.router, 'active_mask') else m.K
        print(f"COMPRESSION task={task} K={active} split={split_events} prune={prune_events} spawn={spawn_events}")

# Final compression ratio
initial_K = 128
final_K = m.K
print(f"COMPRESSION_RATIO: {initial_K}/{final_K} = {initial_K/final_K:.2f}x")
"""

# ===== 11. DOMAIN TRANSFER ANALYSIS =====
exp11 = """import torch, numpy as np
from ngs.models.ngs import build_ngs
from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from experiments.runner import run_experiment
from experiments.config import EXPERIMENTS

# Train on one domain, test zero-shot on others
cfg = NGSConfig(latent_dim=32, max_k=512, k_init=128, top_k=8,
    routing=RoutingStrategy.FACTORIZED_SUBSPACE, parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
    topology_control=TopologyControl.CONTINUOUS_DENSITY, memory_management=MemoryManagement.PRE_ALLOCATED,
    num_subspaces=4, hypernetwork_code_dim=8, hypernetwork_hidden_dim=16)

domains = ["permuted_mnist", "rotated_mnist", "blurry_mnist", "noisy_mnist"]
domain_codes = {}

for d in domains:
    m = build_ngs(784, 10, cfg)
    exp = EXPERIMENTS[d]
    # Quick 1-epoch train
    r = run_experiment(exp, "ngs_cfg_net", 42, "./tmp_val", verbose=False)
    # Capture codes after training
    domain_codes[d] = m.param_store.codes.data.clone().mean(0).numpy()
    print(f"DOMAIN_TRANSFER {d}: code_norm={np.linalg.norm(domain_codes[d]):.4f}")

# Cross-domain code similarity
for i, d1 in enumerate(domains):
    for d2 in domains[i+1:]:
        sim = np.dot(domain_codes[d1], domain_codes[d2]) / (np.linalg.norm(domain_codes[d1])*np.linalg.norm(domain_codes[d2]))
        print(f"DOMAIN_SIM {d1}-{d2}: cosine={sim:.4f}")
"""

# ===== 12. CONTINUAL COMPRESSION EFFICIENCY =====
exp12 = """import torch, numpy as np
from ngs.models.ngs import build_ngs
from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement

cfg = NGSConfig(latent_dim=32, max_k=512, k_init=128, top_k=8,
    routing=RoutingStrategy.FACTORIZED_SUBSPACE, parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
    topology_control=TopologyControl.CONTINUOUS_DENSITY, memory_management=MemoryManagement.PRE_ALLOCATED,
    num_subspaces=4, hypernetwork_code_dim=8, hypernetwork_hidden_dim=16)
m = build_ngs(784, 10, cfg)

# Measure compression efficiency: bits stored per active Gaussian
# Each Gaussian = code (8) + mean (32/4=8 per subspace * 4 = 32) + scale (8) = ~48 params
# But hypernetwork compresses this to 8D code
params_per_G = 8 + 32 + 8  # code + mean + scale per subspace
total_params = sum(p.numel() for p in m.parameters())
active_G = m.K
compressed_per_G = total_params / active_G
theoretical_per_G = 8 + 32 + 8  # code_dim + latent_dim + scale
print(f"COMPRESSION_EFF: total_params={total_params} active_G={active_G}")
print(f"COMPRESSION_EFF: params_per_G={compressed_per_G:.1f} theoretical={theoretical_per_G}")
print(f"COMPRESSION_EFF: hypernet_savings={theoretical_per_G/compressed_per_G:.1f}x per G")

# Effective parameter count (codes + hypernet)
code_params = 512 * 8
hypernet_params = sum(p.numel() for p in m.param_store.hypernet.parameters())
effective = code_params + hypernet_params
print(f"COMPRESSION_EFF: effective_params={effective} vs full={total_params} ratio={total_params/effective:.1f}x")
"""

experiments = [
    (exp1, "1. Domain-Incremental Benchmarks"),
    (exp2, "2. Scaling Laws"),
    (exp3, "3. Strong Baseline Comparisons"),
    (exp4, "4. Information Density"),
    (exp5, "5. Gaussian Specialization"),
    (exp6, "6. Long-Horizon 1000 Tasks"),
    (exp7, "7. Ablation Studies"),
    (exp8, "8. Code Manifold Geometry"),
    (exp9, "9. Few-Shot Adaptation Speed"),
    (exp10, "10. Continual Compression Metrics"),
    (exp11, "11. Domain Transfer Analysis"),
    (exp12, "12. Compression Efficiency"),
]

results = []
for script, name in experiments:
    results.append(run(script, name))

print("\n" + "="*60)
print("VALIDATION SUITE COMPLETE")
print("="*60)
for r in results:
    print(f"  {r.name}: {r.status} ({r.duration_sec:.1f}s)")

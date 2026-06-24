"""
Smoke test for AutopoieticManager - verifies the class compiles and runs.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models.ngs import build_ngs
from ngs.modules.topology_managers import AutopoieticManager

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEVICE}")

# Create config with AUTOPOIETIC topology
cfg = NGSConfig(
    latent_dim=64,
    k_init=16,
    max_k=32,
    top_k=4,
    routing='monolithic_mahalanobis',
    parameter_storage='hypernetwork_generated',
    topology_control='autopoietic',  # Use the new AutopoieticManager
    memory_management='dynamic',
    hypernetwork_hidden_dim=32,
    hypernetwork_code_dim=8,
)

# Add autopoietic-specific config
cfg.extra['entropy_split_threshold'] = 1.5
cfg.extra['overlap_merge_threshold'] = 0.9
cfg.extra['max_tree_depth'] = 5

print("Building NGS model with AutopoieticManager...")
model = build_ngs(784, 10, cfg).to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# Check topology manager type
topology_mgr = model.topology_manager
print(f"Topology manager: {type(topology_mgr).__name__}")
assert isinstance(topology_mgr, AutopoieticManager), "Expected AutopoieticManager"

# Test forward pass and topology adaptation
print("\n--- Testing forward pass ---")
x = torch.randn(32, 784).to(DEVICE)
out = model(x)
print(f"Output shape: {out.logits.shape}")
print(f"Active units: {model.router.active_mask.sum().item()}")

# Test autopoietic step
print("\n--- Testing autopoietic step ---")
z_samples = model.p_down(x)  # Latent representations

# First, check initial state
initial_active = model.router.active_mask.sum().item()
print(f"Initial active units: {initial_active}")

# Run autopoietic step
num_merged, num_split, num_spawned = topology_mgr.step(model, z_samples)
print(f"Step result: merged={num_merged}, split={num_split}, spawned={num_spawned}")
print(f"Active units after step: {model.router.active_mask.sum().item()}")

# Test tree stats
print("\n--- Testing tree stats ---")
tree_stats = topology_mgr.get_tree_stats()
print(f"Tree stats: {tree_stats}")

# Test adapt_topology interface
print("\n--- Testing adapt_topology interface ---")
num_merged, num_split, num_spawned = topology_mgr.adapt_topology(model, z_samples=z_samples)
print(f"Adapt result: merged={num_merged}, split={num_split}, spawned={num_spawned}")

# Test with high entropy scenario (should trigger split)
print("\n--- Testing with synthetic high-entropy routing ---")
# Force high entropy by making routing weights uniform
# We can't easily force this, but we can check the entropy computation
routing = model.router(z_samples)
weights = routing.weights
entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
print(f"Current routing entropy: {entropy.item():.4f}")
print(f"Split threshold: {topology_mgr.tau_split}")
print(f"Merge threshold: {topology_mgr.tau_merge}")

# Test multiple steps
print("\n--- Running 5 autopoietic steps ---")
for i in range(5):
    z = model.p_down(torch.randn(32, 784).to(DEVICE))
    m, s, sp = topology_mgr.step(model, z)
    print(f"  Step {i+1}: merged={m}, split={s}, spawned={sp}, active={model.router.active_mask.sum().item()}")

tree_stats = topology_mgr.get_tree_stats()
print(f"\nFinal tree stats: {tree_stats}")

print("\n✅ AutopoieticManager smoke test PASSED")
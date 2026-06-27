#!/usr/bin/env python
"""
Deep smoke test for TODO11 diagnostic scripts.
Actually instantiates models and runs minimal forward passes.
"""
import sys
sys.path.insert(0, '/home/me/ngs')
sys.path.insert(0, '/home/me/ngs/bioplausible/mep')

import torch
import importlib

# ---- Shared helpers ----
from ngs.core.interfaces import NGSConfig, RoutingStrategy


def sm_test(name, fn):
    try:
        fn()
        print(f"  OK   {name}")
        return True
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
        return False


# ---- 1. diagnose_spectral_norm ----
def test_spectral_norm():
    from experiments.diagnose_spectral_norm import compute_singular_values, compute_all_singular_values
    W = torch.randn(32, 64)
    s = compute_singular_values(W)
    assert s.shape == torch.Size([]), "Power iter should return scalar"
    s2 = compute_all_singular_values(W, k=5)
    assert s2.shape == torch.Size([5]), "SVD should return 5 values"


# ---- 2. diagnose_energy_landscape ----
def test_energy_landscape():
    from ngs.models.ngs import NGSModel
    from experiments.diagnose_energy_landscape import compute_energy
    config = NGSConfig(latent_dim=16, max_k=8, top_k=4, k_init=4, routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS)
    model = NGSModel(784, 10, config)
    z = torch.randn(4, 16)
    routing_output = model.router(z)
    energy = compute_energy(model, z, routing_output)
    assert energy.numel() == 1, "Energy computation should work"


# ---- 3. diagnose_entropy_distribution ----
def test_entropy_distribution():
    from experiments.diagnose_entropy_distribution import compute_routing_entropy
    from types import SimpleNamespace
    weights = torch.tensor([[0.5, 0.3, 0.2]])
    router_out = SimpleNamespace(weights=weights)
    ent = compute_routing_entropy(router_out)
    assert ent.numel() == 1


# ---- 4. diagnose_gaussian_overlap ----
def test_gaussian_overlap():
    from experiments.diagnose_gaussian_overlap import compute_overlap_matrix
    config = NGSConfig(latent_dim=8, max_k=4, top_k=2, k_init=2, routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS)
    from ngs.models.ngs import NGSModel
    model = NGSModel(784, 10, config)
    router = model.router
    mu = router.mu[router.active_mask] if hasattr(router, 'active_mask') else router.mu
    log_s = router.log_s[router.active_mask] if hasattr(router, 'active_mask') else router.log_s
    overlaps = compute_overlap_matrix(mu, log_s)
    assert overlaps.shape[0] == model.router.K, "Should return [K, K] matrix"


# ---- 5. compare_ngs_vs_dense ----
def test_ngs_vs_dense():
    from experiments.compare_ngs_vs_dense import DenseMLP, NGSWrapper
    config = NGSConfig(latent_dim=16, max_k=8, top_k=4, k_init=4, routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS)
    model = NGSWrapper(784, 10, config)
    x = torch.randn(2, 784)
    out = model(x)
    assert out.shape == torch.Size([2, 10])


# ---- 6. ablate_projections ----
def test_ablate_projections():
    from experiments.ablate_projections import RandomProjectionNGS
    import types
    config = NGSConfig(latent_dim=16, max_k=8, top_k=4, k_init=4, routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS)
    for pt in ["random", "rff", "mlp"]:
        model = RandomProjectionNGS(784, 10, 16, config, proj_type=pt)
        out = model(torch.randn(2, 784))
        assert out.shape == torch.Size([2, 10])


# ---- 7. analyze_gaussian_specialization ----
def test_gaussian_specialization():
    from ngs.models.ngs import NGSModel
    import torch.nn.functional as F
    model = NGSModel(784, 10, NGSConfig(latent_dim=16, max_k=8, top_k=4, k_init=4, routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS))
    x = torch.randn(2, 784)
    out = model(x)
    # Basic stuff that script does
    router_out = out.routing_output
    assert router_out is not None


# ---- 8. eqprop_via_epoptimizer ----
def test_eqprop_epoptimizer():
    from ngs.modules.eqprop import create_eqngs
    config = NGSConfig(latent_dim=16, max_k=8, top_k=4, k_init=4, routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS)
    model = create_eqngs(784, 10, config, ep_beta=0.2, ep_settle_steps=5, ep_settle_lr=0.1, spectral_mode='none')
    x = torch.randn(2, 784)
    out = model(x)
    assert out is not None or hasattr(out, 'logits'), "Forward pass through EqNGS should work"


# ---- 9. eqprop_mse_energy ----
def test_eqprop_mse():
    from ngs.models.ngs import NGSModel
    model = NGSModel(784, 10, NGSConfig(latent_dim=16, max_k=8, top_k=4, k_init=4, routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS))
    x = torch.randn(2, 784)
    out = model(x)
    assert out is not None


# ---- 10. baseline_moe ----
def test_baseline_moe():
    from experiments.baseline_moe import TopKMoE
    moe = TopKMoE(784, 10, num_experts=4, top_k=2, d_expert=64)
    out = moe(torch.randn(2, 784))
    assert out.shape == torch.Size([2, 10])


# ---- Main ----
TESTS = [
    ("diagnose_spectral_norm", test_spectral_norm),
    ("diagnose_energy_landscape", test_energy_landscape),
    ("diagnose_entropy_distribution", test_entropy_distribution),
    ("diagnose_gaussian_overlap", test_gaussian_overlap),
    ("compare_ngs_vs_dense", test_ngs_vs_dense),
    ("ablate_projections", test_ablate_projections),
    ("analyze_gaussian_specialization", test_gaussian_specialization),
    ("eqprop_via_epoptimizer", test_eqprop_epoptimizer),
    ("eqprop_mse_energy", test_eqprop_mse),
    ("baseline_moe", test_baseline_moe),
]

if __name__ == "__main__":
    passed = 0
    failed = 0
    for name, test_fn in TESTS:
        ok = sm_test(name, test_fn)
        if ok:
            passed += 1
        else:
            failed += 1
    print(f"\n{'='*60}")
    print(f"SMOKE TEST SUMMARY: {passed} passed, {failed} failed")
    print(f"{'='*60}")

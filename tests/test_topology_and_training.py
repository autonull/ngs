"""Tests for topology management and training workflows."""
import torch
from mngs.profiles import Baseline_LeanNGS, CFG_Net_Full, Ultra_Edge_Sparse, Ablation_Hypernetwork_Only
from mngs.model import build_mngs


def test_heuristic_topology_is_noop_on_factorized_router():
    """Factorized routers don't have active_mask, topology should be no-op."""
    config = CFG_Net_Full()
    model = build_mngs(d_in=784, d_out=10, config=config)
    result = model.adapt_density()
    assert result == (0, 0, 0), "Topology should be a no-op for factorized routers"


def test_heuristic_manager_prune():
    """Test that HeuristicManager can prune units."""
    config = Baseline_LeanNGS()
    model = build_mngs(d_in=784, d_out=10, config=config)
    
    # Initialize some units
    initial_k = model.K
    
    # Artificially reduce log_alpha to trigger pruning
    model.router.log_alpha.data[:] = -10.0  # alpha ~= 0 -> prune
    
    result = model.adapt_density()
    assert result[0] > 0 or initial_k == 0, "Should prune some units"


def test_update_grad_ema():
    """Test that gradient EMA updates work."""
    config = Baseline_LeanNGS()
    model = build_mngs(d_in=784, d_out=10, config=config)
    x = torch.randn(4, 784)
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    old_ema = model.grad_mu_ema[model.router.active_mask].clone()
    model.update_grad_ema()
    new_ema = model.grad_mu_ema[model.router.active_mask]
    
    # EMA should have changed from zero
    assert not torch.allclose(old_ema, new_ema), "EMA should update after backward"


def test_ultra_edge_profile():
    config = Ultra_Edge_Sparse()
    model = build_mngs(d_in=784, d_out=10, config=config)
    x = torch.randn(4, 784)
    out = model(x)
    assert out.shape == (4, 10)
    
    # Edge profile has smaller latent dim
    assert config.latent_dim == 16


def test_ablation_hypernetwork_profile():
    config = Ablation_Hypernetwork_Only()
    model = build_mngs(d_in=784, d_out=10, config=config)
    x = torch.randn(4, 784)
    out = model(x)
    assert out.shape == (4, 10)


def test_all_profiles_smoke():
    """Smoke test all profiles produce correct shapes."""
    from mngs.profiles import profile_all
    for config in profile_all():
        model = build_mngs(d_in=784, d_out=10, config=config)
        x = torch.randn(4, 784)
        out = model(x)
        assert out.shape == (4, 10), f"Profile {config.routing} produced wrong shape"
        # Check gradient flow
        out.sum().backward()
        assert model.p_down.weight.grad is not None


def test_diversity_loss():
    config = Baseline_LeanNGS()
    model = build_mngs(d_in=784, d_out=10, config=config)
    loss = model.diversity_loss()
    assert loss.ndim == 0, "Diversity loss should be scalar"
    assert loss <= 0.0, "Should be negative (we negate the min distance)"


def test_entropy_loss():
    config = Baseline_LeanNGS()
    model = build_mngs(d_in=784, d_out=10, config=config)
    x = torch.randn(4, 784)
    eloss = model.entropy_loss(x)
    assert eloss.ndim == 0, "Entropy loss should be scalar"

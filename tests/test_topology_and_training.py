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
    """Test that gradient EMA updates work (auto-updated via hook)."""
    config = Baseline_LeanNGS()
    model = build_mngs(d_in=784, d_out=10, config=config)
    x = torch.randn(4, 784)
    out = model(x)
    loss = out.sum()
    
    # Capture EMA before backward (hook runs during backward)
    old_ema = model.router.grad_mu_ema[model.router.active_mask].clone()
    loss.backward()
    new_ema = model.router.grad_mu_ema[model.router.active_mask]
    
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


def test_continuous_density_split():
    """Verify ContinuousDensityManager splits units when error density is high."""
    from mngs.core.config import MNGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    # Use monolithic routing (has active_mask) + continuous density topology
    config = MNGSConfig(
        latent_dim=32,
        k_init=128,
        max_k=512,
        top_k=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.PRE_ALLOCATED_MASKED,
        use_lora=True,
    )
    model = build_mngs(d_in=784, d_out=10, config=config)

    initial_active = model.K

    # Set split gates above threshold for all active units
    active_idx = model.router.active_mask.nonzero(as_tuple=True)[0]
    model.split_gate.data[active_idx] = 10.0  # sigmoid(10) ~ 1.0 > threshold

    # Set error density high so the split condition is met
    model.error_density[active_idx] = 1.0  # well above 1e-3 threshold

    # Run forward pass to populate activation_density
    x = torch.randn(4, 784)
    model(x)

    _, num_split, _ = model.adapt_density()
    assert num_split > 0, "ContinuousDensityManager should split when error density is high"
    assert model.K >= initial_active, "Active units should increase after split"

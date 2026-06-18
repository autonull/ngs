"""Integration tests for topology managers."""

import torch
import pytest
from ngs.core.interfaces import NGSConfig, TopologyControl
from ngs.modules.topology_managers import HeuristicManager, ContinuousDensityManager
from ngs.models import build_ngs


class TestTopologyManagers:
    """Test all topology manager implementations."""

    @pytest.fixture(params=[TopologyControl.DISCRETE_HEURISTIC, TopologyControl.CONTINUOUS_DENSITY])
    def topology_control(self, request):
        return request.param

    def test_topology_adapt_invariants(self, topology_control):
        """Test split/prune/spawn invariants."""
        config = NGSConfig(
            topology_control=topology_control,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
            split_threshold=0.05,
            prune_threshold=0.01,
        )

        if topology_control == TopologyControl.DISCRETE_HEURISTIC:
            manager = HeuristicManager(config)
        else:
            manager = ContinuousDensityManager(config)

        # Build minimal model
        model = build_ngs(784, 10, config)

        # Run adaptation
        z_samples = torch.randn(100, 32)
        action = manager.adapt_topology(model, z_samples=z_samples)

        # Check action structure (tuple of num_pruned, num_split, num_spawned)
        assert isinstance(action, tuple)
        assert len(action) == 3
        num_pruned, num_split, num_spawned = action
        assert num_pruned >= 0
        assert num_split >= 0
        assert num_spawned >= 0

        # K bounds invariant
        assert model.K <= config.max_k
        assert model.K >= 0

    def test_topology_parameter_continuity(self, topology_control):
        """Test parameter continuity after topology changes."""
        config = NGSConfig(
            topology_control=topology_control,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
            split_threshold=0.5,  # High to force splits
            prune_threshold=0.01,
        )

        model = build_ngs(784, 10, config)

        # Forward pass to initialize
        x = torch.randn(4, 784)
        out1 = model(x)

        # Force topology adaptation
        z_samples = torch.randn(200, 32)
        model.adapt_density(z_samples=z_samples, split_thresh=0.01, prune_thresh=0.01)

        # Forward pass after adaptation
        out2 = model(x)

        # Output shape should be preserved
        assert out1.logits.shape == out2.logits.shape == (4, 10)

        # Model should still produce valid outputs
        assert not torch.isnan(out2.logits).any()

    def test_topology_compute_losses(self, topology_control):
        """Test compute_topology_losses method returns valid losses."""
        config = NGSConfig(
            topology_control=topology_control,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
        )

        model = build_ngs(784, 10, config)

        losses = model.compute_topology_losses()

        assert isinstance(losses, dict)
        for name, loss in losses.items():
            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0  # Scalar
            assert not torch.isnan(loss)

    def test_heuristic_manager(self):
        """Test HeuristicManager specific behavior."""
        config = NGSConfig(
            topology_control=TopologyControl.DISCRETE_HEURISTIC,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
            split_threshold=0.05,
            prune_threshold=0.01,
        )

        manager = HeuristicManager(config)

        model = build_ngs(784, 10, config)

        z_samples = torch.randn(50, 32)
        action = manager.adapt_topology(model, z_samples=z_samples)

        assert isinstance(action, tuple)
        assert len(action) == 3

    def test_continuous_density_manager(self):
        """Test ContinuousDensityManager specific behavior."""
        config = NGSConfig(
            topology_control=TopologyControl.CONTINUOUS_DENSITY,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
            split_threshold=0.05,
            prune_threshold=0.01,
            ema_decay=0.99,
        )

        manager = ContinuousDensityManager(config)

        model = build_ngs(784, 10, config)

        # Update error densities
        x = torch.randn(100, 784)
        logits = model(x).logits
        targets = torch.randint(0, 10, (100,))
        model.update_unit_errors(logits, targets)

        # Check error_density updated
        assert model.error_density.abs().sum() > 0

        # Adapt
        z_samples = torch.randn(50, 32)
        action = manager.adapt_topology(model, z_samples=z_samples)

        assert isinstance(action, tuple)
        assert len(action) == 3

    def test_all_managers_in_model(self):
        """Test all topology managers work in full NGS model."""
        for control in [TopologyControl.DISCRETE_HEURISTIC, TopologyControl.CONTINUOUS_DENSITY]:
            config = NGSConfig(
                topology_control=control,
                max_k=64,
                k_init=16,
                top_k=8,
                latent_dim=32,
            )

            model = build_ngs(784, 10, config)
            x = torch.randn(4, 784)
            out = model(x)

            assert out.logits.shape == (4, 10)

            # Test adapt_density doesn't crash
            z = torch.randn(20, 32)
            action = model.adapt_density(z_samples=z)
            assert isinstance(action, tuple)
            assert len(action) == 3

            # Test gradients
            out.logits.sum().backward()
            assert model.p_down.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
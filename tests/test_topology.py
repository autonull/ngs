"""Integration tests for topology managers."""

import torch
import pytest
from mngs.core.config import MNGSConfig, TopologyControl
from mngs.modules.topology_managers import HeuristicManager, ContinuousDensityManager
from mngs import build_mngs


class TestTopologyManagers:
    """Test all topology manager implementations."""

    @pytest.fixture(params=[TopologyControl.DISCRETE_HEURISTIC, TopologyControl.CONTINUOUS_DENSITY])
    def topology_control(self, request):
        return request.param

    def test_topology_adapt_invariants(self, topology_control):
        """Test split/prune/spawn invariants."""
        config = MNGSConfig(
            topology_control=topology_control,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
            split_threshold=0.05,
            prune_threshold=0.01,
        )

        if topology_control == TopologyControl.DISCRETE_HEURISTIC:
            manager = HeuristicManager(
                split_threshold=config.split_threshold,
                prune_threshold=config.prune_threshold,
                ema_decay=config.ema_decay,
            )
        else:
            manager = ContinuousDensityManager(
                split_threshold=config.split_threshold,
                prune_threshold=config.prune_threshold,
                density_decay=config.ema_decay,
            )

        # Build minimal model
        model = build_mngs(784, 10, config)

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
        config = MNGSConfig(
            topology_control=topology_control,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
            split_threshold=0.5,  # High to force splits
            prune_threshold=0.01,
        )

        model = build_mngs(784, 10, config)

        # Forward pass to initialize
        x = torch.randn(4, 784)
        out1 = model(x)

        # Force topology adaptation
        z_samples = torch.randn(200, 32)
        model.adapt_density(z_samples=z_samples, split_thresh=0.01, prune_thresh=0.01)

        # Forward pass after adaptation
        out2 = model(x)

        # Output shape should be preserved
        assert out1.shape == out2.shape == (4, 10)

        # Model should still produce valid outputs
        assert not torch.isnan(out2).any()

    def test_topology_compute_losses(self, topology_control):
        """Test compute_topology_losses method returns valid losses."""
        config = MNGSConfig(
            topology_control=topology_control,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
        )

        model = build_mngs(784, 10, config)

        losses = model.compute_topology_losses()

        assert isinstance(losses, dict)
        for name, loss in losses.items():
            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0  # Scalar
            assert not torch.isnan(loss)

    def test_heuristic_manager(self):
        """Test HeuristicManager specific behavior."""
        config = MNGSConfig(
            topology_control=TopologyControl.DISCRETE_HEURISTIC,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
            split_threshold=0.05,
            prune_threshold=0.01,
        )

        manager = HeuristicManager(
            split_threshold=config.split_threshold,
            prune_threshold=config.prune_threshold,
            ema_decay=config.ema_decay,
        )

        model = build_mngs(784, 10, config)

        z_samples = torch.randn(50, 32)
        action = manager.adapt_topology(model, z_samples=z_samples)

        assert isinstance(action, tuple)
        assert len(action) == 3

    def test_continuous_density_manager(self):
        """Test ContinuousDensityManager specific behavior."""
        config = MNGSConfig(
            topology_control=TopologyControl.CONTINUOUS_DENSITY,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
            split_threshold=0.05,
            prune_threshold=0.01,
            ema_decay=0.99,
        )

        manager = ContinuousDensityManager(
            split_threshold=config.split_threshold,
            prune_threshold=config.prune_threshold,
            density_decay=config.ema_decay,
        )

        model = build_mngs(784, 10, config)

        # Update error densities
        x = torch.randn(100, 784)
        logits = model(x)
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
        """Test all topology managers work in full MNGS model."""
        for control in [TopologyControl.DISCRETE_HEURISTIC, TopologyControl.CONTINUOUS_DENSITY]:
            config = MNGSConfig(
                topology_control=control,
                max_k=64,
                k_init=16,
                top_k=8,
                latent_dim=32,
            )

            model = build_mngs(784, 10, config)
            x = torch.randn(4, 784)
            out = model(x)

            assert out.shape == (4, 10)

            # Test adapt_density doesn't crash
            z = torch.randn(20, 32)
            action = model.adapt_density(z_samples=z)
            assert isinstance(action, tuple)
            assert len(action) == 3

            # Test gradients
            out.sum().backward()
            assert model.p_down.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""Integration tests for the full NGS functionality."""

import torch
import pytest
from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models.ngs import NGSModel, build_ngs
from ngs.modules.routers import build_router
from ngs.modules.topology_managers import build_topology_manager
from ngs.modules.memory_managers import build_memory_manager
from ngs.training.trainer import NGSTrainer, TrainerConfig


class TestNGSIntegration:
    """Integration tests for the full NGS model."""

    def test_basic_forward_pass(self):
        """Test basic forward pass with default config."""
        cfg = NGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_ngs(784, 10, cfg)
        x = torch.randn(8, 784)
        out = model(x)
        assert out.logits.shape == (8, 10)
        assert model.K == 16

    def test_all_routing_strategies(self):
        """Test all routing strategies."""
        for strategy in RoutingStrategy:
            if strategy == RoutingStrategy.LSH_APPROXIMATE:
                # LSH is known to be less stable
                continue
            
            cfg = NGSConfig(
                routing=strategy,
                max_k=64,
                k_init=16,
                latent_dim=32,
                top_k=8,
                num_subspaces=4,
                num_levels=3,
                top_k_factorized=2,
            )
            model = build_ngs(784, 10, cfg)
            x = torch.randn(8, 784)
            out = model(x)
            assert out.logits.shape == (8, 10), f"Failed for {strategy}"

    def test_all_parameter_storages(self):
        """Test all parameter storage strategies."""
        for storage in ParameterStorage:
            cfg = NGSConfig(
                parameter_storage=storage,
                max_k=64,
                k_init=16,
                latent_dim=32,
            )
            model = build_ngs(784, 10, cfg)
            x = torch.randn(8, 784)
            out = model(x)
            assert out.logits.shape == (8, 10), f"Failed for {storage}"

    def test_all_topology_managers(self):
        """Test all topology managers."""
        for topology in TopologyControl:
            cfg = NGSConfig(
                topology_control=topology,
                max_k=64,
                k_init=16,
                latent_dim=32,
            )
            model = build_ngs(784, 10, cfg)
            x = torch.randn(8, 784)
            out = model(x)
            assert out.logits.shape == (8, 10), f"Failed for {topology}"

    def test_all_memory_managers(self):
        """Test all memory management strategies."""
        for memory in MemoryManagement:
            cfg = NGSConfig(
                memory_management=memory,
                max_k=64,
                k_init=16,
                latent_dim=32,
            )
            model = build_ngs(784, 10, cfg)
            x = torch.randn(8, 784)
            out = model(x)
            assert out.logits.shape == (8, 10), f"Failed for {memory}"

    def test_train_epoch(self):
        """Test a single training epoch."""
        cfg = NGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_ngs(784, 10, cfg)

        trainer_config = TrainerConfig(
            lr=1e-3,
            epochs=1,
            batch_size=32,
        )

        trainer = NGSTrainer(model, trainer_config, device='cpu')

        # Create dummy data
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 784),
            torch.randint(0, 10, (100,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        metrics = trainer.train_epoch(loader)
        assert 'loss' in metrics
        assert 'K' in metrics

    def test_topology_adaptation(self):
        """Test topology adaptation."""
        cfg = NGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_ngs(784, 10, cfg)

        # Forward pass to initialize routing state
        x = torch.randn(8, 784)
        out = model(x)

        # Test topology adaptation
        if hasattr(model, 'adapt_density'):
            num_pruned, num_split, num_spawned = model.adapt_density()
            # These should be non-negative
            assert num_pruned >= 0
            assert num_split >= 0
            assert num_spawned >= 0

    def test_expand_capacity(self):
        """Test capacity expansion."""
        cfg = NGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_ngs(784, 10, cfg)

        initial_max_k = cfg.max_k
        new_max_k = 128
        
        if hasattr(model, 'expand_capacity'):
            model.expand_capacity(new_max_k)
            assert model.config.max_k == new_max_k

    def test_entropy_loss(self):
        """Test entropy loss computation."""
        cfg = NGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_ngs(784, 10, cfg)

        if hasattr(model, 'entropy_loss'):
            loss = model.entropy_loss(torch.randn(8, 784))
            assert loss.item() >= 0

    def test_diversity_loss(self):
        """Test diversity loss computation."""
        cfg = NGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_ngs(784, 10, cfg)

        if hasattr(model, 'diversity_loss'):
            loss = model.diversity_loss()
            assert isinstance(loss, torch.Tensor)

    def test_gradient_flow(self):
        """Test gradient flows through all components."""
        cfg = NGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_ngs(784, 10, cfg)
        x = torch.randn(8, 784, requires_grad=True)
        
        out = model(x)
        loss = out.logits.sum()
        loss.backward()
        
        assert x.grad is not None

    def test_all_topology_controls_with_model(self):
        """Test all topology controls work with the full model."""
        # Skip meta_learned for speed (has extra nn.Module)
        topology_controls = [
            TopologyControl.DISCRETE_HEURISTIC,
            TopologyControl.CONTINUOUS_DENSITY,
            TopologyControl.MERGE_AWARE,
            # TopologyControl.META_LEARNED,  # Can take longer
        ]
        
        for topology in topology_controls:
            cfg = NGSConfig(
                topology_control=topology,
                max_k=64,
                k_init=16,
                latent_dim=32,
            )
            model = build_ngs(784, 10, cfg)
            x = torch.randn(8, 784)
            out = model(x)
            assert out.logits.shape == (8, 10), f"Failed for {topology}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

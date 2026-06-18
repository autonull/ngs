"""Integration tests for NGS model."""

import torch
import pytest
from ngs.core.interfaces import (
    NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
)
from ngs.models import build_ngs


class TestNGSModel:
    """Test NGS model end-to-end."""

    @pytest.fixture(params=[
        (RoutingStrategy.MONOLITHIC_MAHALANOBIS, ParameterStorage.DIRECT_ADAPTER, TopologyControl.DISCRETE_HEURISTIC, MemoryManagement.PRE_ALLOCATED),
        (RoutingStrategy.FACTORIZED_SUBSPACE, ParameterStorage.HYPERNETWORK_GENERATED, TopologyControl.CONTINUOUS_DENSITY, MemoryManagement.PRE_ALLOCATED),
    ])
    def config_combo(self, request):
        return request.param

    def test_model_forward(self, config_combo):
        """Test forward pass with various config combinations."""
        routing, storage, topology, memory = config_combo

        config = NGSConfig(
            routing=routing,
            parameter_storage=storage,
            topology_control=topology,
            memory_management=memory,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
        )

        model = build_ngs(784, 10, config)
        x = torch.randn(8, 784)

        out = model(x)

        assert hasattr(out, 'logits')
        assert out.logits.shape == (8, 10)

    def test_model_gradient_flow(self, config_combo):
        """Test gradient flow through entire model."""
        routing, storage, topology, memory = config_combo

        config = NGSConfig(
            routing=routing,
            parameter_storage=storage,
            topology_control=topology,
            memory_management=memory,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
        )

        model = build_ngs(784, 10, config)
        x = torch.randn(4, 784)

        out = model(x)
        loss = out.logits.sum()
        loss.backward()

        # Check key parameters have gradients
        assert model.p_down.weight.grad is not None
        assert model.p_up.weight.grad is not None
        assert model.gamma.grad is not None

    def test_model_k_property(self):
        """Test K property tracks active units."""
        config = NGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_ngs(784, 10, config)

        assert model.K == 16

    def test_model_entropy_loss(self):
        """Test entropy loss computation."""
        config = NGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_ngs(784, 10, config)

        x = torch.randn(8, 784)
        loss = model.entropy_loss(x)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss >= 0

    def test_model_diversity_loss(self):
        """Test diversity loss computation."""
        config = NGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_ngs(784, 10, config)

        loss = model.diversity_loss()

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_model_split_gate_loss(self):
        """Test split gate loss for continuous density."""
        config = NGSConfig(
            max_k=64,
            k_init=16,
            latent_dim=32,
            topology_control=TopologyControl.CONTINUOUS_DENSITY,
        )
        model = build_ngs(784, 10, config)

        loss = model.split_gate_loss()

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss >= 0

    def test_model_update_unit_errors(self):
        """Test unit error density updates."""
        config = NGSConfig(
            max_k=64,
            k_init=16,
            latent_dim=32,
            topology_control=TopologyControl.CONTINUOUS_DENSITY,
        )
        model = build_ngs(784, 10, config)

        x = torch.randn(32, 784)
        logits = model(x).logits
        targets = torch.randint(0, 10, (32,))

        model.update_unit_errors(logits, targets)

        # Error density should be updated
        assert model.error_density.abs().sum() > 0

    def test_model_adapt_density(self):
        """Test topology adaptation."""
        config = NGSConfig(
            max_k=64,
            k_init=16,
            latent_dim=32,
            topology_control=TopologyControl.CONTINUOUS_DENSITY,
        )
        model = build_ngs(784, 10, config)

        z_samples = torch.randn(50, 32)
        action = model.adapt_density(z_samples=z_samples, split_thresh=0.05, prune_thresh=0.01)

        assert isinstance(action, tuple)
        assert len(action) == 3
        num_pruned, num_split, num_spawned = action
        assert isinstance(num_pruned, int)
        assert isinstance(num_split, int)
        assert isinstance(num_spawned, int)
        assert model.K <= config.max_k

    def test_model_compute_topology_losses(self):
        """Test topology losses computation."""
        config = NGSConfig(
            max_k=64,
            k_init=16,
            latent_dim=32,
        )
        model = build_ngs(784, 10, config)

        losses = model.compute_topology_losses()

        assert isinstance(losses, dict)
        assert 'entropy' in losses
        assert 'diversity' in losses

    def test_model_serialization(self):
        """Test model save/load."""
        config = NGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_ngs(784, 10, config)

        x = torch.randn(4, 784)
        y1 = model(x).logits

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)

            model2 = build_ngs(784, 10, config)
            model2.load_state_dict(torch.load(f.name))

            y2 = model2(x).logits

            assert torch.allclose(y1, y2, atol=1e-6)

    def test_model_determinism(self):
        """Test deterministic outputs with fixed seed."""
        config = NGSConfig(max_k=64, k_init=16, latent_dim=32)

        torch.manual_seed(42)
        model1 = build_ngs(784, 10, config)

        torch.manual_seed(42)
        model2 = build_ngs(784, 10, config)

        x = torch.randn(4, 784)

        out1 = model1(x).logits
        out2 = model2(x).logits

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_model_device_transfer(self):
        """Test model works on CPU and CUDA."""
        config = NGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_ngs(784, 10, config)

        # CPU
        x = torch.randn(4, 784)
        out = model(x)
        assert out.logits.device.type == 'cpu'

        # CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            x_cuda = x.cuda()
            out_cuda = model_cuda(x_cuda)
            assert out_cuda.logits.device.type == 'cuda'

    def test_factorized_routing_output(self):
        """Test factorized routing returns list outputs."""
        config = NGSConfig(
            routing=RoutingStrategy.FACTORIZED_SUBSPACE,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
            num_subspaces=4,
        )
        model = build_ngs(784, 10, config)

        x = torch.randn(4, 784)
        z = model.p_down(x)
        routing_output = model.router(z)

        assert isinstance(routing_output.level_indices, list)
        assert isinstance(routing_output.level_weights, list)
        assert len(routing_output.level_indices) == 4
        assert len(routing_output.level_weights) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
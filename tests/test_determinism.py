"""Tests for seed reproducibility across devices."""

import torch
import numpy as np
import pytest
from mngs.core.config import MNGSConfig, TopologyControl
from mngs import build_mngs
from mngs.training.trainer import NGSTrainer, TrainConfig


def set_all_seeds(seed: int):
    """Set all random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TestDeterminism:
    """Test deterministic behavior across runs and devices."""

    def test_model_initialization_determinism(self):
        """Test model initialization is deterministic with same seed."""
        config = MNGSConfig(max_k=64, k_init=16, latent_dim=32)

        set_all_seeds(42)
        model1 = build_mngs(784, 10, config)

        set_all_seeds(42)
        model2 = build_mngs(784, 10, config)

        # Compare all parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-7)

    def test_forward_determinism(self):
        """Test forward pass is deterministic."""
        config = MNGSConfig(max_k=64, k_init=16, latent_dim=32)

        set_all_seeds(42)
        model1 = build_mngs(784, 10, config)

        set_all_seeds(42)
        model2 = build_mngs(784, 10, config)

        x = torch.randn(8, 784)

        out1 = model1(x)
        out2 = model2(x)

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_backward_determinism(self):
        """Test backward pass is deterministic."""
        config = MNGSConfig(max_k=64, k_init=16, latent_dim=32)

        set_all_seeds(42)
        model1 = build_mngs(784, 10, config)

        set_all_seeds(42)
        model2 = build_mngs(784, 10, config)

        x = torch.randn(8, 784)
        target = torch.randint(0, 10, (8,))

        out1 = model1(x)
        loss1 = torch.nn.functional.cross_entropy(out1, target)
        loss1.backward()

        out2 = model2(x)
        loss2 = torch.nn.functional.cross_entropy(out2, target)
        loss2.backward()

        # Compare gradients
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if p1.grad is not None and p2.grad is not None:
                assert torch.allclose(p1.grad, p2.grad, atol=1e-5)

    @pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA training not fully deterministic")
    def test_training_determinism(self):
        """Test full training epoch is deterministic (CPU only)."""
        config = MNGSConfig(max_k=64, k_init=16, latent_dim=32)
        
        set_all_seeds(42)
        model1 = build_mngs(784, 10, config)
        trainer1 = NGSTrainer(model1, TrainConfig(lr=1e-3, epochs=1, batch_size=32, device='cpu'))
        
        set_all_seeds(42)
        model2 = build_mngs(784, 10, config)
        trainer2 = NGSTrainer(model2, TrainConfig(lr=1e-3, epochs=1, batch_size=32, device='cpu'))
        
        # Same data
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 784),
            torch.randint(0, 10, (100,))
        )
        loader1 = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        set_all_seeds(42)
        loader2 = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        trainer1.train_epoch(loader1)
        trainer2.train_epoch(loader2)
        
        # Compare model states
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-4)

    def test_topology_adaptation_determinism(self):
        """Test topology adaptation is deterministic."""
        config = MNGSConfig(
            max_k=64,
            k_init=16,
            latent_dim=32,
            topology_control=TopologyControl.CONTINUOUS_DENSITY,
        )

        set_all_seeds(42)
        model1 = build_mngs(784, 10, config)

        set_all_seeds(42)
        model2 = build_mngs(784, 10, config)

        z_samples = torch.randn(50, 32)

        action1 = model1.adapt_density(z_samples=z_samples, split_thresh=0.05, prune_thresh=0.01)
        action2 = model2.adapt_density(z_samples=z_samples, split_thresh=0.05, prune_thresh=0.01)

        assert action1[0] == action2[0]  # num_pruned
        assert action1[1] == action2[1]  # num_split
        assert action1[2] == action2[2]  # num_spawned

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_cuda_consistency(self):
        """Test CPU and CUDA give same results (within tolerance)."""
        config = MNGSConfig(max_k=64, k_init=16, latent_dim=32)

        set_all_seeds(42)
        model_cpu = build_mngs(784, 10, config)

        set_all_seeds(42)
        model_cuda = build_mngs(784, 10, config).cuda()

        x_cpu = torch.randn(8, 784)
        x_cuda = x_cpu.cuda()

        out_cpu = model_cpu(x_cpu)
        out_cuda = model_cuda(x_cuda)

        # Compare on CPU
        assert torch.allclose(out_cpu, out_cuda.cpu(), atol=1e-4)

    def test_different_seeds_produce_different_results(self):
        """Test different seeds produce different models."""
        config = MNGSConfig(max_k=64, k_init=16, latent_dim=32)

        set_all_seeds(42)
        model1 = build_mngs(784, 10, config)

        set_all_seeds(123)
        model2 = build_mngs(784, 10, config)

        # At least some parameters should differ
        same = True
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.allclose(p1, p2, atol=1e-7):
                same = False
                break

        assert not same, "Different seeds produced identical models"

    def test_data_loader_determinism(self):
        """Test DataLoader shuffle is deterministic with seed."""
        dataset = torch.utils.data.TensorDataset(
            torch.arange(100).float(),
            torch.arange(100).long()
        )

        set_all_seeds(42)
        loader1 = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
        batch1 = next(iter(loader1))[0]

        set_all_seeds(42)
        loader2 = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
        batch2 = next(iter(loader2))[0]

        assert torch.equal(batch1, batch2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
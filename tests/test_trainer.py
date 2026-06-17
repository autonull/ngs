"""Integration tests for NGSTrainer."""

import torch
import pytest
from mngs.core.config import MNGSConfig
from mngs import build_mngs
from mngs.training.trainer import NGSTrainer, TrainConfig


class TestTrainer:
    """Test NGSTrainer functionality."""

    def test_trainer_creation(self):
        """Test trainer creation."""
        config = MNGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_mngs(784, 10, config)

        trainer_config = TrainConfig(
            lr=1e-3,
            epochs=1,
            batch_size=32,
        )

        trainer = NGSTrainer(model, trainer_config)

        assert trainer.model is model
        assert trainer.config.lr == 1e-3

    def test_trainer_train_epoch(self):
        """Test single training epoch."""
        config = MNGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_mngs(784, 10, config)

        trainer_config = TrainConfig(
            lr=1e-3,
            epochs=1,
            batch_size=32,
        )

        trainer = NGSTrainer(model, trainer_config)

        # Create dummy data
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 784),
            torch.randint(0, 10, (100,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        # Should not crash
        metrics = trainer.train_epoch(loader)

        assert 'loss' in metrics
        assert 'K' in metrics

    def test_trainer_with_replay_buffer(self):
        """Test trainer with replay buffer."""
        config = MNGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_mngs(784, 10, config)

        from experiments.datasets import ReplayBuffer
        replay_buffer = ReplayBuffer(max_size=1000)

        trainer_config = TrainConfig(
            lr=1e-3,
            epochs=1,
            batch_size=32,
            replay_buffer=replay_buffer,
            replay_ratio=0.5,
        )

        trainer = NGSTrainer(model, trainer_config)

        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 784),
            torch.randint(0, 10, (100,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        trainer.train_epoch(loader)

    def test_trainer_with_old_model_kd(self):
        """Test trainer with knowledge distillation."""
        config = MNGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_mngs(784, 10, config)

        trainer_config = TrainConfig(
            lr=1e-3,
            epochs=1,
            batch_size=32,
            kd_weight=10.0,
            kd_temperature=2.0,
        )

        trainer = NGSTrainer(model, trainer_config)

        # Create old model
        import copy
        old_model = copy.deepcopy(model)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False

        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 784),
            torch.randint(0, 10, (100,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        trainer.train_epoch(loader, old_model=old_model)

    def test_trainer_learning_rate_scheduler(self):
        """Test LR scheduler integration."""
        config = MNGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_mngs(784, 10, config)

        trainer_config = TrainConfig(
            lr=1e-3,
            epochs=3,
            batch_size=32,
            lr_scheduler='cosine',
        )

        trainer = NGSTrainer(model, trainer_config)

        assert trainer.scheduler is not None

        initial_lr = trainer.optimizer.param_groups[0]['lr']

        dataset = torch.utils.data.TensorDataset(
            torch.randn(50, 784),
            torch.randint(0, 10, (50,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for _ in range(3):
            trainer.train_epoch(loader)

        final_lr = trainer.optimizer.param_groups[0]['lr']

        # LR should have changed with cosine scheduler
        assert final_lr <= initial_lr

    def test_trainer_gradient_clipping(self):
        """Test gradient clipping."""
        config = MNGSConfig(max_k=64, k_init=16, latent_dim=32)
        model = build_mngs(784, 10, config)

        trainer_config = TrainConfig(
            lr=1.0,  # High LR to trigger clipping
            epochs=1,
            batch_size=32,
            grad_clip=1.0,
        )

        trainer = NGSTrainer(model, trainer_config)

        dataset = torch.utils.data.TensorDataset(
            torch.randn(50, 784),
            torch.randint(0, 10, (50,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        # Should not crash with gradient clipping
        trainer.train_epoch(loader)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
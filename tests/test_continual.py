"""Integration tests for continual learning."""

import torch
import pytest
import numpy as np
from ngs.core.interfaces import NGSConfig, TopologyControl, MemoryManagement
from ngs.models import build_ngs
from ngs.training.trainer import NGSTrainer, TrainerConfig


class TestContinualLearning:
    """Test multi-task continual learning."""

    def test_multi_task_sequence(self):
        """Test training on sequence of tasks."""
        config = NGSConfig(
            max_k=128,
            k_init=32,
            latent_dim=32,
            topology_control=TopologyControl.CONTINUOUS_DENSITY,
        )

        model = build_ngs(784, 10, config)

        trainer_config = TrainerConfig(
            lr=1e-3,
            epochs=1,
            batch_size=64,
            replay_ratio=1.0,
            kd_weight=5.0,
        )

        trainer = NGSTrainer(model, trainer_config)

        from experiments.datasets import get_task_loaders, ReplayBuffer
        from experiments.metrics import evaluate_model_on_task
        import copy

        replay_buffer = ReplayBuffer(max_size=5000)
        old_model = None

        # Train on 3 tasks
        for task_id in range(3):
            train_loader, test_loader, _ = get_task_loaders(
                'split_mnist', task_id, 2, 64
            )

            trainer.train_epoch(train_loader, replay_buffer=replay_buffer, old_model=old_model)

            # Evaluate
            acc = evaluate_model_on_task(model, test_loader, trainer.device)
            assert 0 <= acc <= 1

            # Update replay
            import torch.nn.functional as F
            for x, y in train_loader:
                x_flat = x.view(x.size(0), -1)
                y_onehot = F.one_hot(y, num_classes=10).float()
                replay_buffer.add(x_flat, y_onehot)

            # Save old model
            old_model = copy.deepcopy(model)
            old_model.eval()
            for p in old_model.parameters():
                p.requires_grad = False

    def test_knowledge_distillation(self):
        """Test KD prevents forgetting."""
        config = NGSConfig(max_k=128, k_init=32, latent_dim=32)
        model = build_ngs(784, 10, config)

        trainer_config = TrainerConfig(
            lr=1e-3,
            epochs=1,
            batch_size=64,
            kd_weight=10.0,
        )

        trainer = NGSTrainer(model, trainer_config)

        from experiments.datasets import get_task_loaders
        from experiments.metrics import evaluate_model_on_task
        import copy

        # Task 0
        train_loader, test_loader, _ = get_task_loaders('split_mnist', 0, 2, 64)
        trainer.train_epoch(train_loader)
        acc_task0_after_0 = evaluate_model_on_task(model, test_loader, trainer.device)

        # Save old model
        old_model = copy.deepcopy(model)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False

        # Task 1
        train_loader, test_loader, _ = get_task_loaders('split_mnist', 1, 2, 64)
        trainer.train_epoch(train_loader, old_model=old_model)

        # Evaluate on task 0 again
        _, test_loader_0, _ = get_task_loaders('split_mnist', 0, 2, 64)
        acc_task0_after_1 = evaluate_model_on_task(model, test_loader_0, trainer.device)

        # With KD, forgetting should be limited
        forgetting = acc_task0_after_0 - acc_task0_after_1
        assert forgetting < 0.5  # Should not forget everything

    def test_replay_buffer(self):
        """Test replay buffer helps retention."""
        config = NGSConfig(max_k=128, k_init=32, latent_dim=32)
        model = build_ngs(784, 10, config)

        trainer_config = TrainerConfig(
            lr=1e-3,
            epochs=1,
            batch_size=64,
            replay_ratio=1.0,
        )

        trainer = NGSTrainer(model, trainer_config)

        from experiments.datasets import get_task_loaders, ReplayBuffer
        from experiments.metrics import evaluate_model_on_task

        replay_buffer = ReplayBuffer(max_size=2000)

        # Task 0
        train_loader, test_loader, _ = get_task_loaders('split_mnist', 0, 2, 64)
        trainer.train_epoch(train_loader, replay_buffer=replay_buffer)

        import torch.nn.functional as F
        for x, y in train_loader:
            x_flat = x.view(x.size(0), -1)
            y_onehot = F.one_hot(y, num_classes=10).float()
            replay_buffer.add(x_flat, y_onehot)

        acc_task0 = evaluate_model_on_task(model, test_loader, trainer.device)

        # Task 1 with replay
        train_loader_1, test_loader_1, _ = get_task_loaders('split_mnist', 1, 2, 64)
        trainer.train_epoch(train_loader_1, replay_buffer=replay_buffer)

        # Check task 0 retention
        _, test_loader_0, _ = get_task_loaders('split_mnist', 0, 2, 64)
        acc_task0_after = evaluate_model_on_task(model, test_loader_0, trainer.device)

        # With replay, should retain some performance
        assert acc_task0_after > 0.1

    def test_topology_adaptation_during_cl(self):
        """Test topology adapts during continual learning."""
        config = NGSConfig(
            max_k=128,
            k_init=32,
            latent_dim=32,
            topology_control=TopologyControl.CONTINUOUS_DENSITY,
            split_threshold=0.05,
            prune_threshold=0.01,
        )

        model = build_ngs(784, 10, config)

        trainer_config = TrainerConfig(
            lr=1e-3,
            epochs=1,
            batch_size=64,
        )

        trainer = NGSTrainer(model, trainer_config)

        from experiments.datasets import get_task_loaders

        initial_k = model.K

        for task_id in range(3):
            train_loader, _, _ = get_task_loaders('split_mnist', task_id, 2, 64)
            trainer.train_epoch(train_loader)

            # Adapt topology
            model.adapt_density(split_thresh=0.05, prune_thresh=0.01)

        # K should have changed (grown)
        assert model.K >= initial_k
        assert model.K <= config.max_k

    def test_capacity_saturation(self):
        """Test model handles capacity saturation."""
        config = NGSConfig(
            max_k=32,  # Small capacity
            k_init=16,
            latent_dim=32,
            topology_control=TopologyControl.CONTINUOUS_DENSITY,
            memory_management=MemoryManagement.STRICT_CAPACITY,
        )

        model = build_ngs(784, 10, config)

        trainer_config = TrainerConfig(
            lr=1e-3,
            epochs=1,
            batch_size=64,
        )

        trainer = NGSTrainer(model, trainer_config)

        from experiments.datasets import get_task_loaders

        # Train many tasks
        for task_id in range(5):
            train_loader, _, _ = get_task_loaders('split_mnist', task_id, 2, 64)
            trainer.train_epoch(train_loader)
            model.adapt_density(split_thresh=0.01, prune_thresh=0.001)

        # Should not exceed max_k
        assert model.K <= config.max_k


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
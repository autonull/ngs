"""
Experiment configuration for LeanNGS continual learning evaluation.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    d_latent: int = 32
    k_init: int = 128
    max_k: int = 448  # ~513K params to match MLP baseline (~534K)
    top_k: int = 8
    lora_rank: int = 4
    gamma_init: float = 0.1
    tau_init: float = 1.0
    mu_init_std: float = 1.0
    w_init_std: float = 1e-4


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs_per_task: int = 2
    batch_size: int = 256
    replay_size: int = 50000
    replay_ratio: float = 1.0  # 1:1 replay
    kd_weight: float = 2.0
    kd_temperature: float = 2.0
    split_thresh: float = 0.005
    prune_thresh: float = 0.01
    max_spawn_per_call: int = 5
    adapt_every_epoch: bool = True


@dataclass
class ExperimentConfig:
    name: str
    dataset: str  # 'split_mnist', 'permuted_mnist', 'split_cifar10', 'split_fashion', 'digits', 'cifar100'
    scenario: str  # 'class_incremental', 'domain_incremental', 'task_incremental'
    n_tasks: int
    classes_per_task: int
    input_dim: int
    output_dim: int
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# Predefined experiment configurations
EXPERIMENTS = {
    'split_mnist': ExperimentConfig(
        name='Split-MNIST',
        dataset='split_mnist',
        scenario='class_incremental',
        n_tasks=5,
        classes_per_task=2,
        input_dim=784,
        output_dim=2,
    ),
    'split_fashion': ExperimentConfig(
        name='Split-FashionMNIST',
        dataset='split_fashion',
        scenario='class_incremental',
        n_tasks=5,
        classes_per_task=2,
        input_dim=784,
        output_dim=2,
    ),
    'permuted_mnist': ExperimentConfig(
        name='Permuted-MNIST',
        dataset='permuted_mnist',
        scenario='domain_incremental',
        n_tasks=10,
        classes_per_task=10,
        input_dim=784,
        output_dim=10,
    ),
    'split_cifar10': ExperimentConfig(
        name='Split-CIFAR10',
        dataset='split_cifar10',
        scenario='class_incremental',
        n_tasks=5,
        classes_per_task=2,
        input_dim=3072,
        output_dim=2,
    ),
    'split_cifar100': ExperimentConfig(
        name='Split-CIFAR100',
        dataset='split_cifar100',
        scenario='class_incremental',
        n_tasks=10,
        classes_per_task=10,
        input_dim=3072,
        output_dim=10,
    ),
    'digits': ExperimentConfig(
        name='Digits',
        dataset='digits',
        scenario='class_incremental',
        n_tasks=5,
        classes_per_task=2,
        input_dim=64,
        output_dim=2,
    ),
    'rotated_mnist': ExperimentConfig(
        name='Rotated-MNIST',
        dataset='rotated_mnist',
        scenario='domain_incremental',
        n_tasks=10,
        classes_per_task=10,
        input_dim=784,
        output_dim=10,
    ),
    'blurry_mnist': ExperimentConfig(
        name='Blurry-MNIST',
        dataset='blurry_mnist',
        scenario='domain_incremental',
        n_tasks=5,
        classes_per_task=10,
        input_dim=784,
        output_dim=10,
    ),
    'noisy_mnist': ExperimentConfig(
        name='Noisy-MNIST',
        dataset='noisy_mnist',
        scenario='domain_incremental',
        n_tasks=5,
        classes_per_task=10,
        input_dim=784,
        output_dim=10,
    ),
    'split_cifar100_20': ExperimentConfig(
        name='Split-CIFAR100-20',
        dataset='split_cifar100',
        scenario='class_incremental',
        n_tasks=20,
        classes_per_task=5,
        input_dim=3072,
        output_dim=5,
    ),
}
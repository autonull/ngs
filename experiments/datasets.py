"""
Dataset loaders for continual learning experiments.
Supports: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, Digits, Permuted MNIST.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import numpy as np
import random
from typing import Tuple, List, Optional, Iterator
from dataclasses import dataclass


@dataclass
class TaskData:
    train_loader: DataLoader
    test_loader: DataLoader
    classes: List[int]
    task_id: int


def get_transform(dataset: str, augment: bool = False):
    """Get transforms for dataset."""
    if dataset in ['mnist', 'fashion', 'permuted_mnist']:
        mean, std = (0.1307,), (0.3081,)
        if augment:
            return transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif dataset in ['cifar10', 'cifar100']:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        if augment:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([transforms.ToTensor()])


def load_digits_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Load digits dataset from torchvision (MNIST subset) or generate synthetic."""
    # Use MNIST as digits proxy since sklearn not available
    from torchvision import datasets
    train_ds = datasets.MNIST('./data', train=True, download=True)
    test_ds = datasets.MNIST('./data', train=False, download=True)
    
    X = torch.cat([train_ds.data, test_ds.data], dim=0).float().numpy().reshape(-1, 784) / 255.0
    y = torch.cat([train_ds.targets, test_ds.targets], dim=0).numpy()
    
    # Use only first 1000 samples per class for efficiency
    X_sub = []
    y_sub = []
    for c in range(10):
        mask = y == c
        X_sub.append(X[mask][:1000])
        y_sub.append(y[mask][:1000])
    
    return np.vstack(X_sub), np.hstack(y_sub)


class PermutedMNIST:
    """Permuted MNIST for domain-incremental learning."""
    def __init__(self, n_tasks: int = 10, seed: int = 42):
        self.n_tasks = n_tasks
        self.permutations = []
        rng = np.random.RandomState(seed)
        for _ in range(n_tasks):
            perm = rng.permutation(784)
            self.permutations.append(perm)

    def get_task_data(self, task_id: int, batch_size: int = 256) -> Tuple[DataLoader, DataLoader]:
        transform = get_transform('permuted_mnist')
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)

        def permute_data(dataset, perm):
            data = dataset.data.view(-1, 784).float() / 255.0
            data = data[:, perm]
            dataset.data = data.view(-1, 1, 28, 28)
            return dataset

        train_ds = permute_data(train_ds, self.permutations[task_id])
        test_ds = permute_data(test_ds, self.permutations[task_id])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader


class RemapLabels:
    """Wrapper to remap labels to 0..C-1."""
    def __init__(self, dataset, classes):
        self.dataset = dataset
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()
        self.indices = [i for i, t in enumerate(targets) if t in classes]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, y = self.dataset[real_idx]
        # y can be int or tensor
        y_val = y.item() if hasattr(y, 'item') else y
        return x, self.class_to_idx[y_val]


def create_split_loaders(
    dataset_name: str,
    task_id: int,
    classes_per_task: int,
    batch_size: int = 256,
    augment: bool = False
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """Create train/test loaders for a specific split task."""
    transform = get_transform(dataset_name, augment)

    if dataset_name == 'mnist':
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    elif dataset_name == 'fashion':
        train_ds = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        train_ds = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))

    train_wrapped = RemapLabels(train_ds, classes)
    test_wrapped = RemapLabels(test_ds, classes)

    train_loader = DataLoader(train_wrapped, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_wrapped, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, classes


def create_digits_loaders(
    task_id: int,
    classes_per_task: int = 2,
    batch_size: int = 256,
    **kwargs
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """Create loaders for digits dataset (using MNIST as proxy)."""
    X, y = load_digits_dataset()
    classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))

    mask = np.isin(y, classes)
    X_task, y_task = X[mask], y[mask]

    # Remap labels to 0, 1, ...
    y_task = np.searchsorted(classes, y_task)

    # Simple split
    n_train = int(0.8 * len(X_task))
    idx = np.random.permutation(len(X_task))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    train_ds = TensorDataset(
        torch.from_numpy(X_task[train_idx]).float(),
        torch.from_numpy(y_task[train_idx]).long()
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_task[test_idx]).float(),
        torch.from_numpy(y_task[test_idx]).long()
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, classes


def get_task_loaders(
    config_name: str,
    task_id: int,
    classes_per_task: int = 2,
    batch_size: int = 256,
    **kwargs
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """Unified interface for getting task loaders."""
    if config_name == 'split_mnist':
        return create_split_loaders('mnist', task_id, classes_per_task, batch_size)
    elif config_name == 'split_fashion':
        return create_split_loaders('fashion', task_id, classes_per_task, batch_size)
    elif config_name == 'split_cifar10':
        return create_split_loaders('cifar10', task_id, classes_per_task, batch_size)
    elif config_name == 'split_cifar100':
        return create_split_loaders('cifar100', task_id, classes_per_task, batch_size)
    elif config_name == 'digits':
        return create_digits_loaders(task_id, classes_per_task, batch_size)
    elif config_name == 'permuted_mnist':
        # Need to handle differently - returns pre-made loaders
        raise NotImplementedError("Use PermutedMNIST class directly")
    else:
        raise ValueError(f"Unknown config: {config_name}")


class ReplayBuffer:
    """Reservoir sampling replay buffer."""
    def __init__(self, max_size: int = 50000, seed: int = 42):
        self.max_size = max_size
        self.buffer = []
        self.rng = random.Random(seed)

    def add(self, x: torch.Tensor, y: torch.Tensor):
        for xi, yi in zip(x, y):
            if len(self.buffer) >= self.max_size:
                idx = self.rng.randrange(len(self.buffer))
                self.buffer[idx] = (xi.clone(), yi.clone())
            else:
                self.buffer.append((xi.clone(), yi.clone()))

    def sample(self, batch_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if len(self.buffer) == 0:
            return None, None
        batch = self.rng.sample(self.buffer, min(batch_size, len(self.buffer)))
        x = torch.stack([b[0] for b in batch])
        y = torch.stack([b[1] for b in batch])
        return x, y

    def __len__(self):
        return len(self.buffer)
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
    elif dataset == 'svhn':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        if augment:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
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
        # Load raw data and create permuted tensors
        train_ds = datasets.MNIST('./data', train=True, download=True)
        test_ds = datasets.MNIST('./data', train=False, download=True)

        perm = self.permutations[task_id]
        
        # Pre-compute permuted data
        train_data = train_ds.data.view(-1, 784).float() / 255.0
        train_data = train_data[:, perm]
        train_targets = train_ds.targets
        
        test_data = test_ds.data.view(-1, 784).float() / 255.0
        test_data = test_data[:, perm]
        test_targets = test_ds.targets

        # Create tensor datasets
        from torch.utils.data import TensorDataset
        train_ds = TensorDataset(train_data, train_targets)
        test_ds = TensorDataset(test_data, test_targets)

        # Wrap to apply normalization
        class NormalizeDataset:
            def __init__(self, dataset):
                self.dataset = dataset
                self.mean = 0.1307
                self.std = 0.3081
            def __len__(self):
                return len(self.dataset)
            def __getitem__(self, idx):
                x, y = self.dataset[idx]
                x = (x - self.mean) / self.std
                return x, y

        train_ds = NormalizeDataset(train_ds)
        test_ds = NormalizeDataset(test_ds)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available(),
            persistent_workers=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available(),
            persistent_workers=False
        )
        return train_loader, test_loader


class RotatedMNIST:
    """Rotated MNIST for domain-incremental learning."""
    def __init__(self, n_tasks: int = 10, max_angle: float = 180.0, seed: int = 42):
        self.n_tasks = n_tasks
        self.max_angle = max_angle
        self.angles = np.linspace(0, max_angle, n_tasks, endpoint=False)
        rng = np.random.RandomState(seed)
        rng.shuffle(self.angles)

    def get_task_data(self, task_id: int, batch_size: int = 256) -> Tuple[DataLoader, DataLoader]:
        angle = self.angles[task_id]
        
        transform = transforms.Compose([
            transforms.Lambda(lambda img: transforms.functional.rotate(img, angle)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available(),
            persistent_workers=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available(),
            persistent_workers=False
        )
        return train_loader, test_loader


class BlurryMNIST:
    """Blurry MNIST for domain-incremental learning."""
    def __init__(self, n_tasks: int = 10, max_kernel: int = 9, seed: int = 42):
        self.n_tasks = n_tasks
        self.kernels = list(range(1, max_kernel + 1, 2))  # 1, 3, 5, 7, 9
        # Extend if needed
        while len(self.kernels) < n_tasks:
            self.kernels.extend(self.kernels)
        self.kernels = self.kernels[:n_tasks]
        rng = np.random.RandomState(seed)
        rng.shuffle(self.kernels)

    def get_task_data(self, task_id: int, batch_size: int = 256) -> Tuple[DataLoader, DataLoader]:
        kernel = self.kernels[task_id]
        
        if kernel == 1:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            transform = transforms.Compose([
                transforms.GaussianBlur(kernel_size=kernel, sigma=(0.5, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available(),
            persistent_workers=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available(),
            persistent_workers=False
        )
        return train_loader, test_loader


class NoisyMNIST:
    """Noisy MNIST for domain-incremental learning."""
    def __init__(self, n_tasks: int = 10, max_noise: float = 0.5, seed: int = 42):
        self.n_tasks = n_tasks
        self.noise_levels = np.linspace(0, max_noise, n_tasks)
        rng = np.random.RandomState(seed)
        rng.shuffle(self.noise_levels)

    def get_task_data(self, task_id: int, batch_size: int = 256) -> Tuple[DataLoader, DataLoader]:
        noise = self.noise_levels[task_id]
        
        class AddNoise:
            def __init__(self, noise_std):
                self.noise_std = noise_std
            def __call__(self, tensor):
                return tensor + torch.randn_like(tensor) * self.noise_std
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            AddNoise(noise),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available(),
            persistent_workers=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available(),
            persistent_workers=False
        )
        return train_loader, test_loader


class RemapLabels:
    """Wrapper to remap labels to 0..C-1."""
    def __init__(self, dataset, classes):
        self.dataset = dataset
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        # Handle different attribute names (targets for MNIST/CIFAR, labels for SVHN)
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        elif hasattr(dataset, 'labels'):
            targets = dataset.labels
        else:
            raise AttributeError(f"Dataset {type(dataset)} has no targets or labels attribute")
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
    augment: bool = False,
    remap_labels: bool = False
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """Create train/test loaders for a specific split task.
    
    Args:
        remap_labels: If True, remap labels to 0..C-1 (task-incremental).
                     If False, keep original labels (class-incremental).
    """
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
    elif dataset_name == 'svhn':
        train_ds = datasets.SVHN('./data', split='train', download=True, transform=transform)
        test_ds = datasets.SVHN('./data', split='test', download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))

    if remap_labels:
        train_wrapped = RemapLabels(train_ds, classes)
        test_wrapped = RemapLabels(test_ds, classes)
    else:
        # Class-incremental: keep original labels, filter by classes
        from torch.utils.data import Subset
        train_idx = [i for i, (_, target) in enumerate(train_ds) if target in classes]
        test_idx = [i for i, (_, target) in enumerate(test_ds) if target in classes]
        train_wrapped = Subset(train_ds, train_idx)
        test_wrapped = Subset(test_ds, test_idx)

    train_loader = DataLoader(
        train_wrapped, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )
    test_loader = DataLoader(
        test_wrapped, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

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

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    return train_loader, test_loader, classes


def get_task_loaders(
    config_name: str,
    task_id: int,
    classes_per_task: int = 2,
    batch_size: int = 256,
    scenario: str = 'class_incremental',
    **kwargs
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """Unified interface for getting task loaders.
    
    Args:
        scenario: 'class_incremental' (keep original labels), 'task_incremental' (remap labels)
    """
    remap = scenario == 'task_incremental'
    
    if config_name == 'split_mnist':
        return create_split_loaders('mnist', task_id, classes_per_task, batch_size, remap_labels=remap)
    elif config_name == 'split_fashion':
        return create_split_loaders('fashion', task_id, classes_per_task, batch_size, remap_labels=remap)
    elif config_name == 'split_cifar10':
        return create_split_loaders('cifar10', task_id, classes_per_task, batch_size, remap_labels=remap)
    elif config_name == 'split_cifar100':
        return create_split_loaders('cifar100', task_id, classes_per_task, batch_size, remap_labels=remap)
    elif config_name == 'digits':
        return create_digits_loaders(task_id, classes_per_task, batch_size)
    elif config_name == 'permuted_mnist':
        permuted = kwargs.get('permuted_obj')
        if permuted is None:
            permuted = PermutedMNIST()
        train_loader, test_loader = permuted.get_task_data(task_id, batch_size)
        classes = list(range(10))
        return train_loader, test_loader, classes
    elif config_name == 'rotated_mnist':
        rotated = kwargs.get('rotated_obj')
        if rotated is None:
            rotated = RotatedMNIST()
        train_loader, test_loader = rotated.get_task_data(task_id, batch_size)
        classes = list(range(10))
        return train_loader, test_loader, classes
    elif config_name == 'blurry_mnist':
        blurry = kwargs.get('blurry_obj')
        if blurry is None:
            blurry = BlurryMNIST()
        train_loader, test_loader = blurry.get_task_data(task_id, batch_size)
        classes = list(range(10))
        return train_loader, test_loader, classes
    elif config_name == 'noisy_mnist':
        noisy = kwargs.get('noisy_obj')
        if noisy is None:
            noisy = NoisyMNIST()
        train_loader, test_loader = noisy.get_task_data(task_id, batch_size)
        classes = list(range(10))
        return train_loader, test_loader, classes
    elif config_name == 'svhn':
        return create_split_loaders('svhn', task_id, classes_per_task, batch_size, remap_labels=remap)
    elif config_name == 'mnist':
        # Full MNIST: all 10 classes in one task
        transform = get_transform('mnist', augment=False)
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
        classes = list(range(10))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, classes
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
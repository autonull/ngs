"""
Fast data loading for ablation experiments - pre-load to tensors to avoid PIL overhead.
Saves ~66% training time by eliminating per-batch transforms.
"""
import torch
import torch.utils.data as data
import numpy as np
from typing import Tuple


def load_cifar10_fast(data_dir: str = './data') -> Tuple[data.TensorDataset, data.TensorDataset, int, int]:
    """Load CIFAR10 as pre-transformed tensors."""
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    
    # Pre-load all to tensors (one-time cost)
    print("Pre-loading CIFAR10 to tensors...")
    train_x = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    train_y = torch.tensor([train_ds[i][1] for i in range(len(train_ds))], dtype=torch.long)
    test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    test_y = torch.tensor([test_ds[i][1] for i in range(len(test_ds))], dtype=torch.long)
    
    train_tensor_ds = data.TensorDataset(train_x, train_y)
    test_tensor_ds = data.TensorDataset(test_x, test_y)
    
    return train_tensor_ds, test_tensor_ds, 3072, 10


def load_mnist_fast(data_dir: str = './data') -> Tuple[data.TensorDataset, data.TensorDataset, int, int]:
    """Load MNIST as pre-transformed tensors."""
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    train_x = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    train_y = torch.tensor([train_ds[i][1] for i in range(len(train_ds))], dtype=torch.long)
    test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    test_y = torch.tensor([test_ds[i][1] for i in range(len(test_ds))], dtype=torch.long)
    
    return data.TensorDataset(train_x, train_y), data.TensorDataset(test_x, test_y), 784, 10


def load_fashion_mnist_fast(data_dir: str = './data') -> Tuple[data.TensorDataset, data.TensorDataset, int, int]:
    """Load Fashion-MNIST as pre-transformed tensors."""
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    
    train_ds = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    
    train_x = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    train_y = torch.tensor([train_ds[i][1] for i in range(len(train_ds))], dtype=torch.long)
    test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    test_y = torch.tensor([test_ds[i][1] for i in range(len(test_ds))], dtype=torch.long)
    
    return data.TensorDataset(train_x, train_y), data.TensorDataset(test_x, test_y), 784, 10


def load_digits_fast() -> Tuple[data.TensorDataset, data.TensorDataset, int, int]:
    """Load Digits as tensors."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return (
        data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        64, 10
    )


def get_fast_loaders(dataset: str, batch_size: int = 256) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, int]:
    """Get fast DataLoaders for any supported dataset."""
    loaders = {
        'cifar10': load_cifar10_fast,
        'mnist': load_mnist_fast,
        'fashion_mnist': load_fashion_mnist_fast,
        'digits': load_digits_fast,
    }
    
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(loaders.keys())}")
    
    train_ds, test_ds, input_dim, output_dim = loaders[dataset]()
    
    # Adaptive batch size
    batch_size = min(batch_size, max(32, len(train_ds) // 50))
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    
    return train_loader, test_loader, input_dim, output_dim
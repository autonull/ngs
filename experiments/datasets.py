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


class MultimodalMNIST:
    """
    Multimodal MNIST dataset with two modalities:
    - Modality 0: Original MNIST images
    - Modality 1: Transformed MNIST (permuted/rotated/noisy)
    
    For testing FactorizedRouter: each modality gets its own subspace.
    """
    def __init__(self, 
                 modality_types: tuple = ("original", "permuted"),
                 seed: int = 42):
        self.modality_types = modality_types
        self.num_modalities = len(modality_types)
        self.seed = seed
        
        # Load base MNIST
        train_ds = datasets.MNIST('./data', train=True, download=True)
        test_ds = datasets.MNIST('./data', train=False, download=True)
        
        self.train_data = train_ds.data.view(-1, 784).float() / 255.0
        self.train_targets = train_ds.targets
        self.test_data = test_ds.data.view(-1, 784).float() / 255.0
        self.test_targets = test_ds.targets
        
        # Generate transforms for each modality
        rng = np.random.RandomState(seed)
        self.perm_indices = {}  # mod_idx -> permutation
        self.angle_map = {}     # mod_idx -> angle
        self.noise_map = {}     # mod_idx -> noise_std
        
        for mod_idx, mod_type in enumerate(modality_types):
            if mod_type == "permuted":
                self.perm_indices[mod_idx] = rng.permutation(784)
            elif mod_type == "rotated":
                self.angle_map[mod_idx] = rng.uniform(0, 180)
            elif mod_type == "noisy":
                self.noise_map[mod_idx] = rng.uniform(0, 0.3)
            # original needs no extra params
    
    def _get_modality_data(self, mod_idx: int, data: torch.Tensor) -> torch.Tensor:
        """Apply modality-specific transform to data."""
        mod_type = self.modality_types[mod_idx]
        
        if mod_type == "original":
            return data
        elif mod_type == "permuted":
            perm = self.perm_indices[mod_idx]
            return data[:, perm]
        elif mod_type == "rotated":
            angle = self.angle_map[mod_idx]
            # Simple rotation approximation: roll pixels
            perm = np.roll(np.arange(784), int(angle * 784 / 180))
            return data[:, perm]
        elif mod_type == "noisy":
            noise_std = self.noise_map[mod_idx]
            noise = torch.randn_like(data) * noise_std
            return torch.clamp(data + noise, 0, 1)
        else:
            return data
    
    def get_loaders(self, batch_size: int = 256) -> tuple:
        """Get train and test loaders returning (modalities_list, labels)."""
        # Prepare multimodal data
        train_modalities = []
        test_modalities = []
        
        for m in range(self.num_modalities):
            train_mod = self._get_modality_data(m, self.train_data)
            test_mod = self._get_modality_data(m, self.test_data)
            
            # Normalize
            train_mod = (train_mod - 0.1307) / 0.3081
            test_mod = (test_mod - 0.1307) / 0.3081
            
            train_modalities.append(train_mod)
            test_modalities.append(test_mod)
        
        # Stack modalities: [num_modalities, N, 784] -> [N, num_modalities, 784]
        train_modalities = torch.stack(train_modalities, dim=1)
        test_modalities = torch.stack(test_modalities, dim=1)
        
        train_ds = TensorDataset(train_modalities, self.train_targets)
        test_ds = TensorDataset(test_modalities, self.test_targets)
        
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


def get_multimodal_loaders(
    modality_types: tuple = ("original", "permuted"),
    batch_size: int = 256,
    seed: int = 42
) -> tuple:
    """Convenience function to get multimodal MNIST loaders."""
    dataset = MultimodalMNIST(modality_types=modality_types, seed=seed)
    return dataset.get_loaders(batch_size=batch_size)


# Add missing imports for Omniglot
import os
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image


class OmniglotDataset:
    """
    Omniglot dataset for few-shot learning.
    Downloads from GitHub if not present.
    """
    def __init__(self, data_dir: str = './data', download: bool = True):
        self.data_dir = data_dir
        self.images_dir = data_dir  # Files extracted directly to data_dir
        
        if download and not (os.path.exists(os.path.join(data_dir, 'images_background')) or os.path.exists(os.path.join(data_dir, 'images_evaluation'))):
            self._download()
        
        self._load_data()
    
    def _download(self):
        """Download Omniglot dataset."""
        import urllib.request
        import zipfile
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        urls = [
            "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip",
            "https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip"
        ]
        
        for url in urls:
            filename = url.split('/')[-1]
            filepath = os.path.join(self.data_dir, filename)
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            os.remove(filepath)
    
    def _load_data(self):
        """Load all images and organize by alphabet/character."""
        self.character_paths = []
        self.character_labels = []
        self.alphabet_to_chars = {}
        
        for split in ['images_background', 'images_evaluation']:
            split_dir = os.path.join(self.images_dir, split)
            if not os.path.exists(split_dir):
                continue
                
            for alphabet in sorted(os.listdir(split_dir)):
                alphabet_dir = os.path.join(split_dir, alphabet)
                if not os.path.isdir(alphabet_dir):
                    continue
                
                for char in sorted(os.listdir(alphabet_dir)):
                    char_dir = os.path.join(alphabet_dir, char)
                    if not os.path.isdir(char_dir):
                        continue
                    
                    label = f"{alphabet}/{char}"
                    if alphabet not in self.alphabet_to_chars:
                        self.alphabet_to_chars[alphabet] = []
                    self.alphabet_to_chars[alphabet].append(label)
                    
                    for img_file in os.listdir(char_dir):
                        if img_file.endswith('.png'):
                            self.character_paths.append(os.path.join(char_dir, img_file))
                            self.character_labels.append(label)
        
        self.unique_labels = sorted(list(set(self.character_labels)))
        self.label_to_idx = {l: i for i, l in enumerate(self.unique_labels)}
        self.idx_to_label = {i: l for i, l in enumerate(self.unique_labels)}
        
        print(f"Loaded Omniglot: {len(self.character_paths)} images, {len(self.unique_labels)} characters")
    
    def get_transform(self, image_size: int = 28):
        """Get transform for Omniglot images."""
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def get_dataloader(self, batch_size: int = 32, image_size: int = 28, shuffle: bool = True):
        """Get dataloader for all images."""
        transform = self.get_transform(image_size)
        
        class OmniglotDatasetTorch(torch.utils.data.Dataset):
            def __init__(self, paths, labels, label_to_idx, transform):
                self.paths = paths
                self.labels = labels
                self.label_to_idx = label_to_idx
                self.transform = transform
            
            def __len__(self):
                return len(self.paths)
            
            def __getitem__(self, idx):
                img = Image.open(self.paths[idx]).convert('L')
                label = self.label_to_idx[self.labels[idx]]
                if self.transform:
                    img = self.transform(img)
                return img, label
        
        dataset = OmniglotDatasetTorch(
            self.character_paths, self.character_labels, self.label_to_idx, transform
        )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def create_omniglot_fewshot_tasks(
    dataset: OmniglotDataset,
    n_way: int = 5,
    k_shot: int = 1,
    n_query: int = 15,
    n_tasks: int = 100,
    image_size: int = 28,
    seed: int = 42
) -> List[Tuple[DataLoader, DataLoader, List[int]]]:
    """
    Create few-shot tasks from Omniglot.
    
    Returns:
        List of (support_loader, query_loader, class_indices)
    """
    rng = np.random.RandomState(seed)
    transform = dataset.get_transform(image_size)
    
    # Group images by character
    char_to_paths = {}
    for path, label in zip(dataset.character_paths, dataset.character_labels):
        if label not in char_to_paths:
            char_to_paths[label] = []
        char_to_paths[label].append(path)
    
    tasks = []
    for _ in range(n_tasks):
        # Sample n_way characters
        selected_chars = rng.choice(list(char_to_paths.keys()), n_way, replace=False)
        
        support_paths, support_labels = [], []
        query_paths, query_labels = [], []
        
        for i, char in enumerate(selected_chars):
            paths = char_to_paths[char]
            rng.shuffle(paths)
            
            support = paths[:k_shot]
            query = paths[k_shot:k_shot + n_query]
            
            support_paths.extend(support)
            support_labels.extend([i] * len(support))
            query_paths.extend(query)
            query_labels.extend([i] * len(query))
        
        # Create datasets
        class FewShotDataset(torch.utils.data.Dataset):
            def __init__(self, paths, labels, transform):
                self.paths = paths
                self.labels = labels
                self.transform = transform
            
            def __len__(self):
                return len(self.paths)
            
            def __getitem__(self, idx):
                img = Image.open(self.paths[idx]).convert('L')
                label = self.labels[idx]
                if self.transform:
                    img = self.transform(img)
                return img, label
        
        support_dataset = FewShotDataset(support_paths, support_labels, transform)
        query_dataset = FewShotDataset(query_paths, query_labels, transform)
        
        support_loader = DataLoader(support_dataset, batch_size=len(support_paths), shuffle=True, num_workers=0)
        query_loader = DataLoader(query_dataset, batch_size=len(query_paths), shuffle=False, num_workers=0)
        
        tasks.append((support_loader, query_loader, list(range(n_way))))
    
    return tasks


def get_omniglot_loaders(
    n_way: int = 5,
    k_shot: int = 1,
    n_query: int = 15,
    n_tasks: int = 100,
    image_size: int = 28,
    seed: int = 42,
    data_dir: str = './data'
) -> List[Tuple[DataLoader, DataLoader, List[int]]]:
    """Convenience function to get Omniglot few-shot tasks."""
    dataset = OmniglotDataset(data_dir=data_dir)
    return create_omniglot_fewshot_tasks(
        dataset, n_way, k_shot, n_query, n_tasks, image_size, seed
    )
"""
TinyShakespeare dataset loader for character-level language modeling.
Downloads from karpathy/char-rnn and splits into 5 tasks by acts.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import requests
import os
from typing import Tuple, List


def download_tinyshakespeare(data_dir: str = './data') -> str:
    """Download TinyShakespeare dataset if not present."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = os.path.join(data_dir, 'tinyshakespeare.txt')
    
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(filepath):
        print(f"Downloading TinyShakespeare from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, 'w') as f:
            f.write(response.text)
        print(f"Saved to {filepath}")
    
    return filepath


def load_tinyshakespeare(data_dir: str = './data', seq_len: int = 64) -> Tuple[List[int], dict, dict]:
    """
    Load TinyShakespeare and return encoded sequence, vocab mappings.
    
    Returns:
        encoded: List of integer token IDs
        stoi: char -> int mapping
        itos: int -> char mapping
    """
    filepath = download_tinyshakespeare(data_dir)
    
    with open(filepath, 'r') as f:
        text = f.read()
    
    # Build vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode text
    encoded = [stoi[ch] for ch in text]
    
    return encoded, stoi, itos


def create_sequences(encoded: List[int], seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input/target sequences for next-character prediction.
    
    Args:
        encoded: List of token IDs
        seq_len: Sequence length
        
    Returns:
        X: [N, seq_len] input sequences
        y: [N] target characters (next char)
    """
    n = len(encoded) - seq_len
    X = np.zeros((n, seq_len), dtype=np.int64)
    y = np.zeros(n, dtype=np.int64)
    
    for i in range(n):
        X[i] = encoded[i:i+seq_len]
        y[i] = encoded[i+seq_len]
    
    return X, y


def split_into_tasks(X: np.ndarray, y: np.ndarray, n_tasks: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split sequences into n_tasks contiguous chunks.
    Each task gets a contiguous portion of the text (like acts in a play).
    """
    task_size = len(X) // n_tasks
    tasks = []
    
    for i in range(n_tasks):
        start = i * task_size
        end = start + task_size if i < n_tasks - 1 else len(X)
        tasks.append((X[start:end], y[start:end]))
    
    return tasks


def create_tinyshakespeare_loaders(
    task_id: int,
    n_tasks: int = 5,
    seq_len: int = 64,
    batch_size: int = 256,
    data_dir: str = './data'
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """
    Create train/test loaders for TinyShakespeare task.
    
    Args:
        task_id: Task index (0 to n_tasks-1)
        n_tasks: Number of tasks to split into
        seq_len: Sequence length
        batch_size: Batch size
        data_dir: Data directory
        
    Returns:
        train_loader, test_loader, classes (vocab indices)
    """
    encoded, stoi, itos = load_tinyshakespeare(data_dir, seq_len)
    vocab_size = len(stoi)
    
    # Create sequences
    X, y = create_sequences(encoded, seq_len)
    
    # Split into tasks
    tasks = split_into_tasks(X, y, n_tasks)
    
    # Get task data
    X_task, y_task = tasks[task_id]
    
    # Train/test split (80/20)
    n_train = int(0.8 * len(X_task))
    idx = np.random.permutation(len(X_task))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    
    # One-hot encode inputs for MLP compatibility
    X_train_oh = np.zeros((len(train_idx), seq_len, vocab_size), dtype=np.float32)
    X_test_oh = np.zeros((len(test_idx), seq_len, vocab_size), dtype=np.float32)
    
    for i, idx in enumerate(train_idx):
        X_train_oh[i, np.arange(seq_len), X_task[idx]] = 1.0
    for i, idx in enumerate(test_idx):
        X_test_oh[i, np.arange(seq_len), X_task[idx]] = 1.0
    
    train_ds = TensorDataset(
        torch.from_numpy(X_train_oh).float(),
        torch.from_numpy(y_task[train_idx]).long()
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test_oh).float(),
        torch.from_numpy(y_task[test_idx]).long()
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Classes are all vocab characters (for next-char prediction)
    classes = list(range(vocab_size))
    
    return train_loader, test_loader, classes


# Integration with experiment framework
def get_tinyshakespeare_loaders(
    config_name: str,
    task_id: int,
    classes_per_task: int = 65,  # vocab size
    batch_size: int = 256,
    **kwargs
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """Unified interface for getting TinyShakespeare task loaders."""
    n_tasks = kwargs.get('n_tasks', 5)
    seq_len = kwargs.get('seq_len', 64)
    data_dir = kwargs.get('data_dir', './data')
    
    return create_tinyshakespeare_loaders(task_id, n_tasks, seq_len, batch_size, data_dir)
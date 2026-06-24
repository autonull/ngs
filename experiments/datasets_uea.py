"""UEA Multivariate Time Series dataset loader."""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple, List
import os


def load_uea_dataset(dataset_name: str, data_dir: str = './data/UEA/Multivariate_ts') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load UEA dataset from .ts files.
    
    Returns:
        X_train, y_train, X_test, y_test
        X shape: [N, seq_len, n_channels]
        y shape: [N] (0-indexed)
    """
    ds_dir = Path(data_dir) / dataset_name
    train_file = ds_dir / f"{dataset_name}_TRAIN.ts"
    test_file = ds_dir / f"{dataset_name}_TEST.ts"
    
    def parse_ts(filepath, max_len: int = 200):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find @data line
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() == '@data':
                data_start = i + 1
                break
        
        X_list = []
        y_list = []
        for line in lines[data_start:]:
            line = line.strip()
            if not line:
                continue
            # Format: channel1:channel2:channel3:label
            parts = line.split(':')
            label = int(float(parts[-1]))
            channels = parts[:-1]
            
            # Parse each channel (comma-separated values)
            channel_data = []
            for ch in channels:
                vals = [float(v) for v in ch.split(',')]
                # Pad or truncate to max_len
                if len(vals) > max_len:
                    vals = vals[:max_len]
                elif len(vals) < max_len:
                    vals = vals + [0.0] * (max_len - len(vals))
                channel_data.append(vals)
            
            # Stack channels: [n_channels, seq_len]
            X_list.append(np.array(channel_data, dtype=np.float32))
            y_list.append(label)
        
        X = np.stack(X_list)  # [N, n_channels, seq_len]
        y = np.array(y_list, dtype=np.int64)
        return X, y
    
    X_train, y_train = parse_ts(train_file)
    X_test, y_test = parse_ts(test_file)
    
    # Convert labels to 0-indexed
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    label_map = {l: i for i, l in enumerate(sorted(unique_labels))}
    y_train = np.array([label_map[l] for l in y_train])
    y_test = np.array([label_map[l] for l in y_test])
    
    return X_train, y_train, X_test, y_test


def get_uea_loaders(
    dataset_name: str,
    batch_size: int = 128,
    data_dir: str = './data/UEA/Multivariate_ts'
) -> Tuple[DataLoader, DataLoader, int, int, int]:
    """
    Get train/test loaders for UEA dataset.
    
    Returns:
        train_loader, test_loader, n_classes, seq_len, n_channels
    """
    X_train, y_train, X_test, y_test = load_uea_dataset(dataset_name, data_dir)
    
    # Transpose to [N, seq_len, n_channels] for factorized router
    X_train = X_train.transpose(0, 2, 1)
    X_test = X_test.transpose(0, 2, 1)
    
    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    seq_len = X_train.shape[1]
    n_channels = X_train.shape[2]
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader, n_classes, seq_len, n_channels


if __name__ == '__main__':
    # Test loading CharacterTrajectories
    tr, te, n_classes, seq_len, n_channels = get_uea_loaders('CharacterTrajectories', batch_size=32)
    print(f"CharacterTrajectories: n_classes={n_classes}, seq_len={seq_len}, n_channels={n_channels}")
    print(f"Train batches: {len(tr)}, Test batches: {len(te)}")
    for x, y in tr:
        print(f"Batch shape: {x.shape}, labels: {y.shape}")
        break
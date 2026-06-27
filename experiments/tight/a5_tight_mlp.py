import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/me/ngs')

from ngs.models.ngs import MultiLayerNGS
from ngs.core.interfaces import NGSConfig, RoutingStrategy
from experiments.fast_data import load_mnist_fast


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def train_eval(model, train_loader, test_loader, device, epochs=10, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    first_epoch_acc = None
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).logits
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).logits
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        if first_epoch_acc is None:
            first_epoch_acc = acc
        best_acc = max(best_acc, acc)
    return best_acc, first_epoch_acc

def run_a5_tight(seed=42, epochs=10, device='cuda'):
    set_seed(seed)
    train_ds, test_ds, d_in, d_out = load_mnist_fast()
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    results = []
    for use_mlp in [False, True]:
        for mlp_mult in [1, 2, 4] if use_mlp else [1]:
            config = NGSConfig(
                latent_dim=4,
                max_k=4,
                top_k=2,
                k_init=2,
                routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
                gamma_residual=0.1,
                beta_residual=0.1,
                use_mlp_projections=use_mlp,
                mlp_hidden_multiplier=mlp_mult,
            )
            model = MultiLayerNGS(d_in, d_out, 2, [config, config])
            best_acc, first_acc = train_eval(model, train_loader, test_loader, device, epochs)
            result = {
                'config': f'd4_K4_tk2_{"MLP" if use_mlp else "Linear"}x{mlp_mult}',
                'use_mlp': use_mlp, 'mlp_mult': mlp_mult,
                'best_acc': best_acc, 'first_acc': first_acc,
            }
            results.append(result)
            print(f"  {result['config']}: best={best_acc:.4f}, 1st={first_acc:.4f}")
    return results

if __name__ == '__main__':
    results = run_a5_tight()
    print(results)

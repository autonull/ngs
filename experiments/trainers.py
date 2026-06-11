"""
Training functions for different model types.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import numpy as np
from copy import deepcopy

from experiments.baselines import MLP, LoRAMultitask, EWCModel, SIModel, LwFModel, ERModel
from experiments.metrics import evaluate_model_on_task
from experiments.datasets import ReplayBuffer


def train_mlp(model: MLP, train_loader: DataLoader, task_id: int, 
              old_model=None, device='cuda', epochs=5, lr=1e-3, weight_decay=1e-4, **kwargs):
    """Standard MLP training (no CL protection)."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        losses = []
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return np.mean(losses)


def train_er(model: ERModel, train_loader: DataLoader, task_id: int,
             old_model=None, device='cuda', epochs=5, lr=1e-3, weight_decay=1e-4,
             replay_buffer: ReplayBuffer = None, replay_ratio=1.0, **kwargs):
    """Experience Replay training."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        losses = []
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)

            # Add replay samples
            if replay_buffer and len(replay_buffer) > x.size(0):
                rx, ry = replay_buffer.sample(int(x.size(0) * replay_ratio))
                if rx is not None:
                    rx, ry = rx.to(device), ry.to(device)
                    x = torch.cat([x, rx], dim=0)
                    # ry is one-hot, convert to class indices
                    y = torch.cat([y, ry.argmax(dim=1)], dim=0)

            optimizer.zero_grad()
            logits = model.mlp(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Update replay buffer
        if replay_buffer:
            for x, y in train_loader:
                x_flat = x.view(x.size(0), -1)
                replay_buffer.add(x_flat, F.one_hot(y, num_classes=model.mlp.net[-1].out_features).float())

    return np.mean(losses)


def train_ewc(model: EWCModel, train_loader: DataLoader, task_id: int,
              old_model=None, device='cuda', epochs=5, lr=1e-3, weight_decay=1e-4,
              ewc_lambda=1000, **kwargs):
    """EWC training."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        losses = []
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss += model.ewc_loss()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return np.mean(losses)


def train_si(model: SIModel, train_loader: DataLoader, task_id: int,
             old_model=None, device='cuda', epochs=5, lr=1e-3, weight_decay=1e-4,
             si_lambda=1.0, **kwargs):
    """SI training."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        losses = []
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss += model.si_loss()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return np.mean(losses)


def train_lwf(model: LwFModel, train_loader: DataLoader, task_id: int,
              old_model=None, device='cuda', epochs=5, lr=1e-3, weight_decay=1e-4,
              lwf_lambda=1.0, temp=2.0, **kwargs):
    """LwF training."""
    model.to(device)
    model.train()

    # Set old model if provided
    if old_model is not None and task_id > 0:
        model.set_old_model(old_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        losses = []
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss += model.lwf_loss(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return np.mean(losses)


def train_lora(model: LoRAMultitask, train_loader: DataLoader, task_id: int,
               old_model=None, device='cuda', epochs=5, lr=1e-3, weight_decay=1e-4, **kwargs):
    """LoRA training (only adapters + head)."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay
    )

    for epoch in range(epochs):
        losses = []
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return np.mean(losses)


# Trainer registry
TRAINERS = {
    'mlp': train_mlp,
    'er': train_er,
    'ewc': train_ewc,
    'si': train_si,
    'lwf': train_lwf,
    'lora': train_lora,
}


def get_trainer(name: str):
    if name not in TRAINERS:
        raise ValueError(f"Unknown trainer: {name}. Choose from {list(TRAINERS.keys())}")
    return TRAINERS[name]
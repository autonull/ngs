"""
LeanNGS trainer integrated with experiment framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import numpy as np
from copy import deepcopy

from lean_ngs import LeanNGS
from experiments.datasets import ReplayBuffer
from experiments.metrics import evaluate_model_on_task


def train_lean_ngs(model: LeanNGS, train_loader: DataLoader, task_id: int,
                   old_model: Optional[LeanNGS] = None, device='cuda',
                   epochs=5, lr=1e-3, weight_decay=1e-4,
                   replay_buffer: ReplayBuffer = None, replay_ratio=1.0,
                   kd_weight=2.0, kd_temperature=2.0,
                   split_thresh=0.005, prune_thresh=0.01,
                   max_spawn_per_call=5, adapt_every_epoch=True,
                   grad_clip: float = 1.0,
                   lr_scheduler: str = 'cosine',
                   warmup_epochs: int = 0,
                   **kwargs):
    """
    Train LeanNGS with replay + KD + adaptive density control.
    
    Args:
        grad_clip: Gradient clipping max norm (0 to disable)
        lr_scheduler: 'cosine', 'step', 'constant'
        warmup_epochs: Number of warmup epochs for LR
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    if lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    elif lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    else:
        scheduler = None
    
    def get_lr(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return lr * (epoch + 1) / warmup_epochs
        return lr

    for epoch in range(epochs):
        model.train()
        losses = []
        kd_losses = []
        
        # Warmup LR
        if warmup_epochs > 0 and epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = get_lr(epoch)

        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            y_onehot = F.one_hot(y, num_classes=model.p_up.out_features).float()

            # Replay
            if replay_buffer and len(replay_buffer) > x.size(0):
                rx, ry = replay_buffer.sample(int(x.size(0) * replay_ratio))
                if rx is not None:
                    rx, ry = rx.to(device), ry.to(device)
                    x = torch.cat([x, rx], dim=0)
                    y = torch.cat([y, ry.argmax(dim=1)], dim=0)
                    y_onehot = torch.cat([y_onehot, ry], dim=0)

            optimizer.zero_grad()
            logits = model(x)

            # CE loss on all samples
            ce_loss = F.cross_entropy(logits, y)

            # KD loss on replay samples only
            kd_loss = 0
            if old_model is not None:
                with torch.no_grad():
                    old_logits = old_model(x)
                n_new = x.size(0) // (1 + int(replay_ratio))
                if n_new < x.size(0):
                    kd_loss = F.kl_div(
                        F.log_softmax(logits[n_new:] / kd_temperature, dim=-1),
                        F.softmax(old_logits[n_new:] / kd_temperature, dim=-1),
                        reduction='batchmean'
                    ) * (kd_temperature ** 2)

            total_loss = ce_loss + kd_weight * kd_loss
            total_loss.backward()
            model.update_grad_ema()
            
            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()

            losses.append(ce_loss.item())
            kd_losses.append(kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss)

        avg_loss = np.mean(losses)
        avg_kd = np.mean(kd_losses)
        
        # Step scheduler (after warmup)
        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step()

        # Adaptive density control
        if adapt_every_epoch:
            model.adapt_density(
                split_thresh=split_thresh,
                prune_thresh=prune_thresh,
                max_spawn_per_call=max_spawn_per_call
            )

    return avg_loss, avg_kd


def create_lean_ngs(input_dim: int, output_dim: int, **config) -> LeanNGS:
    """Create LeanNGS model from config."""
    return LeanNGS(
        d_in=input_dim,
        d_out=output_dim,
        d_latent=config.get('d_latent', 32),
        k_init=config.get('k_init', 128),
        max_k=config.get('max_k', 1024),
        top_k=config.get('top_k', 8),
        lora_rank=config.get('lora_rank', 4),
    )
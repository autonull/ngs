"""
MNGS trainer integrated with experiment framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict
import numpy as np
from copy import deepcopy

from mngs.model import build_mngs
from mngs.core.config import MNGSConfig
from experiments.datasets import ReplayBuffer
from experiments.metrics import evaluate_model_on_task


def train_mngs(model, train_loader: DataLoader, task_id: int,
               old_model=None, device='cuda',
               epochs=5, lr=1e-3, weight_decay=1e-4,
               replay_buffer: ReplayBuffer = None, replay_ratio=1.0,
               kd_weight=2.0, kd_temperature=2.0,
               adapt_every_epoch=True,
               entropy_weight=0.01, diversity_weight=0.0,
               split_thresh=0.005, prune_thresh=0.01,
               max_spawn_per_call=5, **kwargs):
    """
    Train MNGS with replay + KD + adaptive density control.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        losses = []
        kd_losses = []

        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)

            # Replay
            if replay_buffer and len(replay_buffer) > x.size(0):
                rx, ry = replay_buffer.sample(int(x.size(0) * replay_ratio))
                if rx is not None:
                    rx, ry = rx.to(device), ry.to(device)
                    x = torch.cat([x, rx], dim=0)
                    y = torch.cat([y, ry.argmax(dim=1)], dim=0)

            optimizer.zero_grad()
            logits = model(x)

            # CE loss
            ce_loss = F.cross_entropy(logits, y)

            # KD loss
            kd_loss = 0
            if old_model is not None and kd_weight > 0:
                with torch.no_grad():
                    old_logits = old_model(x)
                n_new = x.size(0) // (1 + int(replay_ratio)) if replay_buffer else x.size(0)
                if n_new < x.size(0):
                    kd_loss = F.kl_div(
                        F.log_softmax(logits[n_new:] / kd_temperature, dim=-1),
                        F.softmax(old_logits[n_new:] / kd_temperature, dim=-1),
                        reduction='batchmean'
                    ) * (kd_temperature ** 2)

            # Entropy regularization
            entropy_loss = model.entropy_loss(x)

            total_loss = ce_loss + kd_weight * kd_loss + entropy_weight * entropy_loss
            total_loss.backward()

            # Gradient EMA for topology
            model.update_grad_ema()

            optimizer.step()

            losses.append(ce_loss.item())
            kd_losses.append(kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss)

        avg_loss = np.mean(losses)

        # Adaptive density control (split/prune)
        if adapt_every_epoch:
            # Update grad_ema before density adaptation
            if hasattr(model, 'update_grad_ema'):
                model.update_grad_ema()
            model.adapt_density(
                split_thresh=split_thresh,
                prune_thresh=prune_thresh,
                max_spawn_per_call=max_spawn_per_call,
            )

    return avg_loss, np.mean(kd_losses)


def create_mngs(input_dim: int, output_dim: int, config: 'MNGSConfig' = None, **kwargs) -> torch.nn.Module:
    """Create MNGS model from config or kwargs."""
    if config is None:
        # Use MNGSConfig defaults and override with kwargs if provided
        from mngs.core.config import MNGSConfig
        config = MNGSConfig(
            latent_dim=kwargs.get('d_latent', 32),
            output_dim=kwargs.get('d_out', output_dim),
            k_init=kwargs.get('k_init', 128),
            max_k=kwargs.get('max_k', 1024),
            top_k=kwargs.get('top_k', 8),
        )
    return build_mngs(input_dim, output_dim, config)


def create_mngs_from_profile(name: str, input_dim: int, output_dim: int) -> torch.nn.Module:
    """Create MNGS from a named profile."""
    from mngs.profiles import Baseline_LeanNGS, CFG_Net_Full, Ultra_Edge_Sparse, Ablation_Hypernetwork_Only

    profiles = {
        'baseline': Baseline_LeanNGS(),
        'cfg_net': CFG_Net_Full(),
        'ultra_edge': Ultra_Edge_Sparse(),
        'abl_hyper': Ablation_Hypernetwork_Only(),
    }

    if name not in profiles:
        raise ValueError(f"Unknown profile: {name}. Choose from {list(profiles.keys())}")

    config = profiles[name]
    return build_mngs(input_dim, output_dim, config)


# Profile config overrides for experiments
PROFILE_TRAIN_CONFIGS = {
    'baseline': {
        'lr': 1e-3,
        'split_thresh': 0.005,
        'prune_thresh': 0.01,
        'adapt_every_epoch': True,
    },
    'cfg_net': {
        'lr': 1e-3,
        'split_thresh': 0.005,
        'prune_thresh': 0.01,
        'adapt_every_epoch': True,
    },
    'ultra_edge': {
        'lr': 1e-3,
        'split_thresh': 0.008,
        'prune_thresh': 0.02,
        'adapt_every_epoch': True,
    },
    'abl_hyper': {
        'lr': 1e-3,
        'split_thresh': 0.005,
        'prune_thresh': 0.01,
        'adapt_every_epoch': True,
    },
}

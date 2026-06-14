"""
MNGS trainer integrated with experiment framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
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
                max_spawn_per_call=5,
                grad_clip: float = 1.0,
                lr_scheduler: str = 'cosine',
                warmup_epochs: int = 0,
                **kwargs):
    """
    Train MNGS with replay + KD + adaptive density control.
    
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

            # Replay
            if replay_buffer is not None and len(replay_buffer) > x.size(0):
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
            if diversity_weight > 0:
                total_loss += diversity_weight * model.diversity_loss()
            # Split gate regularization for ContinuousDensityManager
            if hasattr(model, 'split_gate') and hasattr(model, 'error_density'):
                total_loss = total_loss + 0.001 * model.split_gate_loss()
            total_loss.backward()
            
            # Update per-unit error density for ContinuousDensityManager
            if hasattr(model, 'update_unit_errors'):
                model.update_unit_errors(logits, y)
            
            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            losses.append(ce_loss.item())
            kd_losses.append(kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss)

        avg_loss = np.mean(losses)
        
        # Step scheduler (after warmup)
        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step()

        # Adaptive density control (split/prune/spawn)
        if adapt_every_epoch:
            # Collect latent samples for spawn decisions (only for monolithic router)
            z_samples = None
            if hasattr(model, 'update_unit_errors') and not hasattr(model.router, 'subspace_projectors'):
                # Use a small batch to estimate latent coverage
                try:
                    with torch.no_grad():
                        sample_batch = next(iter(train_loader))
                        x = sample_batch[0].view(sample_batch[0].size(0), -1).to(device)
                        # Get latent representation (after p_down projection)
                        z_samples = model.p_down(x)
                except:
                    pass
            
            model.adapt_density(
                split_thresh=split_thresh,
                prune_thresh=prune_thresh,
                spawn_thresh=-5.0,
                z_samples=z_samples,
                max_spawn_per_call=max_spawn_per_call,
            )

    return avg_loss, np.mean(kd_losses)


def initialize_model_weights(model):
    """Initialize model weights for better training stability."""
    for name, param in model.named_parameters():
        if 'p_down' in name or 'p_up' in name:
            nn.init.xavier_uniform_(param)
        elif 'router.mu' in name:
            nn.init.normal_(param, mean=0.0, std=1.0)
        elif 'router.log_s' in name:
            nn.init.constant_(param, 0.0)
        elif 'router.log_alpha' in name:
            nn.init.constant_(param, 0.0)
        elif 'split_gate' in name:
            nn.init.constant_(param, 0.5)  # sigmoid(0.5) ≈ 0.62 > 0.5 threshold
        elif 'param_store.W_A' in name or 'param_store.lora_A' in name:
            nn.init.kaiming_uniform_(param, a=5**0.5)
        elif 'param_store.W_B' in name or 'param_store.lora_B' in name:
            nn.init.zeros_(param)
        elif 'param_store.codes' in name:
            nn.init.normal_(param, mean=0.0, std=0.1)
        elif 'hypernet' in name and 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'hypernet' in name and 'bias' in name:
            nn.init.zeros_(param)
        elif 'subspace_projectors' in name and 'weight' in name:
            nn.init.orthogonal_(param)


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
    model = build_mngs(input_dim, output_dim, config)
    initialize_model_weights(model)
    return model


def create_mngs_from_profile(name: str, input_dim: int, output_dim: int) -> torch.nn.Module:
    """Create MNGS from a named profile."""
    from mngs.profiles import (
        Baseline_LeanNGS, Baseline_LeanNGS_ParamMatched,
        CFG_Net_Full, CFG_Net_Full_ParamMatched,
        Ultra_Edge_Sparse,
        Ablation_Hypernetwork_Only, Ablation_Hypernetwork_Only_ParamMatched
    )

    # Parameter-matched profiles (for fair comparison with MLP/EWC/SI/LwF/ER baselines)
    profiles = {
        'baseline': Baseline_LeanNGS_ParamMatched(),
        'cfg_net': CFG_Net_Full_ParamMatched(),
        'ultra_edge': Ultra_Edge_Sparse(),
        'abl_hyper': Ablation_Hypernetwork_Only_ParamMatched(),
    }

    # LoRA-efficient versions (for efficiency comparisons)
    lora_profiles = {
        'baseline_lora': Baseline_LeanNGS(),
        'cfg_net_lora': CFG_Net_Full(),
        'abl_hyper_lora': Ablation_Hypernetwork_Only(),
    }

    all_profiles = {**profiles, **lora_profiles}

    if name not in all_profiles:
        raise ValueError(f"Unknown profile: {name}. Choose from {list(all_profiles.keys())}")

    config = all_profiles[name]
    model = build_mngs(input_dim, output_dim, config)
    initialize_model_weights(model)
    return model


# Profile config overrides for experiments
PROFILE_TRAIN_CONFIGS = {
    'baseline': {
        'lr': 1e-3,
        'split_thresh': 0.05,
        'prune_thresh': 0.01,
        'adapt_every_epoch': True,
        'kd_weight': 10.0,
    },
    'cfg_net': {
        'lr': 1e-3,
        'split_thresh': 0.02,
        'prune_thresh': 0.01,
        'adapt_every_epoch': True,
        'kd_weight': 10.0,
    },
    'cfg_net_lora': {
        'lr': 1e-3,
        'split_thresh': 0.02,
        'prune_thresh': 0.01,
        'adapt_every_epoch': True,
        'kd_weight': 10.0,
    },
    'ultra_edge': {
        'lr': 1e-3,
        'split_thresh': 0.08,
        'prune_thresh': 0.02,
        'adapt_every_epoch': True,
        'kd_weight': 10.0,
    },
    'abl_hyper': {
        'lr': 1e-3,
        'split_thresh': 0.005,
        'prune_thresh': 0.01,
        'adapt_every_epoch': True,
        'kd_weight': 10.0,
    },
    'baseline_lora': {
        'lr': 1e-3,
        'split_thresh': 0.05,
        'prune_thresh': 0.01,
        'adapt_every_epoch': True,
        'kd_weight': 10.0,
    },
    'cfg_net_lora': {
        'lr': 1e-3,
        'split_thresh': 0.02,
        'prune_thresh': 0.01,
        'adapt_every_epoch': True,
        'kd_weight': 10.0,
    },
    'abl_hyper_lora': {
        'lr': 1e-3,
        'split_thresh': 0.005,
        'prune_thresh': 0.01,
        'adapt_every_epoch': True,
        'kd_weight': 10.0,
    },
}

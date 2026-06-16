"""Unified training framework for NGS models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy
from tqdm import tqdm


@dataclass
class TrainConfig:
    """Training configuration."""
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    batch_size: int = 256
    
    # Continual learning
    replay_buffer: Optional[Any] = None
    replay_ratio: float = 1.0
    kd_weight: float = 10.0
    kd_temperature: float = 2.0
    
    # Regularization
    entropy_weight: float = 0.01
    diversity_weight: float = 0.01
    split_gate_weight: float = 0.001
    merge_weight: float = 0.01
    
    # Topology adaptation
    adapt_every_epoch: bool = True
    split_thresh: float = 0.05
    prune_thresh: float = 0.01
    max_spawn_per_call: int = 5
    
    # Optimization
    grad_clip: float = 1.0
    lr_scheduler: str = 'cosine'
    warmup_epochs: int = 0
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 1


@dataclass
class TrainMetrics:
    """Training metrics tracker."""
    losses: list = field(default_factory=list)
    ce_losses: list = field(default_factory=list)
    kd_losses: list = field(default_factory=list)
    entropy_losses: list = field(default_factory=list)
    topology_losses: list = field(default_factory=list)
    topology_actions: list = field(default_factory=list)
    active_units: list = field(default_factory=list)


class NGSTrainer:
    """Unified trainer for NGS models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        device: str = 'cuda',
        callbacks: Optional[list] = None
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.callbacks = callbacks or []
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        
        self._setup_scheduler()
        self.metrics = TrainMetrics()
        self.epoch = 0
        self.global_step = 0
        
    def _setup_scheduler(self):
        if self.config.lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, self.config.epochs - self.config.warmup_epochs)
            )
        elif self.config.lr_scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, self.config.epochs // 3), gamma=0.5
            )
        else:
            self.scheduler = None
    
    def _get_lr(self, epoch: int) -> float:
        if self.config.warmup_epochs > 0 and epoch < self.config.warmup_epochs:
            return self.config.lr * (epoch + 1) / self.config.warmup_epochs
        return self.config.lr
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        old_model: Optional[nn.Module] = None,
        task_id: int = 0
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_ce = []
        epoch_kd = []
        epoch_entropy = []
        epoch_topo = []
        
        # Warmup LR
        if self.config.warmup_epochs > 0 and self.epoch < self.config.warmup_epochs:
            for pg in self.optimizer.param_groups:
                pg['lr'] = self._get_lr(self.epoch)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}", leave=False)
        
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.view(x.size(0), -1).to(self.device)
            y = y.to(self.device)
            
            # Replay
            if self.config.replay_buffer is not None and len(self.config.replay_buffer) > x.size(0):
                rx, ry = self.config.replay_buffer.sample(int(x.size(0) * self.config.replay_ratio))
                if rx is not None:
                    rx, ry = rx.to(self.device), ry.to(self.device)
                    x = torch.cat([x, rx], dim=0)
                    y = torch.cat([y, ry.argmax(dim=1)], dim=0)
            
            self.optimizer.zero_grad()
            
            # Forward
            output = self.model(x)
            logits = output.logits
            
            # CE loss
            ce_loss = F.cross_entropy(logits, y)
            
            # KD loss
            kd_loss = torch.tensor(0.0, device=self.device)
            if old_model is not None and self.config.kd_weight > 0:
                with torch.no_grad():
                    old_output = old_model(x)
                    old_logits = old_output.logits
                n_new = x.size(0) // (1 + int(self.config.replay_ratio)) if self.config.replay_buffer else x.size(0)
                if n_new < x.size(0):
                    kd_loss = F.kl_div(
                        F.log_softmax(logits[n_new:] / self.config.kd_temperature, dim=-1),
                        F.softmax(old_logits[n_new:] / self.config.kd_temperature, dim=-1),
                        reduction='batchmean'
                    ) * (self.config.kd_temperature ** 2)
            
            # Entropy regularization
            entropy_loss = self.model.entropy_loss(x)
            
            # Topology losses
            topo_losses = self.model.compute_topology_losses()
            topo_loss = sum(topo_losses.values())
            
            # Total loss
            total_loss = (
                ce_loss 
                + self.config.kd_weight * kd_loss
                + self.config.entropy_weight * entropy_loss
                + self.config.split_gate_weight * topo_losses.get('split_gate', 0)
                + self.config.merge_weight * topo_losses.get('merge_reg', 0)
                + self.config.diversity_weight * topo_losses.get('diversity', 0)
            )
            
            total_loss.backward()
            
            # Update per-unit error density
            if hasattr(self.model, 'update_unit_errors'):
                self.model.update_unit_errors(logits, y)
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            self.global_step += 1
            
            epoch_losses.append(ce_loss.item())
            epoch_ce.append(ce_loss.item())
            epoch_kd.append(kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss)
            epoch_entropy.append(entropy_loss.item())
            epoch_topo.append(topo_loss.item() if isinstance(topo_loss, torch.Tensor) else topo_loss)
            
            # Callbacks
            for cb in self.callbacks:
                cb.on_batch_end(self, batch_idx, {
                    'loss': ce_loss.item(),
                    'kd_loss': kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss,
                    'entropy': entropy_loss.item(),
                })
            
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{ce_loss.item():.4f}',
                    'K': self.model.K
                })
        
        # Step scheduler
        if self.scheduler is not None and self.epoch >= self.config.warmup_epochs:
            self.scheduler.step()
        
        # Topology adaptation
        topo_action = None
        if self.config.adapt_every_epoch:
            z_samples = self._collect_latent_samples(train_loader)
            topo_action = self.model.adapt_density(
                z_samples=z_samples,
                split_thresh=self.config.split_thresh,
                prune_thresh=self.config.prune_thresh,
                max_spawn_per_call=self.config.max_spawn_per_call,
            )
        
        # Record metrics
        self.metrics.losses.append(np.mean(epoch_losses))
        self.metrics.ce_losses.append(np.mean(epoch_ce))
        self.metrics.kd_losses.append(np.mean(epoch_kd))
        self.metrics.entropy_losses.append(np.mean(epoch_entropy))
        self.metrics.topology_losses.append(np.mean(epoch_topo))
        if topo_action:
            self.metrics.topology_actions.append(topo_action)
        self.metrics.active_units.append(self.model.K)
        
        self.epoch += 1
        
        return {
            'loss': np.mean(epoch_losses),
            'ce_loss': np.mean(epoch_ce),
            'kd_loss': np.mean(epoch_kd),
            'entropy': np.mean(epoch_entropy),
            'topology': np.mean(epoch_topo),
            'K': self.model.K,
            'topo_action': topo_action,
        }
    
    def _collect_latent_samples(self, train_loader: DataLoader, n_samples: int = 1000) -> Optional[torch.Tensor]:
        """Collect latent samples for spawn decisions."""
        try:
            samples = []
            with torch.no_grad():
                for x, _ in train_loader:
                    x = x.view(x.size(0), -1).to(self.device)
                    z = self.model.p_down(x)
                    samples.append(z)
                    if len(torch.cat(samples)) >= n_samples:
                        break
            if samples:
                return torch.cat(samples)[:n_samples]
        except Exception:
            pass
        return None
    
    def train(
        self,
        train_loader: DataLoader,
        old_model: Optional[nn.Module] = None,
        task_id: int = 0,
        val_loader: Optional[DataLoader] = None,
        eval_fn: Optional[Callable] = None
    ) -> TrainMetrics:
        """Full training loop."""
        for epoch in range(self.config.epochs):
            metrics = self.train_epoch(train_loader, old_model, task_id)
            
            # Validation
            if val_loader is not None and eval_fn is not None and epoch % self.config.eval_interval == 0:
                val_metrics = eval_fn(self.model, val_loader)
                metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
            
            # Epoch callbacks
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, metrics)
        
        return self.metrics


class ContinualTrainer(NGSTrainer):
    """Extended trainer for continual learning scenarios."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_models = {}  # Store old models for KD
    
    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        eval_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Train on a single task with continual learning."""
        old_model = self.task_models.get(task_id - 1) if task_id > 0 else None
        
        metrics = self.train(train_loader, old_model, task_id, val_loader, eval_fn)
        
        # Save model for future KD
        self.task_models[task_id] = deepcopy(self.model).eval()
        
        return {
            'task_id': task_id,
            'metrics': metrics,
            'final_K': self.model.K,
        }


def create_trainer(
    model: nn.Module,
    config: TrainConfig,
    device: str = 'cuda',
    continual: bool = False,
    **kwargs
) -> NGSTrainer:
    """Factory function to create trainer."""
    if continual:
        return ContinualTrainer(model, config, device, **kwargs)
    return NGSTrainer(model, config, device, **kwargs)

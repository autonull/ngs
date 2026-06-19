#!/usr/bin/env python
"""Callbacks for NGSTrainer: early stopping, checkpointing, logging."""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
import json
import time


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""
    monitor: str = "val_avg_final_accuracy"  # metric to monitor
    patience: int = 10
    min_delta: float = 1e-4
    mode: str = "max"  # "max" or "min"
    restore_best_weights: bool = True
    min_epochs: int = 5  # minimum epochs before stopping


class EarlyStopping:
    """Early stopping callback with best model restoration."""
    
    def __init__(self, config: EarlyStoppingConfig = None):
        self.config = config or EarlyStoppingConfig()
        self.best_value = -np.inf if self.config.mode == "max" else np.inf
        self.best_epoch = 0
        self.best_state = None
        self.counter = 0
        self.stopped = False
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        # Check if we have the monitored metric
        monitor_key = self.config.monitor
        if monitor_key not in metrics:
            # Try to find validation metric
            for k in metrics:
                if k.startswith("val_"):
                    monitor_key = k
                    break
        
        if monitor_key not in metrics:
            return
        
        current = metrics[monitor_key]
        improved = False
        
        if self.config.mode == "max":
            improved = current > self.best_value + self.config.min_delta
        else:
            improved = current < self.best_value - self.config.min_delta
        
        if improved:
            self.best_value = current
            self.best_epoch = epoch
            self.counter = 0
            if self.config.restore_best_weights:
                self.best_state = {
                    'model': trainer.model.state_dict(),
                    'optimizer': trainer.optimizer.state_dict(),
                    'epoch': epoch,
                }
        else:
            self.counter += 1
        
        # Check stopping condition
        if epoch >= self.config.min_epochs and self.counter >= self.config.patience:
            self.stopped = True
            print(f"Early stopping at epoch {epoch}. Best {monitor_key}: {self.best_value:.6f} at epoch {self.best_epoch}")
            
            if self.config.restore_best_weights and self.best_state:
                trainer.model.load_state_dict(self.best_state['model'])
                trainer.optimizer.load_state_dict(self.best_state['optimizer'])
    
    def should_stop(self) -> bool:
        return self.stopped


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""
    dir: str = "./checkpoints"
    save_every: int = 5  # epochs
    save_best: bool = True
    monitor: str = "val_avg_final_accuracy"
    mode: str = "max"
    max_keep: int = 3
    prefix: str = "ngs"


class CheckpointCallback:
    """Model checkpointing callback."""
    
    def __init__(self, config: CheckpointConfig = None):
        self.config = config or CheckpointConfig()
        self.checkpoint_dir = Path(self.config.dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_value = -np.inf if self.config.mode == "max" else np.inf
        self.saved_checkpoints: List[Path] = []
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        # Save periodic checkpoint
        if (epoch + 1) % self.config.save_every == 0:
            self._save_checkpoint(trainer, epoch, metrics, is_best=False)
        
        # Save best checkpoint
        if self.config.save_best:
            monitor_key = self.config.monitor
            if monitor_key not in metrics:
                for k in metrics:
                    if k.startswith("val_"):
                        monitor_key = k
                        break
            
            if monitor_key in metrics:
                current = metrics[monitor_key]
                improved = False
                if self.config.mode == "max":
                    improved = current > self.best_value
                else:
                    improved = current < self.best_value
                
                if improved:
                    self.best_value = current
                    self._save_checkpoint(trainer, epoch, metrics, is_best=True)
    
    def _save_checkpoint(self, trainer, epoch: int, metrics: Dict, is_best: bool):
        tag = "best" if is_best else f"epoch_{epoch}"
        path = self.checkpoint_dir / f"{self.config.prefix}_{tag}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'metrics': metrics,
            'config': trainer.config.__dict__ if hasattr(trainer.config, '__dict__') else {},
            'timestamp': time.time(),
        }
        
        torch.save(checkpoint, path)
        self.saved_checkpoints.append(path)
        print(f"Saved checkpoint: {path}")
        
        # Prune old checkpoints
        if len(self.saved_checkpoints) > self.config.max_keep:
            old = self.saved_checkpoints.pop(0)
            if old.exists() and "best" not in old.name:
                old.unlink()
    
    def load_best(self, trainer, device: str = 'cpu') -> Dict:
        """Load best checkpoint."""
        best_path = self.checkpoint_dir / f"{self.config.prefix}_best.pt"
        if not best_path.exists():
            return {}
        
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


class MetricLogger:
    """Logs metrics to JSONL file for streaming analysis."""
    
    def __init__(self, log_path: str = "./logs/metrics.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.step = 0
    
    def on_batch_end(self, trainer, batch_idx: int, metrics: Dict):
        self.step += 1
        if self.step % 100 == 0:  # Log every 100 steps
            record = {
                'step': self.step,
                'epoch': trainer.epoch,
                'batch': batch_idx,
                'metrics': metrics,
                'timestamp': time.time(),
            }
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(record) + '\n')
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict):
        record = {
            'step': self.step,
            'epoch': epoch,
            'metrics': metrics,
            'K': getattr(trainer.model, 'K', 0),
            'timestamp': time.time(),
        }
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')


class ProgressTracker:
    """Tracks training progress for resume capability."""
    
    def __init__(self, state_path: str = "./checkpoints/training_state.json"):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict):
        state = {
            'epoch': epoch + 1,  # next epoch to run
            'global_step': trainer.global_step,
            'metrics': metrics,
            'model_K': getattr(trainer.model, 'K', 0),
            'optimizer_state': {k: v for k, v in trainer.optimizer.state_dict().items() 
                              if not isinstance(v, torch.Tensor)},  # Don't save tensor state here
            'timestamp': time.time(),
        }
        with open(self.state_path, 'w') as f:
            json.dump(state, f, default=str)
    
    def load_state(self) -> Optional[Dict]:
        if self.state_path.exists():
            with open(self.state_path) as f:
                return json.load(f)
        return None


def create_callbacks(
    early_stopping: bool = True,
    checkpointing: bool = True,
    logging: bool = True,
    progress_tracking: bool = True,
    **kwargs
) -> List:
    """Create standard callback list."""
    callbacks = []
    
    if early_stopping:
        callbacks.append(EarlyStopping(EarlyStoppingConfig(**kwargs.get('early_stopping', {}))))
    
    if checkpointing:
        callbacks.append(CheckpointCallback(CheckpointConfig(**kwargs.get('checkpointing', {}))))
    
    if logging:
        callbacks.append(MetricLogger(kwargs.get('log_path', "./logs/metrics.jsonl")))
    
    if progress_tracking:
        callbacks.append(ProgressTracker(kwargs.get('state_path', "./checkpoints/training_state.json")))
    
    return callbacks


if __name__ == "__main__":
    # Demo
    from ngs.training.trainer import NGSTrainer, TrainerConfig
    from ngs.models.ngs import build_ngs
    from ngs.core.interfaces import NGSConfig
    
    config = NGSConfig(latent_dim=32, max_k=128, k_init=32)
    model = build_ngs(784, 10, config)
    train_config = TrainerConfig(epochs=20, lr=1e-3)
    
    callbacks = create_callbacks(
        early_stopping=True,
        checkpointing=True,
        early_stopping_config={'patience': 5},
        checkpointing_config={'save_every': 5, 'dir': './test_checkpoints'},
    )
    
    trainer = NGSTrainer(model, train_config, device='cpu', callbacks=callbacks)
    print(f"Created trainer with {len(callbacks)} callbacks")

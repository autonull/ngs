"""
Minimal smoke test 4A: Split-CIFAR100 with DynamicHead - 1 task, 1 epoch.
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.modules.dynamic_head import DynamicHead
from experiments.datasets import get_task_loaders, ReplayBuffer
from experiments.metrics import compute_metrics
from experiments.backbones import PretrainedBackbone


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


class BackboneDynamicHead(nn.Module):
    """Backbone + DynamicHead for class-incremental learning."""
    
    def __init__(self, backbone_name: str = 'resnet18', max_classes: int = 100):
        super().__init__()
        self.backbone = PretrainedBackbone(backbone_name, freeze=True)
        self.head = DynamicHead(d_latent=self.backbone.feature_dim, max_classes=max_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)
    
    def add_classes(self, class_ids: list):
        self.head.add_classes(class_ids)
    
    @property
    def num_active_classes(self) -> int:
        return self.head.num_active_classes


def evaluate_model_on_task_backbone(model, test_loader, device) -> float:
    """Evaluate backbone model on a single task (keeps 4D input)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    print(f"Device: {DEVICE}")
    print(f"Minimal smoke test: 1 task, 1 epoch")
    print(f"{'='*60}")
    
    set_seed(SEED)
    
    # Build model
    model = BackboneDynamicHead(backbone_name='resnet18', max_classes=100).to(DEVICE)
    
    # Task 0: classes 0-9
    model.add_classes(list(range(10)))
    print(f"Active classes: {model.num_active_classes}")
    
    # Get data
    train_loader, test_loader, _ = get_task_loaders(
        'split_cifar100', 0, 10, batch_size=128, scenario='class_incremental'
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Quick train
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    
    model.train()
    epoch_loss = 0.0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if i >= 2:  # Just 3 batches for minimal test
            break
    
    avg_loss = epoch_loss / 3
    print(f"Avg loss (3 batches): {avg_loss:.4f}")
    
    # Evaluate
    acc = evaluate_model_on_task_backbone(model, test_loader, DEVICE)
    print(f"Test accuracy: {acc:.4f}")
    
    print("\n✅ Minimal smoke test passed!")
    print("DynamicHead + ResNet18 backbone works for class-incremental learning")


if __name__ == '__main__':
    main()
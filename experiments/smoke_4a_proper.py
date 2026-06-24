"""
Proper smoke test 4A: Split-CIFAR100 with DynamicHead - 3 tasks, 5 epochs each.
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

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


def evaluate_model(model, test_loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train_task(model, train_loader, old_model, epochs, lr, weight_decay, 
               replay_buffer, replay_ratio, kd_weight, kd_temperature):
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Replay
            if replay_buffer is not None and len(replay_buffer) > x.size(0):
                rx, ry = replay_buffer.sample(int(x.size(0) * replay_ratio))
                if rx is not None:
                    rx, ry = rx.to(DEVICE), ry.to(DEVICE)
                    x = torch.cat([x, rx], dim=0)
                    y = torch.cat([y, ry.argmax(dim=1)], dim=0)
            
            optimizer.zero_grad()
            logits = model(x)
            
            ce_loss = F.cross_entropy(logits, y)
            
            kd_loss = torch.tensor(0.0, device=DEVICE)
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
            
            total_loss = ce_loss + kd_weight * kd_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += ce_loss.item()
            n_batches += 1
        
        scheduler.step()
    
    return epoch_loss / max(n_batches, 1)


def main():
    print(f"Device: {DEVICE}")
    print(f"Proper smoke test: 3 tasks, 5 epochs each")
    print(f"{'='*60}")
    
    set_seed(SEED)
    
    n_tasks = 3
    classes_per_task = 10
    max_classes = 100
    
    model = BackboneDynamicHead(backbone_name='resnet18', max_classes=max_classes).to(DEVICE)
    
    print(f"Model: ResNet18 backbone (frozen) + DynamicHead(d_latent={model.backbone.feature_dim}, max_classes={max_classes})")
    
    replay_buffer = ReplayBuffer(max_size=5000, seed=SEED)
    old_model = None
    
    accuracy_matrix = np.full((n_tasks, n_tasks), np.nan)
    active_classes_per_task = []
    
    epochs_per_task = 5
    lr = 1e-3
    weight_decay = 1e-4
    replay_ratio = 1.0
    kd_weight = 2.0
    kd_temperature = 2.0
    
    for task_id in range(n_tasks):
        print(f"\n--- Task {task_id + 1}/{n_tasks} ---")
        
        new_classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))
        model.add_classes(new_classes)
        active_classes_per_task.append(model.num_active_classes)
        print(f"  Active classes: {active_classes_per_task[-1]} ({new_classes})")
        
        train_loader, test_loader, _ = get_task_loaders(
            'split_cifar100', task_id, classes_per_task, batch_size=128, scenario='class_incremental'
        )
        
        start = time.time()
        loss = train_task(
            model, train_loader, old_model,
            epochs=epochs_per_task, lr=lr, weight_decay=weight_decay,
            replay_buffer=replay_buffer, replay_ratio=replay_ratio,
            kd_weight=kd_weight, kd_temperature=kd_temperature,
        )
        print(f"  Train loss: {loss:.4f} (time: {time.time()-start:.1f}s)")
        
        for eval_task in range(task_id + 1):
            _, eval_test_loader, _ = get_task_loaders(
                'split_cifar100', eval_task, classes_per_task, batch_size=128, scenario='class_incremental'
            )
            acc = evaluate_model(model, eval_test_loader, DEVICE)
            accuracy_matrix[eval_task, task_id] = acc
            print(f"  Task {eval_task} acc: {acc:.4f}")
        
        for x, y in train_loader:
            replay_buffer.add(x, F.one_hot(y, max_classes).float())
        
        if kd_weight > 0:
            old_model = BackboneDynamicHead(backbone_name='resnet18', max_classes=max_classes).to(DEVICE)
            old_model.head.load_state_dict(model.head.state_dict())
            old_model.eval()
            for p in old_model.parameters():
                p.requires_grad = False
            old_model.add_classes(list(range((task_id + 1) * classes_per_task)))
    
    random_baseline = 1.0 / max_classes
    metrics = compute_metrics(accuracy_matrix, random_baseline)
    
    print(f"\n{'='*60}")
    print("SMOKE TEST 4A RESULTS")
    print(f"{'='*60}")
    
    from experiments.metrics import print_results
    print_results(metrics, "Split-CIFAR100 + DynamicHead (proper)")
    print(f"\nActive classes trajectory: {active_classes_per_task}")
    print(f"Final active classes: {active_classes_per_task[-1]}")
    
    forgetting = metrics.avg_forgetting
    avg_acc = metrics.avg_final_accuracy
    
    print(f"\n{'='*60}")
    print("DECISION GATE CHECK")
    print(f"{'='*60}")
    print(f"Forgetting: {forgetting:.4f} (target: <0.05)")
    print(f"Avg Final Acc: {avg_acc:.4f} (target: >0.85)")
    
    if forgetting < 0.05 and avg_acc > 0.85:
        print("✅ PASS: Continue with full 4A run")
    else:
        print("🔄 PIVOT: Try replay-free head (LoRA per class) or more epochs")
    
    os.makedirs('results/smoke_4a', exist_ok=True)
    results = {
        'accuracy_matrix': accuracy_matrix.tolist(),
        'active_classes': active_classes_per_task,
        'metrics': metrics.to_dict(),
        'config': {
            'n_tasks': n_tasks,
            'classes_per_task': classes_per_task,
            'epochs_per_task': epochs_per_task,
            'batch_size': 128,
            'lr': lr,
            'weight_decay': weight_decay,
            'replay_size': 5000,
            'replay_ratio': replay_ratio,
            'kd_weight': kd_weight,
            'kd_temperature': kd_temperature,
            'backbone': 'resnet18',
            'd_latent': model.backbone.feature_dim,
            'max_classes': max_classes,
            'device': DEVICE,
            'seed': SEED,
        }
    }
    
    with open('results/smoke_4a/proper_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to results/smoke_4a/proper_results.json")


if __name__ == '__main__':
    main()
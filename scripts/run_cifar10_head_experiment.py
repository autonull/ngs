#!/usr/bin/env python3
"""
CIFAR-10 Experiment: NGS vs Dense Linear head on a ConvNet4 backbone.

This script:
1. Defines a simple ConvNet4 backbone
2. Trains with a dense linear head of capacity ~NGS
3. Trains with an NGS head (replacing the final dense layer)
4. Compares test accuracy after 10 epochs
5. Saves results to results/cifar_ngs_head.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ngs.core.interfaces import NGSConfig
from ngs.models import build_ngs


# ───────────────────────── ConvNet4 Backbone ─────────────────────────

class ConvNet4(nn.Module):
    """Simple 4-layer ConvNet for CIFAR-10."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 4x4

            nn.Conv2d(256, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> [B, out_dim, 1, 1]
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # [B, out_dim]


# ───────────────────────── Heads ─────────────────────────

class DenseHead(nn.Module):
    """Dense linear classification head (equivalent capacity to NGS head)."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ConvNetDense(nn.Module):
    """ConvNet4 + Dense Head."""

    def __init__(self, num_classes: int = 10, backbone_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.backbone = ConvNet4(out_dim=backbone_dim)
        self.head = DenseHead(backbone_dim, num_classes, hidden_dim)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


class ConvNetNGS(nn.Module):
    """ConvNet4 + NGS Head."""

    def __init__(self, num_classes: int = 10, backbone_dim: int = 128, ngs_config: NGSConfig = None):
        super().__init__()
        self.backbone = ConvNet4(out_dim=backbone_dim)
        if ngs_config is None:
            ngs_config = NGSConfig(
                latent_dim=128,
                k_init=32,
                max_k=128,
                top_k=8,
                tau=1.0,
                routing="monolithic_mahalanobis",
                parameter_storage="direct_adapter",
                topology_control="discrete_heuristic",
                memory_management="pre_allocated",
                split_threshold=0.05,
                prune_threshold=0.01,
            )
        self.ngs_head = build_ngs(backbone_dim, num_classes, ngs_config)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.ngs_head(feat)
        return out.logits


# ───────────────────────── Training & Evaluation ─────────────────────────

def get_cifar10_loaders(batch_size=128, num_workers=4, data_dir='./data'):
    """Get CIFAR-10 train and test loaders with standard augmentation."""

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainloader, testloader


def evaluate(model, dataloader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item() * labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss


def train_model(model, trainloader, testloader, device, epochs=10, lr=1e-3, weight_decay=1e-4):
    """Train a model and return list of (epoch, test_acc, test_loss)."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        test_acc, test_loss = evaluate(model, testloader, device)
        history.append({'epoch': epoch + 1, 'test_acc': test_acc, 'test_loss': test_loss})
        print(f"Epoch {epoch+1}/{epochs} - Test Acc: {test_acc:.2f}%, Test Loss: {test_loss:.4f}")

    return history


# ───────────────────────── Main ─────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading CIFAR-10 data...")
    trainloader, testloader = get_cifar10_loaders(batch_size=128, num_workers=4)

    epochs = 10
    results = {}

    # Train Dense Head
    print("\n" + "=" * 60)
    print("Training Dense Head (ConvNet4 + Dense)")
    print("=" * 60)
    dense_model = ConvNetDense(num_classes=10, backbone_dim=128, hidden_dim=256)
    start = time.time()
    dense_history = train_model(dense_model, trainloader, testloader, device, epochs=epochs)
    dense_time = time.time() - start
    dense_final_acc = dense_history[-1]['test_acc']
    print(f"Dense Head Final Test Accuracy: {dense_final_acc:.2f}% (time: {dense_time:.1f}s)")
    results['dense'] = {
        'final_test_acc': dense_final_acc,
        'history': dense_history,
        'training_time_s': dense_time,
    }

    # Train NGS Head
    print("\n" + "=" * 60)
    print("Training NGS Head (ConvNet4 + NGS)")
    print("=" * 60)
    ngs_config = NGSConfig(
        latent_dim=128,
        k_init=32,
        max_k=128,
        top_k=8,
        tau=1.0,
        routing="monolithic_mahalanobis",
        parameter_storage="direct_adapter",
        topology_control="discrete_heuristic",
        memory_management="pre_allocated",
        split_threshold=0.05,
        prune_threshold=0.01,
    )
    ngs_model = ConvNetNGS(num_classes=10, backbone_dim=128, ngs_config=ngs_config)
    start = time.time()
    ngs_history = train_model(ngs_model, trainloader, testloader, device, epochs=epochs)
    ngs_time = time.time() - start
    ngs_final_acc = ngs_history[-1]['test_acc']
    print(f"NGS Head Final Test Accuracy: {ngs_final_acc:.2f}% (time: {ngs_time:.1f}s)")
    results['ngs'] = {
        'final_test_acc': ngs_final_acc,
        'history': ngs_history,
        'training_time_s': ngs_time,
    }

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Dense Head Accuracy: {dense_final_acc:.2f}%")
    print(f"NGS Head Accuracy:   {ngs_final_acc:.2f}%")
    print(f"Accuracy Gap (NGS - Dense): {ngs_final_acc - dense_final_acc:.2f}%")

    results['summary'] = {
        'dense_acc': dense_final_acc,
        'ngs_acc': ngs_final_acc,
        'gap': ngs_final_acc - dense_final_acc,
    }

    # Save results
    output_path = Path('results/cifar_ngs_head.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()

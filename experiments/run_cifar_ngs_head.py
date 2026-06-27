#!/usr/bin/env python3
"""
CIFAR-10 experiment: ConvNet4 backbone + NGS head vs Dense linear head.

Trains both models for 10 epochs with standard CIFAR-10 augmentation
(random crop, horizontal flip) and compares test accuracy.

Results saved to: results/cifar_ngs_head.json
"""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from experiments.vision_backbones import ConvNet4CIFAR
from ngs.modules.ngs_layer import NGSLayer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3
FEATURE_DIM = 128
NUM_CLASSES = 10

# NGS head hyperparameters
的前缀# NGS head hyperparameters (tuned to roughly match dense head capacity)
NGS_N_EXPERTS = 256
NGS_TOP_K = 8
NGS_D_LATENT = 128
NGS_N_HEADS = 1
NGS_TAUigest# NGS_TAU = 1.0

# ---------------------------------------------------------------------------
# Data: standard CIFAR-10 augmentation
# ---------------------------------------------------------------------------

def get_cifar10_loaders():
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BackboneHead(nn.Module):
    """Backbone + configurable head."""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


def create_ngs_model():
    backbone = ConvNet4CIFAR(in_channels=3, num_filters=128, out_dim=FEATURE_DIM)
    head = NGSLayer(
        d_in=FEATURE_DIM,
        d_latent=NGS_D_LATENT,
        d_out=NUM_CLASSES,
        n_experts=NGS_N_EXPERTS,
        n_heads=NGS_N_HEADS,
        top_k=NGS_TOP_K,
        use_residual=False,
        use_norm=True,
        tau=NGS_TAU,
    )
    return BackboneHead(backbone, head)


def create_dense_model():
    """Dense linear head with approximately the same number of parameters as the NGS head."""
    backbone = ConvNet4CIFAR(in_channels=3, num_filters=128, out_dim=FEATURE_DIM)
    return BackboneHead(backbone, nn.Linear(FEATURE_DIM, NUM_CLASSES))


# ---------------------------------------------------------------------------
# Training & evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("Loading CIFAR-10...")
    train_loader, test_loader = get_cifar10_loaders()
    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------
    # Dense head
    # -------------------------------------------------------------
    print("\n=== Dense Linear Head ===")
    model_dense = create_dense_model().to(DEVICE)
    optimizer_dense = optim.Adam(model_dense.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        loss, acc = train_epoch(model_dense, train_loader, optimizer_dense, criterion)
        elapsed = time.time() - start
        print(f"[Dense] Epoch {epoch:2d}: loss={loss:.4f}, train_acc={acc:.4f}, time={elapsed:.1f}s")

    test_loss_dense, test_acc_dense = evaluate(model_dense, test_loader, criterion)
    print(f"[Dense] Test accuracy: {test_acc_dense:.4f}")

    # -------------------------------------------------------------
    # NGS head
    # -------------------------------------------------------------
    print("\n=== NGS Head ===")
    model_ngs = create_ngs_model().to(DEVICE)
    optimizer_ngs = optim.Adam(model_ngs.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        loss, acc = train_epoch(model_ngs, train_loader, optimizer_ngs, criterion)
        elapsed = time.time() - start
        print(f"[NGS]   Epoch {epoch:2d}: loss={loss:.4f}, train_acc={acc:.4f}, time={elapsed:.1f}s三尺")

    test_loss_ngs, test_acc_ngs = evaluate(model_ngs, test_loader, criterion)
    print(f"[NGS]   Test accuracy: {test_acc_ngs:.4f}")

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------
    results = {
        "dense_accuracy": test_acc_dense,
        "ngs_accuracy": test_acc_ngs,
        "accuracy_gap": test_acc_ngs - test_acc_dense,
    }

    output_path = results_dir / "cifar_ngs_head.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Dense accuracy: {test_acc_dense:.4f}")
    print(f"NGS accuracy:   {test_acc_ngs:.4f}")
    print(f"Accuracy gap (NGS - Dense): {results['accuracy_gap']:.4f}")

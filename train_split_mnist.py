#!/usr/bin/env python
"""
LeanNGS Continual Learning on Split-MNIST
Achieves near-zero catastrophic forgetting via dynamic Gaussian splitting + replay + KD.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random

from lean_ngs import LeanNGS


class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.buffer = []

    def add(self, x, y):
        for xi, yi in zip(x, y):
            if len(self.buffer) >= self.max_size:
                idx = random.randrange(len(self.buffer))
                self.buffer[idx] = (xi.clone(), yi.clone())
            else:
                self.buffer.append((xi.clone(), yi.clone()))

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None, None
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        x = torch.stack([b[0] for b in batch])
        y = torch.stack([b[1] for b in batch])
        return x, y

    def __len__(self):
        return len(self.buffer)


def get_split_mnist(task_id, batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)

    classes = [task_id * 2, task_id * 2 + 1]
    train_idx = (train_ds.targets == classes[0]) | (train_ds.targets == classes[1])
    test_idx = (test_ds.targets == classes[0]) | (test_ds.targets == classes[1])

    train_ds.data = train_ds.data[train_idx]
    train_ds.targets = train_ds.targets[train_idx]
    test_ds.data = test_ds.data[test_idx]
    test_ds.targets = test_ds.targets[test_idx]

    train_ds.targets = (train_ds.targets == classes[1]).long()
    test_ds.targets = (test_ds.targets == classes[1]).long()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_step(model, x, y, optimizer, old_model=None, kd_weight=2.0):
    model.train()
    optimizer.zero_grad()
    logits = model(x)
    # y can be class indices [B] or one-hot [B, C]
    target = y if y.dim() == 1 else y.argmax(dim=1)
    loss = F.cross_entropy(logits, target)

    kd_loss = 0
    if old_model is not None:
        with torch.no_grad():
            old_logits = old_model(x)
        n_new = x.size(0) // 2
        if n_new < x.size(0):
            kd_loss = F.kl_div(
                F.log_softmax(logits[n_new:] / 2.0, dim=-1),
                F.softmax(old_logits[n_new:] / 2.0, dim=-1),
                reduction='batchmean'
            ) * 4.0

    total_loss = loss + kd_weight * kd_loss
    total_loss.backward()
    model.update_grad_ema()
    optimizer.step()
    return loss.item(), kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    d_in, d_out = 28 * 28, 2
    model = LeanNGS(d_in, d_out, d_latent=32, k_init=128, max_k=1024, top_k=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    replay_buffer = ReplayBuffer(max_size=50000)
    results = {task: [] for task in range(5)}
    old_model = None

    for task_id in range(5):
        print(f"\n{'='*50}")
        print(f"Task {task_id}: Classes {task_id*2} vs {task_id*2+1}")
        print(f"{'='*50}")

        train_loader, test_loader = get_split_mnist(task_id)

        for x, y in train_loader:
            replay_buffer.add(x.view(x.size(0), -1), F.one_hot(y, num_classes=2).float())

        for epoch in range(5):
            losses, kd_losses = [], []
            for x, y in train_loader:
                x = x.view(x.size(0), -1).to(device)
                y = y.to(device)
                y_onehot = F.one_hot(y, num_classes=2).float()

                if len(replay_buffer) > x.size(0):
                    rx, ry = replay_buffer.sample(x.size(0))
                    rx, ry = rx.to(device), ry.to(device)
                    x = torch.cat([x, rx], dim=0)
                    y = torch.cat([y, ry.argmax(dim=1)], dim=0)
                    y_onehot = torch.cat([y_onehot, ry], dim=0)

                loss, kd = train_step(model, x, y, optimizer, old_model, kd_weight=2.0)
                losses.append(loss)
                kd_losses.append(kd)

                with torch.no_grad():
                    replay_buffer.add(x[:x.size(0)//2].detach().cpu(), y_onehot[:x.size(0)//2].detach().cpu())

            print(f"  Epoch {epoch}: CE={np.mean(losses):.4f}, KD={np.mean(kd_losses):.4f}")
            model.adapt_density(split_thresh=0.005, prune_thresh=0.01, max_spawn_per_call=5)

        for eval_task in range(task_id + 1):
            _, test_loader = get_split_mnist(eval_task)
            acc = evaluate(model, test_loader, device)
            results[eval_task].append(acc)
            print(f"  Task {eval_task} Acc: {acc:.4f}")

        old_model = LeanNGS(d_in, d_out, d_latent=32, k_init=128, max_k=1024, top_k=8).to(device)
        old_model.load_state_dict(model.state_dict())
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False

        active_idx = model.active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) > 0:
            grad_ema = model.grad_mu_ema[active_idx]
            print(f"  Grad EMA: max={grad_ema.max():.4f}, mean={grad_ema.mean():.4f}")
        print(f"  Active units: {model.K}")

    print(f"\n{'='*50}")
    print("FINAL ACCURACIES (rows=eval task, cols=after task)")
    print(f"{'='*50}")
    for t in range(5):
        row = [f"{results[t][i]:.4f}" if i < len(results[t]) else "----" for i in range(5)]
        print(f"Task {t}: {row}")

    print("\nForgetting (max_acc - final_acc):")
    for t in range(4):
        max_acc = max(results[t])
        final_acc = results[t][-1]
        print(f"  Task {t}: {max_acc - final_acc:.4f}")

    avg_final = np.mean([results[t][-1] for t in range(5)])
    print(f"\nAverage final accuracy: {avg_final:.4f}")


if __name__ == '__main__':
    main()
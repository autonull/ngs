#!/usr/bin/env python
"""Quick NGS vs Dense baseline - run only NGS (K=32, top_k=8) vs Dense MLP (256x2)."""
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
sys.path.insert(0, '/home/me/ngs')
from ngs.models.ngs import NGSModel
from ngs.core.interfaces import NGSConfig, RoutingStrategy

device = 'cuda'
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
test = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)
train_loader = DataLoader(train, batch_size=128, shuffle=True)
test_loader = DataLoader(test, batch_size=128, shuffle=False)

def train_andEval(name, model, epochs=5):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out if isinstance(out, torch.Tensor) else out.logits, y)
            loss.backward()
            opt.step()
    # eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            logits = out if isinstance(out, torch.Tensor) else out.logits
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / total
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}: {acc:.4f} ({params:,} params)")
    return acc, params

# NGS
config = NGSConfig(latent_dim=64, max_k=32, top_k=8, k_init=8, routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS)
ngs_acc, ngs_params = train_andEval("NGS (K=32, top_k=8)", NGSModel(784, 10, config))

# Dense MLP
class DenseMLP(torch.nn.Module):
    def __init__(self, d_in, d_out, h):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d_out))
    def forward(self, x): return self.net(x)

dense_acc, dense_params = train_andEval("Dense (256x2)", DenseMLP(784, 10, 256))

print(f"\nNGS vs Dense gap: {ngs_acc - dense_acc:+.4f}")

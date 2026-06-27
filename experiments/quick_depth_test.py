import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
sys.path.insert(0, '/home/me/ngs')
from ngs.models.ngs import NGSModel, MultiLayerNGS
from ngs.core.interfaces import NGSConfig, RoutingStrategy

device = 'cuda'
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
test = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)
train_loader = DataLoader(train, batch_size=128, shuffle=True)
test_loader = DataLoader(test, batch_size=128, shuffle=False)

def train_and_eval(name, model, epochs=5):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out.logits if hasattr(out, 'logits') else out, y)
            loss.backward()
            opt.step()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f'{name}: {acc:.4f}')
    return acc

config = NGSConfig(latent_dim=64, max_k=32, top_k=8, k_init=8, routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS)
for depth in [1, 2, 4, 8]:
    configs = [config] * depth
    model = MultiLayerNGS(784, 10, depth, configs)
    train_and_eval(f'MultiLayerNGS depth={depth}', model)

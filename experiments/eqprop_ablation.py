"""
Ablation: Compare spectral constraint modes for EqNGS.
Modes: (a) no SN, (b) SN post-update
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.core.interfaces import NGSConfig
from ngs.modules.eqprop import create_eqngs

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)

print(f"Device: {DEVICE}")

# Load MNIST
print("Loading MNIST...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

# Base config
base_cfg = NGSConfig(
    latent_dim=64,
    k_init=32,
    max_k=256,
    top_k=8,
    routing='monolithic_mahalanobis',
    parameter_storage='direct_adapter',
    topology_control='discrete_heuristic',
    memory_management='dynamic',
    ema_decay=0.99,
)

# Only test the key modes: no SN and post_update (the others are too slow)
ablation_configs = [
    {
        'name': 'no_sn',
        'spectral_mode': 'none',
        'spectral_gamma': 0.95,
    },
    {
        'name': 'sn_post_update',
        'spectral_mode': 'post_update',
        'spectral_gamma': 0.95,
    },
]

EPOCHS = 2

results = {}

for ablation in ablation_configs:
    name = ablation['name']
    print(f"\n{'='*60}")
    print(f"Running ablation: {name}")
    print(f"{'='*60}")
    
    torch.manual_seed(SEED)
    
    model = create_eqngs(
        d_in=784,
        d_out=10,
        config=base_cfg,
        ep_beta=0.5,
        ep_settle_steps=10,
        ep_settle_lr=0.2,
        ep_momentum=0.9,
        spectral_gamma=ablation.get('spectral_gamma', 0.95),
        spectral_mode=ablation['spectral_mode'],
    ).to(DEVICE)
    
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    epoch_results = []
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_batches = 0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [{name}]", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.size(0), -1)
            
            result = model.ep_step(x, y)
            train_loss += result['loss']
            train_acc += result['accuracy']
            train_batches += 1
        
        train_loss /= train_batches
        train_acc /= train_batches
        
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = x.view(x.size(0), -1)
                
                out = model(x)
                logits = out.logits if hasattr(out, 'logits') else out
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                test_loss += F.cross_entropy(logits, y).item()
        
        test_acc = correct / total
        test_loss /= len(test_loader)
        
        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
        })
        
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
    
    results[name] = epoch_results

# Summary
print(f"\n{'='*60}")
print("ABLATION SUMMARY")
print(f"{'='*60}")
print(f"{'Mode':<25} {'Epoch 1':>10} {'Epoch 2':>10}")
print("-" * 45)
for name, epoch_results in results.items():
    accs = [f"{r['test_acc']:.4f}" for r in epoch_results]
    print(f"{name:<25} {accs[0]:>10} {accs[1]:>10}")

import json
with open('eqprop_ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to eqprop_ablation_results.json")
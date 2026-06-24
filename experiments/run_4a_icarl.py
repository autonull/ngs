"""
4A iCaRL-style: Frozen ResNet18 + train ALL active heads on combined data each task.
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.datasets import get_task_loaders, ReplayBuffer
from experiments.backbones import PretrainedBackbone
from experiments.metrics import compute_metrics

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

class LinearHead(nn.Module):
    def __init__(self, d_in=512, max_classes=100):
        super().__init__()
        self.head = nn.Linear(d_in, max_classes, bias=False)
        self.register_buffer('active', torch.zeros(max_classes, dtype=torch.bool))
    
    def add_classes(self, class_ids):
        self.active[class_ids] = True
    
    def forward(self, x):
        logits = self.head(x)
        logits[:, ~self.active] = -1e9
        return logits

class ResNetModel(nn.Module):
    def __init__(self, max_classes=100):
        super().__init__()
        self.backbone = PretrainedBackbone('resnet18', freeze=True).to(DEVICE)
        self.head = LinearHead(d_in=512, max_classes=max_classes).to(DEVICE)
    
    def forward(self, x):
        with torch.no_grad():
            f = self.backbone(x)
        return self.head(f)
    
    def add_classes(self, c): self.head.add_classes(c)

def eval_model(m, loader):
    m.eval(); corr=tot=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            p=m(x).argmax(1); corr+=(p==y).sum().item(); tot+=y.size(0)
    return corr/tot

def train_task(m, loader, replay_buffer, epochs, lr):
    # Train ALL active heads
    opt = torch.optim.AdamW(m.head.parameters(), lr=lr)
    m.train()
    for _ in range(epochs):
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            
            # Replay
            if replay_buffer and len(replay_buffer) > x.size(0):
                rx,ry=replay_buffer.sample(x.size(0))
                if rx is not None:
                    x=torch.cat([x,rx.to(DEVICE)],0)
                    y=torch.cat([y,ry.argmax(1).to(DEVICE)],0)
            
            opt.zero_grad()
            logits = m(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

model = ResNetModel(max_classes=100)
rb = ReplayBuffer(10000, SEED)
acc = np.full((10,10), np.nan)

print("=" * 60)
print("4A iCaRL: Frozen ResNet18 + train ALL heads on combined data")
print("=" * 60)

for t in range(10):
    model.add_classes(list(range(t*10, (t+1)*10)))
    tr, te, _ = get_task_loaders('split_cifar100', t, 10, 128, 'class_incremental')
    start = time.time()
    train_task(model, tr, rb, epochs=20, lr=1e-3)
    for et in range(t+1):
        _, ete, _ = get_task_loaders('split_cifar100', et, 10, 128, 'class_incremental')
        acc[et, t] = eval_model(model, ete)
    for x,y in tr: rb.add(x, F.one_hot(y, 100).float())
    print(f"Task {t+1}/10: {[f'{acc[i,t]:.3f}' for i in range(t+1)]}, time={time.time()-start:.0f}s")

m = compute_metrics(acc, 1/100)
print(f"\nForgetting: {m.avg_forgetting:.3f} (target<0.05)")
print(f"Avg Acc:    {m.avg_final_accuracy:.3f} (target>0.50)")

os.makedirs('results/full', exist_ok=True)
json.dump({'acc':acc.tolist(),'forgetting':m.avg_forgetting,'avg_acc':m.avg_final_accuracy},
          open('results/full/4a_icarl.json','w'))
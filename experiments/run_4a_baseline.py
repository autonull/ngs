"""
4A Baseline: Frozen ResNet18 + separate linear per class + replay.
Standard iCaRL-style approach to establish upper bound.
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.datasets import get_task_loaders, ReplayBuffer
from experiments.backbones import PretrainedBackbone
from experiments.metrics import compute_metrics
from copy import deepcopy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# Separate linear head per class
class SeparateLinearHead(nn.Module):
    def __init__(self, d_in=512, max_classes=100):
        super().__init__()
        self.d_in = d_in
        self.max_classes = max_classes
        self.heads = nn.ModuleList([nn.Linear(d_in, 1, bias=False) for _ in range(max_classes)])
        self.register_buffer('active', torch.zeros(max_classes, dtype=torch.bool))
    
    def add_classes(self, class_ids):
        for c in class_ids:
            self.active[c] = True
    
    def forward(self, x):
        logits = torch.full((x.size(0), self.max_classes), -1e9, device=x.device, dtype=x.dtype)
        for c in range(self.max_classes):
            if self.active[c]:
                logits[:, c:c+1] = self.heads[c](x)
        return logits
    
    def get_current_params(self, task_id, classes_per_task):
        params = []
        current_classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))
        for c in current_classes:
            params.extend(list(self.heads[c].parameters()))
        return params
    
    def get_all_active_params(self):
        params = []
        for c in range(self.max_classes):
            if self.active[c]:
                params.extend(list(self.heads[c].parameters()))
        return params

class ResNetSeparateModel(nn.Module):
    def __init__(self, max_classes=100):
        super().__init__()
        self.backbone = PretrainedBackbone('resnet18', freeze=True).to(DEVICE)
        self.head = SeparateLinearHead(d_in=512, max_classes=max_classes).to(DEVICE)
    
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

def train_task(m, loader, replay_buffer, task_id, classes_per_task, epochs, lr):
    current_classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))
    params = m.head.get_current_params(task_id, classes_per_task)
    opt = torch.optim.AdamW(params, lr=lr)
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
            
            # New data: task-masked CE
            n_new = x.size(0) // 2 if replay_buffer and len(replay_buffer) > 0 else x.size(0)
            if n_new > 0:
                task_logits = logits[:n_new, current_classes]
                y_shifted = y[:n_new] - task_id * classes_per_task
                ce_loss = F.cross_entropy(task_logits, y_shifted)
            else:
                ce_loss = torch.tensor(0., device=DEVICE)
            
            # Replay: full CE
            if n_new < x.size(0):
                replay_logits = logits[n_new:]
                replay_y = y[n_new:]
                ce_loss = ce_loss + F.cross_entropy(replay_logits, replay_y)
            
            ce_loss.backward()
            opt.step()

model = ResNetSeparateModel(max_classes=100)
rb = ReplayBuffer(10000, SEED)
acc = np.full((10,10), np.nan)

print("=" * 60)
print("4A Baseline: Frozen ResNet18 + separate linear + replay")
print("=" * 60)

for t in range(10):
    model.add_classes(list(range(t*10, (t+1)*10)))
    tr, te, _ = get_task_loaders('split_cifar100', t, 10, 128, 'class_incremental')
    start = time.time()
    train_task(model, tr, rb, t, 10, epochs=20, lr=1e-3)
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
          open('results/full/4a_baseline.json','w'))
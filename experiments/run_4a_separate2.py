"""
4A: Separate linear per class + task-masked loss.
Only computes CE on current task's classes to avoid softmax coupling.
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models.ngs import build_ngs
from experiments.datasets import get_task_loaders
from experiments.backbones import PretrainedBackbone
from experiments.metrics import compute_metrics

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# NGS as frozen feature extractor
feat_cfg = NGSConfig(latent_dim=128, k_init=32, max_k=64, top_k=4,
    routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    parameter_storage=ParameterStorage.DIRECT_ADAPTER,
    topology_control=TopologyControl.DISCRETE_HEURISTIC,
    memory_management=MemoryManagement.DYNAMIC)

feat_extractor = build_ngs(512, 128, feat_cfg).to(DEVICE)
for p in feat_extractor.parameters(): p.requires_grad = False
feat_extractor.eval()

# Separate linear head per class
class SeparateLinearHead(nn.Module):
    def __init__(self, d_in=128, max_classes=100):
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
    
    def get_active_params(self):
        params = []
        for c in range(self.max_classes):
            if self.active[c]:
                params.extend(list(self.heads[c].parameters()))
        return params

class NGSSeparateModel(nn.Module):
    def __init__(self, max_classes=100):
        super().__init__()
        self.backbone = PretrainedBackbone('resnet18', freeze=True).to(DEVICE)
        self.feat = feat_extractor
        self.head = SeparateLinearHead(d_in=128, max_classes=max_classes).to(DEVICE)
    
    def forward(self, x):
        with torch.no_grad():
            f = self.backbone(x)
            f = self.feat(f)
            z = f.latent if hasattr(f, 'latent') else f.logits
        return self.head(z)
    
    def add_classes(self, c): self.head.add_classes(c)

def eval_model(m, loader):
    m.eval(); corr=tot=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            p=m(x).argmax(1); corr+=(p==y).sum().item(); tot+=y.size(0)
    return corr/tot

def train_task(m, loader, task_id, classes_per_task, epochs, lr):
    # Only train current task's heads
    current_classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))
    params = []
    for c in current_classes:
        params.extend(list(m.head.heads[c].parameters()))
    opt = torch.optim.AdamW(params, lr=lr)
    m.train()
    for _ in range(epochs):
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad()
            logits = m(x)
            # Masked CE: only compute loss on current task's classes
            # Shift labels to 0..9 for current task
            y_shifted = y - task_id * classes_per_task
            task_logits = logits[:, current_classes]
            loss = F.cross_entropy(task_logits, y_shifted)
            loss.backward()
            opt.step()

model = NGSSeparateModel(max_classes=100)
acc = np.full((10,10), np.nan)

print("=" * 60)
print("4A: Separate linear + task-masked CE loss")
print("=" * 60)

for t in range(10):
    new_classes = list(range(t*10, (t+1)*10))
    model.add_classes(new_classes)
    tr, te, _ = get_task_loaders('split_cifar100', t, 10, 128, 'class_incremental')
    start = time.time()
    train_task(model, tr, t, 10, epochs=15, lr=1e-3)
    for et in range(t+1):
        _, ete, _ = get_task_loaders('split_cifar100', et, 10, 128, 'class_incremental')
        acc[et, t] = eval_model(model, ete)
    print(f"Task {t+1}/10: {[f'{acc[i,t]:.3f}' for i in range(t+1)]}, time={time.time()-start:.0f}s")

m = compute_metrics(acc, 1/100)
print(f"\nForgetting: {m.avg_forgetting:.3f} (target<0.05)")
print(f"Avg Acc:    {m.avg_final_accuracy:.3f} (target>0.50)")
print("✅ PASS" if m.avg_forgetting < 0.05 else "❌ NEEDS WORK")

os.makedirs('results/full', exist_ok=True)
json.dump({'acc':acc.tolist(),'forgetting':m.avg_forgetting,'avg_acc':m.avg_final_accuracy},
          open('results/full/4a_separate2.json','w'))
"""
4A: Separate linear + fine-tune NGS features (small LR) + replay.
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models.ngs import build_ngs
from experiments.datasets import get_task_loaders, ReplayBuffer
from experiments.backbones import PretrainedBackbone
from experiments.metrics import compute_metrics
from copy import deepcopy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# NGS feature extractor (trainable with small LR)
feat_cfg = NGSConfig(latent_dim=128, k_init=32, max_k=64, top_k=4,
    routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    parameter_storage=ParameterStorage.DIRECT_ADAPTER,
    topology_control=TopologyControl.DISCRETE_HEURISTIC,
    memory_management=MemoryManagement.DYNAMIC)

feat_extractor = build_ngs(512, 128, feat_cfg).to(DEVICE)

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
    
    def get_current_params(self, task_id, classes_per_task):
        params = []
        current_classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))
        for c in current_classes:
            params.extend(list(self.heads[c].parameters()))
        return params

class NGSModel(nn.Module):
    def __init__(self, max_classes=100):
        super().__init__()
        self.backbone = PretrainedBackbone('resnet18', freeze=True).to(DEVICE)
        self.feat = feat_extractor
        self.head = SeparateLinearHead(d_in=128, max_classes=max_classes).to(DEVICE)
    
    def forward(self, x):
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

def train_task(m, loader, replay_buffer, old_model, task_id, classes_per_task, epochs, lr_feat, lr_head, kd_w, kd_t, replay_ratio):
    current_classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))
    
    # Optimizer: small LR for features, larger for heads
    feat_params = [p for p in m.feat.parameters() if p.requires_grad]
    head_params = m.head.get_current_params(task_id, classes_per_task)
    
    opt = torch.optim.AdamW([
        {'params': feat_params, 'lr': lr_feat},
        {'params': head_params, 'lr': lr_head},
    ])
    
    m.train()
    for _ in range(epochs):
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            
            # Replay
            if replay_buffer and len(replay_buffer) > x.size(0):
                rx,ry=replay_buffer.sample(int(x.size(0)*replay_ratio))
                if rx is not None:
                    x=torch.cat([x,rx.to(DEVICE)],0)
                    y=torch.cat([y,ry.argmax(1).to(DEVICE)],0)
            
            opt.zero_grad()
            logits = m(x)
            
            # Split new and replay
            n_new = x.size(0) // (1 + int(replay_ratio)) if replay_buffer and len(replay_buffer) > 0 else x.size(0)
            
            # New data: task-masked CE
            if n_new > 0:
                task_logits = logits[:n_new, current_classes]
                y_shifted = y[:n_new] - task_id * classes_per_task
                ce_loss = F.cross_entropy(task_logits, y_shifted)
            else:
                ce_loss = torch.tensor(0., device=DEVICE)
            
            # Replay data: full CE on all active classes
            if n_new < x.size(0):
                replay_logits = logits[n_new:]
                replay_y = y[n_new:]
                # Only compute loss on classes that are active in replay
                replay_active = replay_y < (task_id + 1) * classes_per_task
                if replay_active.any():
                    replay_logits = replay_logits[replay_active]
                    replay_y = replay_y[replay_active]
                    ce_loss = ce_loss + F.cross_entropy(replay_logits, replay_y)
            
            # KD on replay samples - only on overlapping active classes
            kd_loss = torch.tensor(0., device=DEVICE)
            if old_model and kd_w > 0:
                with torch.no_grad():
                    old_logits = old_model(x)
                n_new = x.size(0) // (1 + int(replay_ratio))
                if n_new < x.size(0):
                    # Only KD on classes both models have active
                    overlap = m.head.active & old_model.head.active
                    if overlap.any():
                        overlap_idx = overlap.nonzero(as_tuple=True)[0]
                        kd_loss = F.kl_div(
                            F.log_softmax(logits[n_new:, overlap_idx] / kd_t, -1),
                            F.softmax(old_logits[n_new:, overlap_idx] / kd_t, -1),
                            'batchmean'
                        ) * kd_t ** 2
            
            (ce_loss + kd_w * kd_loss).backward()
            torch.nn.utils.clip_grad_norm_(feat_params + head_params, 1.0)
            opt.step()

model = NGSModel(max_classes=100)
rb = ReplayBuffer(10000, SEED)
old_model = None
acc = np.full((10,10), np.nan)

print("=" * 60)
print("4A: Fine-tune NGS features + separate heads + replay+KD")
print("=" * 60)

for t in range(10):
    model.add_classes(list(range(t*10, (t+1)*10)))
    tr, te, _ = get_task_loaders('split_cifar100', t, 10, 128, 'class_incremental')
    start = time.time()
    train_task(model, tr, rb, old_model, t, 10, epochs=15, 
               lr_feat=1e-5, lr_head=1e-3, kd_w=5.0, kd_t=2.0, replay_ratio=1.0)
    for et in range(t+1):
        _, ete, _ = get_task_loaders('split_cifar100', et, 10, 128, 'class_incremental')
        acc[et, t] = eval_model(model, ete)
    for x,y in tr: rb.add(x, F.one_hot(y, 100).float())
    
    # Save for KD
    old_model = deepcopy(model)
    old_model.eval()
    for p in old_model.parameters(): p.requires_grad = False
    
    print(f"Task {t+1}/10: {[f'{acc[i,t]:.3f}' for i in range(t+1)]}, time={time.time()-start:.0f}s")

m = compute_metrics(acc, 1/100)
print(f"\nForgetting: {m.avg_forgetting:.3f} (target<0.05)")
print(f"Avg Acc:    {m.avg_final_accuracy:.3f} (target>0.50)")

os.makedirs('results/full', exist_ok=True)
json.dump({'acc':acc.tolist(),'forgetting':m.avg_forgetting,'avg_acc':m.avg_final_accuracy},
          open('results/full/4a_finetune.json','w'))
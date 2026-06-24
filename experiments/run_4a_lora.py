"""
4A Pivot: NGS feature extractor + LoRA per class (replay-free).
Each class gets its own LoRA adapter; router selects class-specific adapter.
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models.ngs import build_ngs
from experiments.datasets import get_task_loaders, ReplayBuffer
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

feat_extractor = build_ngs(512, 128, feat_cfg).to(DEVICE)  # ResNet18 feat_dim=512
for p in feat_extractor.parameters():
    p.requires_grad = False
feat_extractor.eval()

# LoRA per class head
class LoRAPerClass(nn.Module):
    def __init__(self, d_in=128, max_classes=100, rank=8):
        super().__init__()
        self.d_in = d_in
        self.max_classes = max_classes
        self.rank = rank
        
        # Base linear (shared)
        self.base = nn.Linear(d_in, max_classes, bias=False)
        
        # LoRA adapters per class: A @ B where A=[d_in, r], B=[r, 1] per class
        self.lora_A = nn.Parameter(torch.randn(max_classes, d_in, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(max_classes, rank, 1))
        
        self.register_buffer('active', torch.zeros(max_classes, dtype=torch.bool))
        self.scaling = 1.0 / rank
    
    def add_classes(self, class_ids):
        self.active[class_ids] = True
    
    def forward(self, x):
        # x: [B, d_in]
        # Base logits
        logits = self.base(x)  # [B, max_classes]
        
        # LoRA for active classes
        active_idx = self.active.nonzero(as_tuple=True)[0]
        if len(active_idx) > 0:
            for c in active_idx:
                lora_out = (x @ self.lora_A[c]) @ self.lora_B[c]  # [B, 1]
                logits[:, c] += self.scaling * lora_out.squeeze(-1)
        
        # Mask inactive
        logits[:, ~self.active] = -1e9
        return logits
    
    def parameters_lora(self):
        return [self.lora_A, self.lora_B]

class NGSLoRAModel(nn.Module):
    def __init__(self, max_classes=100):
        super().__init__()
        self.backbone = PretrainedBackbone('resnet18', freeze=True).to(DEVICE)
        self.feat = feat_extractor
        self.head = LoRAPerClass(d_in=128, max_classes=max_classes).to(DEVICE)
    
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

def train_task(m, loader, epochs, lr):
    opt = torch.optim.AdamW(m.head.parameters_lora(), lr=lr)
    m.train()
    for _ in range(epochs):
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad()
            logits = m(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

model = NGSLoRAModel(max_classes=100)
acc = np.full((10,10), np.nan)

print("=" * 60)
print("4A Pivot: NGS frozen features + LoRA per class")
print("=" * 60)

for t in range(10):
    model.add_classes(list(range(t*10, (t+1)*10)))
    tr, te, _ = get_task_loaders('split_cifar100', t, 10, 128, 'class_incremental')
    start = time.time()
    train_task(model, tr, epochs=15, lr=1e-3)
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
          open('results/full/4a_lora.json','w'))
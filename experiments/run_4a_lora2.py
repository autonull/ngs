"""
4A Pivot v2: Frozen base + LoRA per class (freeze base after each task).
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

# LoRA per class head with FREEZABLE base
class LoRAPerClassFrozen(nn.Module):
    def __init__(self, d_in=128, max_classes=100, rank=16):
        super().__init__()
        self.d_in = d_in
        self.max_classes = max_classes
        self.rank = rank
        
        # Base linear - will be frozen after each task
        self.base = nn.Linear(d_in, max_classes, bias=False)
        
        # LoRA adapters per class
        self.lora_A = nn.Parameter(torch.randn(max_classes, d_in, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(max_classes, rank, 1))
        
        self.register_buffer('active', torch.zeros(max_classes, dtype=torch.bool))
        self.scaling = 1.0 / rank
        
        # Track which base weights are frozen
        self.register_buffer('base_frozen', torch.zeros(max_classes, dtype=torch.bool))
    
    def add_classes(self, class_ids):
        self.active[class_ids] = True
    
    def freeze_base(self, class_ids):
        """Freeze base weights for given classes."""
        with torch.no_grad():
            self.base_frozen[class_ids] = True
    
    def forward(self, x):
        logits = self.base(x)
        active_idx = self.active.nonzero(as_tuple=True)[0]
        if len(active_idx) > 0:
            for c in active_idx:
                lora_out = (x @ self.lora_A[c]) @ self.lora_B[c]
                logits[:, c] += self.scaling * lora_out.squeeze(-1)
        logits[:, ~self.active] = -1e9
        return logits
    
    def get_trainable_params(self):
        params = [self.lora_A, self.lora_B]
        # Only train base for non-frozen classes
        active_not_frozen = self.active & ~self.base_frozen
        if active_not_frozen.any():
            # We need to handle this differently - use a hook or separate optimizer group
            pass
        return params

class NGSLoRAModel(nn.Module):
    def __init__(self, max_classes=100):
        super().__init__()
        self.backbone = PretrainedBackbone('resnet18', freeze=True).to(DEVICE)
        self.feat = feat_extractor
        self.head = LoRAPerClassFrozen(d_in=128, max_classes=max_classes).to(DEVICE)
    
    def forward(self, x):
        with torch.no_grad():
            f = self.backbone(x)
            f = self.feat(f)
            z = f.latent if hasattr(f, 'latent') else f.logits
        return self.head(z)
    
    def add_classes(self, c): self.head.add_classes(c)
    def freeze_base(self, c): self.head.freeze_base(c)

def eval_model(m, loader):
    m.eval(); corr=tot=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            p=m(x).argmax(1); corr+=(p==y).sum().item(); tot+=y.size(0)
    return corr/tot

def train_task(m, loader, epochs, lr, freeze_base=True):
    # Train LoRA + base
    params = [m.head.lora_A, m.head.lora_B, m.head.base.weight]
    opt = torch.optim.AdamW(params, lr=lr)
    m.train()
    for _ in range(epochs):
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad()
            logits = m(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
    # Freeze base for this task's classes by zeroing gradients in future
    if freeze_base:
        active_now = m.head.active.clone()
        m.freeze_base(active_now)
        # Register hook to zero grad for frozen base weights
        def make_hook(mask):
            def hook(grad):
                g = grad.clone()
                g[mask] = 0
                return g
            return hook
        m.head.base.weight.register_hook(make_hook(m.head.base_frozen))

model = NGSLoRAModel(max_classes=100)
acc = np.full((10,10), np.nan)

print("=" * 60)
print("4A Pivot v2: Frozen base after each task + LoRA per class")
print("=" * 60)

for t in range(10):
    new_classes = list(range(t*10, (t+1)*10))
    model.add_classes(new_classes)
    tr, te, _ = get_task_loaders('split_cifar100', t, 10, 128, 'class_incremental')
    start = time.time()
    train_task(model, tr, epochs=15, lr=1e-3, freeze_base=True)
    for et in range(t+1):
        _, ete, _ = get_task_loaders('split_cifar100', et, 10, 128, 'class_incremental')
        acc[et, t] = eval_model(model, ete)
    print(f"Task {t+1}/10: {[f'{acc[i,t]:.3f}' for i in range(t+1)]}, time={time.time()-start:.0f}s")

m = compute_metrics(acc, 1/100)
print(f"\nForgetting: {m.avg_forgetting:.3f} (target<0.05)")
print(f"Avg Acc:    {m.avg_final_accuracy:.3f} (target>0.50)")

os.makedirs('results/full', exist_ok=True)
json.dump({'acc':acc.tolist(),'forgetting':m.avg_forgetting,'avg_acc':m.avg_final_accuracy},
          open('results/full/4a_lora2.json','w'))
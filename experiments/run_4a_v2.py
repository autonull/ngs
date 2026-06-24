"""
4A v2: Freeze router mu for old classes + higher replay/KD.
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.modules.dynamic_head import DynamicHead
from experiments.datasets import get_task_loaders, ReplayBuffer
from experiments.metrics import compute_metrics
from experiments.backbones import PretrainedBackbone
from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

class BackboneDynamicHead(nn.Module):
    def __init__(self, max_classes=100, config=None):
        super().__init__()
        self.backbone = PretrainedBackbone('resnet18', freeze=True).to(DEVICE)
        self.head = DynamicHead(d_latent=self.backbone.feature_dim, max_classes=max_classes, config=config)
    def forward(self, x): return self.head(self.backbone(x))
    def add_classes(self, c): self.head.add_classes(c)
    @property
    def num_active_classes(self): return self.head.num_active_classes

def eval_model(m, loader):
    m.eval(); corr=tot=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            p=m(x).argmax(1); corr+=(p==y).sum().item(); tot+=y.size(0)
    return corr/tot

def train_task(m, loader, old, epochs, lr, wd, rb, rr, kw, kt, freeze_mu_mask=None):
    opt=torch.optim.AdamW(m.head.parameters(), lr=lr, weight_decay=wd)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    m.train()
    
    # Freeze router mu hook for old classes
    mu_hooks = []
    if freeze_mu_mask is not None:
        def make_hook(mask):
            def hook(grad):
                g = grad.clone()
                g[mask] = 0
                return g
            return hook
        hook = make_hook(freeze_mu_mask)
        mu_hooks.append(m.head.ngs.router.mu.register_hook(hook))
    
    for _ in range(epochs):
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            if rb and len(rb)>x.size(0):
                rx,ry=rb.sample(x.size(0))
                if rx is not None:
                    x=torch.cat([x,rx.to(DEVICE)],0)
                    y=torch.cat([y,ry.argmax(1).to(DEVICE)],0)
            opt.zero_grad()
            logits=m(x)
            ce=F.cross_entropy(logits,y)
            kd=torch.tensor(0.,device=DEVICE)
            if old and kw>0:
                with torch.no_grad(): ol=old(x)
                n_new=x.size(0)//(1+int(rr))
                if n_new<x.size(0):
                    kd=F.kl_div(F.log_softmax(logits[n_new:]/kt,-1),F.softmax(ol[n_new:]/kt,-1),'batchmean')*kt**2
            (ce+kw*kd).backward()
            torch.nn.utils.clip_grad_norm_(m.head.parameters(),1)
            opt.step()
        sch.step()
    
    for h in mu_hooks: h.remove()

cfg=NGSConfig(latent_dim=256,k_init=64,max_k=512,top_k=16,
    routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    parameter_storage=ParameterStorage.DIRECT_ADAPTER,
    topology_control=TopologyControl.DISCRETE_HEURISTIC,
    memory_management=MemoryManagement.DYNAMIC,
    tau=1.0,gamma_residual=0.1,ema_decay=0.99)

model=BackboneDynamicHead(max_classes=100, config=cfg).to(DEVICE)
rb=ReplayBuffer(20000,SEED); old=None
acc=np.full((10,10),np.nan)
prev_mu_mask = None

print("=" * 60)
print("4A v2: Freeze router mu + 20k replay + KD=5")
print("=" * 60)

for t in range(10):
    model.add_classes(list(range(t*10,(t+1)*10)))
    
    # Capture current router mu mask before training (for freezing)
    if t > 0:
        prev_mu_mask = model.head.ngs.router.active_mask.clone()
    
    tr,te,_=get_task_loaders('split_cifar100',t,10,128,'class_incremental')
    start=time.time()
    train_task(model,tr,old,epochs=15,lr=1e-3,wd=1e-4,rb=rb,rr=2.0,kw=5.0,kt=2.0,
               freeze_mu_mask=prev_mu_mask if t>0 else None)
    for et in range(t+1):
        _,ete,_=get_task_loaders('split_cifar100',et,10,128,'class_incremental')
        acc[et,t]=eval_model(model,ete)
    for x,y in tr: rb.add(x,F.one_hot(y,100).float())
    old=BackboneDynamicHead(100,cfg).to(DEVICE)
    old.head.load_state_dict(model.head.state_dict())
    old.eval()
    for p in old.parameters(): p.requires_grad=False
    old.add_classes(list(range((t+1)*10)))
    print(f"Task {t+1}/10: {[f'{acc[i,t]:.3f}' for i in range(t+1)]}, time={time.time()-start:.0f}s")

m=compute_metrics(acc,1/100)
print(f"\nForgetting: {m.avg_forgetting:.3f} (target<0.05)")
print(f"Avg Acc:    {m.avg_final_accuracy:.3f} (target>0.50)")

os.makedirs('results/full',exist_ok=True)
json.dump({'acc':acc.tolist(),'forgetting':m.avg_forgetting,'avg_acc':m.avg_final_accuracy},
          open('results/full/4a_v2.json','w'))
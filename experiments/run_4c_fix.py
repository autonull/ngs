"""
4C Fix: Federated sharing ONLY router params (mu, log_s, log_alpha) for 10x+ reduction.
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models.ngs import build_ngs
from experiments.datasets import get_task_loaders

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

cfg = NGSConfig(
    latent_dim=32, k_init=16, max_k=64, top_k=4,
    routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    parameter_storage=ParameterStorage.DIRECT_ADAPTER,
    topology_control=TopologyControl.DISCRETE_HEURISTIC,
    memory_management=MemoryManagement.DYNAMIC,
    tau=1.0, gamma_residual=0.1, ema_decay=0.99,
)

class NGSClient(nn.Module):
    def __init__(self):
        super().__init__()
        self.ngs = build_ngs(784, 10, cfg)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.ngs(x)
        return out.logits if hasattr(out, 'logits') else out
    
    def get_router_params(self):
        """Return ONLY router params: mu, log_s, log_alpha"""
        router_params = {}
        for n, p in self.ngs.named_parameters():
            if 'router.mu' in n or 'router.log_s' in n or 'router.log_alpha' in n:
                router_params[n] = p.data.clone()
        return router_params
    
    def set_router_params(self, router_params):
        """Load shared router params"""
        for n, p in self.ngs.named_parameters():
            if n in router_params:
                p.data.copy_(router_params[n])

def train_local(model, loader, epochs=1, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if x.dim() > 2: x = x.view(x.size(0), -1)
            opt.zero_grad()
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

def eval_model(model, loader):
    model.eval(); corr=tot=0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if x.dim() > 2: x = x.view(x.size(0), -1)
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            p = logits.argmax(1); corr += (p==y).sum().item(); tot += y.size(0)
    return corr/tot

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# Load MNIST split for 2 clients
client_loaders = []
for cid in range(2):
    tr, te, _ = get_task_loaders('split_mnist', cid, 5, 128, scenario='task_incremental')
    client_loaders.append((tr, te))
_, global_test, _ = get_task_loaders('mnist', 0, 10, 128)

print("=" * 50)
print("BASELINE: Centralized training (full model)")
central = build_ngs(784, 10, cfg).to(DEVICE)
for _ in range(20):
    for cid in range(2):
        train_local(central, client_loaders[cid][0], epochs=1)
central_acc = eval_model(central, global_test)
central_params = count_params(central)
print(f"Central acc: {central_acc:.3f}, params: {central_params:,}")

print("\nFEDAVG: Full model averaging (20 rounds)")
fed_models = [build_ngs(784, 10, cfg).to(DEVICE) for _ in range(2)]
for rnd in range(20):
    for cid in range(2):
        train_local(fed_models[cid], client_loaders[cid][0], epochs=1)
    avg_state = {}
    for k in fed_models[0].state_dict():
        avg_state[k] = torch.stack([m.state_dict()[k].float() for m in fed_models]).mean(0)
    for m in fed_models: m.load_state_dict(avg_state)
fed_acc = eval_model(fed_models[0], global_test)
print(f"FedAvg acc: {fed_acc:.3f}")

print("\nNGS ROUTER SHARING: Share ONLY router (mu, log_s, log_alpha)")
clients = [NGSClient().to(DEVICE) for _ in range(2)]
init_router = clients[0].get_router_params()
for c in clients: c.set_router_params(init_router)

for rnd in range(20):
    for cid in range(2):
        train_local(clients[cid], client_loaders[cid][0], epochs=1)
    avg_router = {}
    for k in init_router:
        avg_router[k] = torch.stack([c.get_router_params()[k] for c in clients]).mean(0)
    for c in clients: c.set_router_params(avg_router)

ngs_acc = eval_model(clients[0], global_test)
router_params = sum(p.numel() for n, p in clients[0].ngs.named_parameters() 
                    if 'router.mu' in n or 'router.log_s' in n or 'router.log_alpha' in n)
full_params = count_params(clients[0])
print(f"NGS router acc: {ngs_acc:.3f}, router params: {router_params:,}, full: {full_params:,}, reduction: {full_params/router_params:.1f}x")

ratio = ngs_acc / central_acc if central_acc > 0 else 0
comm_red = full_params / router_params
print(f"\nResults: NGS/Central acc ratio: {ratio:.2f} (target >0.9)")
print(f"Comm reduction: {comm_red:.1f}x (target >10x)")
print("✅ PASS" if ratio > 0.9 and comm_red > 10 else "❌ NEEDS WORK")

os.makedirs('results/full', exist_ok=True)
json.dump({'central_acc': central_acc, 'fed_acc': fed_acc, 'ngs_acc': ngs_acc,
           'comm_reduction': comm_red, 'acc_ratio': ratio},
          open('results/full/4c_fix.json', 'w'))
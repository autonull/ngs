"""
4C: Federated code sharing vs full-model FedAvg.
2 clients, MNIST, 20 rounds, validates 10x comm reduction at ~90% central acc.
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models.ngs import build_ngs
from experiments.datasets import get_task_loaders, ReplayBuffer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# NGS config for code sharing
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
    
    def get_code_params(self):
        """Return router + adapter params (code to share)"""
        code_params = {}
        for n, p in self.ngs.named_parameters():
            if 'router' in n or 'param_store' in n:
                code_params[n] = p.data.clone()
        return code_params
    
    def set_code_params(self, code_params):
        """Load shared code params"""
        for n, p in self.ngs.named_parameters():
            if n in code_params:
                p.data.copy_(code_params[n])

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

# Load MNIST split for 2 clients (5 classes each)
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
    # Average full models
    avg_state = {}
    for k in fed_models[0].state_dict():
        avg_state[k] = torch.stack([m.state_dict()[k].float() for m in fed_models]).mean(0)
    for m in fed_models: m.load_state_dict(avg_state)
fed_acc = eval_model(fed_models[0], global_test)
fed_params = count_params(fed_models[0])
print(f"FedAvg acc: {fed_acc:.3f}, params/client: {fed_params:,}")

print("\nNGS CODE SHARING: Share router + adapter codes only")
# Initialize clients with same random seed for fair comparison
clients = [NGSClient().to(DEVICE) for _ in range(2)]
# Share initial code
init_code = clients[0].get_code_params()
for c in clients: c.set_code_params(init_code)

for rnd in range(20):
    for cid in range(2):
        train_local(clients[cid], client_loaders[cid][0], epochs=1)
    # Average ONLY code params
    avg_code = {}
    for k in init_code:
        avg_code[k] = torch.stack([c.get_code_params()[k] for c in clients]).mean(0)
    for c in clients: c.set_code_params(avg_code)

ngs_acc = eval_model(clients[0], global_test)
# Count code params only
code_params = sum(p.numel() for n, p in clients[0].ngs.named_parameters() 
                  if 'router' in n or 'param_store' in n)
full_params = count_params(clients[0])
print(f"NGS code acc: {ngs_acc:.3f}, code params: {code_params:,}, full: {full_params:,}, reduction: {full_params/code_params:.1f}x")

ratio = ngs_acc / central_acc if central_acc > 0 else 0
comm_red = full_params / code_params
print(f"\nResults: NGS/Central acc ratio: {ratio:.2f} (target >0.9)")
print(f"Comm reduction: {comm_red:.1f}x (target >10x)")
print("✅ PASS" if ratio > 0.9 and comm_red > 10 else "❌ NEEDS WORK")

os.makedirs('results/full', exist_ok=True)
json.dump({'central_acc': central_acc, 'fed_acc': fed_acc, 'ngs_acc': ngs_acc,
           'comm_reduction': comm_red, 'acc_ratio': ratio},
          open('results/full/4c.json', 'w'))
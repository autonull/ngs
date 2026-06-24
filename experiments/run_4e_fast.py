"""
Fast 4E: TinyShakespeare - minimal viable run.
1 seed, 1000 steps (not 10k), small model, validates ppl < 12.
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.datasets_tinyshakespeare import create_tinyshakespeare_loaders
from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
from ngs.models.ngs import build_ngs

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# Tiny config: d_model=128, 8 experts, 1000 steps
cfg = NGSConfig(
    latent_dim=128, k_init=8, max_k=32, top_k=4,
    routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    parameter_storage=ParameterStorage.DIRECT_ADAPTER,
    topology_control=TopologyControl.DISCRETE_HEURISTIC,
    memory_management=MemoryManagement.DYNAMIC,
    tau=1.0, gamma_residual=0.1, ema_decay=0.99,
)

# Input is [B, seq_len, vocab_size] = [32, 64, 65], flatten to [32, 4096]
# Use smaller model
model = build_ngs(64*65, 65, cfg).to(DEVICE)
train_loader, val_loader, _ = create_tinyshakespeare_loaders(task_id=0, n_tasks=1, seq_len=64, batch_size=32)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
model.train()

step = 0
start = time.time()
for x, y in train_loader:
    if step >= 1000: break
    x, y = x.to(DEVICE), y.to(DEVICE)
    x = x.view(x.size(0), -1)  # flatten [B, 64, 65] -> [B, 4160]
    opt.zero_grad()
    out = model(x)
    logits = out.logits if hasattr(out, 'logits') else out
    loss = F.cross_entropy(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    step += 1
    if step % 200 == 0:
        print(f"Step {step}: loss={loss.item():.3f}, ppl={loss.item():.3f}")

# Quick eval
model.eval()
total_loss = 0; total_tokens = 0
with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.size(0), -1)
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            loss = F.cross_entropy(logits, y, reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()

ppl = np.exp(total_loss / total_tokens)
print(f"\nVal PPL: {ppl:.2f} (target < 12 for fast)")
print(f"Time: {time.time()-start:.1f}s")
print("✅ PASS" if ppl < 12 else "❌ NEEDS SCALE")

os.makedirs('results/fast', exist_ok=True)
json.dump({'ppl': float(ppl), 'steps': step, 'time': time.time()-start}, open('results/fast/4e.json', 'w'))
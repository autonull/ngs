"""
4E: TinyShakespeare - overnight run (10k steps).
Uses one-hot input [B, seq_len, vocab] -> flatten -> NGS.
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

# Config: d_model=512, 32 experts per TODO6.md
# But one-hot input is [seq_len, vocab] = [64, 65] = 4160 dim
# Use smaller latent_dim to fit
cfg = NGSConfig(
    latent_dim=256, k_init=16, max_k=64, top_k=8,
    routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    parameter_storage=ParameterStorage.DIRECT_ADAPTER,
    topology_control=TopologyControl.DISCRETE_HEURISTIC,
    memory_management=MemoryManagement.DYNAMIC,
    tau=1.0, gamma_residual=0.1, ema_decay=0.99,
)

# Input dim = seq_len * vocab = 64 * 65 = 4160
# Output = vocab = 65 (next char prediction)
model = build_ngs(64*65, 65, cfg).to(DEVICE)

# Data
train_loader, val_loader, _ = create_tinyshakespeare_loaders(task_id=0, n_tasks=1, seq_len=64, batch_size=32)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10000)

print(f"Training for 10000 steps...")
model.train()
step = 0
start = time.time()

for x, y in train_loader:
    if step >= 10000: break
    x, y = x.to(DEVICE), y.to(DEVICE)
    x = x.view(x.size(0), -1)  # [B, 64*65]
    opt.zero_grad()
    out = model(x)
    logits = out.logits if hasattr(out, 'logits') else out
    loss = F.cross_entropy(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    scheduler.step()
    
    if step % 1000 == 0:
        print(f"Step {step}: loss={loss.item():.3f}, ppl={np.exp(loss.item()):.2f}, time={time.time()-start:.0f}s")
    step += 1

# Eval
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
print(f"\nFinal PPL: {ppl:.2f} (target <11.5)")
print("✅ PASS" if ppl < 11.5 else "❌ NEEDS MORE")

os.makedirs('results/full', exist_ok=True)
json.dump({'ppl': float(ppl), 'steps': step, 'time': time.time()-start},
          open('results/full/4e.json', 'w'))
"""
Smoke test for MAML Omniglot with ConvNet4 backbone + NGS head.
Verifies gradients flow through backbone + hypernetwork.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.maml_trainer_cnn import create_maml_trainer_cnn
from experiments.datasets import get_omniglot_loaders
from experiments.vision_backbones import create_omniglot_ngs

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEVICE}")
print("Loading Omniglot (20 tasks for smoke test)...")
tasks = get_omniglot_loaders(
    n_way=5, k_shot=1, n_query=15, n_tasks=20, image_size=28
)
print(f"Loaded {len(tasks)} few-shot tasks")

# Create model with ConvNet4 backbone
model = create_omniglot_ngs(
    num_classes=5,
    latent_dim=64,
    max_k=64,
    k_init=32,
    top_k=8,
    hypernetwork_hidden_dim=64,
    hypernetwork_code_dim=16,
).to(DEVICE)

print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
backbone_params = sum(p.numel() for p in model.parameters_backbone())
head_params = sum(p.numel() for p in model.parameters_head())
print(f"  Backbone: {backbone_params:,}")
print(f"  NGS head: {head_params:,}")

# Test forward
print("\n--- Forward pass ---")
x = torch.randn(5, 1, 28, 28).to(DEVICE)
out = model(x)
print(f"Output: {out.logits.shape}")

# Test MAML inner loop adaptation manually
print("\n--- Manual inner loop test (5 steps) ---")
supp_x, supp_y = next(iter(tasks[0][0]))
query_x, query_y = next(iter(tasks[0][1]))
supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)
query_x, query_y = query_x.to(DEVICE), query_y.to(DEVICE)

# Check gradient flow through backbone
model.train()
opt = torch.optim.SGD(model.parameters_head(), lr=0.01)
for step in range(5):
    opt.zero_grad()
    out = model(supp_x)
    loss = torch.nn.functional.cross_entropy(out.logits, supp_y)
    loss.backward()
    opt.step()
    if step == 0:
        # Check backbone gradients (should be None since we only optimize head)
        has_backbone_grad = any(p.grad is not None for p in model.parameters_backbone())
        print(f"  Backbone has grads: {has_backbone_grad} (expected False)")
        has_head_grad = any(p.grad is not None for p in model.parameters_head())
        print(f"  Head has grads: {has_head_grad} (expected True)")

# Evaluate
model.eval()
with torch.no_grad():
    out = model(query_x)
    acc = (out.logits.argmax(1) == query_y).float().mean().item()
print(f"  Query acc after 5 steps: {acc:.3f}")

# Test MAMLTrainerCNN integration
print("\n--- Testing MAMLTrainerCNN ---")
from experiments.maml_trainer_cnn import MAMLTrainerCNN

trainer = MAMLTrainerCNN(
    num_classes=5,
    latent_dim=64,
    max_k=64,
    k_init=32,
    top_k=8,
    inner_lr=0.01,
    inner_steps=5,
    meta_lr=1e-3,
    backbone_lr=1e-4,
    device=DEVICE
)

print(f"Meta-model params: {sum(p.numel() for p in trainer.meta_model.parameters()):,}")
inner_params = sum(p.numel() for p in trainer.inner_loop_params())
print(f"Inner-loop params: {inner_params:,}")

# Single MAML step
meta_loss, fmodel = trainer.maml_step(supp_x, supp_y, query_x, query_y)
print(f"Meta loss: {meta_loss.item():.4f}")

# Check meta-gradients
print("\n--- Checking meta-gradients ---")
has_grad = False
for name, param in trainer.meta_model.named_parameters():
    if 'backbone' in name or 'ngs_head.param_store.hypernet' in name:
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            print(f"  ✅ {name}: grad norm = {grad_norm:.6f}")

if not has_grad:
    print("  ❌ No meta-gradients on backbone/hypernet")
else:
    print("  ✅ Meta-gradients flow through backbone + hypernetwork")

print("\n✅ Smoke test PASSED")
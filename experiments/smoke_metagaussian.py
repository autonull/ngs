"""
Smoke test for MetaGaussianPrior - verifies the class compiles and works.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngs.modules.parameter_stores import MetaGaussianPrior

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEVICE}")

# Create MetaGaussianPrior for 20 domains (e.g., Omniglot alphabets)
n_domains = 20
max_k = 32
d_latent = 64

print(f"\nCreating MetaGaussianPrior: {n_domains} domains, {max_k} max units, {d_latent} latent dim")
meta_prior = MetaGaussianPrior(n_domains, max_k, d_latent).to(DEVICE)
print(f"MetaGaussianPrior params: {sum(p.numel() for p in meta_prior.parameters()):,}")

# Test forward pass
print("\n--- Testing forward pass ---")
for domain_id in [0, 5, 19]:
    mu_0, log_sigma_0 = meta_prior(domain_id)
    print(f"Domain {domain_id}: mu_0 shape={mu_0.shape}, log_sigma_0 shape={log_sigma_0.shape}")
    print(f"  mu_0 mean={mu_0.mean().item():.4f}, std={mu_0.std().item():.4f}")
    print(f"  log_sigma_0 mean={log_sigma_0.mean().item():.4f}")

# Test get_all_priors
print("\n--- Testing get_all_priors ---")
all_mu, all_log_sigma = meta_prior.get_all_priors()
print(f"All mu shape: {all_mu.shape}")
print(f"All log_sigma shape: {all_log_sigma.shape}")

# Test gradient flow
print("\n--- Testing gradient flow ---")
domain_id = 0
mu_0, log_sigma_0 = meta_prior(domain_id)
# Simulate a simple loss
loss = mu_0.sum() + log_sigma_0.sum()
loss.backward()

print(f"mu_0.grad norm: {meta_prior.mu_0.grad.norm().item():.6f}")
print(f"log_sigma_0.grad norm: {meta_prior.log_sigma_0.grad.norm().item():.6f}")

# Check gradients are only for the accessed domain
print(f"\n--- Gradient sparsity ---")
mu_grad = meta_prior.mu_0.grad
log_sigma_grad = meta_prior.log_sigma_0.grad
print(f"mu_0 non-zero grads: {(mu_grad != 0).sum().item()} / {mu_grad.numel()}")
print(f"log_sigma_0 non-zero grads: {(log_sigma_grad != 0).sum().item()} / {log_sigma_grad.numel()}")

# Test with optimizer
print("\n--- Testing with optimizer ---")
opt = torch.optim.Adam(meta_prior.parameters(), lr=1e-3)
for step in range(5):
    opt.zero_grad()
    domain_id = step % n_domains
    mu_0, log_sigma_0 = meta_prior(domain_id)
    loss = mu_0.sum() + log_sigma_0.sum()
    loss.backward()
    opt.step()
    print(f"  Step {step}: domain={domain_id}, loss={loss.item():.4f}")

print("\n✅ MetaGaussianPrior smoke test PASSED")
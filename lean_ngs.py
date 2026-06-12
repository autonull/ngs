import torch
import torch.nn as nn
import torch.nn.functional as F


class LeanNGS(nn.Module):
    def __init__(self, d_in, d_out, d_latent=32, k_init=128, max_k=512, top_k=8, lora_rank=4):
        super().__init__()
        self.d_latent = d_latent
        self.top_k = top_k
        self.max_k = max_k
        self.lora_rank = lora_rank
        self.eps = 1e-5

        self.p_down = nn.Linear(d_in, d_latent, bias=False)
        self.p_up = nn.Linear(d_latent, d_out, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.tau = nn.Parameter(torch.tensor(1.0))

        # Pre-allocate max capacity
        self.register_buffer('active_mask', torch.zeros(max_k, dtype=torch.bool))
        self.active_mask[:k_init] = True

        self.mu = nn.Parameter(torch.randn(max_k, d_latent) * 1.0)
        self.log_s = nn.Parameter(torch.zeros(max_k, d_latent))
        self.log_alpha = nn.Parameter(torch.zeros(max_k))
        
        # LoRA-style low-rank adapters: W = A @ B where A: [d, r], B: [r, d]
        self.W_A = nn.Parameter(torch.randn(max_k, d_latent, lora_rank) * 1e-2)
        self.W_B = nn.Parameter(torch.randn(max_k, lora_rank, d_latent) * 1e-2)

        # EMA of gradient magnitudes for splitting
        self.register_buffer('grad_mu_ema', torch.zeros(max_k))
        self.ema_decay = 0.99

    @property
    def K(self):
        return self.active_mask.sum().item()

    def forward(self, x):
        z = self.p_down(x)  # [B, d]
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]

        mu = self.mu[active_idx]           # [K, d]
        log_s = self.log_s[active_idx]     # [K, d]
        log_alpha = self.log_alpha[active_idx]  # [K]
        W_A = self.W_A[active_idx]         # [K, d, r]
        W_B = self.W_B[active_idx]         # [K, r, d]

        # Diagonal Mahalanobis in log-space
        diff = z.unsqueeze(1) - mu.unsqueeze(0)           # [B, K, d]
        s_sq = torch.exp(2 * log_s) + self.eps            # [K, d]
        mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1) # [B, K]

        log_w = log_alpha - (0.5 / self.tau) * mahalanobis_sq  # [B, K]

        topk_vals, topk_rel_idx = torch.topk(log_w, min(self.top_k, self.K), dim=-1)  # [B, top_k]

        weights = F.softmax(topk_vals, dim=-1).unsqueeze(-1)        # [B, top_k, 1]
        
        # LoRA: W = A @ B, so W @ z = A @ (B @ z)
        W_A_topk = W_A[topk_rel_idx]  # [B, top_k, d, r]
        W_B_topk = W_B[topk_rel_idx]  # [B, top_k, r, d]
        
        # B @ z: [B, top_k, r, d] @ [B, d] -> [B, top_k, r]
        Bz = torch.einsum('bkrd,bd->bkr', W_B_topk, z)
        # A @ (Bz): [B, top_k, d, r] @ [B, top_k, r] -> [B, top_k, d]
        local_out = torch.einsum('bkdr,bkr->bkd', W_A_topk, Bz)
        
        blended = (weights * local_out).sum(dim=1)                   # [B, d]

        return self.p_up(blended + self.gamma * z)

    def entropy_loss(self, x):
        z = self.p_down(x)
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        mu = self.mu[active_idx]
        log_s = self.log_s[active_idx]
        log_alpha = self.log_alpha[active_idx]

        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        s_sq = torch.exp(2 * log_s) + self.eps
        mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1)
        log_w = log_alpha - (0.5 / self.tau) * mahalanobis_sq
        log_p = F.log_softmax(log_w, dim=-1)
        p = log_p.exp()
        entropy = -(p * log_p).sum(dim=-1).mean()
        return entropy

    @torch.no_grad()
    def update_grad_ema(self):
        if self.mu.grad is None:
            return
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        grad_mag = self.mu.grad[active_idx].norm(dim=-1)
        self.grad_mu_ema[active_idx] = self.ema_decay * self.grad_mu_ema[active_idx] + (1 - self.ema_decay) * grad_mag

    @torch.no_grad()
    def adapt_density(self, split_thresh=0.05, prune_thresh=0.01, split_scale=0.5, noise_std=0.01,
                      spawn_thresh=-5.0, max_spawn_per_call=10, z_samples=None):
        """Split high-gradient units, prune low-opacity units, spawn in uncovered regions."""
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        K = len(active_idx)
        if K == 0:
            return

        alpha = torch.sigmoid(self.log_alpha[active_idx])
        grad_ema = self.grad_mu_ema[active_idx]
        max_s = torch.exp(self.log_s[active_idx]).max(dim=-1).values

        # Prune
        prune_mask = alpha < prune_thresh
        if prune_mask.any():
            prune_idx = active_idx[prune_mask]
            self.active_mask[prune_idx] = False
            self.grad_mu_ema[prune_idx] = 0
            print(f"Pruned {prune_mask.sum().item()} units (K={self.K})")

        # Split
        split_mask = (grad_ema > split_thresh) & (max_s > split_thresh)
        if split_mask.any():
            split_idx = active_idx[split_mask]
            n_split = len(split_idx)
            free_slots = (~self.active_mask).nonzero(as_tuple=True)[0]
            n_available = len(free_slots)
            n_split = min(n_split, n_available)

            if n_split > 0:
                split_idx = split_idx[:n_split]
                new_idx = free_slots[:n_split]

                self.mu[new_idx] = self.mu[split_idx] + torch.randn_like(self.mu[split_idx]) * noise_std
                self.log_s[new_idx] = self.log_s[split_idx] + torch.log(torch.tensor(split_scale))
                self.log_alpha[new_idx] = self.log_alpha[split_idx].clone()
                alpha_new = torch.sigmoid(self.log_alpha[new_idx])
                self.log_alpha[new_idx] = torch.logit(alpha_new * 0.5, eps=1e-8)
                
                # Initialize LoRA adapters for new units
                self.W_A[new_idx] = torch.randn_like(self.W_A[split_idx]) * 1e-2
                self.W_B[new_idx] = torch.randn_like(self.W_B[split_idx]) * 1e-2
                
                self.grad_mu_ema[new_idx] = 0
                self.active_mask[new_idx] = True

                # Halve scale of original units
                self.log_s[split_idx] = self.log_s[split_idx] + torch.log(torch.tensor(split_scale))
                alpha_split = torch.sigmoid(self.log_alpha[split_idx])
                self.log_alpha[split_idx] = torch.logit(alpha_split * 0.5, eps=1e-8)

                print(f"Split {n_split} units (K={self.K})")

        # Spawn: if z_samples provided, find uncovered regions and spawn new units
        if z_samples is not None:
            free_slots = (~self.active_mask).nonzero(as_tuple=True)[0]
            if len(free_slots) > 0:
                z_samples = z_samples.to(self.mu.device)
                mu_active = self.mu[active_idx]
                log_s_active = self.log_s[active_idx]
                s_sq = torch.exp(2 * log_s_active) + self.eps

                # Compute max weight for each sample across all Gaussians
                diff = z_samples.unsqueeze(1) - mu_active.unsqueeze(0)  # [B, K, d]
                mahalanobis_sq = ((diff ** 2) / s_sq.unsqueeze(0)).sum(dim=-1)  # [B, K]
                log_alpha_active = self.log_alpha[active_idx]
                log_w = log_alpha_active - (0.5 / self.tau) * mahalanobis_sq
                max_log_w, _ = log_w.max(dim=-1)  # [B]

                # Find samples with low max weight (uncovered)
                uncovered_mask = max_log_w < spawn_thresh
                if uncovered_mask.any():
                    uncovered_z = z_samples[uncovered_mask]
                    n_spawn = min(len(uncovered_z), len(free_slots), max_spawn_per_call)
                    if n_spawn > 0:
                        spawn_idx = free_slots[:n_spawn]
                        # Use mean of uncovered samples as new means
                        self.mu[spawn_idx] = uncovered_z[:n_spawn]
                        self.log_s[spawn_idx].fill_(0.0)  # s=1
                        self.log_alpha[spawn_idx].fill_(0.0)  # alpha=0.5
                        
                        # Initialize LoRA adapters for spawned units
                        self.W_A[spawn_idx] = torch.randn_like(self.W_A[spawn_idx]) * 1e-2
                        self.W_B[spawn_idx] = torch.randn_like(self.W_B[spawn_idx]) * 1e-2
                        
                        self.grad_mu_ema[spawn_idx] = 0
                        self.active_mask[spawn_idx] = True
                        print(f"Spawned {n_spawn} units in uncovered regions (K={self.K})")

    def diversity_loss(self):
        """Push Gaussian means apart to encourage coverage."""
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        if len(active_idx) < 2:
            return torch.tensor(0.0, device=self.mu.device)
        mu = self.mu[active_idx]  # [K, d]
        # Pairwise distances
        dist = torch.cdist(mu, mu, p=2)  # [K, K]
        # Exclude diagonal
        mask = ~torch.eye(len(active_idx), dtype=torch.bool, device=mu.device)
        # Want to maximize minimum distance -> minimize negative min distance
        min_dist = dist[mask].min()
        return -min_dist


def train_step(model, x, y, optimizer, lambda_entropy=0.01):
    model.train()
    optimizer.zero_grad()
    logits = model(x)
    # y is one-hot, convert to class indices for CE
    target = y.argmax(dim=1)
    loss = F.cross_entropy(logits, target)
    entropy = model.entropy_loss(x)
    total_loss = loss - lambda_entropy * entropy
    total_loss.backward()
    model.update_grad_ema()
    return loss.item(), entropy.item()


@torch.no_grad()
def eval_step(model, x, y):
    model.eval()
    pred = model(x)
    loss = F.mse_loss(pred, y)
    return loss.item()
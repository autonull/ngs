import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LeanNGS(nn.Module):
    def __init__(self, d_in, d_out, d_latent=32, k_init=128, top_k=8, max_k=512, adc_mode='pre_alloc'):
        super().__init__()
        self.d_latent = d_latent
        self.top_k = top_k
        self.k_init = k_init
        self.max_k = max_k
        self.adc_mode = adc_mode # 'pre_alloc' or 'dynamic'
        
        self.p_down = nn.Linear(d_in, d_latent, bias=False)
        self.p_up = nn.Linear(d_latent, d_out, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.1)) # Residual
        
        # Determine initial sizes based on mode
        actual_k = self.max_k if self.adc_mode == 'pre_alloc' else self.k_init
        
        # G-Unit Params
        self.mu = nn.Parameter(torch.randn(actual_k, d_latent) * 0.1)
        self.log_s = nn.Parameter(torch.zeros(actual_k, d_latent)) # s = exp(log_s)
        self.log_alpha = nn.Parameter(torch.zeros(actual_k))       # alpha = sigmoid(log_alpha)
        # Fixing the bug from prompt code: W should map from d_latent to d_latent so it can be added to z BEFORE P_up
        self.W = nn.Parameter(torch.randn(actual_k, d_latent, d_latent) * 1e-4)
        
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.eps = 1e-5

        # EMA tracker for gradient of mu
        self.register_buffer('grad_mu_ema', torch.zeros(actual_k, d_latent))
        self.register_buffer('active_mask', torch.zeros(actual_k, dtype=torch.bool))
        self.active_mask[:self.k_init] = True
        
        # Hook for gradient EMA
        self.mu.register_hook(self._update_mu_grad_ema)

    def _update_mu_grad_ema(self, grad):
        # grad is [K, d]
        with torch.no_grad():
            alpha_ema = 0.9 # decay factor
            # For simplicity, we track the norm of the gradient per unit as EMA
            grad_norm = torch.norm(grad, dim=-1) # [K]
            
            if not hasattr(self, 'grad_mu_norm_ema'):
                # Also track scalar magnitude per unit for easier thresholding
                self.register_buffer('grad_mu_norm_ema', torch.zeros_like(self.log_alpha))
                
            self.grad_mu_norm_ema = alpha_ema * self.grad_mu_norm_ema + (1 - alpha_ema) * grad_norm

    def forward(self, x):
        z = self.p_down(x) # [B, d]
        
        # Handle active mask for pre_alloc mode
        if self.adc_mode == 'pre_alloc':
            # We must mask out inactive units by setting their log_alpha to -inf
            # so they never get selected in top-k
            mask = self.active_mask
            inf_mask = (~mask).float() * -1e9
            eff_log_alpha = self.log_alpha + inf_mask
            
            eff_mu = self.mu
            eff_log_s = self.log_s
            eff_W = self.W
        else:
            eff_log_alpha = self.log_alpha
            eff_mu = self.mu
            eff_log_s = self.log_s
            eff_W = self.W

        # 1. Diagonal Mahalanobis (Log-space, highly stable)
        diff = z.unsqueeze(1) - eff_mu.unsqueeze(0) # [B, K, d]
        s_sq = torch.exp(2 * eff_log_s) + self.eps  # [K, d]
        mahalanobis_sq = ((diff ** 2) / s_sq).sum(dim=-1) # [B, K]
        
        # 2. Log-weights & Top-K
        log_w = eff_log_alpha - (0.5 / self.tau) * mahalanobis_sq # [B, K]
        
        # Save log_w for entropy loss if needed
        self.last_log_w = log_w
        
        k_actual = min(self.top_k, log_w.shape[1])
        topk_vals, topk_idx = torch.topk(log_w, k_actual, dim=-1) # [B, k_actual]
        
        # 3. Blend
        weights = F.softmax(topk_vals, dim=-1).unsqueeze(-1) # [B, k_actual, 1]
        
        # Save weights for entropy regularizer
        self.last_weights = weights.squeeze(-1) # [B, k_actual]
        
        W_topk = eff_W[topk_idx] # [B, k_actual, d, d]
        
        # z is [B, d]. We need to apply W_topk to z. W_topk is [B, k_actual, d, d].
        local_out = torch.einsum('bd,bkdo->bko', z, W_topk) # [B, k_actual, d]
        blended = (weights * local_out).sum(dim=1) # [B, d]
        
        # 4. Residual + Up
        return self.p_up(blended + self.gamma * z)

    def compute_entropy_loss(self):
        if not hasattr(self, 'last_weights'):
            return torch.tensor(0.0, device=self.mu.device)
        # Entropy bonus: + \lambda \sum w_i \log(w_i)
        # To maximize entropy (prevent a single unit hoarding), we minimize \sum w_i \log(w_i)
        # Adding a small eps to avoid log(0)
        w = self.last_weights
        entropy = - (w * torch.log(w + 1e-8)).sum(dim=-1).mean()
        return -entropy # we return the negative entropy because we want to minimize the returned value (minimize negative entropy = maximize entropy)
        # wait, the prompt says "+ \lambda \sum w_i \log(w_i) to prevent a single unit from hoarding".
        # \sum w_i \log(w_i) is negative entropy. By adding it to the loss, we minimize negative entropy, which maximizes entropy!
        # So we just return (w * torch.log(w + 1e-8)).sum(dim=-1).mean()
        
    @torch.no_grad()
    def adapt_density(self, split_thresh=0.05, prune_thresh=0.01, optimizer=None):
        """Adaptive Density Control: Split high-gradient units, prune low-opacity units."""
        if not hasattr(self, 'grad_mu_norm_ema'):
            return 0, 0 # No gradients tracked yet
            
        alpha = torch.sigmoid(self.log_alpha)
        grad_norm = self.grad_mu_norm_ema
        
        if self.adc_mode == 'pre_alloc':
            return self._adapt_density_pre_alloc(alpha, grad_norm, split_thresh, prune_thresh, optimizer)
        else:
            return self._adapt_density_dynamic(alpha, grad_norm, split_thresh, prune_thresh, optimizer)

    def _adapt_density_pre_alloc(self, alpha, grad_norm, split_thresh, prune_thresh, optimizer):
        active_indices = self.active_mask.nonzero().squeeze(-1)
        if active_indices.numel() == 0: return 0, 0
        
        # 1. Prune
        prune_mask = alpha < prune_thresh
        # Don't prune if it leaves us with 0 units
        prune_mask = prune_mask & self.active_mask
        
        num_pruned = prune_mask.sum().item()
        self.active_mask[prune_mask] = False
        
        # Reset properties of pruned units so they don't mess with EMA next time
        self.grad_mu_norm_ema[prune_mask] = 0.0
        
        active_indices = self.active_mask.nonzero().squeeze(-1)
        if active_indices.numel() == 0: return num_pruned, 0
        
        # 2. Split
        # scale s_i = exp(log_s_i)
        max_s = torch.exp(self.log_s).max(dim=-1).values
        
        # Identify units to split: active AND grad_norm > thresh AND max(scale) > thresh
        # (We use split_thresh for scale as well based on prompt: "If m_i > thresh and max(s_i) > thresh")
        split_candidates = (grad_norm > split_thresh) & (max_s > split_thresh) & self.active_mask
        split_indices = split_candidates.nonzero().squeeze(-1)
        
        num_split = 0
        
        inactive_indices = (~self.active_mask).nonzero().squeeze(-1)
        
        for idx in split_indices:
            if len(inactive_indices) == 0:
                break # out of pre-allocated slots
            
            new_idx = inactive_indices[0]
            inactive_indices = inactive_indices[1:]
            
            # Duplicate
            self.active_mask[new_idx] = True
            
            # Add small Gaussian noise to mu
            noise = torch.randn_like(self.mu[idx]) * 0.01 * torch.exp(self.log_s[idx])
            self.mu.data[new_idx] = self.mu.data[idx] + noise
            self.mu.data[idx] = self.mu.data[idx] - noise # adjust original too (optional, but good practice)
            
            # Halve the scale: log(s / 2) = log(s) - log(2)
            self.log_s.data[new_idx] = self.log_s.data[idx] - math.log(2.0)
            self.log_s.data[idx] = self.log_s.data[idx] - math.log(2.0)
            
            # Copy other params
            self.log_alpha.data[new_idx] = self.log_alpha.data[idx]
            self.W.data[new_idx] = self.W.data[idx]
            
            # Reset EMA for both
            self.grad_mu_norm_ema[idx] = 0.0
            self.grad_mu_norm_ema[new_idx] = 0.0
            
            # If optimizer is provided, we need to zero out momentum for the new unit
            # to avoid it being dragged by old momentum
            if optimizer is not None:
                for param, state_idx in [(self.mu, new_idx), (self.log_s, new_idx), 
                                         (self.log_alpha, new_idx), (self.W, new_idx)]:
                    state = optimizer.state.get(param, None)
                    if state is not None:
                        if 'exp_avg' in state:
                            state['exp_avg'][state_idx].zero_()
                        if 'exp_avg_sq' in state:
                            state['exp_avg_sq'][state_idx].zero_()
                            
            num_split += 1
            
        return num_pruned, num_split

    def _adapt_density_dynamic(self, alpha, grad_norm, split_thresh, prune_thresh, optimizer):
        """Option B: Dynamic parameter resizing."""
        if grad_norm is None: return 0, 0
        
        device = self.mu.device
        
        # 1. Prune
        prune_mask = alpha < prune_thresh
        # Keep at least 1 unit
        if prune_mask.all():
            prune_mask[0] = False
            
        keep_mask = ~prune_mask
        num_pruned = prune_mask.sum().item()
        
        # 2. Split
        max_s = torch.exp(self.log_s).max(dim=-1).values
        # Only consider units that are kept for splitting
        split_mask = (grad_norm > split_thresh) & (max_s > split_thresh) & keep_mask
        num_split = split_mask.sum().item()
        
        if num_pruned == 0 and num_split == 0:
            return 0, 0
            
        with torch.no_grad():
            # Build new tensors
            kept_indices = keep_mask.nonzero().squeeze(-1)
            split_indices = split_mask.nonzero().squeeze(-1)
            
            # For split, we duplicate
            new_k = kept_indices.shape[0] + split_indices.shape[0]
            
            new_mu = torch.empty((new_k, self.d_latent), device=device)
            new_log_s = torch.empty((new_k, self.d_latent), device=device)
            new_log_alpha = torch.empty(new_k, device=device)
            new_W = torch.empty((new_k, self.d_latent, self.d_latent), device=device)
            new_grad_ema = torch.zeros(new_k, device=device) # Reset EMA
            
            # Copy kept
            kept_idx_map = {} # old_idx -> new_idx
            for i, old_idx in enumerate(kept_indices):
                new_mu[i] = self.mu[old_idx]
                new_log_s[i] = self.log_s[old_idx]
                new_log_alpha[i] = self.log_alpha[old_idx]
                new_W[i] = self.W[old_idx]
                # we do NOT copy grad ema, we reset it or decay it? Prompt: reset is safe.
                kept_idx_map[old_idx.item()] = i
                
            # Process splits
            curr_idx = len(kept_indices)
            for old_idx in split_indices:
                mapped_idx = kept_idx_map[old_idx.item()]
                
                # new unit
                new_idx = curr_idx
                
                noise = torch.randn_like(new_mu[mapped_idx]) * 0.01 * torch.exp(new_log_s[mapped_idx])
                new_mu[new_idx] = new_mu[mapped_idx] + noise
                new_mu[mapped_idx] = new_mu[mapped_idx] - noise
                
                # Halve scale
                new_log_s[new_idx] = new_log_s[mapped_idx] - math.log(2.0)
                new_log_s[mapped_idx] = new_log_s[mapped_idx] - math.log(2.0)
                
                new_log_alpha[new_idx] = new_log_alpha[mapped_idx]
                new_W[new_idx] = new_W[mapped_idx]
                
                curr_idx += 1
                
            # Now we must update nn.Parameter data (in-place replacement is tricky because shapes change).
            # We can change .data directly if we also update the optimizer state
            self.mu.data = new_mu
            self.log_s.data = new_log_s
            self.log_alpha.data = new_log_alpha
            self.W.data = new_W
            self.grad_mu_norm_ema = new_grad_ema
            
            # Updating optimizer state
            if optimizer is not None:
                # This requires deeply modifying optimizer dicts
                for p in [self.mu, self.log_s, self.log_alpha, self.W]:
                    state = optimizer.state.get(p, None)
                    if state is not None:
                        if 'exp_avg' in state:
                            # We can't easily resize the existing tensor in place if shapes don't match exactly?
                            # Wait, we can re-assign state['exp_avg']
                            old_exp_avg = state['exp_avg']
                            new_exp_avg = torch.zeros_like(p.data)
                            
                            # copy kept
                            for i, old_idx in enumerate(kept_indices):
                                new_exp_avg[i] = old_exp_avg[old_idx]
                            
                            state['exp_avg'] = new_exp_avg
                            
                        if 'exp_avg_sq' in state:
                            old_exp_avg_sq = state['exp_avg_sq']
                            new_exp_avg_sq = torch.zeros_like(p.data)
                            for i, old_idx in enumerate(kept_indices):
                                new_exp_avg_sq[i] = old_exp_avg_sq[old_idx]
                            state['exp_avg_sq'] = new_exp_avg_sq
                            
        return num_pruned, num_split

    def get_num_active(self):
        if self.adc_mode == 'pre_alloc':
            return self.active_mask.sum().item()
        return self.mu.shape[0]

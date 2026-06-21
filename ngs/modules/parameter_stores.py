"""Parameter storage implementations for NGS."""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from ngs.core.interfaces import NGSConfig, ParameterStorage, BaseParameterStore


class DirectAdapterStore(BaseParameterStore):
    """Direct adapter parameter storage (original LeanNGS)."""

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.use_lora = config.use_lora
        self.lora_rank = config.lora_rank
        
        if self.use_lora:
            self.W_A = nn.Parameter(torch.randn(config.max_k, config.latent_dim, config.lora_rank) * 1e-2)
            self.W_B = nn.Parameter(torch.randn(config.max_k, config.lora_rank, config.latent_dim) * 1e-2)
        else:
            self.W = nn.Parameter(torch.randn(config.max_k, config.latent_dim, config.latent_dim) * 1e-2)

    def forward(self, active_indices: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, K = active_indices.shape
        
        if self.use_lora:
            W_A = self.W_A[active_indices]  # [B, K, d, r]
            W_B = self.W_B[active_indices]  # [B, K, r, d]
            
            Bz = torch.einsum('bkrd,bd->bkr', W_B, z)  # [B, K, r]
            out = torch.einsum('bkdr,bkr->bkd', W_A, Bz)
        else:
            W = self.W[active_indices]  # [B, K, d, d]
            out = torch.einsum('bkdo,bd->bko', W, z)  # [B, K, d]
        
        return out

    def get_parameters_for_indices(self, indices: torch.Tensor):
        if self.use_lora:
            return {'W_A': self.W_A[indices], 'W_B': self.W_B[indices]}
        else:
            return {'W': self.W[indices]}

    def expand_capacity(self, new_max_k: int):
        if new_max_k <= self.max_k:
            return
        
        if self.use_lora:
            new_W_A = torch.randn(new_max_k, self.d_latent, self.lora_rank) * 1e-2
            new_W_B = torch.randn(new_max_k, self.lora_rank, self.d_latent) * 1e-2
            new_W_A[:self.max_k] = self.W_A.data
            new_W_B[:self.max_k] = self.W_B.data
            self.W_A = nn.Parameter(new_W_A)
            self.W_B = nn.Parameter(new_W_B)
        else:
            new_W = torch.randn(new_max_k, self.d_latent, self.d_latent) * 1e-2
            new_W[:self.max_k] = self.W.data
            self.W = nn.Parameter(new_W)
        
        self.max_k = new_max_k


class HypernetworkStore(BaseParameterStore):
    """Hypernetwork-generated parameter storage (CFG-Net).
    
    Two implementations provided:
    - High-performance (default): Direct matrix output with efficient einsum
    - Reference: Explicit batched matmul operations
    """

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.code_dim = config.hypernetwork_code_dim
        self.use_lora = config.use_lora
        self.lora_rank = config.lora_rank if config.use_lora else None
        
        self.codes = nn.Parameter(torch.randn(config.max_k, self.code_dim) * 0.1)
        
        if self.use_lora:
            out_size = self.d_latent * self.lora_rank * 2
        else:
            out_size = self.d_latent * self.d_latent
        
        self.hypernet = nn.Sequential(
            nn.Linear(self.code_dim + config.latent_dim, config.hypernetwork_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hypernetwork_hidden_dim, config.hypernetwork_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hypernetwork_hidden_dim, out_size),
        )

    def forward(self, active_indices: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Apply hypernetwork-generated transformations.
        
        High-performance implementation using batched matmul.
        Reference: Explicit einsum formulation in comments.
        """
        B, K = active_indices.shape
        
        codes = self.codes[active_indices]  # [B, K, code_dim]
        z_expanded = z.unsqueeze(1)  # [B, 1, d] - broadcast across K later
        combined = torch.cat([codes, z_expanded.expand(B, K, -1)], dim=-1)  # [B, K, code_dim + d]
        
        combined_flat = combined.view(B * K, -1)
        weights_flat = self.hypernet(combined_flat)
        
        if self.use_lora:
            r = self.lora_rank
            # Hypernet outputs: [W_A, W_B] concatenated
            weights_A = weights_flat[..., :self.d_latent * r]
            weights_B = weights_flat[..., self.d_latent * r:]
            
            # LoRA: output = W_A @ (W_B @ z)
            # W_A: [B, K, d, r], W_B: [B, K, r, d]
            W_A = weights_A.view(B, K, self.d_latent, r)
            W_B = weights_B.view(B, K, r, self.d_latent)
            
            # Batched matmul: W_B @ z for each (b, k)
            # z_kd: [B, K, d] -> [B*K, d] for batch matmul
            z_kd = z.unsqueeze(1).expand(B, K, -1)  # [B, K, d]
            Bz = torch.bmm(
                W_B.view(B * K, r, self.d_latent),  # [B*K, r, d]
                z_kd.reshape(B * K, self.d_latent, 1)  # [B*K, d, 1]
            ).squeeze(-1)  # [B*K, r]
            Bz = Bz.view(B, K, r)  # [B, K, r]
            
            # Then: W_A @ Bz -> [B, K, d]
            out = torch.bmm(
                W_A.view(B * K, self.d_latent, r),  # [B*K, d, r]
                Bz.view(B * K, r, 1)  # [B*K, r, 1]
            ).squeeze(-1)  # [B*K, d]
            out = out.view(B, K, self.d_latent)
        else:
            W = weights_flat.view(B, K, self.d_latent, self.d_latent)
            z_kd = z.unsqueeze(1).expand(B, K, -1)  # [B, K, d]
            out = torch.bmm(
                W.view(B * K, self.d_latent, self.d_latent),
                z_kd.reshape(B * K, self.d_latent, 1)
            ).squeeze(-1)
            out = out.view(B, K, self.d_latent)
        
        return out

    def get_parameters_for_indices(self, indices: torch.Tensor):
        return {'codes': self.codes[indices]}

    def expand_capacity(self, new_max_k: int):
        if new_max_k <= self.max_k:
            return
        
        new_codes = torch.randn(new_max_k, self.code_dim) * 0.1
        new_codes[:self.max_k] = self.codes.data
        self.codes = nn.Parameter(new_codes)
        self.max_k = new_max_k


class LoRAStore(BaseParameterStore):
    """LoRA-based parameter storage with shared bases."""

    def __init__(self, config: NGSConfig):
        super().__init__(config)
        self.lora_rank = config.lora_rank
        
        # Shared LoRA bases (few bases shared across all units)
        self.num_bases = 8
        self.base_A = nn.Parameter(torch.randn(self.num_bases, config.latent_dim, self.lora_rank) * 1e-2)
        self.base_B = nn.Parameter(torch.randn(self.num_bases, self.lora_rank, config.latent_dim) * 1e-2)
        
        # Per-unit combination coefficients
        self.coeff_A = nn.Parameter(torch.randn(config.max_k, self.num_bases) * 0.1)
        self.coeff_B = nn.Parameter(torch.randn(config.max_k, self.num_bases) * 0.1)
        
        # Optional bias
        self.bias = nn.Parameter(torch.zeros(config.max_k, config.latent_dim))

    def forward(self, active_indices: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, K = active_indices.shape
        
        coeff_A = self.coeff_A[active_indices]  # [B, K, num_bases]
        coeff_B = self.coeff_B[active_indices]  # [B, K, num_bases]
        bias = self.bias[active_indices]  # [B, K, d]
        
        # Combine bases: W_A = sum_i coeff_A[:,:,i] * base_A[i]
        W_A = torch.einsum('bki,ird->bkrd', coeff_A, self.base_A)  # [B, K, d, r]
        W_B = torch.einsum('bki,ird->bkrd', coeff_B, self.base_B)  # [B, K, r, d]
        
        Bz = torch.einsum('bkrd,bd->bkr', W_B, z)  # [B, K, r]
        out = torch.einsum('bkdr,bkr->bkd', W_A, Bz)  # [B, K, d]
        
        out = out + bias
        
        return out

    def get_parameters_for_indices(self, indices: torch.Tensor):
        return {
            'coeff_A': self.coeff_A[indices],
            'coeff_B': self.coeff_B[indices],
            'bias': self.bias[indices],
        }

    def expand_capacity(self, new_max_k: int):
        if new_max_k <= self.max_k:
            return
        
        new_coeff_A = torch.randn(new_max_k, self.num_bases) * 0.1
        new_coeff_B = torch.randn(new_max_k, self.num_bases) * 0.1
        new_bias = torch.zeros(new_max_k, self.d_latent)
        
        new_coeff_A[:self.max_k] = self.coeff_A.data
        new_coeff_B[:self.max_k] = self.coeff_B.data
        new_bias[:self.max_k] = self.bias.data
        
        self.coeff_A = nn.Parameter(new_coeff_A)
        self.coeff_B = nn.Parameter(new_coeff_B)
        self.bias = nn.Parameter(new_bias)
        self.max_k = new_max_k


def build_parameter_store(config: NGSConfig) -> BaseParameterStore:
    """Factory function to build parameter store from config."""
    storage = config.parameter_storage
    
    if storage == ParameterStorage.DIRECT_ADAPTER:
        return DirectAdapterStore(config)
    elif storage == ParameterStorage.HYPERNETWORK_GENERATED:
        return HypernetworkStore(config)
    elif storage == ParameterStorage.LORA:
        return LoRAStore(config)
    else:
        raise ValueError(f"Unknown parameter storage: {storage}")


__all__ = [
    "DirectAdapterStore",
    "HypernetworkStore",
    "LoRAStore",
    "build_parameter_store",
    "BaseParameterStore",
]

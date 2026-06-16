"""Parameter storage implementations for NGS."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ngs.core.interfaces import BaseParameterStore, NGSConfig, ParameterStorage


class DirectAdapterStore(BaseParameterStore):
    """Direct adapter parameter storage (full or LoRA)."""

    def __init__(self, config: NGSConfig):
        super().__init__()
        self.config = config
        self.max_k = config.max_k
        self.d_latent = config.latent_dim
        self.use_lora = config.use_lora
        self.lora_rank = config.lora_rank

        if config.use_lora:
            self.W_A = nn.Parameter(torch.randn(config.max_k, config.latent_dim, config.lora_rank) * 1e-2)
            self.W_B = nn.Parameter(torch.randn(config.max_k, config.lora_rank, config.latent_dim) * 1e-2)
        else:
            self.W = nn.Parameter(torch.randn(config.max_k, config.latent_dim, config.latent_dim) * 1e-2)

    def forward(self, indices: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, K = indices.shape

        if self.use_lora:
            W_A = self.W_A[indices]  # [B, K, d, r]
            W_B = self.W_B[indices]  # [B, K, r, d]

            Bz = torch.einsum('bkrd,bd->bkr', W_B, z)
            out = torch.einsum('bkdr,bkr->bkd', W_A, Bz)
        else:
            W = self.W[indices]  # [B, K, d, d]
            out = torch.einsum('bkdo,bd->bko', W, z)

        return out

    def get_parameters(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.use_lora:
            return {'W_A': self.W_A[indices], 'W_B': self.W_B[indices]}
        else:
            return {'W': self.W[indices]}

    def init_unit(self, index: int, source_index: Optional[int] = None) -> None:
        with torch.no_grad():
            if source_index is not None:
                if self.use_lora:
                    self.W_A[index].copy_(self.W_A[source_index])
                    self.W_B[index].copy_(self.W_B[source_index])
                else:
                    self.W[index].copy_(self.W[source_index])
            else:
                if self.use_lora:
                    nn.init.kaiming_uniform_(self.W_A[index], a=5**0.5)
                    nn.init.zeros_(self.W_B[index])
                else:
                    nn.init.xavier_uniform_(self.W[index])

    def merge_units(self, target_idx: int, source_idx: int, weight: float = 0.5) -> None:
        with torch.no_grad():
            if self.use_lora:
                self.W_A[target_idx].lerp_(self.W_A[source_idx], weight)
                self.W_B[target_idx].lerp_(self.W_B[source_idx], weight)
            else:
                self.W[target_idx].lerp_(self.W[source_idx], weight)


class HypernetworkStore(BaseParameterStore):
    """Hypernetwork-generated parameter storage."""

    def __init__(self, config: NGSConfig):
        super().__init__()
        self.config = config
        self.max_k = config.max_k
        self.d_latent = config.latent_dim
        self.code_dim = config.hypernetwork_code_dim
        self.use_lora = config.use_lora

        # Latent codes for each unit
        self.codes = nn.Parameter(torch.randn(config.max_k, config.hypernetwork_code_dim) * 0.1)

        # Shared hypernetwork
        if config.use_lora:
            out_size = config.latent_dim * config.lora_rank * 2
            self.lora_rank = config.lora_rank
        else:
            out_size = config.latent_dim * config.latent_dim
            self.lora_rank = None

        self.hypernet = nn.Sequential(
            nn.Linear(config.hypernetwork_code_dim + config.latent_dim, config.hypernetwork_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hypernetwork_hidden_dim, config.hypernetwork_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hypernetwork_hidden_dim, out_size),
        )

    def forward(self, indices: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, K = indices.shape

        codes = self.codes[indices]  # [B, K, code_dim]
        z_expanded = z.unsqueeze(1).expand(B, K, -1)  # [B, K, d]
        combined = torch.cat([codes, z_expanded], dim=-1)  # [B, K, code_dim + d]

        combined_flat = combined.view(B * K, -1)
        weights_flat = self.hypernet(combined_flat)

        if self.use_lora:
            r = self.lora_rank
            W_A = weights_flat.view(B, K, self.d_latent, r * 2)
            W_A_half = W_A[..., :r]
            W_B_half = W_A[..., r:]

            Bz = torch.einsum('bkrd,bd->bkr', W_B_half.transpose(-1, -2), z)
            out = torch.einsum('bkdr,bkr->bkd', W_A_half, Bz)
        else:
            W = weights_flat.view(B, K, self.d_latent, self.d_latent)
            out = torch.einsum('bkdo,bd->bko', W, z)

        return out

    def get_parameters(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        return {'codes': self.codes[indices]}

    def init_unit(self, index: int, source_index: Optional[int] = None) -> None:
        with torch.no_grad():
            if source_index is not None:
                self.codes[index].copy_(self.codes[source_index])
            else:
                nn.init.normal_(self.codes[index], mean=0.0, std=0.1)

    def merge_units(self, target_idx: int, source_idx: int, weight: float = 0.5) -> None:
        with torch.no_grad():
            self.codes[target_idx].lerp_(self.codes[source_idx], weight)


class LoRAStore(BaseParameterStore):
    """Pure LoRA storage with optional hypernetwork code generation."""

    def __init__(self, config: NGSConfig):
        super().__init__()
        self.config = config
        self.max_k = config.max_k
        self.d_latent = config.latent_dim
        self.lora_rank = config.lora_rank

        self.lora_A = nn.Parameter(torch.randn(config.max_k, config.latent_dim, config.lora_rank) * 1e-2)
        self.lora_B = nn.Parameter(torch.randn(config.max_k, config.lora_rank, config.latent_dim) * 1e-2)

        # Optional: hypernetwork for generating LoRA from codes
        if config.parameter_storage == ParameterStorage.HYPERNETWORK:
            self.codes = nn.Parameter(torch.randn(config.max_k, config.hypernetwork_code_dim) * 0.1)
            self.lora_generator = nn.Sequential(
                nn.Linear(config.hypernetwork_code_dim, config.hypernetwork_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hypernetwork_hidden_dim, config.latent_dim * config.lora_rank * 2),
            )
        else:
            self.codes = None
            self.lora_generator = None

    def forward(self, indices: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, K = indices.shape

        if self.codes is not None:
            codes = self.codes[indices]
            lora_params = self.lora_generator(codes)  # [B, K, d*r*2]
            r = self.lora_rank
            W_A = lora_params[..., :self.d_latent * r].view(B, K, self.d_latent, r)
            W_B = lora_params[..., self.d_latent * r:].view(B, K, r, self.d_latent)
        else:
            W_A = self.lora_A[indices]
            W_B = self.lora_B[indices]

        Bz = torch.einsum('bkrd,bd->bkr', W_B, z)
        out = torch.einsum('bkdr,bkr->bkd', W_A, Bz)

        return out

    def get_parameters(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.codes is not None:
            return {'codes': self.codes[indices]}
        return {'lora_A': self.lora_A[indices], 'lora_B': self.lora_B[indices]}

    def init_unit(self, index: int, source_index: Optional[int] = None) -> None:
        with torch.no_grad():
            if source_index is not None:
                if self.codes is not None:
                    self.codes[index].copy_(self.codes[source_index])
                else:
                    self.lora_A[index].copy_(self.lora_A[source_index])
                    self.lora_B[index].copy_(self.lora_B[source_index])
            else:
                if self.codes is not None:
                    nn.init.normal_(self.codes[index], mean=0.0, std=0.1)
                else:
                    nn.init.kaiming_uniform_(self.lora_A[index], a=5**0.5)
                    nn.init.zeros_(self.lora_B[index])

    def merge_units(self, target_idx: int, source_idx: int, weight: float = 0.5) -> None:
        with torch.no_grad():
            if self.codes is not None:
                self.codes[target_idx].lerp_(self.codes[source_idx], weight)
            else:
                self.lora_A[target_idx].lerp_(self.lora_A[source_idx], weight)
                self.lora_B[target_idx].lerp_(self.lora_B[source_idx], weight)


def build_parameter_store(config: NGSConfig) -> BaseParameterStore:
    """Factory function to build parameter store from config."""
    if config.parameter_storage == ParameterStorage.DIRECT:
        return DirectAdapterStore(config)
    elif config.parameter_storage == ParameterStorage.HYPERNETWORK:
        return HypernetworkStore(config)
    elif config.parameter_storage == ParameterStorage.LORA:
        return LoRAStore(config)
    else:
        raise ValueError(f"Unknown parameter storage: {config.parameter_storage}")
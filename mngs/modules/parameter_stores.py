"""Parameter storage implementations for MNGS."""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseParameterStore(nn.Module, ABC):
    """Base class for all parameter storage strategies."""
    
    @abstractmethod
    def forward(self, active_indices: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Generate or retrieve transformations for active units.
        
        Args:
            active_indices: [B, K] indices of active units
            x: [B, d] input latent features
            
        Returns:
            [B, K, d_latent] transformed features for each active unit
        """
        pass
    
    @abstractmethod
    def get_parameters_for_indices(self, indices: torch.Tensor):
        """Get transformation parameters for specific units."""
        pass


class DirectAdapterStore(BaseParameterStore):
    """
    Direct adapter parameter storage (original LeanNGS).
    
    Each unit stores its own full adapter weights.
    Memory: O(max_k * d^2)
    """
    
    def __init__(self, max_k: int, d_latent: int, use_lora: bool = True, lora_rank: int = 4):
        super().__init__()
        self.max_k = max_k
        self.d_latent = d_latent
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        
        if use_lora:
            # LoRA-style: W = A @ B where A: [d, r], B: [r, d]
            self.W_A = nn.Parameter(torch.randn(max_k, d_latent, lora_rank) * 1e-2)
            self.W_B = nn.Parameter(torch.randn(max_k, lora_rank, d_latent) * 1e-2)
        else:
            # Full weight matrices
            self.W = nn.Parameter(torch.randn(max_k, d_latent, d_latent) * 1e-2)
    
    def forward(self, active_indices: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Apply direct adapter transformations.
        
        Args:
            active_indices: [B, K] unit indices
            z: [B, d_latent] input features
            
        Returns:
            [B, K, d_latent] transformed outputs
        """
        B, K = active_indices.shape
        
        if self.use_lora:
            # Gather LoRA weights for active units
            W_A = self.W_A[active_indices]  # [B, K, d, r]
            W_B = self.W_B[active_indices]  # [B, K, r, d]
            
            # B @ z: [B, K, r, d] @ [B, d] -> [B, K, r]
            Bz = torch.einsum('bkrd,bd->bkr', W_B, z)  # [B, K, r]
            
            # A @ (Bz): [B, K, d, r] @ [B, K, r] -> [B, K, d]
            out = torch.einsum('bkdr,bkr->bkd', W_A, Bz)
        else:
            # Full matrix multiply
            W = self.W[active_indices]  # [B, K, d, d]
            out = torch.einsum('bkdo,bd->bko', W, z)  # [B, K, d]
        
        return out
    
    def get_parameters_for_indices(self, indices: torch.Tensor):
        """Get parameters for specific unit indices."""
        if self.use_lora:
            return {
                'W_A': self.W_A[indices],
                'W_B': self.W_B[indices],
            }
        else:
            return {
                'W': self.W[indices],
            }


class HypernetworkStore(BaseParameterStore):
    """
    Hypernetwork-generated parameter storage (CFG-Net).
    
    Units store tiny latent codes; a shared MLP generates weights.
    Memory: O(max_k * code_dim)
    """
    
    def __init__(self, max_k: int, d_latent: int, code_dim: int = 8, 
                 hidden_dim: int = 16, use_lora: bool = True):
        super().__init__()
        self.max_k = max_k
        self.d_latent = d_latent
        self.code_dim = code_dim
        self.use_lora = use_lora
        
        # Latent codes for each unit
        self.codes = nn.Parameter(torch.randn(max_k, code_dim) * 0.1)
        
        # Shared hypernetwork: generates weights from code + latent input
        if use_lora:
            lora_rank = 4  # Default LoRA rank
            out_size = d_latent * lora_rank * 2
            self.lora_rank = lora_rank
        else:
            out_size = d_latent * d_latent
            self.lora_rank = None
        
        self.hypernet = nn.Sequential(
            nn.Linear(code_dim + d_latent, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_size),
        )
    
    def forward(self, active_indices: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Generate and apply hypernetwork weights.
        
        Args:
            active_indices: [B, K] unit indices
            z: [B, d_latent] input features
            
        Returns:
            [B, K, d_latent] transformed outputs
        """
        B, K = active_indices.shape
        
        # Gather codes for active units
        codes = self.codes[active_indices]  # [B, K, code_dim]
        
        # Expand input for each active unit
        z_expanded = z.unsqueeze(1).expand(B, K, -1)  # [B, K, d]
        
        # Combine code and input
        combined = torch.cat([codes, z_expanded], dim=-1)  # [B, K, code_dim + d]
        
        # Generate weights via hypernetwork
        # Need to process each (B, K) element
        combined_flat = combined.view(B * K, -1)
        weights_flat = self.hypernet(combined_flat)  # [B*K, out_size]
        
        if self.use_lora:
            # Reshape into LoRA components
            r = self.lora_rank
            W_A = weights_flat.view(B, K, self.d_latent, r * 2)
            W_A_half = W_A[..., :r]  # [B, K, d, r]
            W_B_half = W_A[..., r:]  # [B, K, d, r]
            
            # Apply: A @ (B.T @ z)
            # B: [B, K, d, r], B.T: [B, K, r, d]
            Bz = torch.einsum('bkrd,bd->bkr', W_B_half.transpose(-1, -2), z)  # [B, K, r]
            out = torch.einsum('bkdr,bkr->bkd', W_A_half, Bz)
        else:
            W = weights_flat.view(B, K, self.d_latent, self.d_latent)
            out = torch.einsum('bkdo,bd->bko', W, z)
        
        return out
    
    def get_parameters_for_indices(self, indices: torch.Tensor):
        """Get latent codes for specific unit indices."""
        return {
            'codes': self.codes[indices],
        }


def build_parameter_store(config, max_k: int = None, d_latent: int = None):
    """Factory function to build parameter store from config."""
    from mngs.core.config import ParameterStorage
    
    storage = config.parameter_storage if hasattr(config, 'parameter_storage') else config
    max_k = max_k or config.max_k
    d_latent = d_latent or config.latent_dim
    use_lora = getattr(config, 'use_lora', True)
    lora_rank = getattr(config, 'lora_rank', 4)
    code_dim = getattr(config, 'hypernetwork_code_dim', 8)
    hidden_dim = getattr(config, 'hypernetwork_hidden_dim', 16)
    
    if storage == ParameterStorage.DIRECT_ADAPTER:
        return DirectAdapterStore(
            max_k=max_k,
            d_latent=d_latent,
            use_lora=use_lora,
            lora_rank=lora_rank
        )
    elif storage == ParameterStorage.HYPERNETWORK_GENERATED:
        return HypernetworkStore(
            max_k=max_k,
            d_latent=d_latent,
            code_dim=code_dim,
            hidden_dim=hidden_dim,
            use_lora=use_lora
        )
    else:
        raise ValueError(f"Unknown parameter storage: {storage}")

"""Dynamic Head module for class-incremental learning.

A DynamicHead wraps an NGS model and manages active class masking
for class-incremental scenarios where new classes arrive over time.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, List, Union

from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement


def _build_ngs(d_in: int, d_out: int, config: NGSConfig):
    """Lazy import to avoid circular dependency."""
    from ngs.models.ngs import build_ngs as _build_ngs_impl
    return _build_ngs_impl(d_in, d_out, config)


def default_dynamic_config() -> NGSConfig:
    """Default configuration for DynamicHead NGS."""
    return NGSConfig(
        latent_dim=64,
        k_init=32,
        max_k=256,
        top_k=8,
        top_k_factorized=2,
        num_subspaces=4,
        routing=RoutingStrategy.FACTORIZED_SUBSPACE,
        parameter_storage=ParameterStorage.LORA,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.PRE_ALLOCATED,
        use_lora=True,
        lora_rank=4,
        tau=1.0,
        gamma_residual=0.1,
        ema_decay=0.99,
    )


class DynamicHead(nn.Module):
    """Dynamic classification head for class-incremental learning.

    Wraps an NGS model and maintains an active class mask to support
    incremental addition of new classes without catastrophic forgetting.

    Args:
        d_latent: Input latent dimension from backbone
        max_classes: Maximum number of classes to support
        config: Optional NGSConfig. If None, uses default_dynamic_config()

    Example:
        head = DynamicHead(d_latent=512, max_classes=100)
        # After task 1 (classes 0-9):
        head.add_classes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        logits = head(features)  # only classes 0-9 active
        # After task 2 (classes 10-19):
        head.add_classes(list(range(10, 20)))
        logits = head(features)  # classes 0-19 active
    """

    def __init__(
        self,
        d_latent: int,
        max_classes: int = 200,
        config: Optional[NGSConfig] = None,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.max_classes = max_classes

        self.ngs = _build_ngs(d_latent, max_classes, config or default_dynamic_config())

        self.register_buffer("active_mask", torch.zeros(max_classes, dtype=torch.bool))
        self.register_buffer("classes_seen", torch.tensor(0, dtype=torch.long))

    def add_classes(self, class_ids: Union[List[int], torch.Tensor, int]) -> None:
        """Activate new classes in the head.

        Args:
            class_ids: Single class index, list of indices, or tensor of indices
        """
        if isinstance(class_ids, int):
            class_ids = [class_ids]
        elif isinstance(class_ids, torch.Tensor):
            class_ids = class_ids.tolist()

        self.active_mask[class_ids] = True
        self.classes_seen.fill_(int(self.active_mask.sum().item()))

    def remove_classes(self, class_ids: Union[List[int], torch.Tensor, int]) -> None:
        """Deactivate classes (for debugging/ablation)."""
        if isinstance(class_ids, int):
            class_ids = [class_ids]
        elif isinstance(class_ids, torch.Tensor):
            class_ids = class_ids.tolist()

        self.active_mask[class_ids] = False
        self.classes_seen.fill_(int(self.active_mask.sum().item()))

    def reset(self) -> None:
        """Reset all classes to inactive."""
        self.active_mask.zero_()
        self.classes_seen.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with active class masking.

        Args:
            x: [B, d_latent] input features from backbone

        Returns:
            [B, max_classes] logits with inactive classes masked to -inf
        """
        out = self.ngs(x)
        logits = out.logits

        # Mask inactive classes to large negative value
        logits = logits.clone()
        logits[:, ~self.active_mask] = -1e9

        return logits

    @property
    def num_active_classes(self) -> int:
        """Number of currently active classes."""
        return int(self.active_mask.sum().item())

    @property
    def active_class_indices(self) -> torch.Tensor:
        """Indices of active classes."""
        return self.active_mask.nonzero(as_tuple=True)[0]

    def extra_repr(self) -> str:
        return f'd_latent={self.d_latent}, max_classes={self.max_classes}, active={self.num_active_classes}'


def build_dynamic_head(
    d_latent: int,
    max_classes: int = 200,
    config: Optional[NGSConfig] = None,
) -> DynamicHead:
    """Factory function for DynamicHead.

    Args:
        d_latent: Input latent dimension
        max_classes: Maximum number of classes
        config: Optional NGSConfig

    Returns:
        Configured DynamicHead module
    """
    return DynamicHead(d_latent, max_classes, config)


__all__ = [
    "DynamicHead",
    "build_dynamic_head",
    "default_dynamic_config",
]
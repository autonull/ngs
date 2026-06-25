"""
NGS-specific Model Inspector for EP.

Extends bioplausible's ModelInspector to recognize NGS modules
(router, param_store) as state-producing layers.
"""

from typing import Any, Dict, List, Optional
import torch.nn as nn

from ngs.modules.routers import MonolithicRouter
from ngs.modules.parameter_stores import DirectAdapterStore


class NGSModelInspector:
    """
    Extracts sequence of layers from NGS model for EP state tracking.
    
    Recognizes:
    - p_down, p_up as 'layer' (projection layers)
    - router (MonolithicRouter) as 'layer' (stateful Gaussian states)
    - param_store (DirectAdapterStore) as 'layer' (parameter adaptation)
    """

    def __init__(self) -> None:
        self._cache: Dict[int, List[Dict[str, Any]]] = {}

    def inspect(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Extract model structure for EP."""
        model_id = id(model)
        if model_id in self._cache:
            return self._cache[model_id]

        structure: List[Dict[str, Any]] = []
        self._inspect_recursive(model, structure)

        self._cache[model_id] = structure
        return structure

    def _inspect_recursive(
        self, module: nn.Module, structure: List[Dict[str, Any]]
    ) -> None:
        item_type = self._get_module_type(module)

        if item_type:
            structure.append({"type": item_type, "module": module})
            return

        for child in module.children():
            self._inspect_recursive(child, structure)

    def _get_module_type(self, m: nn.Module) -> Optional[str]:
        """Determine module type for EP."""
        # Standard layers
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return "layer"

        # NGS-specific modules
        if isinstance(m, MonolithicRouter):
            return "layer"  # Router has Gaussian states (mu, log_s, log_alpha)
        
        if isinstance(m, DirectAdapterStore):
            return "layer"  # Param store adapts Gaussian parameters

        # Activations
        if isinstance(m, (
            nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid,
            nn.LeakyReLU, nn.ELU, nn.SELU, nn.Mish,
        )):
            return "act"

        return None


# Global instance
ngs_inspector = NGSModelInspector()
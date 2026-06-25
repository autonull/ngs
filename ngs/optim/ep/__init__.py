"""
EP Optimizer for NGS - Unified Equilibrium Propagation with EWC support.

Usage:
    from ngs.optim.ep import EPOptimizer

    # Fast EP (default settings)
    opt = EPOptimizer(model.parameters(), model=model)

    # EP with EWC for continual learning
    opt = EPOptimizer(model.parameters(), model=model, ewc_lambda=100)

    # Backprop (for comparison)
    opt = EPOptimizer(model.parameters(), model=model, mode='backprop')
"""

from .ep_optimizer import EPOptimizer, smep, smep_fast, sdmep, local_ep, natural_ep, muon_backprop

__all__ = [
    "EPOptimizer",
    "smep",
    "smep_fast",
    "sdmep",
    "local_ep",
    "natural_ep",
    "muon_backprop",
]
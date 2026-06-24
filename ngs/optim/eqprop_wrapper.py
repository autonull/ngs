"""
Thin wrapper to import EqProp components from bioplausible.
bioplausible is at /home/me/ngs/bioplausible/mep
"""
import sys
sys.path.insert(0, '/home/me/ngs/bioplausible/mep')

# Core EP optimizer with smep/smep_fast/muon_backprop presets
from mep.optimizers import EPOptimizer, smep, smep_fast, muon_backprop, EWCState

# Spectral constraint for contraction guarantee
from mep.optimizers.strategies import SpectralConstraint, SettlingSpectralPenalty

__all__ = [
    "EPOptimizer",
    "smep",
    "smep_fast", 
    "muon_backprop",
    "SpectralConstraint",
    "SettlingSpectralPenalty",
    "EWCState",
]


def get_ep_optimizer(model, preset='smep_fast', **kwargs):
    """
    Get EP optimizer with preset.
    
    Args:
        model: NGS model (or any nn.Module)
        preset: 'smep' (30 steps, high acc) | 'smep_fast' (10 steps, fast) | 'muon_backprop'
        **kwargs: overrides for lr, beta, settle_steps, etc.
    """
    if preset == 'smep':
        return smep(model.parameters(), model=model, **kwargs)
    elif preset == 'smep_fast':
        return smep_fast(model.parameters(), model=model, **kwargs)
    elif preset == 'muon_backprop':
        return muon_backprop(model.parameters(), model=model, **kwargs)
    else:
        raise ValueError(f"Unknown preset: {preset}. Available: smep, smep_fast, muon_backprop")


def add_spectral_constraint(model, gamma=0.95, timing='post_update'):
    """
    Add SpectralConstraint to all 2D+ parameters in model.
    Returns list of constraint objects for manual enforcement.
    """
    constraints = []
    for name, param in model.named_parameters():
        if param.ndim >= 2:
            constraint = SpectralConstraint(gamma=gamma, timing=timing)
            constraints.append((name, param, constraint))
    return constraints
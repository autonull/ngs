"""
Thin wrapper to import EqProp components from bioplausible.
"""
import sys
from pathlib import Path

# Try to find bioplausible relative to this file
_bioplausible_path = Path(__file__).parent.parent.parent / 'bioplausible' / 'mep'
if _bioplausible_path.exists():
    sys.path.insert(0, str(_bioplausible_path))
else:
    # Fallback to common locations
    for p in ['/home/me/ngs/bioplausible/mep', '/home/me/bioplausible/mep', '../bioplausible/mep']:
        if Path(p).exists():
            sys.path.insert(0, p)
            break

# Core EP optimizer with smep/smep_fast/muon_backprop presets
from mep.optimizers import EPOptimizer, smep, smep_fast, muon_backprop, EWCState, EWCRegularizer

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
    "EWCRegularizer",
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


def add_settling_spectral_penalty(model, gamma=0.95, lambda_penalty=1.0):
    """Add SettlingSpectralPenalty for soft spectral constraint during settling."""
    return SettlingSpectralPenalty(gamma=gamma, lambda_penalty=lambda_penalty)


def create_ewc_regularizer(model, ewc_lambda=100.0):
    """Create EWC regularizer for continual learning."""
    return EWCRegularizer(model, ewc_lambda=ewc_lambda)
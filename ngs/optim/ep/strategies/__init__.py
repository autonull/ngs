"""Strategy implementations for EP optimizer."""

from .base import GradientStrategy, UpdateStrategy, ConstraintStrategy, FeedbackStrategy
from .constraint import NoConstraint, SpectralConstraint
from .update import MuonUpdate, DionUpdate, PlainUpdate
from .gradient import BackpropGradient, EPGradient, LocalEPGradient, NaturalGradient
from .feedback import NoFeedback, ErrorFeedback

__all__ = [
    "GradientStrategy",
    "UpdateStrategy", 
    "ConstraintStrategy",
    "FeedbackStrategy",
    "NoConstraint",
    "SpectralConstraint",
    "MuonUpdate",
    "DionUpdate",
    "PlainUpdate",
    "BackpropGradient",
    "EPGradient",
    "LocalEPGradient",
    "NaturalGradient",
    "NoFeedback",
    "ErrorFeedback",
]
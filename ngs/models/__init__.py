"""NGS Model definitions."""

from mngs.model import MNGS as NGSModel, build_mngs as build_ngs
from .llm_wrapper import LLMWrapper, LLMWrapperConfig, build_llm_wrapper

__all__ = [
    "NGSModel",
    "build_ngs",
    "LLMWrapper",
    "LLMWrapperConfig",
    "build_llm_wrapper",
]
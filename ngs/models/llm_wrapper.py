"""LLM Wrapper: Frozen LLM + NGS Residual Adapters ('Liquefaction')."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMWrapperConfig:
    """Configuration for LLM Wrapper."""
    model_name: str = "gpt2"
    freeze_base: bool = True
    adapter_layers: List[int] = None
    ngs_latent_dim: int = 64
    ngs_k_init: int = 16
    ngs_max_k: int = 128
    ngs_top_k: int = 4
    ngs_routing: str = "factorized_subspace"
    ngs_param_storage: str = "hypernetwork_generated"
    ngs_topology: str = "continuous_density"
    use_lora: bool = True
    lora_rank: int = 4
    hypernetwork_code_dim: int = 16
    residual_scale: float = 0.1
    gradient_checkpointing: bool = False


class NGSAdapter(nn.Module):
    """NGS-based residual adapter for a single transformer layer."""

    def __init__(self, d_model: int, ngs_config):
        super().__init__()
        self.d_model = d_model
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
        from ngs.models import build_ngs
        self.ngs = build_ngs(d_model, d_model, ngs_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_flat = x.view(B * L, D)
        out = self.ngs(x_flat)
        return x + self.residual_scale * out.view(B, L, D)


class LLMWrapper(nn.Module):
    """Frozen LLM with NGS residual adapters at selected layers."""

    def __init__(self, config: LLMWrapperConfig):
        super().__init__()
        self.config = config
        
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
        except ImportError:
            raise ImportError("transformers required for LLMWrapper. Install with: pip install transformers")
        
        hf_config = AutoConfig.from_pretrained(config.model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.d_model = hf_config.hidden_size
        self.n_layers = hf_config.num_hidden_layers
        
        if config.freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        if config.adapter_layers is None:
            adapter_layers = list(range(self.n_layers))
        else:
            adapter_layers = config.adapter_layers
        
        self.adapter_layers = adapter_layers
        
        from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
        routing_map = {
            "monolithic_mahalanobis": RoutingStrategy.MONOLITHIC_MAHALANOBIS,
            "factorized_subspace": RoutingStrategy.FACTORIZED_SUBSPACE,
            "lsh_approximate": RoutingStrategy.LSH_APPROXIMATE,
        }
        storage_map = {
            "direct_adapter": ParameterStorage.DIRECT_ADAPTER,
            "hypernetwork_generated": ParameterStorage.HYPERNETWORK_GENERATED,
        }
        topology_map = {
            "discrete_heuristic": TopologyControl.DISCRETE_HEURISTIC,
            "continuous_density": TopologyControl.CONTINUOUS_DENSITY,
        }
        memory_map = {
            "dynamic_growth": MemoryManagement.DYNAMIC_GROWTH,
            "pre_allocated_masked": MemoryManagement.PRE_ALLOCATED_MASKED,
            "strict_capacity": MemoryManagement.STRICT_CAPACITY,
        }
        
        ngs_cfg = NGSConfig(
            latent_dim=config.ngs_latent_dim,
            k_init=config.ngs_k_init,
            max_k=config.ngs_max_k,
            top_k=config.ngs_top_k,
            routing=routing_map.get(config.ngs_routing, RoutingStrategy.FACTORIZED_SUBSPACE),
            parameter_storage=storage_map.get(config.ngs_param_storage, ParameterStorage.HYPERNETWORK_GENERATED),
            topology_control=topology_map.get(config.ngs_topology, TopologyControl.CONTINUOUS_DENSITY),
            memory_management=memory_map.get("dynamic_growth", MemoryManagement.DYNAMIC_GROWTH),
            use_lora=config.use_lora,
            lora_rank=config.lora_rank,
            hypernetwork_code_dim=config.hypernetwork_code_dim,
            hypernetwork_hidden_dim=64,
            num_subspaces=4,
            tau=1.0,
            gamma_residual=config.residual_scale,
            ema_decay=0.99,
        )
        
        self.adapters = nn.ModuleDict()
        for layer_idx in adapter_layers:
            self.adapters[str(layer_idx)] = NGSAdapter(self.d_model, ngs_cfg)
        
        if config.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if hasattr(self.base_model, "transformer"):
            # GPT-2 style
            transformer = self.base_model.transformer
            hidden_states = transformer.wte(input_ids)
            if hasattr(transformer, "wpe"):
                pos = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
                hidden_states = hidden_states + transformer.wpe(pos)
            hidden_states = transformer.drop(hidden_states)
            
            for i, block in enumerate(transformer.h):
                if str(i) in self.adapters:
                    residual = hidden_states
                    hidden_states = self.adapters[str(i)](hidden_states)
                    hidden_states = hidden_states + residual
                
                block_outputs = block(hidden_states)
                hidden_states = block_outputs[0] if isinstance(block_outputs, tuple) else block_outputs
            
            hidden_states = transformer.ln_f(hidden_states)
            
            if hasattr(self.base_model, "lm_head"):
                logits = self.base_model.lm_head(hidden_states)
            else:
                logits = hidden_states @ self.base_model.transformer.wte.weight.T
                
        elif hasattr(self.base_model, "model"):
            # LLaMA/Mistral style
            hidden_states = self.base_model.model.embed_tokens(input_ids)
            
            for i, layer in enumerate(self.base_model.model.layers):
                if str(i) in self.adapters:
                    residual = hidden_states
                    hidden_states = self.adapters[str(i)](hidden_states)
                    hidden_states = hidden_states + residual
                
                layer_outputs = layer(hidden_states)
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            
            hidden_states = self.base_model.model.norm(hidden_states)
            logits = self.base_model.lm_head(hidden_states)
            
        else:
            raise NotImplementedError(f"Unsupported model architecture: {type(self.base_model)}")
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {"logits": logits, "loss": loss}

    def adapt_topology(self, z_samples: torch.Tensor, **kwargs):
        """Adapt NGS topology across all adapters."""
        for adapter in self.adapters.values():
            if hasattr(adapter.ngs, "adapt_density"):
                adapter.ngs.adapt_density(z_samples=z_samples, **kwargs)

    def get_trainable_params(self):
        """Get only trainable parameters (NGS adapters)."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_trainable_params(self):
        """Count trainable vs total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def build_llm_wrapper(config: LLMWrapperConfig) -> LLMWrapper:
    """Factory function to build LLM wrapper from config."""
    return LLMWrapper(config)
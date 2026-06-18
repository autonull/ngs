"""Advanced research features for NGS."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from abc import ABC, abstractmethod
import random
import math

from ngs.core.interfaces import NGSConfig, RoutingOutput


class SymbolicExtractor(nn.Module):
    """
    Extract symbolic predicates from split-gate activations.
    
    Analyzes routing decisions to discover interpretable patterns:
    - Split gate conditions (when units split)
    - Merge conditions (when units merge)
    - Routing predicates (which inputs route to which units)
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_predicates: int = 16,
        hidden_dim: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_predicates = num_predicates
        self.temperature = temperature
        
        self.predicate_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_predicates),
        )
        
        self.threshold_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_predicates),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract predicates from input.
        
        Returns:
            predicate_logits: [batch, num_predicates] - raw predicate activations
            thresholds: [batch, num_predicates] - learned thresholds per predicate
        """
        predicate_logits = self.predicate_net(x)
        thresholds = self.threshold_net(x)
        return predicate_logits, thresholds
    
    def extract_rules(
        self,
        routing_output: RoutingOutput,
        inputs: torch.Tensor,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Extract human-readable rules from routing behavior.
        
        Args:
            routing_output: RoutingOutput from NGS forward pass
            inputs: Input tensor that produced the routing
            top_k: Number of top rules to extract
            
        Returns:
            List of rule dictionaries with condition, action, confidence
        """
        with torch.no_grad():
            predicate_logits, thresholds = self.forward(inputs)
            predicate_activations = torch.sigmoid(predicate_logits / self.temperature)
            
            active_indices = routing_output.indices.squeeze(-1)
            responsibilities = routing_output.weights
            
            rules = []
            for i in range(min(top_k, self.num_predicates)):
                pred_act = predicate_activations[:, i]
                thresh = thresholds[:, i].mean().item()
                
                high_act_mask = pred_act > thresh
                if high_act_mask.any():
                    routed_units = active_indices[high_act_mask]
                    if routed_units.numel() > 0:
                        most_common = routed_units.mode().values.item()
                        confidence = pred_act[high_act_mask].mean().item()
                        
                        rules.append({
                            "predicate_id": i,
                            "condition": f"predicate_{i} > {thresh:.3f}",
                            "action": f"route_to_unit_{most_common}",
                            "confidence": confidence,
                            "coverage": high_act_mask.float().mean().item(),
                        })
            
            rules.sort(key=lambda r: r["confidence"], reverse=True)
            return rules[:top_k]
    
    def compute_symbolic_loss(
        self,
        routing_output: RoutingOutput,
        inputs: torch.Tensor,
        entropy_weight: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute loss to encourage interpretable symbolic rules.
        
        Encourages:
        - High predicate activation entropy (diverse predicates)
        - Sparse threshold activations (sharp rules)
        """
        predicate_logits, thresholds = self.forward(inputs)
        predicate_probs = torch.softmax(predicate_logits / self.temperature, dim=-1)
        
        entropy = -(predicate_probs * torch.log(predicate_probs + 1e-8)).sum(dim=-1).mean()
        threshold_sparsity = (thresholds * (1 - thresholds)).mean()
        
        return -entropy_weight * entropy + threshold_sparsity


class CrossModalFusion(nn.Module):
    """
    Align factorized routing across modalities.
    
    For multi-modal inputs (e.g., vision + language), learns to align
    subspace projections so that semantically similar concepts
    route to corresponding units across modalities.
    """
    
    def __init__(
        self,
        modality_dims: List[int],
        shared_latent_dim: int,
        num_subspaces: int,
        alignment_weight: float = 1.0,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.shared_latent_dim = shared_latent_dim
        self.num_subspaces = num_subspaces
        self.alignment_weight = alignment_weight
        
        self.modality_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, shared_latent_dim),
                nn.LayerNorm(shared_latent_dim),
                nn.ReLU(),
            )
            for dim in modality_dims
        ])
        
        self.subspace_aligners = nn.ModuleList([
            nn.Linear(shared_latent_dim, shared_latent_dim // num_subspaces)
            for _ in range(num_subspaces)
        ])
        
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=shared_latent_dim,
            num_heads=4,
            batch_first=True,
        )
    
    def forward(
        self,
        modality_inputs: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Fuse multi-modal inputs with cross-modal alignment.
        
        Args:
            modality_inputs: List of [batch, modality_dim] tensors
            
        Returns:
            fused_features: List of [batch, shared_latent_dim] per modality
            alignment_loss: Scalar alignment loss
        """
        batch_size = modality_inputs[0].shape[0]
        
        encoded = []
        for i, x in enumerate(modality_inputs):
            encoded.append(self.modality_encoders[i](x))
        
        stacked = torch.stack(encoded, dim=1)
        attended, _ = self.cross_modal_attention(stacked, stacked, stacked)
        
        fused = []
        for i in range(len(modality_inputs)):
            fused.append(attended[:, i] + encoded[i])
        
        subspace_features = []
        for aligner in self.subspace_aligners:
            subspace_feats = [aligner(f) for f in fused]
            subspace_features.append(torch.stack(subspace_feats, dim=1))
        
        alignment_loss = 0.0
        for sf in subspace_features:
            for i in range(len(modality_inputs)):
                for j in range(i + 1, len(modality_inputs)):
                    alignment_loss += F.mse_loss(sf[:, i], sf[:, j])
        
        alignment_loss = alignment_loss / len(subspace_features)
        alignment_loss = alignment_loss * self.alignment_weight
        
        return fused, alignment_loss
    
    def get_subspace_projections(
        self,
        fused_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Get subspace projections for each modality."""
        projections = []
        for aligner in self.subspace_aligners:
            proj = [aligner(f) for f in fused_features]
            projections.append(torch.stack(proj, dim=1))
        return projections


class MetaMetaLearner:
    """
    Evolutionary search over NGSConfig space.
    
    Uses evolutionary algorithms to discover optimal NGS configurations
    for a given task/distribution.
    """
    
    def __init__(
        self,
        population_size: int = 20,
        num_generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_fraction: float = 0.2,
    ):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        
        self.param_ranges = {
            "latent_dim": (16, 128),
            "k_init": (8, 64),
            "max_k": (32, 512),
            "top_k": (2, 16),
            "num_subspaces": (2, 8),
            "hypernetwork_hidden_dim": (8, 64),
            "hypernetwork_code_dim": (4, 32),
            "split_threshold": (0.01, 0.2),
            "prune_threshold": (0.001, 0.05),
            "tau": (0.1, 5.0),
            "gamma_residual": (0.01, 0.5),
            "ema_decay": (0.9, 0.999),
            "diversity_weight": (0.001, 0.1),
            "entropy_weight": (0.001, 0.1),
        }
    
    def random_config(self) -> NGSConfig:
        """Generate a random NGSConfig within parameter ranges."""
        config_dict = {}
        for param, (low, high) in self.param_ranges.items():
            if isinstance(low, int):
                config_dict[param] = random.randint(low, high)
            else:
                config_dict[param] = random.uniform(low, high)
        
        config_dict["routing"] = random.choice(list(NGSConfig.__dataclass_fields__["routing"].default.__class__))
        config_dict["parameter_storage"] = random.choice(list(NGSConfig.__dataclass_fields__["parameter_storage"].default.__class__))
        config_dict["topology_control"] = random.choice(list(NGSConfig.__dataclass_fields__["topology_control"].default.__class__))
        config_dict["memory_management"] = random.choice(list(NGSConfig.__dataclass_fields__["memory_management"].default.__class__))
        
        return NGSConfig(**config_dict)
    
    def mutate_config(self, config: NGSConfig) -> NGSConfig:
        """Mutate a configuration."""
        config_dict = config.__dict__.copy()
        
        for param, (low, high) in self.param_ranges.items():
            if random.random() < self.mutation_rate:
                if isinstance(low, int):
                    config_dict[param] = random.randint(low, high)
                else:
                    config_dict[param] = random.uniform(low, high)
        
        return NGSConfig(**config_dict)
    
    def crossover_configs(self, config1: NGSConfig, config2: NGSConfig) -> NGSConfig:
        """Crossover two configurations."""
        config_dict = {}
        for param in self.param_ranges.keys():
            if random.random() < self.crossover_rate:
                config_dict[param] = getattr(config1, param)
            else:
                config_dict[param] = getattr(config2, param)
        
        for param in ["routing", "parameter_storage", "topology_control", "memory_management"]:
            config_dict[param] = random.choice([getattr(config1, param), getattr(config2, param)])
        
        return NGSConfig(**config_dict)
    
    def evolve(
        self,
        fitness_fn: callable,
        initial_population: Optional[List[NGSConfig]] = None,
    ) -> Tuple[NGSConfig, List[float]]:
        """
        Run evolutionary search.
        
        Args:
            fitness_fn: Function that takes NGSConfig and returns fitness score
            initial_population: Optional initial population
            
        Returns:
            Best config found, history of best fitness per generation
        """
        if initial_population is None:
            population = [self.random_config() for _ in range(self.population_size)]
        else:
            population = initial_population[:self.population_size]
            while len(population) < self.population_size:
                population.append(self.random_config())
        
        fitness_history = []
        
        for generation in range(self.num_generations):
            fitness_scores = []
            for config in population:
                try:
                    score = fitness_fn(config)
                    fitness_scores.append(score)
                except Exception:
                    fitness_scores.append(float('-inf'))
            
            sorted_pop = sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)
            fitness_history.append(sorted_pop[0][0])
            
            elite_count = int(self.population_size * self.elite_fraction)
            elites = [cfg for _, cfg in sorted_pop[:elite_count]]
            
            new_population = elites.copy()
            
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate and len(elites) >= 2:
                    parent1, parent2 = random.sample(elites, 2)
                    child = self.crossover_configs(parent1, parent2)
                else:
                    parent = random.choice(elites)
                    child = self.mutate_config(parent)
                
                new_population.append(child)
            
            population = new_population
        
        best_config = max(population, key=lambda c: fitness_fn(c))
        return best_config, fitness_history


@dataclass
class TritonKernelConfig:
    """Configuration for Triton kernels."""
    block_size: int = 256
    num_warps: int = 4
    num_stages: int = 3


class MahalanobisKernel:
    """
    Triton kernel for batched Mahalanobis distance computation.
    
    Computes: (x - mu)^T * Sigma^-1 * (x - mu) for batches of Gaussians.
    Optimized for routing where we compute distances to all units.
    """
    
    @staticmethod
    def get_kernel_source() -> str:
        return '''
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" __global__ void mahalanobis_kernel(
    const float* __restrict__ x,        // [B, D]
    const float* __restrict__ mu,       // [K, D]
    const float* __restrict__ sigma_inv, // [K, D, D] or [K, D] for diagonal
    float* __restrict__ output,         // [B, K]
    int B, int K, int D,
    bool is_diagonal
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * K) return;
    
    int b = idx / K;
    int k = idx % K;
    
    float dist = 0.0f;
    
    if (is_diagonal) {
        for (int d = 0; d < D; ++d) {
            float diff = x[b * D + d] - mu[k * D + d];
            float inv_var = sigma_inv[k * D + d];
            dist += diff * diff * inv_var;
        }
    } else {
        for (int i = 0; i < D; ++i) {
            float diff_i = x[b * D + i] - mu[k * D + i];
            float sum = 0.0f;
            for (int j = 0; j < D; ++j) {
                float diff_j = x[b * D + j] - mu[k * D + j];
                sum += sigma_inv[k * D * D + i * D + j] * diff_j;
            }
            dist += diff_i * sum;
        }
    }
    
    output[idx] = dist;
}
'''
    
    @staticmethod
    def launch(
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma_inv: torch.Tensor,
        output: torch.Tensor,
        config: TritonKernelConfig,
        is_diagonal: bool = True,
    ):
        """Launch the Mahalanobis kernel (placeholder - requires Triton JIT)."""
        B, D = x.shape
        K = mu.shape[0]
        
        if is_diagonal:
            diff = x.unsqueeze(1) - mu.unsqueeze(0)
            output.copy_((diff ** 2 * sigma_inv.unsqueeze(0)).sum(dim=-1))
        else:
            diff = x.unsqueeze(1) - mu.unsqueeze(0)
            output.copy_(torch.einsum('bkd,kij,bkj->bk', diff, sigma_inv, diff))


class LoRAKernel:
    """
    Triton kernel for batched LoRA matrix multiplication.
    
    Computes: x @ (W + A @ B) where A: [D, r], B: [r, D], r << D
    """
    
    @staticmethod
    def get_kernel_source() -> str:
        return '''
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" __global__ void lora_matmul_kernel(
    const float* __restrict__ x,      // [B, D]
    const float* __restrict__ W,      // [D, D] or [D_out, D_in]
    const float* __restrict__ A,      // [D, r] or [D_out, r]
    const float* __restrict__ B,      // [r, D] or [r, D_in]
    float* __restrict__ output,       // [B, D_out]
    int B_size, int D_in, int D_out, int r,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B_size * D_out) return;
    
    int b = idx / D_out;
    int d_out = idx % D_out;
    
    float result = 0.0f;
    for (int d_in = 0; d_in < D_in; ++d_in) {
        float w = W[d_out * D_in + d_in];
        float lora = 0.0f;
        for (int i = 0; i < r; ++i) {
            lora += A[d_out * r + i] * B[i * D_in + d_in];
        }
        result += x[b * D_in + d_in] * (w + alpha * lora);
    }
    
    output[idx] = result;
}
'''
    
    @staticmethod
    def launch(
        x: torch.Tensor,
        W: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        output: torch.Tensor,
        config: TritonKernelConfig,
        alpha: float = 1.0,
    ):
        """Launch the LoRA kernel (placeholder - requires Triton JIT)."""
        B_size, D_in = x.shape
        D_out = W.shape[0]
        r = A.shape[1]
        
        base = x @ W.t()
        lora = (x @ A) @ B
        output.copy_(base + alpha * lora)


def build_symbolic_extractor(
    latent_dim: int,
    num_predicates: int = 16,
    hidden_dim: int = 64,
) -> SymbolicExtractor:
    """Factory for SymbolicExtractor."""
    return SymbolicExtractor(latent_dim, num_predicates, hidden_dim)


def build_cross_modal_fusion(
    modality_dims: List[int],
    shared_latent_dim: int,
    num_subspaces: int,
    alignment_weight: float = 1.0,
) -> CrossModalFusion:
    """Factory for CrossModalFusion."""
    return CrossModalFusion(modality_dims, shared_latent_dim, num_subspaces, alignment_weight)


def build_meta_meta_learner(
    population_size: int = 20,
    num_generations: int = 50,
) -> MetaMetaLearner:
    """Factory for MetaMetaLearner."""
    return MetaMetaLearner(population_size, num_generations)
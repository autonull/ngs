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

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None

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


if TRITON_AVAILABLE:
    @triton.jit
    def _mahalanobis_diagonal_kernel(
        x_ptr, mu_ptr, sigma_inv_ptr, out_ptr,
        B, K, D,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid = tl.num_programs(0)
        
        for idx in range(pid, B * K, num_pid):
            b = idx // K
            k = idx % K
            
            dist = 0.0
            for d in range(0, D, BLOCK_SIZE):
                d_end = min(d + BLOCK_SIZE, D)
                x_slice = tl.load(x_ptr + b * D + d)
                mu_slice = tl.load(mu_ptr + k * D + d)
                sigma_slice = tl.load(sigma_inv_ptr + k * D + d)
                
                diff = x_slice - mu_slice
                dist += tl.sum(diff * diff * sigma_slice)
            
            tl.store(out_ptr + idx, dist)

    @triton.jit
    def _mahalanobis_full_kernel(
        x_ptr, mu_ptr, sigma_inv_ptr, out_ptr,
        B, K, D,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid = tl.num_programs(0)
        
        for idx in range(pid, B * K, num_pid):
            b = idx // K
            k = idx % K
            
            dist = 0.0
            for i in range(D):
                diff_i = tl.load(x_ptr + b * D + i) - tl.load(mu_ptr + k * D + i)
                sum_j = 0.0
                for j in range(D):
                    diff_j = tl.load(x_ptr + b * D + j) - tl.load(mu_ptr + k * D + j)
                    sigma_ij = tl.load(sigma_inv_ptr + k * D * D + i * D + j)
                    sum_j += sigma_ij * diff_j
                dist += diff_i * sum_j
            
            tl.store(out_ptr + idx, dist)

    @triton.jit
    def _lora_matmul_kernel(
        x_ptr, W_ptr, A_ptr, B_ptr, out_ptr,
        B_size, D_in, D_out, r, alpha,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid = tl.num_programs(0)
        
        for idx in range(pid, B_size * D_out, num_pid):
            b = idx // D_out
            d_out = idx % D_out
            
            result = 0.0
            for d_in in range(D_in):
                w = tl.load(W_ptr + d_out * D_in + d_in)
                lora = 0.0
                for i in range(r):
                    a_val = tl.load(A_ptr + d_out * r + i)
                    b_val = tl.load(B_ptr + i * D_in + d_in)
                    lora += a_val * b_val
                x_val = tl.load(x_ptr + b * D_in + d_in)
                result += x_val * (w + alpha * lora)
            
            tl.store(out_ptr + idx, result)


class MahalanobisKernel:
    """
    Triton kernel for batched Mahalanobis distance computation.
    
    Computes: (x - mu)^T * Sigma^-1 * (x - mu) for batches of Gaussians.
    Optimized for routing where we compute distances to all units.
    """
    
    @staticmethod
    def launch(
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma_inv: torch.Tensor,
        output: torch.Tensor,
        config: TritonKernelConfig,
        is_diagonal: bool = True,
    ):
        """Launch the Mahalanobis kernel."""
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available. Install with `pip install triton`")
        
        B, D = x.shape
        K = mu.shape[0]
        grid = (B * K,)
        
        x = x.contiguous()
        mu = mu.contiguous()
        sigma_inv = sigma_inv.contiguous()
        output = output.contiguous()
        
        if is_diagonal:
            _mahalanobis_diagonal_kernel[grid](
                x, mu, sigma_inv, output,
                B, K, D,
                BLOCK_SIZE=config.block_size,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
        else:
            _mahalanobis_full_kernel[grid](
                x, mu, sigma_inv, output,
                B, K, D,
                BLOCK_SIZE=config.block_size,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )


class LoRAKernel:
    """
    Triton kernel for batched LoRA matrix multiplication.
    
    Computes: x @ (W + A @ B) where A: [D, r], B: [r, D], r << D
    """
    
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
        """Launch the LoRA kernel."""
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available. Install with `pip install triton`")
        
        B_size, D_in = x.shape
        D_out = W.shape[0]
        r = A.shape[1]
        grid = (B_size * D_out,)
        
        x = x.contiguous()
        W = W.contiguous()
        A = A.contiguous()
        B = B.contiguous()
        output = output.contiguous()
        
        _lora_matmul_kernel[grid](
            x, W, A, B, output,
            B_size, D_in, D_out, r, alpha,
            BLOCK_SIZE=config.block_size,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )


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
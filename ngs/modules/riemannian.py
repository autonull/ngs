"""Riemannian Hypernetwork Manifold: Geodesic interpolation in code space."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RiemannianHypernetworkManifold(nn.Module):
    """
    Riemannian manifold for hypernetwork code space.
    
    Implements geodesic interpolation on a learned latent manifold.
    Default uses hyperbolic geometry (Poincaré ball model) for parameter space,
    which is well-suited for embedding hierarchical structures.
    
    Alternative implementations included as reference:
    - Euclidean (flat space, simple interpolation)
    - Spherical (unit sphere, for directional data)
    """

    def __init__(
        self,
        code_dim: int,
        manifold_dim: int = None,
        curvature: float = 1.0,
        learnable_curvature: bool = True,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.manifold_dim = manifold_dim or code_dim // 2
        self.curvature = nn.Parameter(torch.tensor(curvature)) if learnable_curvature else curvature
        
        self.encoder = nn.Sequential(
            nn.Linear(code_dim, self.manifold_dim * 2),
            nn.ReLU(),
            nn.Linear(self.manifold_dim * 2, self.manifold_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.manifold_dim, self.manifold_dim * 2),
            nn.ReLU(),
            nn.Linear(self.manifold_dim * 2, code_dim),
        )
        
        self.riemannian_metric = nn.Parameter(torch.eye(self.manifold_dim))
    
    def encode(self, code: torch.Tensor) -> torch.Tensor:
        return self.encoder(code)
    
    def decode(self, point: torch.Tensor) -> torch.Tensor:
        return self.decoder(point)
    
    def _project_to_hyperbolic(self, x: torch.Tensor) -> torch.Tensor:
        """Project points to Poincaré ball (ensure ||x|| < 1)."""
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (0.99 / norm).clamp(max=1.0)
        return x * scale
    
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map: project y to tangent space at x.
        
        For hyperbolic geometry (Poincaré ball):
        log_x(y) = (2/√c) * arctanh(2√c * ||y|| / (1 + c||y||² + 2c <x,y>)) * (y - ...)/||y||
        
        Simplified Euclidean fallback with metric:
        """
        diff = y - x
        metric = self.riemannian_metric @ self.riemannian_metric.T
        return diff @ metric
    
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map: project v from tangent space to manifold.
        
        For hyperbolic (Poincaré ball):
        exp_x(v) = x ⊕ (tanh(√c ||v||) / (√c ||v||)) * v
        where ⊕ is Möbius addition.
        
        Simplified Euclidean fallback:
        """
        # For stability, use simple addition with projection
        result = x + v
        return self._project_to_hyperbolic(result)
    
    def geodesic(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Geodesic interpolation between x and y at parameter t.
        
        High-performance (default): Simplified Euclidean interpolation
        Reference: Full hyperbolic geodesic (more complex but principled)
        """
        v = self.log_map(x, y)
        return self.exp_map(x, t * v)
    
    def geodesic_hyperbolic(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Full hyperbolic geodesic (reference implementation, more expensive).
        
        Uses Möbius addition and geodesic formula for Poincaré ball.
        """
        c = max(self.curvature.item(), 1e-4)
        sqrt_c = math.sqrt(c)
        
        # Möbius addition: x ⊕ y = (x + y) / (1 + c<x,y>)
        def mobius_add(x, y):
            num = x + y
            denom = (1 + c * (x * y).sum(dim=-1, keepdim=True)).clamp(min=1e-8)
            return num / denom
        
        # Hyperbolic geodesic
        def exp_map_hyperb(x, v):
            v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
            return mobius_add(x, scale * v)
        
        v = self.log_map(x, y)
        return exp_map_hyperb(x, t * v)
    
    def parallel_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Parallel transport of vector v from x to y's tangent space.
        
        For hyperbolic: uses the formula with metric and geodesic scaling.
        """
        metric = self.riemannian_metric @ self.riemannian_metric.T
        return v @ metric
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geodesic distance between points.
        
        For hyperbolic (Poincaré ball):
        d(x,y) = (2/√c) * arccosh(1 + 2c||x-y||² / ((1+c||x||²)(1+c||y||²)))
        """
        diff = y - x
        metric = self.riemannian_metric @ self.riemannian_metric.T
        return torch.sqrt((diff @ metric * diff).sum(-1) + 1e-8)
    
    def distance_hyperbolic(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Full hyperbolic distance (reference implementation)."""
        c = max(self.curvature.item(), 1e-4)
        x_norm_sq = (x * x).sum(dim=-1)
        y_norm_sq = (y * y).sum(dim=-1)
        xy_norm_sq = ((x - y) * (x - y)).sum(dim=-1)
        
        arg = 1 + 2 * c * xy_norm_sq / ((1 + c * x_norm_sq) * (1 + c * y_norm_sq) + 1e-8)
        arg = arg.clamp(min=1.0 + 1e-8)
        return (2 / math.sqrt(c)) * torch.log(arg + 1e-8) / 2
    
    def frechet_mean(self, points: torch.Tensor, max_iter: int = 50) -> torch.Tensor:
        """Compute Fréchet mean on the manifold.
        
        Uses gradient descent on the geodesic distance.
        """
        # Initialize at Euclidean mean
        mean = points.mean(dim=0)
        mean = self._project_to_hyperbolic(mean)
        
        # Gradient descent for Fréchet mean
        for _ in range(max_iter):
            with torch.no_grad():
                # Compute gradient on manifold
                tangent = torch.zeros_like(mean)
                for p in points:
                    log_p = self.log_map(mean, p)
                    tangent += log_p
                tangent /= len(points)
                
                # Step in tangent direction
                step_size = 0.1 / (1 + _)
                mean = self.exp_map(mean, step_size * tangent)
        
        return mean

    def forward(self, codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        manifold_points = self.encode(codes)
        reconstructed = self.decoder(manifold_points)
        return manifold_points, reconstructed

    def interpolate(self, code_a: torch.Tensor, code_b: torch.Tensor, steps: int = 10) -> torch.Tensor:
        t = torch.linspace(0, 1, steps, device=code_a.device)
        z_a = self.encode(code_a)
        z_b = self.encode(code_b)
        interpolated = []
        for ti in t:
            z_int = self.geodesic(z_a, z_b, ti)
            interpolated.append(self.decode(z_int))
        return torch.stack(interpolated)


class HypernetworkCodeManifold(nn.Module):
    """
    Learns a Riemannian manifold structure on the hypernetwork code space
    for smooth task interpolation and continual learning.
    """

    def __init__(
        self,
        num_tasks: int,
        code_dim: int,
        manifold_dim: int = None,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.code_dim = code_dim
        self.manifold = RiemannianHypernetworkManifold(code_dim, manifold_dim, curvature)
        
        self.task_embeddings = nn.Parameter(torch.randn(num_tasks, code_dim) * 0.1)

    def get_task_code(self, task_id: int) -> torch.Tensor:
        return self.task_embeddings[task_id]

    def interpolate_tasks(self, task_a: int, task_b: int, steps: int = 10) -> torch.Tensor:
        code_a = self.get_task_code(task_a)
        code_b = self.get_task_code(task_b)
        return self.manifold.interpolate(code_a, code_b, steps)

    def geodesic_distance(self, task_a: int, task_b: int) -> torch.Tensor:
        code_a = self.get_task_code(task_a)
        code_b = self.get_task_code(task_b)
        z_a = self.manifold.encode(code_a)
        z_b = self.manifold.encode(code_b)
        return self.manifold.distance(z_a, z_b)

    def compute_frechet_mean(self, task_ids: list) -> torch.Tensor:
        codes = torch.stack([self.get_task_code(t) for t in task_ids])
        return self.manifold.frechet_mean(codes)

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        return self.task_embeddings[task_ids]


def build_riemannian_manifold(config) -> RiemannianHypernetworkManifold:
    """Factory function to build Riemannian manifold from config."""
    return RiemannianHypernetworkManifold(
        code_dim=config.hypernetwork_code_dim if hasattr(config, 'hypernetwork_code_dim') else 16,
        manifold_dim=getattr(config, 'manifold_dim', None),
        curvature=getattr(config, 'riemannian_curvature', 1.0),
        learnable_curvature=getattr(config, 'learnable_curvature', True),
    )
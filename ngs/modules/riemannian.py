"""Riemannian Hypernetwork Manifold: Geodesic interpolation in code space."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RiemannianHypernetworkManifold(nn.Module):
    """
    Riemannian manifold for hypernetwork code space.
    
    Implements geodesic interpolation on a learned latent manifold
    for smooth transitions between task-specific codes.
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

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = y - x
        metric = self.riemannian_metric @ self.riemannian_metric.T
        return diff @ metric

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return x + v

    def geodesic(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v = self.log_map(x, y)
        return self.exp_map(x, t * v)

    def parallel_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        metric = self.riemannian_metric @ self.riemannian_metric.T
        return v @ metric

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = y - x
        metric = self.riemannian_metric @ self.riemannian_metric.T
        return torch.sqrt((diff @ metric * diff).sum(-1) + 1e-8)

    def frechet_mean(self, points: torch.Tensor, max_iter: int = 50) -> torch.Tensor:
        mean = points.mean(dim=0)
        for _ in range(max_iter):
            tangent = self.log_map(mean.unsqueeze(0), points).mean(dim=0)
            mean = self.exp_map(mean, tangent)
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
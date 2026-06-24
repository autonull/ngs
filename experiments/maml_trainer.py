"""MAML Trainer with `higher` library integration for differentiable inner loops."""

import torch
import torch.nn.functional as F
import higher
from typing import Tuple, List, Optional, Callable
from ngs.core.interfaces import NGSConfig
from ngs.models.ngs import build_ngs


class MAMLTrainer:
    """MAML meta-trainer using `higher` for differentiable inner-loop adaptation."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: NGSConfig,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        meta_lr: float = 1e-3,
        device: str = 'cuda'
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_lr = meta_lr
        self.device = device
        
        # Meta-model (shared initialization)
        self.meta_model = build_ngs(input_dim, output_dim, config).to(device)
        
        # Only meta-optimize parameters that should NOT be adapted in inner loop
        # Inner loop adapts: router (mu, log_s, log_alpha) + hypernet codes
        # Outer loop meta-learns: feature extractor, hypernet weights, etc.
        meta_params = []
        for name, param in self.meta_model.named_parameters():
            if 'router' not in name and 'code' not in name:
                meta_params.append(param)
        
        self.meta_opt = torch.optim.AdamW(meta_params, lr=meta_lr)
    
    def inner_loop_params(self) -> List[torch.nn.Parameter]:
        """Get parameters to adapt in inner loop."""
        params = []
        for name, param in self.meta_model.named_parameters():
            if 'router' in name or 'code' in name:
                params.append(param)
        return params
    
    def maml_step(
        self,
        supp_x: torch.Tensor,
        supp_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        copy_initial_weights: bool = False
    ) -> Tuple[torch.Tensor, higher.MonkeyPatchBase]:
        """
        Perform one MAML step with differentiable inner loop using `higher`.
        
        Returns:
            meta_loss: Loss on query set after inner adaptation
            fmodel: The adapted functional model (for inspection)
        """
        # Create functional model with `higher`
        fmodel = higher.monkeypatch(
            self.meta_model,
            copy_initial_weights=copy_initial_weights,
            track_higher_grads=True
        )
        
        # Inner loop optimizer (only adapts router + codes)
        inner_params = self.inner_loop_params()
        inner_opt = torch.optim.SGD(inner_params, lr=self.inner_lr)
        
        fmodel.train()
        for _ in range(self.inner_steps):
            inner_opt.zero_grad()
            out = fmodel(supp_x)
            logits = out.logits if hasattr(out, 'logits') else out
            loss = F.cross_entropy(logits, supp_y)
            loss.backward()
            inner_opt.step()
        
        # Query loss (meta-objective) - MUST have gradients for meta-update
        fmodel.eval()
        out = fmodel(query_x)
        logits = out.logits if hasattr(out, 'logits') else out
        meta_loss = F.cross_entropy(logits, query_y)
        
        return meta_loss, fmodel
    
    def meta_update(self, meta_loss: torch.Tensor):
        """Perform meta-gradient update."""
        self.meta_opt.zero_grad()
        meta_loss.backward()
        self.meta_opt.step()
    
    def evaluate(
        self,
        supp_x: torch.Tensor,
        supp_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        inner_steps: Optional[int] = None
    ) -> float:
        """Evaluate adapted model on query set."""
        if inner_steps is None:
            inner_steps = self.inner_steps
        
        # Use non-differentiable inner loop for evaluation (faster)
        adapted = self.adapt_model(supp_x, supp_y, inner_steps)
        
        adapted.eval()
        with torch.no_grad():
            out = adapted(query_x)
            logits = out.logits if hasattr(out, 'logits') else out
            pred = logits.argmax(1)
            acc = (pred == query_y).float().mean().item()
        
        return acc
    
    def adapt_model(
        self,
        supp_x: torch.Tensor,
        supp_y: torch.Tensor,
        inner_steps: int = 10,
        inner_lr: Optional[float] = None
    ):
        """Fast adaptation on support set (non-differentiable, for eval)."""
        if inner_lr is None:
            inner_lr = self.inner_lr
        
        adapted = torch.nn.Module.__new__(type(self.meta_model))
        adapted.__dict__.update(self.meta_model.__dict__)
        adapted._parameters = self.meta_model._parameters.copy()
        adapted._buffers = self.meta_model._buffers.copy()
        adapted._modules = self.meta_model._modules.copy()
        
        inner_opt = torch.optim.SGD(
            [p for n, p in adapted.named_parameters() if 'router' in n or 'code' in n],
            lr=inner_lr
        )
        
        adapted.train()
        for _ in range(inner_steps):
            inner_opt.zero_grad()
            out = adapted(supp_x)
            logits = out.logits if hasattr(out, 'logits') else out
            loss = F.cross_entropy(logits, supp_y)
            loss.backward()
            inner_opt.step()
        
        return adapted


def create_maml_trainer(
    input_dim: int,
    output_dim: int,
    latent_dim: int = 64,
    max_k: int = 32,
    k_init: int = 16,
    top_k: int = 4,
    inner_lr: float = 0.01,
    inner_steps: int = 5,
    meta_lr: float = 1e-3,
    device: str = 'cuda'
) -> MAMLTrainer:
    """Factory function to create MAML trainer with default NGS config."""
    config = NGSConfig(
        latent_dim=latent_dim,
        max_k=max_k,
        k_init=k_init,
        top_k=top_k,
        parameter_storage='hypernetwork_generated',
        topology_control='discrete_heuristic',
        routing='monolithic_mahalanobis',
        memory_management='dynamic',
        hypernetwork_hidden_dim=32,
        hypernetwork_code_dim=8,
    )
    return MAMLTrainer(input_dim, output_dim, config, inner_lr, inner_steps, meta_lr, device)
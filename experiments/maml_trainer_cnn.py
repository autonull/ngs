"""MAML Trainer with CNN backbone for Omniglot."""

import torch
import torch.nn.functional as F
import higher
from typing import Tuple, List, Optional
from ngs.core.interfaces import NGSConfig
from experiments.omniglot_backbone import create_cnn_ngs_maml, CNNNGS


class MAMLTrainerCNN:
    """MAML meta-trainer with CNN backbone + NGS head."""
    
    def __init__(
        self,
        num_classes: int = 5,
        latent_dim: int = 64,
        max_k: int = 64,
        k_init: int = 32,
        top_k: int = 8,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        meta_lr: float = 1e-3,
        backbone_lr: float = 1e-4,
        device: str = 'cuda'
    ):
        self.num_classes = num_classes
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_lr = meta_lr
        self.backbone_lr = backbone_lr
        self.device = device
        
        # Meta-model: CNN backbone + NGS head
        self.meta_model = create_cnn_ngs_maml(
            num_filters=64,
            latent_dim=latent_dim,
            max_k=max_k,
            k_init=k_init,
            top_k=top_k,
            hypernetwork_hidden_dim=64,
            hypernetwork_code_dim=16,
            num_classes=num_classes
        ).to(device)
        
        # Separate optimizers for backbone (lower LR) and head
        self.meta_opt = torch.optim.AdamW([
            {'params': self.meta_model.parameters_backbone(), 'lr': backbone_lr},
            {'params': self.meta_model.parameters_head(), 'lr': meta_lr},
        ])
    
    def inner_loop_params(self) -> List[torch.nn.Parameter]:
        """Get NGS head parameters to adapt in inner loop (router + codes)."""
        params = []
        for name, param in self.meta_model.ngs_head.named_parameters():
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
        """Perform one MAML step with differentiable inner loop using `higher`."""
        # Create functional model with `higher`
        fmodel = higher.monkeypatch(
            self.meta_model,
            copy_initial_weights=copy_initial_weights,
            track_higher_grads=True
        )
        
        # Inner loop optimizer (only adapts NGS head router + codes)
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
        
        # Query loss (meta-objective) - MUST have gradients
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
        
        # Deep copy the model
        import copy
        adapted = copy.deepcopy(self.meta_model)
        
        inner_opt = torch.optim.SGD(
            [p for n, p in adapted.ngs_head.named_parameters() if 'router' in n or 'code' in n],
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


def create_maml_trainer_cnn(
    num_classes: int = 5,
    latent_dim: int = 64,
    max_k: int = 64,
    k_init: int = 32,
    top_k: int = 8,
    inner_lr: float = 0.01,
    inner_steps: int = 5,
    meta_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    device: str = 'cuda'
) -> MAMLTrainerCNN:
    """Factory function to create CNN+NGS MAML trainer."""
    return MAMLTrainerCNN(
        num_classes=num_classes,
        latent_dim=latent_dim,
        max_k=max_k,
        k_init=k_init,
        top_k=top_k,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        meta_lr=meta_lr,
        backbone_lr=backbone_lr,
        device=device
    )
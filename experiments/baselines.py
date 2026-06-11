"""
Baseline models for continual learning comparison.
Includes: MLP, LoRA, EWC, SI, LwF, ER (Experience Replay).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from copy import deepcopy


class MLP(nn.Module):
    """Standard MLP baseline."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [512, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LoRAMultitask(nn.Module):
    """LoRA-style adapter on frozen backbone."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, rank: int = 16):
        super().__init__()
        # Frozen backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        # LoRA adapters per layer
        self.lora_A1 = nn.Linear(input_dim, rank, bias=False)
        self.lora_B1 = nn.Linear(rank, hidden_dim, bias=False)
        self.lora_A2 = nn.Linear(hidden_dim, rank, bias=False)
        self.lora_B2 = nn.Linear(rank, hidden_dim, bias=False)

        # Task-specific head
        self.head = nn.Linear(hidden_dim, output_dim)

        # Initialize LoRA
        nn.init.kaiming_uniform_(self.lora_A1.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B1.weight)
        nn.init.kaiming_uniform_(self.lora_A2.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B2.weight)

    def forward(self, x):
        h = self.backbone[0](x)
        h = h + self.lora_B1(self.lora_A1(x))
        h = F.relu(h)

        h2 = self.backbone[2](h)
        h2 = h2 + self.lora_B2(self.lora_A2(h))
        h2 = F.relu(h2)

        return self.head(h2)


class EWCModel(nn.Module):
    """MLP with Elastic Weight Consolidation."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [512, 256], ewc_lambda: float = 1000):
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.mlp = MLP(input_dim, output_dim, hidden_dims)
        self.register_buffer('fisher', None)
        self.register_buffer('opt_params', None)

    def forward(self, x):
        return self.mlp(x)

    def consolidate(self, dataloader, device):
        """Compute Fisher information and store optimal params."""
        self.mlp.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.mlp.named_parameters() if p.requires_grad}
        opt_params = {n: p.clone().detach() for n, p in self.mlp.named_parameters() if p.requires_grad}

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            self.mlp.zero_grad()
            logits = self.mlp(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            for n, p in self.mlp.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2

        # Average over dataset
        for n in fisher:
            fisher[n] /= len(dataloader.dataset)

        self.fisher = fisher
        self.opt_params = opt_params

    def ewc_loss(self):
        if self.fisher is None:
            return 0
        loss = 0
        for n, p in self.mlp.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.opt_params[n]) ** 2).sum()
        return self.ewc_lambda * loss


class SIModel(nn.Module):
    """MLP with Synaptic Intelligence."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [512, 256], si_lambda: float = 1.0):
        super().__init__()
        self.si_lambda = si_lambda
        self.mlp = MLP(input_dim, output_dim, hidden_dims)
        self.register_buffer('omega', {})
        self.register_buffer('prev_params', {})

    def forward(self, x):
        return self.mlp(x)

    def update_omega(self, dataloader, device, epsilon=1e-3):
        """Compute path integral of gradients (importance weights)."""
        self.mlp.eval()
        omega = {n: torch.zeros_like(p) for n, p in self.mlp.named_parameters() if p.requires_grad}
        prev_params = {n: p.clone().detach() for n, p in self.mlp.named_parameters() if p.requires_grad}

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            self.mlp.zero_grad()
            logits = self.mlp(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            for n, p in self.mlp.named_parameters():
                if p.grad is not None:
                    omega[n] += p.grad.data * (p.data - prev_params[n])

        for n in omega:
            omega[n] = omega[n] / (len(dataloader.dataset) + epsilon)

        self.omega = omega
        self.prev_params = prev_params

    def si_loss(self):
        if not self.omega:
            return 0
        loss = 0
        for n, p in self.mlp.named_parameters():
            if n in self.omega:
                loss += (self.omega[n] * (p - self.prev_params[n]) ** 2).sum()
        return self.si_lambda * loss


class LwFModel(nn.Module):
    """Learning without Forgetting - knowledge distillation from old model."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [512, 256], 
                 lwf_lambda: float = 1.0, temp: float = 2.0):
        super().__init__()
        self.lwf_lambda = lwf_lambda
        self.temp = temp
        self.mlp = MLP(input_dim, output_dim, hidden_dims)
        self.old_model: Optional[nn.Module] = None

    def forward(self, x):
        return self.mlp(x)

    def set_old_model(self, old_model):
        self.old_model = deepcopy(old_model)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False

    def lwf_loss(self, x):
        if self.old_model is None:
            return 0
        with torch.no_grad():
            old_logits = self.old_model(x)
        new_logits = self.mlp(x)
        return F.kl_div(
            F.log_softmax(new_logits / self.temp, dim=-1),
            F.softmax(old_logits / self.temp, dim=-1),
            reduction='batchmean'
        ) * (self.temp ** 2) * self.lwf_lambda


class ERModel(nn.Module):
    """Experience Replay baseline."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [512, 256]):
        super().__init__()
        self.mlp = MLP(input_dim, output_dim, hidden_dims)

    def forward(self, x):
        return self.mlp(x)


# Factory for creating baselines
def create_baseline(name: str, input_dim: int, output_dim: int, **kwargs):
    """Create baseline model by name."""
    baselines = {
        'mlp': MLP,
        'lora': LoRAMultitask,
        'ewc': EWCModel,
        'si': SIModel,
        'lwf': LwFModel,
        'er': ERModel,
    }
    if name not in baselines:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(baselines.keys())}")
    return baselines[name](input_dim, output_dim, **kwargs)
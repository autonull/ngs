import torch
import torch.nn as nn
from lean_ngs.model import LeanNGS

class StandardMLP(nn.Module):
    def __init__(self, d_in, d_out, d_hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out)
        )
        
    def forward(self, x):
        return self.net(x)

class FixedLeanNGS(nn.Module):
    def __init__(self, d_in, d_out, k=128):
        super().__init__()
        # Pre-allocate exactly K units and never call adapt_density
        self.model = LeanNGS(d_in, d_out, k_init=k, max_k=k, adc_mode='pre_alloc')
        
    def forward(self, x):
        return self.model(x)

    def compute_entropy_loss(self):
        return self.model.compute_entropy_loss()

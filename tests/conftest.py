"""Test configuration with automatic seed setting."""
import pytest
import torch
import numpy as np
import random


@pytest.fixture(autouse=True)
def seed_all():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

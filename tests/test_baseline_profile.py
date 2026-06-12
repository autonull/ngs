"""Test the baseline LeanNGS profile."""
import torch
import pytest
from mngs.profiles import Baseline_LeanNGS
from mngs.model import build_mngs

def test_monolithic_forward_shape():
    config = Baseline_LeanNGS()
    model = build_mngs(d_in=784, d_out=10, config=config)
    x = torch.randn(4, 784)
    out = model(x)
    assert out.shape == (4, 10)

def test_monolithic_routing_output():
    config = Baseline_LeanNGS()
    model = build_mngs(d_in=784, d_out=10, config=config)
    x = torch.randn(4, 784)
    z = model.p_down(x)
    indices, weights = model.router(z)
    assert indices.shape == (4, config.top_k)
    assert weights.shape == (4, config.top_k)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)

def test_baseline_routers_and_stores():
    config = Baseline_LeanNGS()
    model = build_mngs(d_in=784, d_out=10, config=config)
    assert model.router is not None
    assert model.param_store is not None
    assert model.topology_manager is not None

def test_gradient_flow():
    config = Baseline_LeanNGS()
    model = build_mngs(d_in=784, d_out=10, config=config)
    x = torch.randn(4, 784)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert model.p_down.weight.grad is not None
    assert model.p_up.weight.grad is not None

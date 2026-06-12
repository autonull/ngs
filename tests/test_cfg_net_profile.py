"""Test the CFG-Net full profile."""
import torch
from mngs.profiles import CFG_Net_Full
from mngs.model import build_mngs

def test_cfg_net_forward_shape():
    config = CFG_Net_Full()
    model = build_mngs(d_in=784, d_out=10, config=config)
    x = torch.randn(4, 784)
    out = model(x)
    assert out.shape == (4, 10)

def test_cfg_net_routing_output():
    config = CFG_Net_Full()
    model = build_mngs(d_in=784, d_out=10, config=config)
    x = torch.randn(4, 784)
    z = model.p_down(x)
    subspace_indices, subspace_weights = model.router(z)
    assert len(subspace_indices) == config.num_subspaces
    assert len(subspace_weights) == config.num_subspaces
    for indices in subspace_indices:
        assert indices.shape[0] == 4
        assert indices.shape[1] == config.top_k_factorized or indices.shape[1] <= config.k_init // config.num_subspaces

def test_cfg_net_gradient_flow():
    config = CFG_Net_Full()
    model = build_mngs(d_in=784, d_out=10, config=config)
    x = torch.randn(4, 784)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert model.p_down.weight.grad is not None
    assert model.p_up.weight.grad is not None

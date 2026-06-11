import unittest
import torch
from lean_ngs.model import LeanNGS
from lean_ngs.baselines import StandardMLP, FixedLeanNGS
from lean_ngs.dataset import get_mnist_dataloaders

class TestLeanNGS(unittest.TestCase):
    def test_lean_ngs_forward(self):
        model = LeanNGS(784, 10, d_latent=32, k_init=10, adc_mode='pre_alloc')
        x = torch.randn(4, 784)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))

    def test_lean_ngs_entropy(self):
        model = LeanNGS(784, 10, k_init=10, adc_mode='pre_alloc')
        x = torch.randn(4, 784)
        _ = model(x)
        ent = model.compute_entropy_loss()
        self.assertTrue(torch.is_tensor(ent))
        self.assertEqual(ent.dim(), 0) # scalar

    def test_adc_pre_alloc(self):
        model = LeanNGS(784, 10, k_init=5, max_k=10, adc_mode='pre_alloc')
        # Setup fake state
        model.grad_mu_norm_ema = torch.ones_like(model.log_alpha)
        model.active_mask[0] = True
        model.log_s.data[0] = 0.0 # exp(0) = 1.0 > 0.05
        model.log_alpha.data[1] = -10.0 # sigmoid(-10) ~ 0
        pruned, split = model.adapt_density(split_thresh=0.05, prune_thresh=0.01)
        self.assertTrue(split > 0 or pruned > 0)

    def test_adc_dynamic(self):
        model = LeanNGS(784, 10, k_init=5, adc_mode='dynamic')
        model.grad_mu_norm_ema = torch.ones_like(model.log_alpha)
        model.log_s.data[0] = 0.0
        model.log_alpha.data[1] = -10.0
        pruned, split = model.adapt_density(split_thresh=0.05, prune_thresh=0.01)
        self.assertTrue(split > 0 or pruned > 0)

if __name__ == '__main__':
    unittest.main()

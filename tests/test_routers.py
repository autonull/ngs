"""Integration tests for all routers."""

import torch
import pytest
from ngs.core.interfaces import NGSConfig, RoutingStrategy
from ngs.modules.routers import (
    MonolithicRouter, FactorizedRouter, LSRRouter,
    HierarchicalRouter, GaussianAttentionRouter, UncertaintyAwareRouter, build_router
)
from ngs.models.ngs import build_ngs


class TestRouters:
    """Test all router implementations."""

    @pytest.fixture(params=[
        RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        RoutingStrategy.FACTORIZED_SUBSPACE,
        RoutingStrategy.LSH_APPROXIMATE,
        RoutingStrategy.HIERARCHICAL,
        RoutingStrategy.GAUSSIAN_ATTENTION,
        RoutingStrategy.UNCERTAINTY_AWARE,
    ])
    def routing_strategy(self, request):
        return request.param

    def _make_config(self, routing_strategy):
        """Create appropriate config for the routing strategy."""
        base = dict(max_k=64, k_init=16, latent_dim=32, tau=1.0, ema_decay=0.99)
        if routing_strategy == RoutingStrategy.FACTORIZED_SUBSPACE:
            base.update(num_subspaces=4, top_k=8, top_k_factorized=2)
        elif routing_strategy == RoutingStrategy.HIERARCHICAL:
            base.update(num_levels=3, level_capacity_ratio=0.5, level_top_k=4)
        elif routing_strategy == RoutingStrategy.GAUSSIAN_ATTENTION:
            base.update(attention_heads=4, attention_dropout=0.1, sparse_top_k=8)
        elif routing_strategy == RoutingStrategy.UNCERTAINTY_AWARE:
            base.update(evidential_prior=1.0, uncertainty_weight=0.1)
        elif routing_strategy == RoutingStrategy.LSH_APPROXIMATE:
            base.update(top_k=8)
        else:  # MONOLITHIC
            base.update(top_k=8)
        return NGSConfig(routing=routing_strategy, **base)

    def _make_router(self, routing_strategy, config):
        """Create router instance for the given strategy."""
        if routing_strategy == RoutingStrategy.MONOLITHIC_MAHALANOBIS:
            return MonolithicRouter(config)
        elif routing_strategy == RoutingStrategy.FACTORIZED_SUBSPACE:
            return FactorizedRouter(config)
        elif routing_strategy == RoutingStrategy.LSH_APPROXIMATE:
            return LSRRouter(config)
        elif routing_strategy == RoutingStrategy.HIERARCHICAL:
            return HierarchicalRouter(config)
        elif routing_strategy == RoutingStrategy.GAUSSIAN_ATTENTION:
            return GaussianAttentionRouter(config)
        elif routing_strategy == RoutingStrategy.UNCERTAINTY_AWARE:
            return UncertaintyAwareRouter(config)
        else:
            raise ValueError(f"Unknown strategy: {routing_strategy}")

    def test_router_forward_backward(self, routing_strategy):
        """Test forward and backward pass for each router."""
        # LSH router uses random pseudo-distances without gradients - skip
        if routing_strategy == RoutingStrategy.LSH_APPROXIMATE:
            pytest.skip("LSH router uses random pseudo-distances, no gradient flow")

        config = self._make_config(routing_strategy)
        router = self._make_router(routing_strategy, config)
        router.initialize_units(config.k_init)

        x = torch.randn(8, 32, requires_grad=True)
        out = router(x)

        # Check output structure (RoutingOutput)
        assert hasattr(out, 'indices')
        assert hasattr(out, 'weights')
        indices, weights = out.indices, out.weights

        # Check shapes
        if isinstance(indices, list):
            # Factorized/Hierarchical routing - per-subspace/level outputs
            assert len(indices) == len(weights)
            for idx, w in zip(indices, weights):
                assert idx.shape[0] == 8  # batch size
                assert w.shape[0] == 8
                assert idx.shape == w.shape
                # Each subspace/level should sum to 1
                assert torch.allclose(w.sum(dim=-1), torch.ones(8), atol=1e-4)
        else:
            # Standard routing - flat indices concatenated across all levels/spaces
            assert indices.shape[0] == 8  # batch size
            assert weights.shape[0] == 8
            assert indices.shape == weights.shape
            # For standard routers (Monolithic, UncertaintyAware), weights sum to 1
            # For concatenated routers (Factorized, Hierarchical, LSH, GaussianAttention), 
            # weights sum to num_subspaces/num_levels/num_buckets
            if routing_strategy in (RoutingStrategy.MONOLITHIC_MAHALANOBIS, RoutingStrategy.UNCERTAINTY_AWARE):
                assert torch.allclose(weights.sum(dim=-1), torch.ones(8), atol=1e-4)

        # Test backward
        loss = weights.sum() if not isinstance(weights, list) else sum(w.sum() for w in weights)
        loss.backward()
        assert x.grad is not None

    def test_router_gradient_flow(self, routing_strategy):
        """Test gradient flows through router parameters."""
        # LSH router uses random pseudo-distances without gradients - skip
        if routing_strategy == RoutingStrategy.LSH_APPROXIMATE:
            pytest.skip("LSH router uses random pseudo-distances, no gradient flow")

        config = self._make_config(routing_strategy)
        router = self._make_router(routing_strategy, config)
        router.initialize_units(config.k_init)

        x = torch.randn(4, 32)
        out = router(x)

        # Compute loss and backward
        indices, weights = out.indices, out.weights
        loss = weights.sum() if not isinstance(weights, list) else sum(w.sum() for w in weights)

        loss.backward()

        # Check router parameters have gradients
        has_grad = False
        for name, param in router.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, f"No gradients in {routing_strategy} router"

    def test_router_numerical_stability(self, routing_strategy):
        """Test router handles edge cases without NaN."""
        config = self._make_config(routing_strategy)
        router = self._make_router(routing_strategy, config)
        router.initialize_units(config.k_init)

        # Test with extreme values
        test_cases = [
            torch.zeros(4, 32),  # zeros
            torch.ones(4, 32) * 100,  # large values
            torch.randn(4, 32) * 100,  # very large
            torch.randn(4, 32) * 1e-5,  # very small
        ]

        for x in test_cases:
            out = router(x)
            weights = out.weights
            if isinstance(weights, list):
                assert all(not torch.isnan(w).any() for w in weights)
            else:
                assert not torch.isnan(weights).any()

    def test_router_active_units_property(self, routing_strategy):
        """Test num_active_units and max_units properties."""
        config = self._make_config(routing_strategy)
        router = self._make_router(routing_strategy, config)

        assert router.max_units == 64

        router.initialize_units(16)
        # HierarchicalRouter initializes per_level = ceil(k_init/num_levels) per level
        if routing_strategy == RoutingStrategy.HIERARCHICAL:
            expected = -(-config.k_init // config.num_levels) * config.num_levels
        else:
            expected = config.k_init
        assert router.K == expected

    def test_all_routers_in_model(self):
        """Test all routers work in full NGS model."""
        for strategy in RoutingStrategy:
            config = self._make_config(strategy)
            model = build_ngs(784, 10, config)
            x = torch.randn(4, 784)
            out = model(x)

            assert out.logits.shape == (4, 10)

            # Check gradients
            out.logits.sum().backward()
            assert model.p_down.weight.grad is not None

    def test_uncertainty_aware_returns_uncertainty(self):
        """Test UncertaintyAwareRouter returns uncertainty field."""
        config = self._make_config(RoutingStrategy.UNCERTAINTY_AWARE)
        router = self._make_router(RoutingStrategy.UNCERTAINTY_AWARE, config)
        router.initialize_units(config.k_init)

        x = torch.randn(4, 32)
        out = router(x)

        assert out.uncertainty is not None
        assert out.uncertainty.shape == (4,)
        assert (out.uncertainty >= 0).all()

    def test_hierarchical_router_levels(self):
        """Test HierarchicalRouter returns level indices/weights."""
        config = self._make_config(RoutingStrategy.HIERARCHICAL)
        router = self._make_router(RoutingStrategy.HIERARCHICAL, config)
        router.initialize_units(config.k_init)

        x = torch.randn(4, 32)
        out = router(x)

        assert out.level_indices is not None
        assert out.level_weights is not None
        assert len(out.level_indices) == config.num_levels
        assert len(out.level_weights) == config.num_levels

    def test_factorized_router_levels(self):
        """Test FactorizedRouter returns level indices/weights."""
        config = self._make_config(RoutingStrategy.FACTORIZED_SUBSPACE)
        router = self._make_router(RoutingStrategy.FACTORIZED_SUBSPACE, config)
        router.initialize_units(config.k_init)

        x = torch.randn(4, 32)
        out = router(x)

        assert out.level_indices is not None
        assert out.level_weights is not None
        assert len(out.level_indices) == config.num_subspaces
        assert len(out.level_weights) == config.num_subspaces

    def test_build_router_factory(self):
        """Test build_router factory function."""
        for strategy in RoutingStrategy:
            config = self._make_config(strategy)
            router = build_router(config)
            assert router is not None
            assert hasattr(router, 'forward')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""Integration tests for all routers."""

import torch
import pytest
from mngs.core.config import MNGSConfig, RoutingStrategy
from mngs.modules.routers import MonolithicRouter, FactorizedRouter, LSRRouter
from mngs import build_mngs


class TestRouters:
    """Test all router implementations."""

    @pytest.fixture(params=[
        RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        RoutingStrategy.FACTORIZED_SUBSPACE,
        # LSH_APPROXIMATE is a placeholder implementation
    ])
    def routing_strategy(self, request):
        return request.param

    def test_router_forward_backward(self, routing_strategy):
        """Test forward and backward pass for each router."""
        if routing_strategy == RoutingStrategy.FACTORIZED_SUBSPACE:
            config = MNGSConfig(
                routing=routing_strategy,
                max_k=64,
                k_init=16,
                top_k=8,
                latent_dim=32,
                num_subspaces=4,
                top_k_factorized=2,
            )
        elif routing_strategy == RoutingStrategy.LSH_APPROXIMATE:
            config = MNGSConfig(
                routing=routing_strategy,
                max_k=64,
                k_init=16,
                top_k=8,
                latent_dim=32,
            )
        else:
            config = MNGSConfig(
                routing=routing_strategy,
                max_k=64,
                k_init=16,
                top_k=8,
                latent_dim=32,
            )

        if routing_strategy == RoutingStrategy.MONOLITHIC_MAHALANOBIS:
            router = MonolithicRouter(
                max_k=config.max_k,
                d_latent=config.latent_dim,
                top_k=config.top_k,
                tau=config.tau,
                ema_decay=config.ema_decay,
            )
        elif routing_strategy == RoutingStrategy.FACTORIZED_SUBSPACE:
            units_per_space = config.max_k // config.num_subspaces
            router = FactorizedRouter(
                d_latent=config.latent_dim,
                num_subspaces=config.num_subspaces,
                units_per_space=units_per_space,
                top_k=config.top_k_factorized,
                tau=config.tau,
            )
        elif routing_strategy == RoutingStrategy.LSH_APPROXIMATE:
            router = LSRRouter(
                d_latent=config.latent_dim,
                num_buckets=config.max_k // 4,
                num_hash_functions=4,
                top_k=config.top_k,
            )

        router.initialize_units(config.k_init)

        x = torch.randn(8, 32, requires_grad=True)
        out = router(x)

        # Check output structure (tuple of indices and weights)
        assert isinstance(out, tuple)
        assert len(out) == 2
        indices, weights = out

        # Check shapes
        if isinstance(indices, list):
            # Factorized routing
            assert len(indices) == config.num_subspaces
            assert len(weights) == config.num_subspaces
            for idx, w in zip(indices, weights):
                assert idx.shape[0] == 8  # batch size
                assert w.shape[0] == 8
                assert idx.shape == w.shape
        else:
            # Standard routing
            assert indices.shape == (8, config.top_k)
            assert weights.shape == (8, config.top_k)
            assert torch.allclose(weights.sum(dim=-1), torch.ones(8), atol=1e-4)

        # Test backward
        if isinstance(weights, list):
            loss = sum(w.sum() for w in weights)
        else:
            loss = weights.sum()
        loss.backward()
        assert x.grad is not None

    def test_router_gradient_flow(self, routing_strategy):
        """Test gradient flows through router parameters."""
        if routing_strategy == RoutingStrategy.FACTORIZED_SUBSPACE:
            config = MNGSConfig(
                routing=routing_strategy,
                max_k=64,
                k_init=16,
                top_k=8,
                latent_dim=32,
                num_subspaces=4,
                top_k_factorized=2,
            )
        elif routing_strategy == RoutingStrategy.LSH_APPROXIMATE:
            config = MNGSConfig(
                routing=routing_strategy,
                max_k=64,
                k_init=16,
                top_k=8,
                latent_dim=32,
            )
        else:
            config = MNGSConfig(
                routing=routing_strategy,
                max_k=64,
                k_init=16,
                top_k=8,
                latent_dim=32,
            )

        if routing_strategy == RoutingStrategy.MONOLITHIC_MAHALANOBIS:
            router = MonolithicRouter(
                max_k=config.max_k,
                d_latent=config.latent_dim,
                top_k=config.top_k,
                tau=config.tau,
                ema_decay=config.ema_decay,
            )
        elif routing_strategy == RoutingStrategy.FACTORIZED_SUBSPACE:
            units_per_space = config.max_k // config.num_subspaces
            router = FactorizedRouter(
                d_latent=config.latent_dim,
                num_subspaces=config.num_subspaces,
                units_per_space=units_per_space,
                top_k=config.top_k_factorized,
                tau=config.tau,
            )
        elif routing_strategy == RoutingStrategy.LSH_APPROXIMATE:
            router = LSRRouter(
                d_latent=config.latent_dim,
                num_buckets=config.max_k // 4,
                num_hash_functions=4,
                top_k=config.top_k,
            )

        router.initialize_units(config.k_init)

        x = torch.randn(4, 32)
        out = router(x)

        # Compute loss and backward
        indices, weights = out
        if isinstance(weights, list):
            loss = sum(w.sum() for w in weights)
        else:
            loss = weights.sum()

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
        if routing_strategy == RoutingStrategy.FACTORIZED_SUBSPACE:
            config = MNGSConfig(
                routing=routing_strategy,
                max_k=64,
                k_init=16,
                top_k=8,
                latent_dim=32,
                num_subspaces=4,
                top_k_factorized=2,
            )
        elif routing_strategy == RoutingStrategy.LSH_APPROXIMATE:
            config = MNGSConfig(
                routing=routing_strategy,
                max_k=64,
                k_init=16,
                top_k=8,
                latent_dim=32,
            )
        else:
            config = MNGSConfig(
                routing=routing_strategy,
                max_k=64,
                k_init=16,
                top_k=8,
                latent_dim=32,
            )

        if routing_strategy == RoutingStrategy.MONOLITHIC_MAHALANOBIS:
            router = MonolithicRouter(
                max_k=config.max_k,
                d_latent=config.latent_dim,
                top_k=config.top_k,
                tau=config.tau,
                ema_decay=config.ema_decay,
            )
        elif routing_strategy == RoutingStrategy.FACTORIZED_SUBSPACE:
            units_per_space = config.max_k // config.num_subspaces
            router = FactorizedRouter(
                d_latent=config.latent_dim,
                num_subspaces=config.num_subspaces,
                units_per_space=units_per_space,
                top_k=config.top_k_factorized,
                tau=config.tau,
            )
        elif routing_strategy == RoutingStrategy.LSH_APPROXIMATE:
            router = LSRRouter(
                d_latent=config.latent_dim,
                num_buckets=config.max_k // 4,
                num_hash_functions=4,
                top_k=config.top_k,
            )

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
            indices, weights = out
            if isinstance(weights, list):
                assert all(not torch.isnan(w).any() for w in weights)
            else:
                assert not torch.isnan(weights).any()

    def test_router_active_units_property(self, routing_strategy):
        """Test num_active_units and max_units properties."""
        if routing_strategy == RoutingStrategy.FACTORIZED_SUBSPACE:
            config = MNGSConfig(
                routing=routing_strategy,
                max_k=64,
                k_init=16,
                top_k=8,
                latent_dim=32,
                num_subspaces=4,
                top_k_factorized=2,
            )
        else:
            config = MNGSConfig(
                routing=routing_strategy,
                max_k=64,
                k_init=16,
                top_k=8,
                latent_dim=32,
            )

        if routing_strategy == RoutingStrategy.MONOLITHIC_MAHALANOBIS:
            router = MonolithicRouter(
                max_k=config.max_k,
                d_latent=config.latent_dim,
                top_k=config.top_k,
                tau=config.tau,
                ema_decay=config.ema_decay,
            )
        elif routing_strategy == RoutingStrategy.FACTORIZED_SUBSPACE:
            units_per_space = config.max_k // config.num_subspaces
            router = FactorizedRouter(
                d_latent=config.latent_dim,
                num_subspaces=config.num_subspaces,
                units_per_space=units_per_space,
                top_k=config.top_k_factorized,
                tau=config.tau,
            )
        elif routing_strategy == RoutingStrategy.LSH_APPROXIMATE:
            router = LSRRouter(
                d_latent=config.latent_dim,
                num_buckets=config.max_k // 4,
                num_hash_functions=4,
                top_k=config.top_k,
            )

        assert router.max_units == 64

        router.initialize_units(16)
        assert router.K == 16

    def test_all_routers_in_model(self):
        """Test all routers work in full MNGS model."""
        for strategy in [
            RoutingStrategy.MONOLITHIC_MAHALANOBIS,
            RoutingStrategy.FACTORIZED_SUBSPACE,
            # LSH_APPROXIMATE is a placeholder implementation
        ]:
            if strategy == RoutingStrategy.FACTORIZED_SUBSPACE:
                config = MNGSConfig(
                    routing=strategy,
                    max_k=64,
                    k_init=16,
                    top_k=8,
                    latent_dim=32,
                    num_subspaces=4,
                    top_k_factorized=2,
                )
            else:
                config = MNGSConfig(
                    routing=strategy,
                    max_k=64,
                    k_init=16,
                    top_k=8,
                    latent_dim=32,
                )

            model = build_mngs(784, 10, config)
            x = torch.randn(4, 784)
            out = model(x)

            assert out.shape == (4, 10)

            # Check gradients
            out.sum().backward()
            assert model.p_down.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""Integration tests for parameter stores."""

import torch
import pytest
from mngs.core.config import MNGSConfig, ParameterStorage
from mngs.modules.parameter_stores import DirectAdapterStore, HypernetworkStore
from mngs import build_mngs


class TestParameterStores:
    """Test all parameter store implementations."""

    @pytest.fixture(params=[ParameterStorage.DIRECT_ADAPTER, ParameterStorage.HYPERNETWORK_GENERATED])
    def param_storage(self, request):
        return request.param

    def test_param_store_init(self, param_storage):
        """Test parameter store initialization."""
        config = MNGSConfig(
            parameter_storage=param_storage,
            max_k=64,
            latent_dim=32,
            hypernetwork_code_dim=8,
            hypernetwork_hidden_dim=16,
            use_lora=True,
            lora_rank=4,
        )

        if param_storage == ParameterStorage.DIRECT_ADAPTER:
            store = DirectAdapterStore(
                max_k=config.max_k,
                d_latent=config.latent_dim,
                use_lora=config.use_lora,
                lora_rank=config.lora_rank,
            )
        else:
            store = HypernetworkStore(
                max_k=config.max_k,
                d_latent=config.latent_dim,
                code_dim=config.hypernetwork_code_dim,
                hidden_dim=config.hypernetwork_hidden_dim,
                use_lora=config.use_lora,
            )

        assert store.max_k == 64

    def test_param_store_forward(self, param_storage):
        """Test parameter store forward pass."""
        config = MNGSConfig(
            parameter_storage=param_storage,
            max_k=64,
            latent_dim=32,
            hypernetwork_code_dim=8,
            hypernetwork_hidden_dim=16,
            use_lora=True,
            lora_rank=4,
        )

        if param_storage == ParameterStorage.DIRECT_ADAPTER:
            store = DirectAdapterStore(
                max_k=config.max_k,
                d_latent=config.latent_dim,
                use_lora=config.use_lora,
                lora_rank=config.lora_rank,
            )
        else:
            store = HypernetworkStore(
                max_k=config.max_k,
                d_latent=config.latent_dim,
                code_dim=config.hypernetwork_code_dim,
                hidden_dim=config.hypernetwork_hidden_dim,
                use_lora=config.use_lora,
            )

        # Test with different index shapes
        indices = torch.randint(0, 16, (4, 8))  # [B, K]
        z = torch.randn(4, 32)

        out = store(indices, z)

        assert out.shape == (4, 8, 32)
        assert not torch.isnan(out).any()

    def test_param_store_get_parameters(self, param_storage):
        """Test get_parameters_for_indices method."""
        config = MNGSConfig(
            parameter_storage=param_storage,
            max_k=64,
            latent_dim=32,
        )

        if param_storage == ParameterStorage.DIRECT_ADAPTER:
            store = DirectAdapterStore(
                max_k=config.max_k,
                d_latent=config.latent_dim,
                use_lora=config.use_lora,
                lora_rank=config.lora_rank,
            )
        else:
            store = HypernetworkStore(
                max_k=config.max_k,
                d_latent=config.latent_dim,
                code_dim=config.hypernetwork_code_dim,
                hidden_dim=config.hypernetwork_hidden_dim,
                use_lora=config.use_lora,
            )

        indices = torch.tensor([[0, 1, 2], [3, 4, 5]])
        params = store.get_parameters_for_indices(indices)

        assert isinstance(params, dict)
        assert len(params) > 0

    def test_all_stores_in_model(self):
        """Test all parameter stores work in full MNGS model."""
        for storage in [ParameterStorage.DIRECT_ADAPTER, ParameterStorage.HYPERNETWORK_GENERATED]:
            config = MNGSConfig(
                parameter_storage=storage,
                max_k=64,
                k_init=16,
                top_k=8,
                latent_dim=32,
                hypernetwork_code_dim=8,
                use_lora=True,
                lora_rank=4,
            )

            model = build_mngs(784, 10, config)
            x = torch.randn(4, 784)
            out = model(x)

            assert out.shape == (4, 10)
            out.sum().backward()
            assert model.p_down.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""Integration tests for parameter stores."""

import torch
import pytest
from ngs.core.interfaces import NGSConfig, ParameterStorage
from ngs.modules.parameter_stores import DirectAdapterStore, HypernetworkStore, LoRAStore
from ngs.models import build_ngs


class TestParameterStores:
    """Test all parameter store implementations."""

    @pytest.fixture(params=[ParameterStorage.DIRECT_ADAPTER, ParameterStorage.HYPERNETWORK_GENERATED, ParameterStorage.LORA])
    def param_storage(self, request):
        return request.param

    def _create_store(self, param_storage, config):
        if param_storage == ParameterStorage.DIRECT_ADAPTER:
            return DirectAdapterStore(config)
        elif param_storage == ParameterStorage.HYPERNETWORK_GENERATED:
            return HypernetworkStore(config)
        else:
            return LoRAStore(config)

    def test_param_store_init(self, param_storage):
        """Test parameter store initialization."""
        config = NGSConfig(
            parameter_storage=param_storage,
            max_k=64,
            latent_dim=32,
            hypernetwork_code_dim=8,
            hypernetwork_hidden_dim=16,
            lora_rank=4,
            use_lora=True,
        )

        store = self._create_store(param_storage, config)

        assert store.max_k == 64

    def test_param_store_forward(self, param_storage):
        """Test parameter store forward pass."""
        config = NGSConfig(
            parameter_storage=param_storage,
            max_k=64,
            latent_dim=32,
            hypernetwork_code_dim=8,
            hypernetwork_hidden_dim=16,
            lora_rank=4,
            use_lora=True,
        )

        store = self._create_store(param_storage, config)

        # Test with different index shapes
        indices = torch.randint(0, 16, (4, 8))  # [B, K]
        z = torch.randn(4, 32)

        out = store(indices, z)

        assert out.shape == (4, 8, 32)
        assert not torch.isnan(out).any()

    def test_param_store_get_parameters(self, param_storage):
        """Test get_parameters_for_indices method."""
        config = NGSConfig(
            parameter_storage=param_storage,
            max_k=64,
            latent_dim=32,
            lora_rank=4,
        )

        store = self._create_store(param_storage, config)

        indices = torch.tensor([[0, 1, 2], [3, 4, 5]])
        params = store.get_parameters_for_indices(indices)

        assert isinstance(params, dict)
        assert len(params) > 0

    def test_all_stores_in_model(self):
        """Test all parameter stores work in full NGS model."""
        for storage in [ParameterStorage.DIRECT_ADAPTER, ParameterStorage.HYPERNETWORK_GENERATED, ParameterStorage.LORA]:
            config = NGSConfig(
                parameter_storage=storage,
                max_k=64,
                k_init=16,
                top_k=8,
                latent_dim=32,
                hypernetwork_code_dim=8,
                lora_rank=4,
                use_lora=True,
            )

            model = build_ngs(784, 10, config)
            x = torch.randn(4, 784)
            out = model(x)

            assert out.logits.shape == (4, 10)
            out.logits.sum().backward()
            assert any(p.grad is not None for p in model.parameters())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
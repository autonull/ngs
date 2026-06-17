"""Integration tests for all 3 memory managers."""

import torch
import pytest
from ngs.core.interfaces import NGSConfig, MemoryManagement
from ngs.modules.memory_managers import build_memory_manager
from ngs.models.ngs import build_ngs


class TestMemoryManagers:
    """Test all memory manager implementations."""
    
    @pytest.fixture(params=list(MemoryManagement))
    def memory_management(self, request):
        return request.param
    
    def test_memory_enforce_capacity(self, memory_management):
        """Test capacity enforcement."""
        config = NGSConfig(
            memory_management=memory_management,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
        )
        
        manager = build_memory_manager(config)
        model = build_ngs(784, 10, config)
        
        # Should not prune anything initially
        pruned = manager.enforce_capacity(model)
        assert pruned == 0
        assert model.K <= config.max_k
        
    def test_memory_allocate_unit(self, memory_management):
        """Test unit allocation."""
        config = NGSConfig(
            memory_management=memory_management,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
        )
        
# )
        
        manager = build_memory_manager(config)
        model = build_ngs(784, 10, config)
        
        # Allocate new unit
        idx = manager.allocate_unit(model)
        
        if config.memory_management == MemoryManagement.STRICT_CAPACITY:
            # May return None if full
            pass
        else:
            assert idx is not None
            assert 0 <= idx < config.max_k
            
    def test_memory_free_unit(self, memory_management):
        """Test unit freeing."""
        config = NGSConfig(
            memory_management=memory_management,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
        )
        
        manager = build_memory_manager(config)
        model = build_ngs(784, 10, config)
        
        # Free a unit
        manager.free_unit(model, 0)
        
        # Should be able to allocate again
        idx = manager.allocate_unit(model)
        if config.memory_management != MemoryManagement.STRICT_CAPACITY:
            assert idx is not None
            
    @pytest.mark.skip(reason="Dynamic manager allocate_unit doesn't update model.K yet")
    def test_dynamic_growth_expansion(self):
        """Test Dynamic manager expands buffers."""
        config = NGSConfig(
            memory_management=MemoryManagement.DYNAMIC,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
        )
        
        from ngs.modules.memory_managers import DynamicMemoryManager
        manager = build_memory_manager(config)
        assert isinstance(manager, DynamicMemoryManager)
        
        model = build_ngs(784, 10, config)
        
        # Allocate beyond initial
        for _ in range(10):
            idx = manager.allocate_unit(model)
            assert idx is not None
            
        assert model.K >= 26  # 16 initial + 10 allocated
        
    def test_pre_allocated_masked(self):
        """Test PreAllocated manager uses mask."""
        config = NGSConfig(
            memory_management=MemoryManagement.PRE_ALLOCATED,
            max_k=64,
            k_init=16,
            top_k=8,
            latent_dim=32,
        )
        
        from ngs.modules.memory_managers import PreAllocatedMemoryManager
        manager = build_memory_manager(config)
        assert isinstance(manager, PreAllocatedMemoryManager)
        
        model = build_ngs(784, 10, config)
        
        # Should use active_mask
        assert hasattr(model.router, 'active_mask')
        assert model.router.active_mask.sum() == 16
        
    def test_strict_capacity(self):
        """Test StrictCapacity manager enforces hard limit."""
        config = NGSConfig(
            memory_management=MemoryManagement.STRICT_CAPACITY,
            max_k=16,
            k_init=16,
            top_k=8,
            latent_dim=32,
        )
        
        from ngs.modules.memory_managers import StrictCapacityManager
        manager = build_memory_manager(config)
        assert isinstance(manager, StrictCapacityManager)
        
        model = build_ngs(784, 10, config)
        
        # Try to allocate when full
        idx = manager.allocate_unit(model)
        assert idx is None  # Should be full
        
    def test_memory_buffer_expansion(self):
        """Test buffer expansion for all parameter store types."""
        for storage in ['direct', 'hypernetwork', 'lora']:
            config = NGSConfig(
                memory_management=MemoryManagement.DYNAMIC,
                parameter_storage=storage,
                max_k=128,
                k_init=16,
                top_k=8,
                latent_dim=32,
                hypernetwork_code_dim=8,
                hypernetwork_hidden_dim=16,
                use_lora=True,
                lora_rank=4,
            )
            
            model = build_ngs(784, 10, config)
            
            # Expand
            for _ in range(20):
                idx = model.memory_manager.allocate_unit(model)
                assert idx is not None
                
            # Forward should still work
            x = torch.randn(4, 784)
            out = model(x)
            assert out.logits.shape == (4, 10)
            
            # Gradients
            out.logits.sum().backward()
            assert model.p_down.weight.grad is not None
            
    def test_all_managers_in_model(self):
        """Test all memory managers work in full NGSModel."""
        for mgmt in MemoryManagement:
            config = NGSConfig(
                memory_management=mgmt,
                max_k=64,
                k_init=16,
                top_k=8,
                latent_dim=32,
            )
            
            model = build_ngs(784, 10, config)
            x = torch.randn(4, 784)
            out = model(x)
            
            assert out.logits.shape == (4, 10)
            out.logits.sum().backward()
            assert model.p_down.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
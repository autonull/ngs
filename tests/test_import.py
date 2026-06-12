"""Test that all modules can be imported."""

def test_import_config():
    from mngs.core.config import MNGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    assert RoutingStrategy.MONOLITHIC_MAHALANOBIS is not None

def test_import_model():
    from mngs.model import MNGS, build_mngs
    assert MNGS is not None
    assert build_mngs is not None

def test_import_profiles():
    from mngs.profiles import Baseline_LeanNGS, CFG_Net_Full, Ultra_Edge_Sparse, Ablation_Hypernetwork_Only
    assert all([Baseline_LeanNGS, CFG_Net_Full, Ultra_Edge_Sparse, Ablation_Hypernetwork_Only])

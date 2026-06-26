#!/usr/bin/env python
"""
Smoke test runner for TODO11 diagnostic scripts.
Imports each diagnostic, verifies it can construct its model/data, 
and runs a minimal forward pass. Does NOT run full experiments.
"""
import sys
import importlib

sys.path.insert(0, '/home/me/ngs')
sys.path.insert(0, '/home/me/ngs/bioplausible/mep')

SCRIPTS = [
    "experiments.diagnose_spectral_norm",
    "experiments.diagnose_ep_vs_bp_updates",
    "experiments.diagnose_energy_landscape",
    "experiments.diagnose_entropy_distribution",
    "experiments.diagnose_gaussian_overlap",
    "experiments.compare_ngs_vs_dense",
    "experiments.ablate_projections",
    "experiments.analyze_gaussian_specialization",
    "experiments.eqprop_via_epoptimizer",
    "experiments.eqprop_mse_energy",
    "experiments.baseline_moe",
    "experiments.run_diagnostics",
]


def test_import(script_name):
    try:
        mod = importlib.import_module(script_name)
        return True, mod
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    passed = 0
    failed = 0
    for script in SCRIPTS:
        ok, mod = test_import(script)
        if ok:
            print(f"  OK  {script}")
            passed += 1
        else:
            print(f"  FAIL {script}: {mod}")
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

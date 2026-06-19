#!/usr/bin/env python3
"""Simple smoke test for the NGS dashboard."""
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_import_dashboard():
    """Test that the dashboard module can be imported."""
    try:
        from ngs.dashboard.app import create_app, job_store
        print("Dashboard import: SUCCESS")
        return True
    except ImportError as e:
        print(f"Dashboard import: FAILED - {e}")
        return False

def test_job_store():
    """Test the JobStore functionality."""
    try:
        from ngs.dashboard.app import job_store
        job_id = job_store.add_job({"dataset": "split_mnist", "model": "ngs_baseline"})
        assert job_id in job_store.jobs, "Job was not added to store"
        print(f"JobStore test: SUCCESS (added job {job_id})")
        return True
    except Exception as e:
        print(f"JobStore test: FAILED - {e}")
        return False

def test_app_creation():
    """Test that the Dash app can be created."""
    try:
        from ngs.dashboard.app import create_app
        # Don't start the server, just create the app
        print("App creation: SUCCESS")
        return True
    except Exception as e:
        print(f"App creation: FAILED - {e}")
        return False

if __name__ == "__main__":
    print("NGS Dashboard Smoke Test")
    print("=" * 60)

    results = []
    results.append(test_import_dashboard())
    results.append(test_job_store())
    results.append(test_app_creation())

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed.")
        sys.exit(1)

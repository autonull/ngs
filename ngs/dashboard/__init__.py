"""
NGS Dashboard Package

Provides a comprehensive web-based dashboard for configuring,
running, and analyzing NGS continual learning experiments.

Usage:
    from ngs.dashboard import create_app
    app = create_app()
    app.run(host="127.0.0.1", port=8050)

Command-line:
    python -m ngs.dashboard.app --host 127.0.0.1 --port 8050
    # Or use the convenience script:
    ./dashboard.sh
"""

from .app import create_app

__all__ = ["create_app"]

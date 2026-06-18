"""
NGS Dashboard Package

Provides web-based dashboards for configuring, running, and analyzing
NGS continual learning experiments.

Dashboards:
- Full Dashboard (app): Comprehensive with results explorer, visualizations, server management
- Simple Dashboard (simple_app): Streamlined for config + live progress + experiment history
- Component Demos (demos_app): Interactive 3D visualizations of NGS components for exploration

Usage:
    from ngs.dashboard import create_app, create_simple_app, create_demos_app
    app = create_app()  # Full dashboard
    app = create_simple_app()  # Simple dashboard
    app = create_demos_app()  # Component demos

Command-line:
    python -m ngs.dashboard.app --host 127.0.0.1 --port 8050          # Full dashboard
    python -m ngs.dashboard.simple_app --host 127.0.0.1 --port 8051   # Simple dashboard
    python -m ngs.dashboard.demos_app --host 127.0.0.1 --port 8052    # Component demos
    ./dashboard.sh [--simple] [--demos]  # Launch script
"""

from .app import create_app
from .simple_app import create_app as create_simple_app
from .demos_app import create_app as create_demos_app

__all__ = ["create_app", "create_simple_app", "create_demos_app"]
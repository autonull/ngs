"""
NGS Simple Experiment Dashboard

A streamlined dashboard for configuring and monitoring NGS continual learning experiments.
Focuses on: config sidebar + live progress + results for the current experiment.
"""

import os
import json
import time
import datetime
import traceback
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

from dash import Dash, dcc, html, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import numpy as np
import plotly.graph_objects as go

# Local imports - handle both module and script execution
try:
    from .components import (
        config_sidebar, live_monitor_graphs, job_status_table,
        experiment_card, empty_figure, accuracy_matrix_figure,
        active_units_figure, metrics_figure
    )
except ImportError:
    # Direct script execution fallback
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from components import (
        config_sidebar, live_monitor_graphs, job_status_table,
        experiment_card, empty_figure, accuracy_matrix_figure,
        active_units_figure, metrics_figure
    )

# Enable running from both source and pip-installed package
try:
    from experiments.config import EXPERIMENTS
    from experiments.runner import run_experiment
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from experiments.config import EXPERIMENTS
    from experiments.runner import run_experiment


# ------------------------------------------------------------------------------
# Model Options
# ------------------------------------------------------------------------------

MODEL_OPTIONS = [
    ("NGS Baseline", "ngs_baseline"),
    ("NGS CFG-Net", "ngs_cfg_net"),
    ("NGS Abl-Hyper", "ngs_abl_hyper"),
    ("NGS Ultra-Edge", "ngs_ultra_edge"),
    ("NGS Baseline LoRA", "ngs_baseline_lora"),
    ("NGS CFG-Net LoRA", "ngs_cfg_net_lora"),
    ("NGS Abl-Hyper LoRA", "ngs_abl_hyper_lora"),
]


# ------------------------------------------------------------------------------
# Global State (single-process, no external dependencies)
# ------------------------------------------------------------------------------

class ExperimentStore:
    """In-memory experiment store with persistence to disk."""
    
    def __init__(self, results_dir: Path = Path("./results")):
        self._lock = threading.Lock()
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True)
        self.experiments: Dict[str, dict] = {}
        self._load_existing()
        
    def _load_existing(self):
        """Load completed experiments from results directory."""
        for f in self.results_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                job_id = f.stem
                self.experiments[job_id] = {
                    "id": job_id,
                    "config": {
                        "dataset": data.get("config", ""),
                        "model": data.get("model", ""),
                        "seed": data.get("seed", 42),
                    },
                    "status": "completed",
                    "progress": 1.0,
                    "created_at": datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    "result": data,
                }
            except Exception:
                pass
    
    def add_experiment(self, config: dict) -> str:
        job_id = f"exp_{len(self.experiments)}_{int(time.time())}"
        with self._lock:
            self.experiments[job_id] = {
                "id": job_id,
                "config": config,
                "status": "pending",
                "progress": 0.0,
                "created_at": datetime.datetime.now().isoformat(),
                "result": None,
            }
        return job_id
    
    def update_status(self, job_id: str, status: str, progress: float = None, result: dict = None, error: str = None):
        with self._lock:
            if job_id in self.experiments:
                self.experiments[job_id]["status"] = status = status
                if progress is not None:
                    self.experiments[job_id]["progress"] = progress
                if result is not None:
                    self.experiments[job_id]["result"] = result
                if error is not None:
                    self.experiments[job_id]["error"] = error
    
    def get_all(self) -> List[dict]:
        with self._lock:
            return list(self.experiments.values())
    
    def get_latest(self) -> Optional[dict]:
        with self._lock:
            if not self.experiments:
                return None
            return max(self.experiments.values(), key=lambda x: x["created_at"])


exp_store = ExperimentStore()


# ------------------------------------------------------------------------------
# App Layout
# ------------------------------------------------------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "NGS Experiment Dashboard"

app.layout = dbc.Container([
    dcc.Interval(id="interval-component", interval=2000, n_intervals=0),
    dcc.Store(id="selected-exp-store"),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("NGS Experiment Dashboard", className="text-light mt-3 mb-1"),
            html.P("Configure, launch, and monitor continual learning experiments", className="text-muted mb-4"),
        ], width=12)
    ]),
    
    # Main layout: Sidebar + Content
    dbc.Row([
        # Sidebar: Configuration
        dbc.Col([
            config_sidebar(EXPERIMENTS, MODEL_OPTIONS),
        ], width=3),
        
        # Main Content
        dbc.Col([
            # Live Monitor
            live_monitor_graphs(),
            
            html.Hr(className="my-4"),
            
            # Experiment History (focus on list of experiments)
            html.H4("Experiment History", className="text-info mb-3"),
            html.Div(id="experiment-list-container"),
        ], width=9),
    ]),
], fluid=True, style={"backgroundColor": "#0f0f1a", "minHeight": "100vh", "padding": "20px"})


# ------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------

@app.callback(
    [Output("btn-launch", "children"), Output("btn-launch", "disabled"), Output("launch-status", "children")],
    Input("btn-launch", "n_clicks"),
    State("dataset-selector", "value"),
    State("model-profile-selector", "value"),
    State("seed-selector", "value"),
    State("epochs-slider", "value"),
    State("lr-slider", "value"),
    State("wd-slider", "value"),
    State("batch-size-selector", "value"),
    State("topk-slider", "value"),
    State("max-k-slider", "value"),
    State("split-thresh-slider", "value"),
    State("prune-thresh-slider", "value"),
    prevent_initial_call=True,
)
def launch_experiment(n_clicks, dataset, model_profile, seed, epochs, lr, wd, batch_size,
                       top_k, max_k, split_thresh, prune_thresh):
    if n_clicks is None:
        raise PreventUpdate
    
    config = {
        "dataset": dataset,
        "model": model_profile,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": wd,
        "batch_size": batch_size,
        "top_k": top_k,
        "max_k": max_k,
        "split_thresh": split_thresh,
        "prune_thresh": prune_thresh,
    }
    
    job_id = exp_store.add_experiment(config)
    
    def run_job():
        try:
            exp_store.update_status(job_id, "running", progress=0.1)
            exp_config = EXPERIMENTS[dataset]
            result = run_experiment(
                config=exp_config,
                model_name=model_profile,
                seed=seed,
                output_dir="./results",
                verbose=True,
            )
            exp_store.update_status(job_id, "completed", progress=1.0, result=result)
        except Exception as e:
            exp_store.update_status(job_id, "failed", error=str(e))
            traceback.print_exc()
    
    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()
    
    return "Launched!", True, html.Span(f"Started {job_id}", className="text-success")


@app.callback(
    Output("live-accuracy-graph", "figure"),
    Output("live-loss-graph", "figure"),
    Output("live-k-graph", "figure"),
    Output("experiment-list-container", "children"),
    Input("interval-component", "n_intervals"),
)
def update_live_view(n):
    experiments = exp_store.get_all()
    
    # Find running experiment for live graphs
    running = [e for e in experiments if e["status"] == "running"]
    latest_completed = None
    for e in sorted(experiments, key=lambda x: x["created_at"], reverse=True):
        if e["status"] == "completed":
            latest_completed = e
            break
    
    # Live graphs - show running or latest completed
    active = running[0] if running else latest_completed
    
    if active and active.get("result"):
        result = active["result"]
        acc_matrix = np.array(result.get("accuracy_matrix", [[0]]))
        active_units = result.get("active_units", [])
        metrics = result.get("metrics", {})
        
        fig_acc = accuracy_matrix_figure(acc_matrix)
        fig_units = active_units_figure(active_units) if active_units else empty_figure("Active Units")
        fig_metrics = metrics_figure(metrics) if metrics else empty_figure("Metrics")
    else:
        fig_acc = empty_figure("Accuracy Matrix")
        fig_units = empty_figure("Active Units")
        fig_metrics = empty_figure("Metrics")
    
    # Experiment list - show all experiments, newest first
    if not experiments:
        exp_list = html.P("No experiments yet. Launch one from the sidebar.", className="text-muted")
    else:
        sorted_exps = sorted(experiments, key=lambda x: x["created_at"], reverse=True)
        cards = []
        for e in sorted_exps:
            cards.append(experiment_card(
                e["id"], e["status"], e.get("progress", 0),
                e["config"], e["created_at"], e.get("result")
            ))
        exp_list = html.Div(cards)
    
    return fig_acc, fig_units, fig_metrics, exp_list


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def create_app():
    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NGS Simple Experiment Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    app.run(host=args.host, port=args.port, debug=args.debug)
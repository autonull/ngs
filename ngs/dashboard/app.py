"""
NGS Interactive Experiment Dashboard

A comprehensive Dash application for running, monitoring, and analyzing
Neural Gaussian System (NGS) continual learning experiments.

Features:
- Experiment Config Panel (Sidebar): Configure model, training, and dataset parameters
- Task Launcher (Main Area — Top): Select and launch training jobs
- Live Training Monitor (Main Area — Middle): Real-time accuracy/loss curves
- Visualization Suite (Tabbed): Topology, routing, Gaussian means, etc.
- Result Explorer (Tabbed): Browse and compare completed runs
- Server Management: Configurable port, host, CPU/GPU device selection

No external dependencies like Redis; all state managed in-process.
"""

import os
import json
import time
import datetime
import traceback
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque

from dash import Dash, dcc, html, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Enable running from both source and pip-installed package
try:
    from experiments.config import EXPERIMENTS, ModelConfig, TrainConfig
    from experiments.runner import run_experiment
    from experiments.ngs_trainer import create_ngs_from_profile, PROFILE_TRAIN_CONFIGS
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from experiments.config import EXPERIMENTS, ModelConfig, TrainConfig
    from experiments.runner import run_experiment
    from experiments.ngs_trainer import create_ngs_from_profile, PROFILE_TRAIN_CONFIGS


# ------------------------------------------------------------------------------
# Global State (single-process, no external dependencies)
# ------------------------------------------------------------------------------

class JobStore:
    """In-memory job queue and result store."""

    def __init__(self):
        self._lock = threading.Lock()
        self.jobs: Dict[str, dict] = {}  # job_id -> job dict
        self.results_dir = Path("./results")
        self.results_dir.mkdir(exist_ok=True)
        self._callbacks = []

    def add_job(self, config: dict) -> str:
        job_id = f"job_{len(self.jobs)}_{int(time.time())}"
        self.jobs[job_id] = {
            "id": job_id,
            "config": config,
            "status": "pending",
            "progress": 2,
            "created_at": datetime.datetime.now().isoformat(),
        }
        return job_id


job_store = JobStore()

# ------------------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "NGS Experiment Dashboard"

SIDEBAR_WIDTH = "320px"

app.layout = dbc.Container([
    dcc.Interval(id="interval-component", interval=2000, n_intervals=0),

    # Header
    dbc.Row([
        dbc.Col(html.H1("NGS Experiment Dashboard", className="text-light mt-3 mb-4"), width=12)
    ]),

    dbc.Row([
        # Sidebar: Experiment Config Panel
        dbc.Col([
            html.H4("Experiment Config", className="text-info mb-3"),

            # Dataset / Benchmark selector
            html.Label("Dataset / Benchmark", className="text-light mt-2"),
            dcc.Dropdown(
                id="dataset-selector",
                options=[{"label": cfg.name, "value": name} for name, cfg in EXPERIMENTS.items()],
                value="split_mnist",
                clearable=False,
                className="mb-3",
            ),

            # Model Profile selector
            html.Label("Model Profile", className="text-light mt-2"),
            dcc.Dropdown(
                id="model-profile-selector",
                options=[
                    {"label": "NGS Baseline", "value": "ngs_baseline"},
                    {"label": "NGS CFG-Net", "value": "ngs_cfg_net"},
                    {"label": "NGS Abl-Hyper", "value": "ngs_abl_hyper"},
                    {"label": "NGS Ultra-Edge", "value": "ngs_ultra_edge"},
                    {"label": "NGS Baseline LoRA", "value": "ngs_baseline_lora"},
                    {"label": "NGS CFG-Net LoRA", "value": "ngs_cfg_net_lora"},
                    {"label": "NGS Abl-Hyper LoRA", "value": "ngs_abl_hyper_lora"},
                ],
                value="ngs_baseline",
                clearable=False,
                className="mb-3",
            ),

            # Seeds and epochs
            html.Label("Seed", className="text-light mt-2"),
            dcc.Dropdown(
                id="seed-selector",
                options=[{"label": str(s), "value": s} for s in [42, 123, 456, 789, 2024]],
                value=42,
                clearable=False,
                className="mb-3",
            ),

            html.Label("Epochs per Task", className="text-light mt-2"),
            dcc.Slider(
                id="epochs-slider",
                min=1, max=20, step=1, value=2,
                marks={1: "1", 5: "5", 10: "10", 15: "15", 20: "20"},
                className="mb-4",
            ),

            # Advanced: hyperparameters
            html.Hr(),
            html.H6("Hyperparameters", className="text-secondary"),

            html.Label("Learning Rate", className="text-light mt-2"),
            dcc.Slider(
                id="lr-slider",
                min=3e-4, max=3e-3, step=None,
                marks={3e-4: "3e-4", 1e-3: "1e-3", 3e-3: "3e-3"},
                value=1e-3,
                className="mb-3",
            ),

            html.Label("Weight Decay", className="text-light mt-2"),
            dcc.Slider(
                id="wd-slider",
                min=1e-5, max=1e-2, step=None,
                marks={1e-5: "1e-5", 1e-4: "1e-4", 1e-3: "1e-3", 1e-2: "1e-2"},
                value=1e-4,
                className="mb-3",
            ),

            html.Label("Batch Size", className="text-light mt-2"),
            dcc.Dropdown(
                id="batch-size-selector",
                options=[{"label": str(b), "value": b} for b in [32, 64, 128, 256, 512]],
                value=256,
                clearable=False,
                className="mb-3",
            ),

            html.Label("Top-K", className="text-light mt-2"),
            dcc.Slider(
                id="topk-slider",
                min=1, max=64, step=1, value=8,
                marks={1: "1", 8: "8", 16: "16", 32: "32", 64: "64"},
                className="mb-3",
            ),

            html.Label("Max Units (K)", className="text-light mt-2"),
            dcc.Slider(
                id="max-k-slider",
                min=64, max=1024, step=64, value=448,
                marks={64: "64", 256: "256", 448: "448", 768: "768", 1024: "1024"},
                className="mb-3",
            ),

            # Training controls
            html.Hr(),
            html.H6("Topology Controls", className="text-secondary"),

            html.Label("Split Threshold", className="text-light mt-2"),
            dcc.Slider(
                id="split-thresh-slider",
                min=0.0, max=0.2, step=0.005, value=0.05,
                marks={0: "0", 0.05: "0.05", 0.1: "0.1", 0.15: "0.15", 0.2: "0.2"},
                className="mb-3",
            ),

            html.Label("Prune Threshold", className="text-light mt-2"),
            dcc.Slider(
                id="prune-thresh-slider",
                min=0.0, max=0.1, step=0.005, value=0.01,
                marks={0: "0", 0.025: "0.025", 0.05: "0.05", 0.075: "0.075", 0.1: "0.1"},
                className="mb-3",
            ),

            # Export config button
            html.Hr(),
            dbc.Button("Save Config as YAML", id="btn-save-config", color="info", className="w-100 mb-2"),
            dcc.Download(id="download-config"),
            dbc.Button("Launch Training", id="btn-launch", color="success", className="w-100 mt-2"),

        ], width=3, style={"backgroundColor": "#1a1a2e", "padding": "15px", "borderRadius": "8px"}),

        # Main Area
        dbc.Col([
            dcc.Tabs([
                # Tab 1: Task Launcher & Running Jobs
                dcc.Tab(label="Task Launcher", children=[
                    html.H4("Running Jobs", className="text-info mt-3 mb-3"),
                    html.Div(id="jobs-table-container", children=[
                        html.P("No jobs running. Configure and launch a job from the sidebar.", className="text-muted")
                    ]),
                ]),

                # Tab 2: Live Training Monitor
                dcc.Tab(label="Live Monitor", children=[
                    html.H4("Live Training Monitor", className="text-info mt-3 mb-3"),
                    dcc.Graph(id="live-accuracy-graph", config={"displayModeBar": True}, className="mb-4"),
                    dcc.Graph(id="live-loss-graph", config={"displayModeBar": True}, className="mb-4"),
                    dcc.Graph(id="live-k-graph", config={"displayModeBar": True}),
                ]),

                # Tab 3: Visualization Suite
                dcc.Tab(label="Visualizations", children=[
                    html.H4("Experiment Visualizations", className="text-info mt-3 mb-3"),

                    html.Label("Select Result File:", className="text-light mt-2"),
                    dcc.Dropdown(id="result-file-selector", className="mb-4"),

                    dcc.Tabs([
                        dcc.Tab(label="Accuracy Matrix", children=[
                            dcc.Graph(id="viz-accuracy-matrix", config={"displayModeBar": True})
                        ]),
                        dcc.Tab(label="Active Units", children=[
                            dcc.Graph(id="viz-active-units", config={"displayModeBar": True})
                        ]),
                        dcc.Tab(label="Metrics", children=[
                            dcc.Graph(id="viz-metrics", config={"displayModeBar": True})
                        ]),
                    ]),
                ]),

                # Tab 4: Result Explorer
                dcc.Tab(label="Result Explorer", children=[
                    html.H4("Result Explorer", className="text-info mt-3 mb-3"),

                    html.Label("Filter by Model:", className="text-light mt-2"),
                    dcc.Dropdown(id="filter-model", multi=True, className="mb-3"),

                    html.Label("Filter by Dataset:", className="text-light mt-2"),
                    dcc.Dropdown(id="filter-dataset", multi=True, className="mb-3"),

                    html.Div(id="results-table-container", className="mt-4"),

                    dbc.Button("Export CSV", id="btn-export-csv", color="primary", className="mt-3"),
                    dcc.Download(id="download-csv"),
                ]),

                # Tab 5: Server Management
                dcc.Tab(label="Server", children=[
                    html.H4("Server Management", className="text-info mt-3 mb-3"),

                    html.Label("Device:", className="text-light mt-2"),
                    dcc.Dropdown(
                        id="device-selector",
                        options=[{"label": "CPU", "value": "cpu"}, {"label": "GPU (if available)", "value": "cuda"}],
                        value="cuda",
                        clearable=False,
                        className="mb-3",
                    ),

                    html.Label("Results Directory:", className="text-light mt-2"),
                    dcc.Input(id="results-dir", value="./results", className="mb-3", style={"width": "100%"}),

                    html.Hr(),
                    html.H6("Server Info", className="text-secondary"),
                    html.P(id="server-info", className="text-light"),
                ]),
            ]),
        ], width=9),
    ]),
], fluid=True, style={"backgroundColor": "#0f0f1a", "minHeight": "100vh", "padding": "20px"})


# ------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------

@app.callback(
    Output("live-accuracy-graph", "figure"),
    Output("live-loss-graph", "figure"),
    Output("live-k-graph", "figure"),
    Output("jobs-table-container", "children"),
    Input("interval-component", "n_intervals"),
)
def update_live_monitor(n):
    fig_acc = go.Figure()
    fig_acc.update_layout(
        title="Live Accuracy per Task",
        xaxis_title="Task",
        yaxis_title="Accuracy",
        template="plotly_dark",
        height=350,
    )

    fig_loss = go.Figure()
    fig_loss.update_layout(
        title="Live Loss per Task",
        xaxis_title="Task",
        yaxis_title="Loss",
        template="plotly_dark",
        height=350,
    )

    fig_k = go.Figure()
    fig_k.update_layout(
        title="Active Units (K) over Tasks",
        xaxis_title="Task",
        yaxis_title="Active Units",
        template="plotly_dark",
        height=350,
    )

    active_jobs = [j for j in job_store.jobs.values() if j["status"] in ("pending", "running")]
    if not active_jobs:
        jobs_div = html.P("No jobs running. Configure and launch a job from the sidebar.", className="text-muted")
    else:
        rows = []
        for j in job_store.jobs.values():
            rows.append(html.Tr([
                html.Td(j["id"]),
                html.Td(j["status"]),
                html.Td(f"{j['progress']:.0%}"),
                html.Td(j["created_at"]),
            ]))
        jobs_div = dbc.Table([
            html.Thead(html.Tr([html.Th("Job ID"), html.Th("Status"), html.Th("Progress"), html.Th("Created")])),
            html.Tbody(rows),
        ], striped=True, bordered=True, hover=True, responsive=True)

    return fig_acc, fig_loss, fig_k, jobs_div


@app.callback(
    [Output("btn-launch", "children"), Output("btn-launch", "disabled")],
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
def launch_training(n_clicks, dataset, model_profile, seed, epochs, lr, wd, batch_size,
                     top_k, max_k, split_thresh, prune_thresh):
    if n_clicks is None:
        raise PreventUpdate

    config = {
        "dataset": dataset,
        "model_profile": model_profile,
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

    job_id = job_store.add_job(config)

    def run_job():
        try:
            job_store.jobs[job_id]["status"] = "running"
            exp_config = EXPERIMENTS[dataset]
            result = run_experiment(
                config=exp_config,
                model_name=model_profile,
                seed=seed,
                output_dir="./results",
                verbose=True,
            )
            job_store.jobs[job_id]["status"] = "completed"
            job_store.jobs[job_id]["result"] = result
            job_store.jobs[job_id]["progress"] = 1.0
        except Exception as e:
            job_store.jobs[job_id]["status"] = "failed"
            job_store.jobs[job_id]["error"] = str(e)
            traceback.print_exc()

    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()

    return "Training Launched!", True


@app.callback(
    Output("result-file-selector", "options"),
    Input("interval-component", "n_intervals"),
)
def update_result_files(n):
    results_dir = Path("./results")
    if not results_dir.exists():
        return []
    files = sorted([f.name for f in results_dir.glob("*.json") if f.is_file()])
    return [{"label": f, "value": f} for f in files]


@app.callback(
    Output("viz-accuracy-matrix", "figure"),
    Output("viz-active-units", "figure"),
    Output("viz-metrics", "figure"),
    Input("result-file-selector", "value"),
)
def update_visualizations(filename):
    if not filename:
        empty = go.Figure()
        empty.update_layout(template="plotly_dark", title="No data selected")
        return empty, empty, empty

    file_path = Path("./results") / filename
    try:
        with open(file_path) as f:
            data = json.load(f)
    except Exception:
        error_fig = go.Figure()
        error_fig.update_layout(template="plotly_dark", title="Error loading data")
        return error_fig, error_fig, error_fig

    acc_matrix = np.array(data["accuracy_matrix"])
    fig_acc = go.Figure(data=go.Heatmap(
        z=acc_matrix,
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        text=np.round(acc_matrix, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig_acc.update_layout(
        title="Accuracy Matrix (Task x Task)",
        xaxis_title="Evaluated After Task",
        yaxis_title="Task",
        template="plotly_dark",
        height=500,
    )

    active_units = data.get("active_units", [])
    if active_units:
        fig_units = go.Figure(data=go.Scatter(
            x=list(range(len(active_units))),
            y=active_units,
            mode="lines+markers",
            line={"color": "cyan"},
        ))
    else:
        fig_units = go.Figure()
    fig_units.update_layout(
        title="Active Units over Time",
        xaxis_title="Task",
        yaxis_title="Active Units",
        template="plotly_dark",
        height=350,
    )

    metrics = data.get("metrics", {})
    metric_names = ["avg_final_accuracy", "avg_forgetting", "bwt", "fwt", "la"]
    metric_values = [metrics.get(m, 0) for m in metric_names]
    fig_metrics = go.Figure(data=go.Bar(
        x=metric_names,
        y=metric_values,
        marker_color=["#2E86AB", "#DC3545", "#FD7E14", "#28A745", "#6C4AB6"],
    ))
    fig_metrics.update_layout(
        title="CL Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        template="plotly_dark",
        height=350,
    )

    return fig_acc, fig_units, fig_metrics


@app.callback(
    Output("results-table-container", "children"),
    Output("filter-model", "options"),
    Output("filter-dataset", "options"),
    Input("interval-component", "n_intervals"),
    State("filter-model", "value"),
    State("filter-dataset", "value"),
)
def update_result_explorer(n, model_filter, dataset_filter):
    results_dir = Path("./results")
    if not results_dir.exists():
        return html.P("No results found.", className="text-muted"), [], []

    all_results = []
    for f in results_dir.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            all_results.append(data)
        except Exception:
            continue

    models = sorted(set(r.get("model", "") for r in all_results))
    datasets = sorted(set(r.get("config", "") for r in all_results))

    if model_filter:
        all_results = [r for r in all_results if r.get("model") in model_filter]
    if dataset_filter:
        all_results = [r for r in all_results if r.get("config") in dataset_filter]

    if not all_results:
        return html.P("No results match the filters.", className="text-muted"), [], []

    rows = []
    for r in all_results:
        metrics = r.get("metrics", {})
        rows.append(html.Tr([
            html.Td(r.get("config", "")),
            html.Td(r.get("model", "")),
            html.Td(r.get("seed", "")),
            html.Td(f"{metrics.get('avg_final_accuracy', 0):.4f}"),
            html.Td(f"{metrics.get('avg_forgetting', 0):.4f}"),
            html.Td(f"{metrics.get('bwt', 0):.4f}"),
        ]))

    table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Dataset"), html.Th("Model"), html.Th("Seed"),
            html.Th("Accuracy"), html.Th("Forgetting"), html.Th("BWT"),
        ])),
        html.Tbody(rows),
    ], striped=True, bordered=True, hover=True, responsive=True)

    model_options = [{"label": m, "value": m} for m in models]
    dataset_options = [{"label": d, "value": d} for d in datasets]

    return table, model_options, dataset_options


@app.callback(
    Output("server-info", "children"),
    Input("interval-component", "n_intervals"),
)
def update_server_info(n):
    import platform
    info = [
        f"Platform: {platform.platform()}",
        f"Python: {platform.python_version()}",
        f"Working Directory: {os.getcwd()}",
        f"Total Jobs: {len(job_store.jobs)}",
        f"Results Directory: {job_store.results_dir}",
    ]
    return html.Ul([html.Li(line, className="text-light") for line in info])


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def create_app():
    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NGS Experiment Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)
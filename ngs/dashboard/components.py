"""
Shared UI components for NGS dashboards.
"""
from dash import dcc, html
import dash_bootstrap_components as dbc


def config_sidebar(experiment_options, model_options, default_experiment="split_mnist", default_model="ngs_baseline"):
    """Reusable experiment configuration sidebar."""
    return html.Div([
        html.H4("Experiment Config", className="text-info mb-3"),
        
        html.Label("Dataset / Benchmark", className="text-light mt-2"),
        dcc.Dropdown(
            id="dataset-selector",
            options=[{"label": exp.name, "value": key} for key, exp in experiment_options.items()],
            value=default_experiment,
            clearable=False,
            className="mb-3",
        ),
        
        html.Label("Model Profile", className="text-light mt-2"),
        dcc.Dropdown(
            id="model-profile-selector",
            options=[{"label": label, "value": value} for label, value in model_options],
            value=default_model,
            clearable=False,
            className="mb-3",
        ),
        
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
        
        html.Hr(),
        dbc.Button("Launch Training", id="btn-launch", color="success", className="w-100 mt-2"),
        html.Div(id="launch-status", className="mt-2 text-light small"),
    ], style={"backgroundColor": "#1a1a2e", "padding": "15px", "borderRadius": "8px"})


def live_monitor_graphs():
    """Reusable live monitoring graphs."""
    return html.Div([
        html.H4("Live Training Monitor", className="text-info mt-3 mb-3"),
        dcc.Graph(id="live-accuracy-graph", config={"displayModeBar": True}, className="mb-4"),
        dcc.Graph(id="live-loss-graph", config={"displayModeBar": True}, className="mb-4"),
        dcc.Graph(id="live-k-graph", config={"displayModeBar": True}),
    ])


def job_status_table():
    """Reusable running jobs table."""
    return html.Div([
        html.H4("Running Experiments", className="text-info mt-3 mb-3"),
        html.Div(id="jobs-table-container", children=[
            html.P("No experiments running. Launch one from the sidebar.", className="text-muted")
        ]),
    ])


def experiment_card(job_id, status, progress, config, created_at, result=None):
    """Card showing a single experiment with its status and results."""
    status_colors = {
        "pending": "secondary",
        "running": "primary",
        "completed": "success",
        "failed": "danger",
    }
    color = status_colors.get(status, "secondary")
    
    progress_bar = dbc.Progress(value=progress*100, color=color, className="mb-2", style={"height": "8px"})
    
    config_items = []
    for k, v in config.items():
        config_items.append(html.Li(f"{k}: {v}"))
    
    card_body = [
        html.H6(job_id, className="text-light"),
        html.Span(f"Status: {status}", className=f"badge bg-{color} me-2"),
        html.Small(f"Created: {created_at}", className="text-muted ms-2"),
        progress_bar,
        html.Ul(config_items, className="small text-light mt-2"),
    ]
    
    if result and status == "completed":
        metrics = result.get("metrics", {})
        card_body.append(html.Hr())
        card_body.append(html.H6("Results", className="text-info"))
        for metric in ["avg_final_accuracy", "avg_forgetting", "bwt", "fwt", "la"]:
            if metric in metrics:
                card_body.append(html.P(f"{metric}: {metrics[metric]:.4f}", className="small text-light mb-1"))
    
    return dbc.Card([
        dbc.CardBody(card_body)
    ], className="mb-3")


def empty_figure(title="No data"):
    """Create an empty plotly figure with a message."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=350,
        annotations=[{
            "text": "Run an experiment to see live data",
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5, "showarrow": False,
            "font": {"size": 16, "color": "gray"}
        }]
    )
    return fig


def accuracy_matrix_figure(acc_matrix):
    """Create accuracy matrix heatmap."""
    import plotly.graph_objects as go
    import numpy as np
    arr = np.array(acc_matrix)
    fig = go.Figure(data=go.Heatmap(
        z=arr,
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        text=np.round(arr, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig.update_layout(
        title="Accuracy Matrix (Task × Task)",
        xaxis_title="Evaluated After Task",
        yaxis_title="Task",
        template="plotly_dark",
        height=500,
    )
    return fig


def active_units_figure(active_units):
    """Create active units timeline."""
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Scatter(
        x=list(range(len(active_units))),
        y=active_units,
        mode="lines+markers",
        line={"color": "cyan"},
    ))
    fig.update_layout(
        title="Active Units over Time",
        xaxis_title="Task",
        yaxis_title="Active Units",
        template="plotly_dark",
        height=350,
    )
    return fig


def metrics_figure(metrics):
    """Create CL metrics bar chart."""
    import plotly.graph_objects as go
    metric_names = ["avg_final_accuracy", "avg_forgetting", "bwt", "fwt", "la"]
    metric_values = [metrics.get(m, 0) for m in metric_names]
    fig = go.Figure(data=go.Bar(
        x=metric_names,
        y=metric_values,
        marker_color=["#2E86AB", "#DC3545", "#FD7E14", "#28A745", "#6C4AB6"],
    ))
    fig.update_layout(
        title="CL Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        template="plotly_dark",
        height=350,
    )
    return fig
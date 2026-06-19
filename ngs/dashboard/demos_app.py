"""
NGS Component Demos Dashboard - Clean, Intuitive Design

Organized as a guided exploration:
1. Model Selector (persistent left)
2. Visualization Tabs (logical flow: Core -> Routing -> Structure -> Advanced)
3. Contextual Controls (right panel, changes per tab)
"""

import os
import time
import threading
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output, State, callback_context, ALL, MATCH
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

try:
    from .components import empty_figure
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from components import empty_figure

try:
    from ngs.models.ngs import build_ngs
    from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from ngs.models.ngs import build_ngs
    from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement

import torch

# ========================================================================
# STATE
# ========================================================================

class DemoStore:
    def __init__(self):
        self._lock = threading.Lock()
        self.models = {}
        self.current_key = None
    
    def get_or_create_model(self, config_dict):
        key = str(sorted(config_dict.items()))
        with self._lock:
            if key not in self.models:
                config = NGSConfig(**config_dict)
                model = build_ngs(784, 10, config)
                self._init_weights(model)
                self.models[key] = model
            self.current_key = key
            return self.models[key]
    
    def _init_weights(self, model):
        for name, param in model.named_parameters():
            if 'router.mu' in name:
                torch.nn.init.normal_(param, mean=0.0, std=1.0)
            elif 'router.log_s' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'router.log_alpha' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'split_gate' in name:
                torch.nn.init.constant_(param, 0.5)
            elif 'param_store.W_A' in name or 'param_store.lora_A' in name:
                torch.nn.init.kaiming_uniform_(param, a=5**0.5)
            elif 'param_store.W_B' in name or 'param_store.lora_B' in name:
                torch.nn.init.zeros_(param)
            elif 'param_store.codes' in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'hypernet' in name and 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'hypernet' in name and 'bias' in name:
                torch.nn.init.zeros_(param)
            elif 'subspace_projectors' in name and 'weight' in name:
                torch.nn.init.orthogonal_(param)

demo_store = DemoStore()

# ========================================================================
# MODEL CONFIGS
# ========================================================================

ROUTING_STRATEGIES = {
    "factorized": {
        "name": "Factorized Subspace",
        "desc": "4 independent subspaces × hypernetwork adapters. Best for continual learning.",
        "icon": "🔷",
        "config": {
            "routing": RoutingStrategy.FACTORIZED_SUBSPACE,
            "parameter_storage": ParameterStorage.HYPERNETWORK_GENERATED,
            "topology_control": TopologyControl.CONTINUOUS_DENSITY,
            "memory_management": MemoryManagement.PRE_ALLOCATED,
            "latent_dim": 32, "max_k": 256, "k_init": 64, "top_k": 8,
            "num_subspaces": 4, "hypernetwork_code_dim": 8, "hypernetwork_hidden_dim": 16,
        }
    },
    "monolithic": {
        "name": "Monolithic Mahalanobis",
        "desc": "Single Gaussian mixture with full covariance. Baseline for comparison.",
        "icon": "🔴",
        "config": {
            "routing": RoutingStrategy.MONOLITHIC_MAHALANOBIS,
            "parameter_storage": ParameterStorage.DIRECT_ADAPTER,
            "topology_control": TopologyControl.DISCRETE_HEURISTIC,
            "memory_management": MemoryManagement.PRE_ALLOCATED,
            "latent_dim": 32, "max_k": 256, "k_init": 64, "top_k": 8,
        }
    },
    "lsh": {
        "name": "LSH Approximate",
        "desc": "Locality-sensitive hashing for extreme scale (1000+ units).",
        "icon": "🟡",
        "config": {
            "routing": RoutingStrategy.LSH_APPROXIMATE,
            "parameter_storage": ParameterStorage.LORA,
            "topology_control": TopologyControl.DISCRETE_HEURISTIC,
            "memory_management": MemoryManagement.STRICT_CAPACITY,
            "latent_dim": 32, "max_k": 1024, "k_init": 128, "top_k": 16,
            "lora_rank": 4,
        }
    },
    "hierarchical": {
        "name": "Hierarchical",
        "desc": "Multi-level coarse-to-fine routing. Good for structured data.",
        "icon": "🟣",
        "config": {
            "routing": RoutingStrategy.HIERARCHICAL,
            "parameter_storage": ParameterStorage.HYPERNETWORK_GENERATED,
            "topology_control": TopologyControl.CONTINUOUS_DENSITY,
            "memory_management": MemoryManagement.PRE_ALLOCATED,
            "latent_dim": 32, "max_k": 256, "k_init": 64, "top_k": 8,
            "num_subspaces": 4, "hypernetwork_code_dim": 8,
        }
    },
    "gaussian_attention": {
        "name": "Gaussian Attention",
        "desc": "Attention-based routing with uncertainty estimates.",
        "icon": "🟠",
        "config": {
            "routing": RoutingStrategy.GAUSSIAN_ATTENTION,
            "parameter_storage": ParameterStorage.DIRECT_ADAPTER,
            "topology_control": TopologyControl.CONTINUOUS_DENSITY,
            "memory_management": MemoryManagement.PRE_ALLOCATED,
            "latent_dim": 32, "max_k": 256, "k_init": 64, "top_k": 8,
        }
    },
}

# Tab definitions in logical learning order
VIS_TABS = [
    ("gaussians", "🧠 Gaussian Means", "Core representation — where units live in latent space"),
    ("routing", "🎯 Routing", "How inputs select their top-K units"),
    ("codes", "🔮 Hypernetwork Codes", "Generated adapter parameters (factorized/hierarchical only)"),
    ("subspaces", "📐 Subspaces", "Subspace alignment & independence (factorized/hierarchical only)"),
    ("uncertainty", "📊 Uncertainty", "Calibration & confidence (gaussian_attention only)"),
    ("geodesics", "🌈 Geodesics", "Manifold paths between units"),
    ("topology", "🏗️ Topology", "Active units vs capacity"),
]

# ========================================================================
# DATA EXTRACTION
# ========================================================================

def extract_data(model):
    router = model.router
    data = {}
    
    if hasattr(router, "mu") and hasattr(router, "active_mask"):
        mu = router.mu.detach().cpu().numpy()
        active_mask = router.active_mask.detach().cpu().numpy()
        if mu.ndim == 3:
            mu_flat = mu.reshape(-1, mu.shape[-1])
        else:
            mu_flat = mu
        active_idx = np.where(active_mask)[0]
        if len(active_idx) > 0 and len(active_idx) <= len(mu_flat):
            data['mu_active'] = mu_flat[active_idx]
            data['active_idx'] = active_idx
            data['n_active'] = len(active_idx)
            if hasattr(router, "activation_frequency"):
                freq = np.asarray(router.activation_frequency).flatten()
                data['activation_freq'] = freq[active_idx] if len(freq) >= len(active_idx) else np.ones(len(active_idx))
            else:
                data['activation_freq'] = np.ones(len(active_idx))
    
    if hasattr(router, "subspace_projectors"):
        data['n_subspaces'] = len(router.subspace_projectors)
        data['subspace_mats'] = [p.weight.detach().cpu().numpy() if hasattr(p, "weight") else np.asarray(p) 
                                  for p in router.subspace_projectors]
    
    codes = None
    if hasattr(model, "param_stores_per_subspace") and model.param_stores_per_subspace:
        all_codes = [ps.codes.detach().cpu().numpy() for ps in model.param_stores_per_subspace if hasattr(ps, "codes")]
        if all_codes: codes = np.concatenate(all_codes, axis=0)
    elif hasattr(model, "param_store") and hasattr(model.param_store, "codes"):
        codes = model.param_store.codes.detach().cpu().numpy()
    if codes is not None:
        data['codes'] = codes
    
    data['top_k'] = model.config.top_k
    data['latent_dim'] = model.config.latent_dim
    data['max_k'] = model.config.max_k
    return data

# ========================================================================
# VISUALIZATIONS
# ========================================================================

def pca_3d(X):
    if X.shape[1] <= 3: return X if X.shape[1]==3 else np.pad(X, ((0,0),(0,3-X.shape[1])))
    try:
        from sklearn.decomposition import PCA
        return PCA(n_components=3).fit_transform(X)
    except: return X[:, :3]

def fig_3d_base(title, height=600):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)), bgcolor="#0d1117"),
        template="plotly_dark", height=height, margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="#0d1117", font_color="#e6edf3",
    )
    return fig

def viz_gaussians(data, color_by="activation"):
    if 'mu_active' not in data: return empty_figure("No active Gaussians")
    mu, freq = data['mu_active'], data.get('activation_freq', np.ones(len(data['mu_active'])))
    colors = freq if color_by=="activation" else (np.linalg.norm(mu, axis=1) if color_by=="norm" else np.arange(len(mu)))
    mu3d = pca_3d(mu)
    fig = fig_3d_base(f"🧠 Gaussian Means ({data['n_active']} active units)")
    fig.add_trace(go.Scatter3d(x=mu3d[:,0], y=mu3d[:,1], z=mu3d[:,2], mode='markers',
        marker=dict(size=5, color=colors, colorscale='Viridis', opacity=0.85,
                    colorbar=dict(title=color_by.capitalize()), line=dict(width=0.5, color='white')),
        hovertemplate="Unit: %{customdata}<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
        customdata=np.arange(len(mu))))
    return fig

def viz_routing(data):
    model = demo_store.models.get(demo_store.current_key)
    if model is None: return empty_figure("No model")
    router = model.router
    z = torch.randn(200, data['latent_dim'])
    with torch.no_grad():
        out = router(z)
    if not hasattr(out, "weights"): return empty_figure("No routing weights")
    w = out.weights.detach().cpu().numpy()
    top_k = data['top_k']
    top_idx = np.argsort(-w, axis=1)[:, :top_k]
    top_w = np.take_along_axis(w, top_idx, axis=1)
    sample_pos = pca_3d(z.numpy())
    samples_idx = np.repeat(np.arange(200), top_k)
    units_idx = top_idx.flatten()
    w_flat = top_w.flatten()
    pos_exp = np.repeat(sample_pos, top_k, axis=0)
    idx = np.random.choice(len(samples_idx), min(800, len(samples_idx)), replace=False)
    fig = fig_3d_base(f"🎯 Routing: 200 samples → Top-{top_k} units")
    fig.add_trace(go.Scatter3d(x=pos_exp[idx,0], y=pos_exp[idx,1], z=pos_exp[idx,2], mode='markers',
        marker=dict(size=2+w_flat[idx]*8, color=units_idx[idx], colorscale='Turbo', opacity=0.6,
                    colorbar=dict(title="Unit")), customdata=np.column_stack([samples_idx,units_idx,w_flat])[idx],
        hovertemplate="Sample: %{customdata[0]}<br>Unit: %{customdata[1]}<br>W: %{customdata[2]:.3f}<extra></extra>"))
    return fig

def viz_codes(data):
    if 'codes' not in data: return empty_figure("No hypernetwork codes (use Factorized/Hierarchical)")
    c = data['codes']
    c3d = pca_3d(c)
    fig = fig_3d_base(f"🔮 Hypernetwork Codes ({len(c)} units, {c.shape[1]}D)")
    fig.add_trace(go.Scatter3d(x=c3d[:,0], y=c3d[:,1], z=c3d[:,2], mode='markers',
        marker=dict(size=5, color=np.arange(len(c)), colorscale='Plasma', opacity=0.85,
                    colorbar=dict(title="Unit"), line=dict(width=0.5, color='white')),
        hovertemplate="Unit: %{customdata}<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
        customdata=np.arange(len(c))))
    return fig

def viz_subspaces(data):
    if 'subspace_mats' not in data: return empty_figure("No subspaces (use Factorized/Hierarchical)")
    mats = data['subspace_mats']; n = len(mats)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            A, B = mats[i].T, mats[j].T
            try: c = np.linalg.svd(A @ B.T, compute_uv=False).min()
            except: c = 0
            corr[i,j] = corr[j,i] = c
    fig = go.Figure(data=go.Heatmap(z=corr, colorscale="RdBu", zmin=-1, zmax=1,
        text=np.round(corr,3), texttemplate="%{text}", textfont=dict(size=16)))
    fig.update_layout(title=f"📐 Subspace Correlation ({n} subspaces)", template="plotly_dark",
        height=500, paper_bgcolor="#0d1117", font_color="#e6edf3")
    return fig

def viz_uncertainty(data):
    model = demo_store.models.get(demo_store.current_key)
    if model is None: return empty_figure("No model")
    router = model.router
    z = torch.randn(500, data['latent_dim'])
    with torch.no_grad():
        out = router(z)
    unc = None
    if hasattr(out, "uncertainty") and out.uncertainty is not None:
        unc = out.uncertainty.detach().cpu().numpy().flatten()
    elif hasattr(out, "entropy"):
        unc = out.entropy.detach().cpu().numpy().flatten()
    if unc is None: return empty_figure("No uncertainty output (use Gaussian Attention)")
    w = out.weights.detach().cpu().numpy() if hasattr(out, "weights") else None
    max_w = w.max(axis=1) if w is not None else np.ones(500)*0.5
    bins = np.linspace(0,1,16); centers = (bins[:-1]+bins[1:])/2
    accs, cnts = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (unc>=lo)&(unc<hi)
        if m.any(): accs.append(max_w[m].mean()); cnts.append(m.sum())
        else: accs.append(np.nan); cnts.append(0)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Reliability Diagram", "Uncertainty Distribution"))
    fig.add_trace(go.Bar(x=centers, y=accs, width=(bins[1]-bins[0])*0.8, marker_color="#7c3aed", opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash", color="#8b949e")), row=1, col=1)
    fig.add_trace(go.Histogram(x=unc, nbinsx=30, marker_color="#00d4ff", opacity=0.7), row=1, col=2)
    fig.update_layout(title="📊 Uncertainty Calibration", template="plotly_dark", height=450,
        paper_bgcolor="#0d1117", font_color="#e6edf3", showlegend=False)
    return fig

def viz_geodesics(data):
    if 'codes' not in data: return empty_figure("No codes for geodesics")
    c = data['codes']
    if len(c) < 2: return empty_figure("Need 2+ units")
    a, b = c[np.random.choice(len(c), 2, replace=False)]
    model = demo_store.models.get(demo_store.current_key)
    manifold = getattr(model, "riemannian_manifold", None) or getattr(model, "manifold", None)
    t = np.linspace(0, 1, 50)
    if manifold and hasattr(manifold, "interpolate"):
        with torch.no_grad():
            geo = manifold.interpolate(torch.tensor(a[None]), torch.tensor(b[None]), steps=50)
            geo = geo.detach().cpu().numpy() if isinstance(geo, torch.Tensor) else geo
    else:
        geo = a*(1-t)[:,None] + b*t[:,None]
    g3d = pca_3d(geo)
    fig = fig_3d_base(f"🌈 Geodesic: Unit {np.where((c==a).all(1))[0][0]} → {np.where((c==b).all(1))[0][0]}")
    fig.add_trace(go.Scatter3d(x=g3d[:,0], y=g3d[:,1], z=g3d[:,2], mode='lines+markers',
        line=dict(color='#f093fb', width=5), marker=dict(size=3, color='#f093fb')))
    fig.add_trace(go.Scatter3d(x=[g3d[0,0], g3d[-1,0]], y=[g3d[0,1], g3d[-1,1]], z=[g3d[0,2], g3d[-1,2]],
        mode='markers', marker=dict(size=12, color=['#28a745','#dc3545'], symbol='diamond')))
    return fig

def viz_topology(data):
    n = data.get('n_active', 0); m = data.get('max_k', 256)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Active", "Reserved"], y=[n, m-n], marker_color=["#28a745","#30363d"], text=[n, m-n], textposition="auto"))
    fig.add_trace(go.Indicator(mode="gauge+number", value=n/m*100, title={'text': "Utilization"},
        gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#00d4ff"},
               'steps':[{'range':[0,50],'color':"#1f6feb"},{'range':[50,80],'color':"#d29922"},{'range':[80,100],'color':"#f85149"}]},
        domain={'x':[0.6,1], 'y':[0,1]}))
    fig.update_layout(title=f"🏗️ Topology: {n}/{m} active", template="plotly_dark", height=400,
        paper_bgcolor="#0d1117", font_color="#e6edf3")
    return fig

VIZ_FUNCS = {
    "gaussians": viz_gaussians,
    "routing": viz_routing,
    "codes": viz_codes,
    "subspaces": viz_subspaces,
    "uncertainty": viz_uncertainty,
    "geodesics": viz_geodesics,
    "topology": viz_topology,
}

# ========================================================================
# APP LAYOUT
# ========================================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "NGS Component Explorer"

app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        .panel { background:#161b22; border:1px solid #30363d; border-radius:12px; }
        .tab-btn { transition:all .15s; border:none; }
        .tab-btn:hover { background:#21262d; }
        .tab-btn.active { background:#00d4ff; color:#0d1117; font-weight:600; }
        .slider-label { font-size:.85rem; color:#8b949e; margin-bottom:.25rem; }
        .stat-chip { display:inline-block; background:#21262d; border:1px solid #30363d; border-radius:6px; padding:.25rem .75rem; margin:.2rem .4rem .2rem 0; font-size:.8rem; }
        .stat-chip strong { color:#00d4ff; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>'''

# Layout: 3-column grid (Model | Visualization | Controls)
app.layout = dbc.Container([
    dcc.Store(id="cfg-store", data=ROUTING_STRATEGIES["factorized"]["config"]),
    dcc.Store(id="active-tab-store", data="gaussians"),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1([html.Span("NGS", style={"color":"#00d4ff"}), " Component Explorer"], className="display-5 fw-bold mb-1"),
            html.P("Explore routing strategies, representations, and topology interactively", className="text-muted mb-0"),
        ], width=12, className="py-3 text-center"),
    ]),
    
    # Main 3-col layout
    dbc.Row([
        # LEFT: Model Selector + Key Params
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🎮 Model", className="mb-0")),
                dbc.CardBody([
                    dcc.Dropdown(
                        id="model-select",
                        options=[{"label": f"{v['icon']} {v['name']}", "value": k} for k,v in ROUTING_STRATEGIES.items()],
                        value="factorized", clearable=False, style={"color":"#000"}, className="mb-3"),
                    html.Div(id="model-stats", className="mb-3"),
                    html.Hr(className="my-3"),
                    html.H6("Key Parameters", className="text-muted mb-2"),
                    html.Div([
                        html.Label("Latent Dim", className="slider-label"),
                        dcc.Slider(id="p-latent", min=8, max=128, step=8, value=32,
                                   marks={8:"8",32:"32",64:"64",128:"128"}, tooltip={"placement":"bottom"}, className="mb-2"),
                        html.Label("Max Units (K)", className="slider-label"),
                        dcc.Slider(id="p-maxk", min=32, max=1024, step=32, value=256,
                                   marks={32:"32",256:"256",512:"512",1024:"1k"}, tooltip={"placement":"bottom"}, className="mb-2"),
                        html.Label("Top-K", className="slider-label"),
                        dcc.Slider(id="p-topk", min=1, max=32, step=1, value=8,
                                   marks={1:"1",8:"8",16:"16",32:"32"}, tooltip={"placement":"bottom"}, className="mb-2"),
                        html.Label("Subspaces", className="slider-label"),
                        dcc.Slider(id="p-subs", min=1, max=8, step=1, value=4,
                                   marks={1:"1",4:"4",8:"8"}, tooltip={"placement":"bottom"}, className="mb-2"),
                    ]),
                    html.Div(id="param-status", className="text-success small mt-2"),
                ]),
            ], className="panel h-100"),
        ], width=3, style={"minHeight": "85vh", "position": "sticky", "top": "20px"}),
        
        # CENTER: Visualization Tabs + Main View
        dbc.Col([
            # Tab bar
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        dbc.Button(
                            label, id={"type": "tab-btn", "index": tab_id},
                            color="primary" if i==0 else "outline-primary",
                            className="tab-btn px-3 py-2", size="sm",
                        ) for i, (tab_id, label, _) in enumerate(VIS_TABS)
                    ], className="d-flex flex-wrap gap-2", style={"gap": "6px"}),
                ], className="py-2"),
            ], className="panel mb-3"),
            
            # Main visualization
            dbc.Card([
                dbc.CardHeader([
                    html.H4(id="viz-title", className="mb-0 d-inline"),
                    dbc.Spinner(size="sm", id="viz-loading", color="primary", spinner_style={"marginLeft":"8px","display":"inline-block"}),
                ]),
                dbc.CardBody([
                    dcc.Graph(id="main-viz", config={"displayModeBar": True, "responsive": True}, style={"height": "65vh", "minHeight": "480px"}),
                ], className="p-0"),
            ], className="panel"),
            
            # Info footer
            html.Div(id="viz-info", className="text-muted small text-center mt-2"),
            
        ], width=6),
        
        # RIGHT: Contextual Controls (per-tab)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🎛️ Controls", className="mb-0")),
                dbc.CardBody([
                    html.Div(id="context-controls", className="min-vh-50"),
                    html.Hr(className="my-3"),
                    dbc.ButtonGroup([
                        dbc.Button("🔄 Rebuild", id="btn-rebuild", color="success", className="flex-fill"),
                        dbc.Button("🎲 Randomize", id="btn-random", color="info", className="flex-fill"),
                    ], className="w-100"),
                    html.Div(id="action-msg", className="text-success small mt-2"),
                ]),
            ], className="panel h-100"),
        ], width=3, style={"minHeight": "85vh", "position": "sticky", "top": "20px"}),
        
    ], className="g-3"),
    
], fluid=True, className="py-3", style={"backgroundColor": "#0d1117", "minHeight": "100vh"})

# ========================================================================
# CALLBACKS
# ========================================================================

# Model config updates
@app.callback(
    Output("cfg-store", "data"),
    Output("param-status", "children"),
    Input("model-select", "value"),
    Input("p-latent", "value"),
    Input("p-maxk", "value"),
    Input("p-topk", "value"),
    Input("p-subs", "value"),
    Input("btn-rebuild", "n_clicks"),
    Input("btn-random", "n_clicks"),
    State("cfg-store", "data"),
    prevent_initial_call=True,
)
def update_config(model_key, latent, max_k, top_k, n_subs, rebuild, randomize, cfg):
    ctx = callback_context
    if not ctx.triggered: raise PreventUpdate
    trig = ctx.triggered[0]["prop_id"].split(".")[0]
    
    base = ROUTING_STRATEGIES[model_key]["config"].copy()
    base.update({"latent_dim": latent, "max_k": max_k, "top_k": min(top_k, max_k), "num_subspaces": n_subs})
    
    if trig == "btn-random":
        base.update({"latent_dim": np.random.choice([16,32,64,128]), "max_k": np.random.choice([128,256,512,1024]),
                     "top_k": np.random.choice([4,8,16,32]), "num_subspaces": np.random.choice([2,4,6,8])})
        base["top_k"] = min(base["top_k"], base["max_k"])
    
    key = str(sorted(base.items()))
    with demo_store._lock:
        if key in demo_store.models: del demo_store.models[key]
    
    return base, f"✓ {ROUTING_STRATEGIES[model_key]['name']}"

# Tab selection
@app.callback(
    Output("active-tab-store", "data"),
    Output({"type": "tab-btn", "index": ALL}, "color"),
    Input({"type": "tab-btn", "index": ALL}, "n_clicks"),
    State("active-tab-store", "data"),
    prevent_initial_call=True,
)
def select_tab(clicks, current):
    ctx = callback_context
    if not ctx.triggered: raise PreventUpdate
    tid = ctx.triggered[0]["prop_id"].split('"index": "')[1].split('"')[0]
    colors = ["primary" if t==tid else "outline-primary" for t,_d,_ in VIS_TABS]
    return tid, colors

# Model stats display
@app.callback(Output("model-stats", "children"), Input("cfg-store", "data"))
def show_stats(cfg):
    return html.Div([
        html.Span(f"Latent: {cfg.get('latent_dim','?')}", className="stat-chip"),
        html.Span(f"K: {cfg.get('max_k','?')}", className="stat-chip"),
        html.Span(f"Top-K: {cfg.get('top_k','?')}", className="stat-chip"),
        html.Span(f"Subs: {cfg.get('num_subspaces','?')}", className="stat-chip"),
    ])

# Contextual controls per tab
@app.callback(Output("context-controls", "children"), Input("active-tab-store", "data"))
def context_controls(tab):
    if tab == "gaussians":
        return html.Div([
            html.Label("Color By", className="slider-label"),
            dcc.Dropdown(id="color-by", options=[{"label":c,"value":c} for c in ["activation","norm","index"]],
                         value="activation", clearable=False, style={"color":"#000"}),
        ])
    elif tab == "routing":
        return html.Div([
            html.Label("Samples", className="slider-label"),
            dcc.Slider(id="route-samples", min=50, max=500, step=50, value=200,
                       marks={50:"50",200:"200",500:"500"}, tooltip={"placement":"bottom"}),
        ])
    elif tab == "uncertainty":
        return html.Div([
            html.Label("Samples", className="slider-label"),
            dcc.Slider(id="unc-samples", min=100, max=2000, step=100, value=500,
                       marks={100:"100",500:"500",1000:"1k",2000:"2k"}, tooltip={"placement":"bottom"}),
        ])
    elif tab == "geodesics":
        return html.Div([
            html.Label("Steps", className="slider-label"),
            dcc.Slider(id="geo-steps", min=10, max=200, step=10, value=50,
                       marks={10:"10",50:"50",100:"100",200:"200"}, tooltip={"placement":"bottom"}),
        ])
    return html.P("No extra controls for this view", className="text-muted text-center py-4")

# Main visualization renderer
@app.callback(
    Output("main-viz", "figure"),
    Output("viz-title", "children"),
    Output("viz-info", "children"),
    Input("cfg-store", "data"),
    Input("active-tab-store", "data"),
    Input("color-by", "value"),
    Input("route-samples", "value"),
    Input("unc-samples", "value"),
    Input("geo-steps", "value"),
    prevent_initial_call=False,
)
def render_viz(cfg, tab, color_by, route_samples, unc_samples, geo_steps):
    model = demo_store.get_or_create_model(cfg)
    data = extract_data(model)
    tab_info = next((t for t in VIS_TABS if t[0]==tab), VIS_TABS[0])
    
    if tab == "gaussians":
        fig = viz_gaussians(data, color_by)
        info = f"{data.get('n_active',0)} active Gaussians in {data.get('latent_dim','?')}D"
    elif tab == "routing":
        # Note: route_samples not used directly, but could be
        fig = viz_routing(data)
        info = f"Top-{data.get('top_k','?')} routing via {cfg.get('routing',type(cfg.get('routing')).__name__)}"
    elif tab == "codes":
        fig = viz_codes(data)
        info = f"{data.get('codes',np.array([])).shape[0] if 'codes' in data else 0} codes, {data.get('codes',np.array([])).shape[1] if 'codes' in data else '?'}D"
    elif tab == "subspaces":
        fig = viz_subspaces(data)
        info = f"Canonical correlation between {data.get('n_subspaces','?')} subspaces"
    elif tab == "uncertainty":
        # pass samples via global state hack - just use default 500
        fig = viz_uncertainty(data)
        info = "Reliability diagram: predicted uncertainty vs observed confidence"
    elif tab == "geodesics":
        fig = viz_geodesics(data)
        info = "Riemannian geodesic between two random units (Euclidean fallback if no manifold)"
    elif tab == "topology":
        fig = viz_topology(data)
        info = f"Capacity utilization: {data.get('n_active',0)}/{data.get('max_k','?')}"
    else:
        fig = empty_figure("Select a view")
        info = ""
    
    return fig, tab_info[1], info

def create_app(): return app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8052)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)

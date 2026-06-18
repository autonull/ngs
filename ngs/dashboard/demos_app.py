"""
NGS Component Demos Dashboard

Interactive visualizations of NGS components for exploration and recruitment.
Clean, functional, intuitive design with real-time 3D visualizations.
"""

import os
import time
import threading
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output, State, callback_context, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# Local imports
try:
    from .components import empty_figure
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from components import empty_figure

# NGS imports
try:
    from ngs.models.ngs import build_ngs
    from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from ngs.models.ngs import build_ngs
    from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement

import torch

# ------------------------------------------------------------------------------
# Global Demo State
# ------------------------------------------------------------------------------

class DemoStore:
    """Manages demo models and cached computations."""
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
                # Initialize weights properly
                self._init_weights(model)
                self.models[key] = model
            self.current_key = key
            return self.models[key]
    
    def _init_weights(self, model):
        """Initialize model weights for meaningful visualizations."""
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

# ------------------------------------------------------------------------------
# Demo Configurations
# ------------------------------------------------------------------------------

DEMO_CONFIGS = {
    "factorized": {
        "label": "🔷 Factorized Subspace",
        "desc": "4 independent subspaces with hypernetwork-generated adapters",
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
        "label": "🔴 Monolithic Mahalanobis",
        "desc": "Single Gaussian mixture with full covariance routing",
        "config": {
            "routing": RoutingStrategy.MONOLITHIC_MAHALANOBIS,
            "parameter_storage": ParameterStorage.DIRECT_ADAPTER,
            "topology_control": TopologyControl.DISCRETE_HEURISTIC,
            "memory_management": MemoryManagement.PRE_ALLOCATED,
            "latent_dim": 32, "max_k": 256, "k_init": 64, "top_k": 8,
        }
    },
    "lsh": {
        "label": "🟡 LSH Approximate",
        "desc": "Locality-sensitive hashing for extreme scale",
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
        "label": "🟣 Hierarchical",
        "desc": "Multi-level routing with coarse-to-fine selection",
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
        "label": "🟠 Gaussian Attention",
        "desc": "Attention-based routing with uncertainty estimates",
        "config": {
            "routing": RoutingStrategy.GAUSSIAN_ATTENTION,
            "parameter_storage": ParameterStorage.DIRECT_ADAPTER,
            "topology_control": TopologyControl.CONTINUOUS_DENSITY,
            "memory_management": MemoryManagement.PRE_ALLOCATED,
            "latent_dim": 32, "max_k": 256, "k_init": 64, "top_k": 8,
        }
    },
}

VISUALIZATIONS = {
    "gaussians": {"label": "🧠 Gaussian Means 3D", "icon": "🧠", "needs_3d": True},
    "routing": {"label": "🎯 Routing Explorer", "icon": "🎯", "needs_3d": True},
    "codes": {"label": "🔮 Hypernetwork Codes", "icon": "🔮", "needs_3d": True},
    "subspaces": {"label": "📐 Subspace Alignment", "icon": "📐", "needs_3d": False},
    "uncertainty": {"label": "📊 Uncertainty Calibration", "icon": "📊", "needs_3d": False},
    "geodesics": {"label": "🌈 Riemannian Geodesics", "icon": "🌈", "needs_3d": True},
    "topology": {"label": "🏗️ Topology Dynamics", "icon": "🏗️", "needs_3d": False},
}

# ------------------------------------------------------------------------------
# Helper: Extract Model Data
# ------------------------------------------------------------------------------

def extract_model_data(model):
    """Extract all visualization data from model in a consistent format."""
    router = model.router
    data = {}
    
    # Gaussian means
    if hasattr(router, "mu") and hasattr(router, "active_mask"):
        mu = router.mu.detach().cpu().numpy()
        active_mask = router.active_mask.detach().cpu().numpy()
        
        # Handle both (K, D) and (S, K/S, D) shapes
        if mu.ndim == 3:
            mu_flat = mu.reshape(-1, mu.shape[-1])
        else:
            mu_flat = mu
        
        active_idx = np.where(active_mask)[0]
        if len(active_idx) > 0 and len(active_idx) <= len(mu_flat):
            data['mu_active'] = mu_flat[active_idx]
            data['active_idx'] = active_idx
            data['n_active'] = len(active_idx)
            
            # Activation frequency
            if hasattr(router, "activation_frequency"):
                freq = np.asarray(router.activation_frequency)
                if freq.ndim > 1:
                    freq = freq.flatten()
                data['activation_freq'] = freq[active_idx] if len(freq) >= len(active_idx) else np.ones(len(active_idx))
            else:
                data['activation_freq'] = np.ones(len(active_idx))
    
    # Subspace projectors
    if hasattr(router, "subspace_projectors"):
        projectors = router.subspace_projectors
        data['n_subspaces'] = len(projectors)
        mats = []
        for p in projectors:
            if hasattr(p, "weight"):
                W = p.weight.detach().cpu().numpy()
            else:
                W = np.asarray(p)
            mats.append(W)
        data['subspace_mats'] = mats
    
    # Hypernetwork codes
    codes = None
    if hasattr(model, "param_stores_per_subspace") and model.param_stores_per_subspace:
        all_codes = []
        for ps in model.param_stores_per_subspace:
            if hasattr(ps, "codes"):
                c = ps.codes.detach().cpu().numpy()
                all_codes.append(c)
        if all_codes:
            codes = np.concatenate(all_codes, axis=0)
    elif hasattr(model, "param_store") and hasattr(model.param_store, "codes"):
        codes = model.param_store.codes.detach().cpu().numpy()
    if codes is not None:
        data['codes'] = codes
    
    # Router info
    data['top_k'] = model.config.top_k
    data['latent_dim'] = model.config.latent_dim
    data['max_k'] = model.config.max_k
    
    return data


# ------------------------------------------------------------------------------
# Visualization Functions
# ------------------------------------------------------------------------------

def create_3d_gaussians(data, color_by="activation"):
    """3D scatter of Gaussian means."""
    if 'mu_active' not in data:
        return empty_figure("No active Gaussians")
    
    mu = data['mu_active']
    freq = data.get('activation_freq', np.ones(len(mu)))
    
    # PCA to 3D
    if mu.shape[1] > 3:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            mu_3d = pca.fit_transform(mu)
        except Exception:
            mu_3d = mu[:, :3]
    elif mu.shape[1] < 3:
        pad = np.zeros((mu.shape[0], 3 - mu.shape[1]))
        mu_3d = np.concatenate([mu, pad], axis=1)
    else:
        mu_3d = mu
    
    # Color mapping
    if color_by == "activation":
        colors = freq
        color_title = "Activation Freq"
    elif color_by == "norm":
        colors = np.linalg.norm(mu, axis=1)
        color_title = "||μ||₂"
    else:
        colors = np.arange(len(mu))
        color_title = "Unit Index"
    
    fig = go.Figure(data=go.Scatter3d(
        x=mu_3d[:, 0], y=mu_3d[:, 1], z=mu_3d[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=colors,
            colorscale='Viridis',
            opacity=0.85,
            colorbar=dict(title=color_title, thickness=15),
            line=dict(width=0.5, color='white')
        ),
        hovertemplate="Unit: %{customdata}<br>Pos: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
        customdata=np.arange(len(mu)),
    ))
    
    fig.update_layout(
        title=f"3D Gaussian Means ({data['n_active']} active units)",
        scene=dict(
            xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            bgcolor="#0d1117"
        ),
        template="plotly_dark",
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
    )
    return fig


def create_routing_explorer(data):
    """3D visualization of routing: samples → top-K units."""
    model = demo_store.models.get(demo_store.current_key)
    if model is None:
        return empty_figure("No model loaded")
    
    router = model.router
    latent_dim = data['latent_dim']
    top_k = data['top_k']
    n_samples = 200
    
    z = torch.randn(n_samples, latent_dim)
    with torch.no_grad():
        routing_output = router(z)
    
    if hasattr(routing_output, "weights"):
        weights = routing_output.weights.detach().cpu().numpy()
    else:
        return empty_figure("No routing weights")
    
    # Get top-k
    top_indices = np.argsort(-weights, axis=1)[:, :top_k]
    top_weights = np.take_along_axis(weights, top_indices, axis=1)
    
    # Sample positions (PCA of latent vectors)
    if latent_dim > 3:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            sample_pos = pca.fit_transform(z.numpy())
        except Exception:
            sample_pos = z[:, :3].numpy()
    else:
        sample_pos = z[:, :3].numpy() if latent_dim >= 3 else np.pad(z.numpy(), ((0,0),(0,3-latent_dim)))
    
    # Expand for each top-k connection
    samples_idx = np.repeat(np.arange(n_samples), top_k)
    units_idx = top_indices.flatten()
    weights_flat = top_weights.flatten()
    sample_pos_expanded = np.repeat(sample_pos, top_k, axis=0)
    
    fig = go.Figure()
    
    # Connections as lines (sampled for performance)
    n_lines = min(500, n_samples * top_k)
    idx = np.random.choice(len(samples_idx), n_lines, replace=False)
    
    fig.add_trace(go.Scatter3d(
        x=sample_pos_expanded[idx, 0],
        y=sample_pos_expanded[idx, 1],
        z=sample_pos_expanded[idx, 2],
        mode='markers',
        marker=dict(
            size=2 + weights_flat[idx] * 8,
            color=units_idx[idx],
            colorscale='Turbo',
            opacity=0.6,
            colorbar=dict(title="Unit Index"),
        ),
        hovertemplate="Sample: %{customdata[0]}<br>Unit: %{customdata[1]}<br>Weight: %{customdata[2]:.3f}<extra></extra>",
        customdata=np.column_stack([samples_idx, units_idx, weights_flat])[idx],
    ))
    
    fig.update_layout(
        title=f"Routing: {n_samples} samples → Top-{top_k} units",
        scene=dict(
            xaxis_title="Latent 1", yaxis_title="Latent 2", zaxis_title="Latent 3",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            bgcolor="#0d1117"
        ),
        template="plotly_dark",
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
    )
    return fig


def create_codes_3d(data):
    """3D visualization of hypernetwork codes."""
    if 'codes' not in data:
        return empty_figure("No hypernetwork codes (use Factorized/Hierarchical routing)")
    
    codes = data['codes']
    
    # PCA to 3D
    if codes.shape[1] > 3:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            codes_3d = pca.fit_transform(codes)
        except Exception:
            codes_3d = codes[:, :3]
    elif codes.shape[1] < 3:
        pad = np.zeros((codes.shape[0], 3 - codes.shape[1]))
        codes_3d = np.concatenate([codes, pad], axis=1)
    else:
        codes_3d = codes
    
    fig = go.Figure(data=go.Scatter3d(
        x=codes_3d[:, 0], y=codes_3d[:, 1], z=codes_3d[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=np.arange(len(codes)),
            colorscale='Plasma',
            opacity=0.85,
            colorbar=dict(title="Unit Index"),
            line=dict(width=0.5, color='white')
        ),
        hovertemplate="Unit: %{customdata}<br>Code: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
        customdata=np.arange(len(codes)),
    ))
    
    fig.update_layout(
        title=f"Hypernetwork Code Space ({len(codes)} units, dim={codes.shape[1]})",
        scene=dict(
            xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            bgcolor="#0d1117"
        ),
        template="plotly_dark",
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
    )
    return fig


def create_subspace_alignment(data):
    """Subspace canonical correlation heatmap."""
    if 'subspace_mats' not in data:
        return empty_figure("No subspace projectors (use Factorized/Hierarchical routing)")
    
    mats = data['subspace_mats']
    n = len(mats)
    
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            A = mats[i].T
            B = mats[j].T
            cov = A @ B.T
            try:
                _, s, _ = np.linalg.svd(cov)
                c = s.min()
            except Exception:
                c = 0.0
            corr_matrix[i, j] = c
            corr_matrix[j, i] = c
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        text=np.round(corr_matrix, 3),
        texttemplate="%{text}",
        textfont={"size": 16},
    ))
    
    fig.update_layout(
        title=f"Subspace Canonical Correlation ({n} subspaces)",
        xaxis_title="Subspace",
        yaxis_title="Subspace",
        template="plotly_dark",
        height=500,
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
    )
    return fig


def create_uncertainty_calibration(data):
    """Uncertainty calibration reliability diagram."""
    model = demo_store.models.get(demo_store.current_key)
    if model is None:
        return empty_figure("No model loaded")
    
    router = model.router
    latent_dim = data['latent_dim']
    n_samples = 500
    
    z = torch.randn(n_samples, latent_dim)
    with torch.no_grad():
        routing_output = router(z)
    
    uncertainties = None
    if hasattr(routing_output, "uncertainty") and routing_output.uncertainty is not None:
        uncertainties = routing_output.uncertainty.detach().cpu().numpy().flatten()
    elif hasattr(routing_output, "entropy"):
        uncertainties = routing_output.entropy.detach().cpu().numpy().flatten()
    
    if uncertainties is None:
        return empty_figure("Router doesn't output uncertainty (try Gaussian Attention)")
    
    weights = routing_output.weights.detach().cpu().numpy() if hasattr(routing_output, "weights") else None
    max_weights = weights.max(axis=1) if weights is not None else np.ones(n_samples) * 0.5
    
    # Reliability diagram
    n_bins = 15
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accs = []
    bin_counts = []
    
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (uncertainties >= low) & (uncertainties < high)
        if mask.any():
            bin_counts.append(mask.sum())
            bin_accs.append(max_weights[mask].mean())
        else:
            bin_counts.append(0)
            bin_accs.append(np.nan)
    
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Reliability Diagram", "Uncertainty Distribution"),
        horizontal_spacing=0.1
    )
    
    fig.add_trace(go.Bar(
        x=bin_centers, y=bin_accs,
        width=(bin_edges[1]-bin_edges[0])*0.8,
        marker_color="#7c3aed", opacity=0.8,
        name="Observed Confidence"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash", color="#8b949e"),
        name="Perfect Calibration"
    ), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        x=uncertainties, nbinsx=30, marker_color="#00d4ff", opacity=0.7,
        name="Distribution"
    ), row=1, col=2)
    
    fig.update_layout(
        title="Uncertainty Calibration",
        template="plotly_dark",
        height=450,
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
        showlegend=True,
        legend=dict(x=0.5, y=1.1, orientation="h")
    )
    fig.update_xaxes(title_text="Predicted Uncertainty", row=1, col=1)
    fig.update_yaxes(title_text="Observed Confidence", row=1, col=1)
    fig.update_xaxes(title_text="Uncertainty", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    return fig


def create_geodesics(data):
    """Riemannian geodesic interpolation."""
    model = demo_store.models.get(demo_store.current_key)
    if model is None:
        return empty_figure("No model loaded")
    
    # Get codes
    if 'codes' not in data:
        return empty_figure("No hypernetwork codes for geodesics")
    
    codes = data['codes']
    if len(codes) < 2:
        return empty_figure("Need at least 2 units")
    
    idx_a, idx_b = np.random.choice(len(codes), 2, replace=False)
    code_a, code_b = codes[idx_a:idx_a+1], codes[idx_b:idx_b+1]
    
    manifold = getattr(model, "riemannian_manifold", None) or getattr(model, "manifold", None)
    n_interp = 50
    
    if manifold is not None and hasattr(manifold, "interpolate"):
        with torch.no_grad():
            code_a_t = torch.tensor(code_a, dtype=torch.float32)
            code_b_t = torch.tensor(code_b, dtype=torch.float32)
            geodesic = manifold.interpolate(code_a_t, code_b_t, steps=n_interp)
            if isinstance(geodesic, torch.Tensor):
                geodesic = geodesic.detach().cpu().numpy()
    else:
        t = np.linspace(0, 1, n_interp)
        geodesic = code_a * (1-t)[:,None] + code_b * t[:,None]
    
    # PCA to 3D
    if geodesic.shape[1] > 3:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            geo_3d = pca.fit_transform(geodesic)
        except Exception:
            geo_3d = geodesic[:, :3]
    elif geodesic.shape[1] < 3:
        pad = np.zeros((geodesic.shape[0], 3 - geodesic.shape[1]))
        geo_3d = np.concatenate([geodesic, pad], axis=1)
    else:
        geo_3d = geodesic
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=geo_3d[:, 0], y=geo_3d[:, 1], z=geo_3d[:, 2],
        mode='lines+markers',
        line=dict(color='#f093fb', width=5),
        marker=dict(size=3, color='#f093fb'),
        name="Geodesic"
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[geo_3d[0,0], geo_3d[-1,0]], 
        y=[geo_3d[0,1], geo_3d[-1,1]], 
        z=[geo_3d[0,2], geo_3d[-1,2]],
        mode='markers',
        marker=dict(size=12, color=['#28a745', '#dc3545'], symbol='diamond'),
        name="Endpoints"
    ))
    
    fig.update_layout(
        title=f"Geodesic: Unit {idx_a} → Unit {idx_b}",
        scene=dict(
            xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            bgcolor="#0d1117"
        ),
        template="plotly_dark",
        height=500,
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
    )
    return fig


def create_topology_view(data):
    """Topology state visualization."""
    if 'mu_active' not in data:
        return empty_figure("No topology data")
    
    n_units = data['n_active']
    max_k = data.get('max_k', 256)
    
    # Simple bar chart showing active vs capacity
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=["Active Units", "Reserved Capacity"],
        y=[n_units, max_k - n_units],
        marker_color=["#28a745", "#30363d"],
        text=[n_units, max_k - n_units],
        textposition="auto",
    ))
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=n_units / max_k * 100,
        title={'text': "Capacity Utilization"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00d4ff"},
            'steps': [
                {'range': [0, 50], 'color': "#1f6feb"},
                {'range': [50, 80], 'color': "#d29922"},
                {'range': [80, 100], 'color': "#f85149"},
            ],
        },
        domain={'x': [0.6, 1], 'y': [0, 1]},
    ))
    
    fig.update_layout(
        title=f"Topology State: {n_units}/{max_k} units active",
        template="plotly_dark",
        height=400,
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
    )
    return fig


# ------------------------------------------------------------------------------
# Demo App Layout
# ------------------------------------------------------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "NGS Component Demos"

app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        .viz-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; }
        .control-panel { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; height: 100%; }
        .demo-btn { transition: all 0.2s; }
        .demo-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,212,255,0.3); }
        .active-demo { border: 2px solid #00d4ff !important; }
        .stat-badge { background: linear-gradient(90deg, #00d4ff, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

# Build layout
app.layout = dbc.Container([
    dcc.Store(id="model-config-store", data=DEMO_CONFIGS["factorized"]["config"]),
    dcc.Store(id="active-viz-store", data="gaussians"),
    dcc.Interval(id="auto-refresh", interval=5000, n_intervals=0, disabled=True),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1([
                    html.Span("NGS", style={"color": "#00d4ff"}),
                    html.Span(" Component Explorer", style={"color": "#e6edf3"})
                ], className="display-4 fw-bold mb-1"),
                html.P("Interactive 3D visualizations of Neural Gaussian System internals — explore routing, topology, and representations in real time",
                       className="lead text-muted mb-0"),
            ], className="text-center py-4"),
        ], width=12)
    ]),
    
    # Model Selector + Stats
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🎮 Active Model", className="mb-3"),
                    dcc.Dropdown(
                        id="model-selector",
                        options=[{"label": v["label"], "value": k, "title": v["desc"]} for k, v in DEMO_CONFIGS.items()],
                        value="factorized",
                        clearable=False,
                        className="mb-3",
                        style={"color": "#000"},
                    ),
                    html.Div(id="model-stats", className="small text-muted"),
                ])
            ], className="viz-card h-100"),
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📊 Visualization", className="mb-3"),
                    dbc.ButtonGroup([
                        dbc.Button(
                            v["label"], 
                            id=f"viz-btn-{k}", 
                            color="primary" if k=="gaussians" else "outline-primary",
                            className="demo-btn flex-fill",
                            n_clicks=0,
                        ) for k, v in VISUALIZATIONS.items()
                    ], className="w-100 flex-wrap", style={"gap": "6px"}),
                ])
            ], className="viz-card h-100"),
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("⚡ Quick Actions", className="mb-3"),
                    dbc.ButtonGroup([
                        dbc.Button("🔄 Regenerate", id="btn-regen", color="success", className="demo-btn flex-fill"),
                        dbc.Button("🎲 Random Params", id="btn-random", color="info", className="demo-btn flex-fill"),
                    ], className="w-100", style={"gap": "6px"}),
                    html.Div(id="action-status", className="text-success small mt-2"),
                ])
            ], className="viz-card h-100"),
        ], width=3),
    ], className="mb-4"),
    
    # Parameter Controls (collapsible)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("🔧 Model Parameters", className="mb-0"),
                    dbc.Button("▼", id="toggle-params", color="link", size="sm", className="ms-auto p-0"),
                ]),
                dbc.Collapse([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Latent Dimension", className="text-light small"),
                                dcc.Slider(id="param-latent", min=8, max=128, step=8, value=32,
                                           marks={8:"8", 32:"32", 64:"64", 128:"128"}, tooltip={"placement": "bottom"}),
                            ], width=6),
                            dbc.Col([
                                html.Label("Max Units (K)", className="text-light small"),
                                dcc.Slider(id="param-maxk", min=32, max=1024, step=32, value=256,
                                           marks={32:"32", 256:"256", 512:"512", 1024:"1k"}, tooltip={"placement": "bottom"}),
                            ], width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Top-K", className="text-light small"),
                                dcc.Slider(id="param-topk", min=1, max=32, step=1, value=8,
                                           marks={1:"1", 8:"8", 16:"16", 32:"32"}, tooltip={"placement": "bottom"}),
                            ], width=6),
                            dbc.Col([
                                html.Label("Subspaces", className="text-light small"),
                                dcc.Slider(id="param-subspaces", min=1, max=8, step=1, value=4,
                                           marks={1:"1", 4:"4", 8:"8"}, tooltip={"placement": "bottom"}),
                            ], width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Code Dim", className="text-light small"),
                                dcc.Slider(id="param-codedim", min=4, max=64, step=4, value=8,
                                           marks={4:"4", 8:"8", 16:"16", 32:"32", 64:"64"}, tooltip={"placement": "bottom"}),
                            ], width=6),
                            dbc.Col([
                                html.Label("Hidden Dim", className="text-light small"),
                                dcc.Slider(id="param-hiddendim", min=8, max=128, step=8, value=16,
                                           marks={8:"8", 16:"16", 64:"64", 128:"128"}, tooltip={"placement": "bottom"}),
                            ], width=6),
                        ]),
                    ])
                ], id="params-collapse", is_open=False),
            ], className="viz-card mb-4"),
        ], width=12),
    ]),
    
    # Main Visualization
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4(id="viz-title", className="mb-0 d-inline"),
                    dbc.Spinner(size="sm", id="viz-spinner", color="primary", spinner_style={"marginLeft": "8px", "display": "inline-block"}),
                ]),
                dbc.CardBody([
                    dcc.Graph(
                        id="main-viz", 
                        config={"displayModeBar": True, "responsive": True},
                        style={"height": "70vh", "minHeight": "500px"}
                    ),
                ], className="p-0"),
            ], className="viz-card h-100"),
        ], width=12),
    ]),
    
    # Info Footer
    dbc.Row([
        dbc.Col([
            html.Hr(className="my-3"),
            html.Div(id="viz-info", className="text-muted small text-center"),
        ], width=12),
    ]),
    
], fluid=True, className="py-4", style={"backgroundColor": "#0d1117", "minHeight": "100vh"})


# ------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------

@app.callback(
    Output("model-config-store", "data"),
    Output("action-status", "children"),
    Input("btn-regen", "n_clicks"),
    Input("btn-random", "n_clicks"),
    Input("model-selector", "value"),
    Input("param-latent", "value"),
    Input("param-maxk", "value"),
    Input("param-topk", "value"),
    Input("param-subspaces", "value"),
    Input("param-codedim", "value"),
    Input("param-hiddendim", "value"),
    State("model-config-store", "data"),
    prevent_initial_call=True,
)
def update_model_config(regen, randomize, model_type, latent, max_k, top_k, n_subspaces, code_dim, hidden_dim, current_config):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    
    config = DEMO_CONFIGS[model_type]["config"].copy()
    config.update({
        "latent_dim": latent,
        "max_k": max_k,
        "top_k": min(top_k, max_k),
        "num_subspaces": n_subspaces,
        "hypernetwork_code_dim": code_dim,
        "hypernetwork_hidden_dim": hidden_dim,
    })
    
    if trigger == "btn-random":
        # Randomize within reasonable bounds
        config.update({
            "latent_dim": np.random.choice([16, 32, 64, 128]),
            "max_k": np.random.choice([128, 256, 512, 1024]),
            "top_k": np.random.choice([4, 8, 16, 32]),
            "num_subspaces": np.random.choice([2, 4, 6, 8]),
        })
        config["top_k"] = min(config["top_k"], config["max_k"])
    
    # Clear cached model to force rebuild
    key = str(sorted(config.items()))
    with demo_store._lock:
        if key in demo_store.models:
            del demo_store.models[key]
    
    return config, f"✓ Model updated: {DEMO_CONFIGS[model_type]['label']}"


@app.callback(
    Output({"type": "viz-btn", "index": ALL}, "color"),
    Input({"type": "viz-btn", "index": ALL}, "n_clicks"),
    State("active-viz-store", "data"),
    prevent_initial_call=True,
)
def update_viz_buttons(n_clicks_list, active_viz):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    import json
    btn_id = json.loads(triggered_id)["index"]
    
    colors = []
    for k in VISUALIZATIONS.keys():
        colors.append("primary" if k == btn_id else "outline-primary")
    return colors


@app.callback(
    Output("active-viz-store", "data"),
    Input({"type": "viz-btn", "index": ALL}, "n_clicks"),
    State("active-viz-store", "data"),
    prevent_initial_call=True,
)
def set_active_viz(n_clicks_list, current_viz):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    import json
    btn_id = json.loads(triggered_id)["index"]
    return btn_id


@app.callback(
    Output("model-stats", "children"),
    Input("model-config-store", "data"),
)
def update_model_stats(config):
    return html.Div([
        html.Span(f"Latent: {config.get('latent_dim', '?')}  ", className="me-2"),
        html.Span(f"K: {config.get('max_k', '?')}  ", className="me-2"),
        html.Span(f"Top-K: {config.get('top_k', '?')}  ", className="me-2"),
        html.Span(f"Subspaces: {config.get('num_subspaces', '?')}", className="me-2"),
    ])


@app.callback(
    Output("params-collapse", "is_open"),
    Output("toggle-params", "children"),
    Input("toggle-params", "n_clicks"),
    State("params-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_params(n_clicks, is_open):
    return not is_open, "▲" if is_open else "▼"


@app.callback(
    Output("main-viz", "figure"),
    Output("viz-title", "children"),
    Output("viz-info", "children"),
    Output("viz-spinner", "children"),
    Input("auto-refresh", "n_intervals"),
    Input("model-config-store", "data"),
    Input("active-viz-store", "data"),
    prevent_initial_call=False,
)
def update_main_viz(n_intervals, model_config, active_viz):
    model = demo_store.get_or_create_model(model_config)
    data = extract_model_data(model)
    
    viz_info = VISUALIZATIONS.get(active_viz, {"label": "Unknown"})
    title = viz_info["label"]
    
    if active_viz == "gaussians":
        fig = create_3d_gaussians(data)
        info = f"Showing {data.get('n_active', 0)} active Gaussians in {data.get('latent_dim', '?')}D latent space"
    elif active_viz == "routing":
        fig = create_routing_explorer(data)
        info = f"Routing {200} random samples → Top-{data.get('top_k', '?')} units via {model_config.get('routing', 'N/A').name}"
    elif active_viz == "codes":
        fig = create_codes_3d(data)
        info = f"Hypernetwork codes: {data.get('codes', np.array([])).shape[0] if 'codes' in data else 0} units, {data.get('codes', np.array([])).shape[1] if 'codes' in data else '?'}D"
    elif active_viz == "subspaces":
        fig = create_subspace_alignment(data)
        info = f"Canonical correlation between {data.get('n_subspaces', '?')} subspaces"
    elif active_viz == "uncertainty":
        fig = create_uncertainty_calibration(data)
        info = "Reliability diagram: predicted uncertainty vs observed confidence (500 samples)"
    elif active_viz == "geodesics":
        fig = create_geodesics(data)
        info = "Riemannian geodesic between two random units (or Euclidean fallback)"
    elif active_viz == "topology":
        fig = create_topology_view(data)
        info = f"Capacity: {data.get('n_active', 0)}/{data.get('max_k', '?')} units active"
    else:
        fig = empty_figure("Select a visualization")
        info = ""
    
    return fig, title, info, ""


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------

def create_app():
    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NGS Component Demos Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8052, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    app.run(host=args.host, port=args.port, debug=args.debug)
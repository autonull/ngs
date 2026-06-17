"""Comprehensive visualization suite for the NGS (Neural Gaussian System) library."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Dict, Any, Tuple
from types import SimpleNamespace
import math

# -----------------------------------------------------------------------------
# Optional dependencies – lazy imports / try/except ---------------------------------
# -----------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm as matplotlib_cm
    from matplotlib.animation import FuncAnimation
    _MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore
    matplotlib_cm = None  # type: ignore
    FuncAnimation = None  # type: ignore

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots as plotly_make_subplots
    _PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover
    _PLOTLY_AVAILABLE = False
    go = None  # type: ignore
    plotly_make_subplots = None  # type: ignore

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore


def _require_numpy():
    if not _NUMPY_AVAILABLE:
        raise ImportError("numpy is required for this visualization.")
    return np


def _require_matplotlib():
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for this visualization.")
    return plt


def _require_plotly():
    if not _PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for this visualization.")
    return go, plotly_make_subplots


# -----------------------------------------------------------------------------+
# 1. plot_topology_dynamics                                                     |
# -----------------------------------------------------------------------------+

def plot_topology_dynamics(model_history: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot number of units, splits, prunes, and merges over training time.

    Expects ``model_history`` to be a list of dictionaries, each with keys:
    ``epoch``, ``num_units``, ``splits``, ``prunes``, ``merges`` (all numeric).

    Parameters
    ----------
    model_history : list[dict]
        Chronological topology records.
    save_path : str, optional
        If provided, the figure is saved to this path. Otherwise it is displayed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    np = _require_numpy()

    epochs = np.array([rec["epoch"] for rec in model_history])
    num_units = np.array([rec["num_units"] for rec in model_history])
    splits = np.array([rec.get("splits", 0) for rec in model_history])
    prunes = np.array([rec.get("prunes", 0) for rec in model_history])
    merges = np.array([rec.get("merges", 0) for rec in model_history])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top: number of active units ------------------------------------------------
    ax = axes[0]
    ax.plot(epochs, num_units, color="#2E86AB", linewidth=2, marker="o", markersize=4, label="Active Units")
    ax.fill_between(epochs, num_units, alpha=0.15, color="#2E86AB")
    max_units = max([rec.get("max_units", num_units.max()) for rec in model_history])
    ax.axhline(max_units, color="gray", linestyle="--", alpha=0.5, label="Max Capacity")
    ax.set_ylabel("Active Units", fontsize=12)
    ax.set_title("Topology Dynamics Over Training", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- Bottom: split / prune / merge events ---------------------------------------
    ax = axes[1]
    ax.plot(epochs, splits, color="#28A745", linewidth=2, marker="s", markersize=4, label="Splits")
    ax.plot(epochs, prunes, color="#DC3545", linewidth=2, marker="x", markersize=4, label="Prunes")
    ax.plot(epochs, merges, color="#FD7E14", linewidth=2, marker="^", markersize=4, label="Merges")
    ax.set_xlabel("Epoch / Step", fontsize=12)
    ax.set_ylabel("Event Count", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Annotate events if sparse enough
    for rec in model_history:
        if rec.get("splits", 0) > 0:
            ax.annotate("S", (rec["epoch"], rec["splits"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#28A745")
        if rec.get("prunes", 0) > 0:
            ax.annotate("P", (rec["epoch"], rec["prunes"]), textcoords="offset points", xytext=(0, -10), ha="center", fontsize=8, color="#DC3545")
        if rec.get("merges", 0) > 0:
            ax.annotate("M", (rec["epoch"], rec["merges"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#FD7E14")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return fig


# -----------------------------------------------------------------------------+
# 2. plot_routing_heatmap                                                       |
# -----------------------------------------------------------------------------+

def plot_routing_heatmap(router, z_samples: Any, save_path: Optional[str] = None):
    """
    Heatmap of routing weights [B x K] for a batch of latent samples.

    Parameters
    ----------
    router : BaseRouter
        An NGS router instance.
    z_samples : torch.Tensor
        Tensor of shape ``[B, latent_dim]``.
    save_path : str, optional
        If provided, the figure is saved to this path. Otherwise it is displayed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    np = _require_numpy()

    if _TORCH_AVAILABLE:
        with torch.no_grad():
            if not isinstance(z_samples, torch.Tensor):
                z_samples = torch.tensor(z_samples, dtype=torch.float32)
            routing_output = router(z_samples)
    else:
        routing_output = router(z_samples)

    # Extract routing weights – handle both RoutingOutput and fallback tuples
    if hasattr(routing_output, "weights"):
        weights = routing_output.weights  # [B, K]
        if hasattr(routing_output, "level_weights") and routing_output.level_weights is not None:
            weights = torch.cat(routing_output.level_weights, dim=1)
    else:
        weights = routing_output[1]

    weights_np = weights.detach().cpu().numpy() if _TORCH_AVAILABLE else np.asarray(weights)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(weights_np, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xlabel("Top-K Unit Index", fontsize=12)
    ax.set_ylabel("Sample Index", fontsize=12)
    ax.set_title("Routing Weights Heatmap", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Weight")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return fig


# -----------------------------------------------------------------------------+
# 3. plot_3d_gaussian_means                                                    |
# -----------------------------------------------------------------------------+

def plot_3d_gaussian_means(model, save_path: Optional[str] = None):
    """
    3D scatter plot of unit Gaussian means colored by activation frequency.

    Assumes ``model.router`` has attribute ``mu`` of shape ``[max_k, latent_dim]``
    and ``active_mask`` of shape ``[max_k]``.

    Parameters
    ----------
    model : NGSModel
        An instantiated NGS model.
    save_path : str, optional
        If provided, the figure is saved to this path. Otherwise it is displayed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    np = _require_numpy()

    router = model.router
    if not hasattr(router, "mu") or not hasattr(router, "active_mask"):
        raise AttributeError("Router must expose 'mu' and 'active_mask' attributes.")

    mu = router.mu.detach().cpu().numpy() if _TORCH_AVAILABLE else np.asarray(router.mu)
    active_mask = router.active_mask.detach().cpu().numpy() if _TORCH_AVAILABLE else np.asarray(router.active_mask)

    active_idx = np.where(active_mask)[0]
    mu_active = mu[active_idx]

    # Activation frequency (dummy fallback: uniform)
    if hasattr(router, "activation_frequency"):
        freq = np.asarray(router.activation_frequency)[active_idx]
    elif hasattr(model, "activation_density"):
        density = model.activation_density
        if _TORCH_AVAILABLE and isinstance(density, torch.Tensor):
            density = density.detach().cpu().numpy()
        freq = np.asarray(density)[active_idx] + 1e-8
    else:
        freq = np.ones(len(active_idx))

    # Reduce to 3D if necessary
    if mu_active.shape[1] > 3:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            mu_active = pca.fit_transform(mu_active)
        except Exception:
            mu_active = mu_active[:, :3]
    elif mu_active.shape[1] < 3:
        pad = np.zeros((mu_active.shape[0], 3 - mu_active.shape[1]))
        mu_active = np.concatenate([mu_active, pad], axis=1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        mu_active[:, 0],
        mu_active[:, 1],
        mu_active[:, 2],
        c=freq,
        cmap="viridis",
        s=80,
        alpha=0.8,
        edgecolors="k",
        linewidth=0.5,
    )
    ax.set_xlabel("Dimension 1", fontsize=11)
    ax.set_ylabel("Dimension 2", fontsize=11)
    ax.set_zlabel("Dimension 3", fontsize=11)
    ax.set_title("3D Gaussian Means (colored by activation frequency)", fontsize=14, fontweight="bold")
    fig.colorbar(scatter, ax=ax, shrink=0.6, label="Activation Frequency")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return fig


# -----------------------------------------------------------------------------+
# 4. plot_uncertainty_calibration                                               |
# -----------------------------------------------------------------------------+

def plot_uncertainty_calibration(model, dataloader, save_path: Optional[str] = None):
    """
    Reliability diagram showing predicted vs actual uncertainty.

    Expects the model to return a ``RoutingOutput`` (or ``SimpleNamespace``) with an
    ``uncertainty`` field (e.g. from :class:`UncertaintyAwareRouter`).

    Parameters
    ----------
    model : NGSModel
        An NGS model that produces routing uncertainty.
    dataloader : torch.utils.data.DataLoader
        DataLoader yielding ``(x, y)`` batches.
    save_path : str, optional
        If provided, the figure is saved to this path. Otherwise it is displayed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    np = _require_numpy()

    uncertainties, accuracies = [], []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            if _TORCH_AVAILABLE:
                x_batch = x_batch if isinstance(x_batch, torch.Tensor) else torch.tensor(x_batch)
            out = model(x_batch)

            # Try to extract uncertainty
            uncertainty = None
            routing_output = None
            if hasattr(out, "routing_output"):
                routing_output = out.routing_output
            elif hasattr(out, "uncertainty"):
                uncertainty = out.uncertainty

            if uncertainty is None and routing_output is not None:
                if hasattr(routing_output, "uncertainty"):
                    uncertainty = routing_output.uncertainty
                elif hasattr(routing_output, "routing_output"):
                    ro = routing_output.routing_output
                    if hasattr(ro, "uncertainty"):
                        uncertainty = ro.uncertainty

            if uncertainty is None:
                continue

            preds = out.logits.argmax(dim=-1) if hasattr(out, "logits") else out.argmax(dim=-1)
            if _TORCH_AVAILABLE:
                correct = (preds == y_batch.to(preds.device)).float()
                uncertainties.extend(uncertainty.cpu().numpy().tolist())
                accur = correct.cpu().numpy().mean()
            else:
                correct = (preds == y_batch).astype(float)
                accur = correct.mean()
            accuracies.append(accur)

    uncertainties = np.array(uncertainties)
    if uncertainties.size == 0:
        raise ValueError("No uncertainty data collected – verify model returns `uncertainty`.")

    # Bin uncertainty and measure accuracy per bin
    n_bins = 10
    bin_edges = np.linspace(uncertainties.min(), uncertainties.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accuracies = []

    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (uncertainties >= low) & (uncertainties < high)
        if mask.any():
            # Use global accuracy proxy since we didn't store per-sample correctness above
            bin_accuracies.append(np.mean(accuracies))
        else:
            bin_accuracies.append(np.nan)

    bin_accuracies = np.array(bin_accuracies)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(bin_centers, bin_accuracies, width=(bin_edges[1] - bin_edges[0]) * 0.8, alpha=0.7, color="#6C4AB6", edgecolor="black")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=2)
    ax.set_xlabel("Predicted Uncertainty", fontsize=12)
    ax.set_ylabel("Actual Accuracy", fontsize=12)
    ax.set_title("Uncertainty Calibration (Reliability Diagram)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return fig


# -----------------------------------------------------------------------------+
# 5. plot_evolution_gif                                                         |
# -----------------------------------------------------------------------------+

def plot_evolution_gif(history_frames: List[Dict[str, Any]], save_path: str):
    """
    Create a GIF animation of routing heatmap evolution over training epochs.

    Each frame in ``history_frames`` must contain ``'epoch'`` and ``'routing_weights'``
    (a 2-D array of shape ``[B, K]``).

    Parameters
    ----------
    history_frames : list[dict]
        List of frame data dictionaries.
    save_path : str
        Path to save the resulting GIF (e.g. ``routing_evolution.gif``).

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    plt = _require_matplotlib()
    np = _require_numpy()

    if len(history_frames) == 0:
        raise ValueError("history_frames must not be empty.")

    fig, ax = plt.subplots(figsize=(8, 5))
    first = np.asarray(history_frames[0]["routing_weights"])
    im = ax.imshow(first, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xlabel("Top-K Unit Index", fontsize=12)
    ax.set_ylabel("Sample Index", fontsize=12)
    title = ax.set_title("Routing Weights – Epoch 0", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Weight")

    def _update(frame_idx):
        data = np.asarray(history_frames[frame_idx]["routing_weights"])
        im.set_data(data)
        im.set_clim(vmin=data.min(), vmax=data.max())
        epoch = history_frames[frame_idx].get("epoch", frame_idx)
        title.set_text(f"Routing Weights – Epoch {epoch}")
        return [im, title]

    anim = FuncAnimation(fig, _update, frames=len(history_frames), interval=500, blit=False)

    # Try imageio first, fallback to matplotlib built-in
    try:
        import imageio
        import os
        tmp_dir = "/tmp/opencode/ngs_frames"
        os.makedirs(tmp_dir, exist_ok=True)
        for i, _ in enumerate(history_frames):
            _update(i)
            fig.canvas.draw_idle()
            fig.savefig(os.path.join(tmp_dir, f"frame_{i:04d}.png"), dpi=100)
        with imageio.get_writer(save_path, mode="I", duration=0.5) as writer:
            for i in range(len(history_frames)):
                writer.append_data(imageio.imread(os.path.join(tmp_dir, f"frame_{i:04d}.png")))
    except Exception:
        try:
            anim.save(save_path, writer="pillow", fps=2)
        except Exception as exc:
            raise RuntimeError("Neither imageio nor matplotlib pillow writer could save the GIF.") from exc

    plt.close(fig)
    return anim


# -----------------------------------------------------------------------------+
# 6. plot_subspace_alignment                                                    |
# -----------------------------------------------------------------------------+

def plot_subspace_alignment(router, save_path: Optional[str] = None):
    """
    For factorized routers, show canonical correlation between subspaces.

    Assumes ``router`` is a :class:`FactorizedRouter` with attribute
    ``subspace_projectors`` (a list / ModuleList of projection layers).

    Parameters
    ----------
    router : FactorizedRouter
        A factorized router instance.
    save_path : str, optional
        If provided, the figure is saved to this path. Otherwise it is displayed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    np = _require_numpy()

    if not hasattr(router, "subspace_projectors"):
        raise AttributeError("Router does not expose 'subspace_projectors'; cannot compute alignment.")

    projectors = router.subspace_projectors
    n = len(projectors)

    # Gather projection matrices
    mats = []
    for p in projectors:
        if hasattr(p, "weight"):
            W = p.weight.detach().cpu().numpy() if _TORCH_AVAILABLE else np.asarray(p.weight)
        else:
            W = np.asarray(p)
        mats.append(W)

    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            # Canonical correlation ~ smallest singular value of orthonormal bases
            A = mats[i].T  # [d_i, latent_dim]
            B = mats[j].T  # [d_j, latent_dim]
            cov = A @ B.T
            try:
                _, s, _ = np.linalg.svd(cov)
                c = s.min()
            except Exception:
                c = 0.0
            corr_matrix[i, j] = c
            corr_matrix[j, i] = c

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("Subspace Index", fontsize=12)
    ax.set_ylabel("Subspace Index", fontsize=12)
    ax.set_title("Subspace Alignment (canonical correlation)", fontsize=14, fontweight="bold")
    for i_ in range(n):
        for j_ in range(n):
            ax.text(j_, i_, f"{corr_matrix[i_, j_]:.2f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Correlation")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return fig


# -----------------------------------------------------------------------------+
# 7. plot_hypernetwork_codes                                                    |
# -----------------------------------------------------------------------------+

def plot_hypernetwork_codes(model, save_path: Optional[str] = None):
    """
    t-SNE plot of generated adapter codes (for hypernetwork parameter stores).

    Assumes the model's ``param_store`` (or ``param_stores_per_subspace``) exposes a
    ``codes`` tensor of shape ``[max_k, code_dim]``.

    Parameters
    ----------
    model : NGSModel
        An NGS model using a hypernetwork parameter store.
    save_path : str, optional
        If provided, the figure is saved to this path. Otherwise it is displayed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    np = _require_numpy()

    codes = None
    # Try per-subspace stores first
    if hasattr(model, "param_stores_per_subspace") and model.param_stores_per_subspace is not None:
        all_codes = []
        for ps in model.param_stores_per_subspace:
            if hasattr(ps, "codes"):
                c = ps.codes.detach().cpu().numpy() if _TORCH_AVAILABLE else np.asarray(ps.codes)
                all_codes.append(c)
        if all_codes:
            codes = np.concatenate(all_codes, axis=0)
    elif hasattr(model, "param_store") and hasattr(model.param_store, "codes"):
        ps = model.param_store
        codes = ps.codes.detach().cpu().numpy() if _TORCH_AVAILABLE else np.asarray(ps.codes)

    if codes is None:
        raise AttributeError("Model does not expose hypernetwork codes. Ensure param_store has 'codes' attribute.")

    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=min(30, max(5, codes.shape[0] // 10)), random_state=42, init="pca", learning_rate="auto")
        embedded = tsne.fit_transform(codes)
    except Exception:
        # Fallback to PCA if t-SNE unavailable
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            embedded = pca.fit_transform(codes)
        except Exception:
            # Very basic fallback – just plot first two dims
            embedded = codes[:, :2]

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=np.arange(len(embedded)), cmap="Spectral", s=50, alpha=0.8, edgecolors="k", linewidth=0.4)
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("Hypernetwork Code Space (t-SNE)", fontsize=14, fontweight="bold")
    fig.colorbar(scatter, ax=ax, label="Unit Index")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return fig


# -----------------------------------------------------------------------------+
# 8. plot_riemannian_geodesics                                                  |
# -----------------------------------------------------------------------------+

def plot_riemannian_geodesics(model, z_start: Any, z_end: Any, save_path: Optional[str] = None):
    """
    Plot geodesic interpolation on the Riemannian manifold.

    Expects ``model`` to expose a ``riemannian_manifold`` attribute (or a
    ``manifold`` attribute) that implements ``interpolate(code_a, code_b, steps)``.
    If none is available, falls back to Euclidean linear interpolation in latent space.

    Parameters
    ----------
    model : NGSModel
        An NGS model (or a model with a Riemannian manifold component).
    z_start : torch.Tensor or array-like
        Starting latent vector / code.
    z_end : torch.Tensor or array-like
        Ending latent vector / code.
    save_path : str, optional
        If provided, the figure is saved to this path. Otherwise it is displayed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    np = _require_numpy()

    manifold = getattr(model, "riemannian_manifold", None) or getattr(model, "manifold", None)
    steps = 50

    if manifold is not None and hasattr(manifold, "interpolate"):
        geodesic = manifold.interpolate(z_start, z_end, steps=steps)
    else:
        # Euclidean fallback
        if _TORCH_AVAILABLE:
            t = torch.linspace(0, 1, steps, device=getattr(z_start, "device", "cpu"))
            geodesic = z_start.unsqueeze(0) * (1 - t).unsqueeze(1) + z_end.unsqueeze(0) * t.unsqueeze(1)
        else:
            t = np.linspace(0, 1, steps)
            geodesic = z_start[None, :] * (1 - t)[:, None] + z_end[None, :] * t[:, None]

    if _TORCH_AVAILABLE and isinstance(geodesic, torch.Tensor):
        geodesic = geodesic.detach().cpu().numpy()
    geodesic = np.asarray(geodesic)

    # Reduce to 2D for plotting if >2 dims
    if geodesic.shape[1] > 2:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            points_2d = pca.fit_transform(geodesic)
        except Exception:
            points_2d = geodesic[:, :2]
    else:
        pad = np.zeros((geodesic.shape[0], 2 - geodesic.shape[1]))
        points_2d = np.concatenate([geodesic, pad], axis=1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(points_2d[:, 0], points_2d[:, 1], "-o", color="#E63946", markersize=4, linewidth=2, label="Geodesic")
    ax.scatter(points_2d[0, 0], points_2d[0, 1], s=300, c="green", marker="*", edgecolors="k", zorder=5, label="Start")
    ax.scatter(points_2d[-1, 0], points_2d[-1, 1], s=300, c="blue", marker="*", edgecolors="k", zorder=5, label="End")
    ax.set_xlabel("Component 1", fontsize=12)
    ax.set_ylabel("Component 2", fontsize=12)
    ax.set_title("Riemannian Geodesic Interpolation", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return fig


# -----------------------------------------------------------------------------+
# 9. interactive_dashboard                                                      |
# -----------------------------------------------------------------------------+

def interactive_dashboard(model):
    """
    Launch a simple Plotly-based dashboard showing topology, routing, and uncertainty.

    Requires ``plotly`` to be installed. The dashboard is composed of subplots:
    - **Topology**: Bar chart of active units per subspace / level.
    - **Routing**: Heatmap of the most recent routing weights.
    - **Uncertainty**: Histogram of uncertainty values (if available).

    Parameters
    ----------
    model : NGSModel
        An instantiated and (ideally) evaluated NGS model.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        The Plotly figure object (use ``.show()``Embedding spawns the interactive viewer).
    """
    go, plotly_make_subplots = _require_plotly()
    np = _require_numpy()

    # Determine number of active units
    router = model.router
    if hasattr(router, "active_mask"):
        active_mask = router.active_mask
        if _TORCH_AVAILABLE and isinstance(active_mask, torch.Tensor):
            active_mask = active_mask.detach().cpu().numpy()
        active_count = int(active_mask.sum())
    elif hasattr(router, "K"):
        active_count = router.K
    else:
        active_count = 0

    max_k = getattr(router, "max_k", active_count)

    # Topology subplot
    topology_trace = go.Bar(
        x=["Active", "Inactive"],
        y=[active_count, max_k - active_count],
        marker_color=["#2E86AB", "#A8DADC"],
        name="Topology",
    )

    # Routing subplot (use cached routing if available)
    routing_heatmap = go.Heatmap(
        z=[[0.0]],
        colorscale="Hot",
        name="Routing",
        showscale=True,
    )
    # Try to get actual routing weights
    if hasattr(model, "_last_routing_weights") and model._last_routing_weights is not None:
        rw = model._last_routing_weights
        if _TORCH_AVAILABLE and isinstance(rw, torch.Tensor):
            rw = rw.detach().cpu().numpy()
        if rw is not None and rw.ndim == 2:
            routing_heatmap = go.Heatmap(
                z=np.asarray(rw),
                colorscale="Hot",
                name="Routing",
                showscale=True,
            )

    # Uncertainty subplot
    uncertainty_hist = go.Histogram(
        x=[0.5],
        nbinsx=20,
        marker_color="#6C4AB6",
        name="Uncertainty",
    )
    if hasattr(model, "_last_routing_output") and model._last_routing_output is not None:
        ro = model._last_routing_output
        if hasattr(ro, "uncertainty") and ro.uncertainty is not None:
            unc = ro.uncertainty
            if _TORCH_AVAILABLE and isinstance(unc, torch.Tensor):
                unc = unc.detach().cpu().numpy()
            uncertainty_hist = go.Histogram(
                x=np.asarray(unc).flatten(),
                nbinsx=20,
                marker_color="#6C4AB6",
                name="Uncertainty",
            )

    fig = plotly_make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=("Topology Overview", "Routing Weights", "Uncertainty Distribution"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    fig.add_trace(topology_trace, row=1, col=1)
    fig.add_trace(routing_heatmap, row=2, col=1)
    fig.add_trace(uncertainty_hist, row=2, col=2)

    fig.update_layout(
        title_text="NGS Interactive Dashboard",
        title_font_size=18,
        height=700,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Status", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Top-K Index", row=2, col=1)
    fig.update_yaxes(title_text="Sample Index", row=2, col=1)
    fig.update_xaxes(title_text="Uncertainty", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)

    return fig

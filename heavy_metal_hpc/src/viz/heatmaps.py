"""Spatial concentration heatmap utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_concentration(
    concentration: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Heavy Metal Concentration (µg/L)",
    cmap: str = "YlOrRd",
    vmin: float | None = None,
    vmax: float | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Render a 2-D concentration heatmap.

    Parameters
    ----------
    concentration:
        (nx, ny) concentration field (µg L⁻¹).
    x, y:
        1-D coordinate arrays for axes.
    title:
        Figure title.
    cmap:
        Matplotlib colormap name.
    vmin, vmax:
        Color scale limits; defaults to data range.
    save_path:
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(x, y, concentration.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    fig.colorbar(im, ax=ax, label="µg L⁻¹")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_time_series_snapshots(
    concentration_stack: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    n_cols: int = 4,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a grid of concentration snapshots at successive times.

    Parameters
    ----------
    concentration_stack:
        (T, nx, ny) array.
    x, y:
        Coordinate arrays.
    times:
        1-D array of snapshot times (s).
    n_cols:
        Number of columns in the subplot grid.
    save_path:
        Optional output path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    T = concentration_stack.shape[0]
    n_rows = (T + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    vmax = concentration_stack.max()
    for idx, ax in enumerate(axes.flat):
        if idx < T:
            ax.pcolormesh(x, y, concentration_stack[idx].T, cmap="YlOrRd", vmin=0, vmax=vmax, shading="auto")
            ax.set_title(f"t = {times[idx]:.0f} s")
            ax.set_aspect("equal")
            ax.axis("off")
        else:
            ax.set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

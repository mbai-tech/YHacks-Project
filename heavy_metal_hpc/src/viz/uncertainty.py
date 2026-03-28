"""Ensemble uncertainty visualization (spread maps, percentile envelopes)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_ensemble_spread(
    mean: np.ndarray,
    std: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    time_label: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Side-by-side ensemble mean and standard deviation maps.

    Parameters
    ----------
    mean:
        (nx, ny) ensemble mean concentration (µg L⁻¹).
    std:
        (nx, ny) ensemble standard deviation (µg L⁻¹).
    x, y:
        Coordinate arrays.
    time_label:
        String appended to subplot titles.
    save_path:
        Optional output file path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    im1 = ax1.pcolormesh(x, y, mean.T, cmap="YlOrRd", shading="auto")
    fig.colorbar(im1, ax=ax1, label="µg L⁻¹")
    ax1.set_title(f"Ensemble Mean {time_label}")
    ax1.set_aspect("equal")

    im2 = ax2.pcolormesh(x, y, std.T, cmap="Blues", shading="auto")
    fig.colorbar(im2, ax=ax2, label="µg L⁻¹")
    ax2.set_title(f"Ensemble Std Dev {time_label}")
    ax2.set_aspect("equal")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_percentile_envelope(
    times: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    location_label: str = "monitoring station",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Time-series percentile envelope at a single monitoring location.

    Parameters
    ----------
    times:
        1-D array of times (s).
    p10, p50, p90:
        10th, 50th, and 90th percentile concentration time series (µg L⁻¹).
    location_label:
        Label for the plot legend.
    save_path:
        Optional output path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(times, p10, p90, alpha=0.3, label="10–90th pctile")
    ax.plot(times, p50, lw=2, label="Median")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Concentration (µg L⁻¹)")
    ax.set_title(f"Uncertainty Envelope — {location_label}")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

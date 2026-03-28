"""General diagnostic plots: convergence, parameter distributions, cost curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_history(
    loss_history: list[float],
    log_scale: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the optimiser loss curve over iterations.

    Parameters
    ----------
    loss_history:
        Ordered list of loss values.
    log_scale:
        Use log scale on the y-axis.
    save_path:
        Optional output path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history, lw=1.5)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Optimiser Convergence")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_parameter_posterior(
    samples: np.ndarray,
    param_names: list[str],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Histogram matrix of posterior parameter samples.

    Parameters
    ----------
    samples:
        (M, n_params) array of ensemble / MCMC samples.
    param_names:
        List of parameter names for axis labels.
    save_path:
        Optional output path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = samples.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(samples[:, i], bins=30, edgecolor="white", linewidth=0.4)
        ax.set_title(name)
        ax.set_xlabel("Value")
        if i == 0:
            ax.set_ylabel("Count")
    fig.suptitle("Parameter Posterior Distributions", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_obs_vs_predicted(
    observed: np.ndarray,
    predicted: np.ndarray,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter plot of observed vs model-predicted concentrations.

    Parameters
    ----------
    observed, predicted:
        1-D concentration arrays.
    save_path:
        Optional output path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(observed, predicted, alpha=0.6, s=20)
    lims = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]
    ax.plot(lims, lims, "k--", lw=1, label="1:1 line")
    ax.set_xlabel("Observed (µg L⁻¹)")
    ax.set_ylabel("Predicted (µg L⁻¹)")
    ax.set_title("Calibration Fit")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

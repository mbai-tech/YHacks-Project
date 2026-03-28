"""Reusable finite-difference / finite-volume stencil operators."""

from __future__ import annotations

import numpy as np


def gradient_x(field: np.ndarray, dx: float) -> np.ndarray:
    """Central-difference x-gradient on interior, one-sided at boundaries.

    Parameters
    ----------
    field:
        (nx, ny) scalar field.
    dx:
        Grid spacing in x (m).

    Returns
    -------
    np.ndarray
        (nx, ny) ∂field/∂x.
    """
    return np.gradient(field, dx, axis=0)


def gradient_y(field: np.ndarray, dy: float) -> np.ndarray:
    """Central-difference y-gradient.

    Parameters
    ----------
    field:
        (nx, ny) scalar field.
    dy:
        Grid spacing in y (m).

    Returns
    -------
    np.ndarray
        (nx, ny) ∂field/∂y.
    """
    return np.gradient(field, dy, axis=1)


def laplacian(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Second-order isotropic Laplacian (∂²/∂x² + ∂²/∂y²).

    Parameters
    ----------
    field:
        (nx, ny) scalar field.
    dx, dy:
        Grid spacings (m).

    Returns
    -------
    np.ndarray
        (nx, ny) ∇²field.
    """
    d2x = np.gradient(np.gradient(field, dx, axis=0), dx, axis=0)
    d2y = np.gradient(np.gradient(field, dy, axis=1), dy, axis=1)
    return d2x + d2y


def divergence(fx: np.ndarray, fy: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Divergence of a vector field (∂fx/∂x + ∂fy/∂y).

    Parameters
    ----------
    fx, fy:
        (nx, ny) x- and y-components of the flux vector.
    dx, dy:
        Grid spacings (m).

    Returns
    -------
    np.ndarray
        (nx, ny) ∇·F.
    """
    return np.gradient(fx, dx, axis=0) + np.gradient(fy, dy, axis=1)


def apply_neumann_bc(field: np.ndarray) -> np.ndarray:
    """Enforce zero-flux (Neumann) boundary conditions via ghost-cell padding.

    Copies edge-cell values into the halo, then trims back to original shape.

    Parameters
    ----------
    field:
        (nx, ny) array (interior only, no ghost cells).

    Returns
    -------
    np.ndarray
        Same shape as input with boundary gradients zeroed.
    """
    out = field.copy()
    out[0, :] = out[1, :]
    out[-1, :] = out[-2, :]
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out

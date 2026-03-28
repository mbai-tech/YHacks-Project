"""Gradient-based and gradient-free optimizers for parameter calibration."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize, OptimizeResult


class GradientOptimizer:
    """Wrapper around ``scipy.optimize.minimize`` for differentiable loss functions.

    Parameters
    ----------
    method:
        SciPy minimization method (e.g. ``"L-BFGS-B"``, ``"SLSQP"``).
    bounds:
        Optional list of (min, max) bounds for each parameter.
    tol:
        Convergence tolerance.
    max_iter:
        Maximum number of iterations.
    """

    def __init__(
        self,
        method: str = "L-BFGS-B",
        bounds: list[tuple[float, float]] | None = None,
        tol: float = 1e-6,
        max_iter: int = 500,
    ) -> None:
        self.method = method
        self.bounds = bounds
        self.tol = tol
        self.max_iter = max_iter

    def minimize(
        self,
        loss_fn: Callable[[np.ndarray], float],
        x0: np.ndarray,
    ) -> OptimizeResult:
        """Run gradient-based minimisation.

        Parameters
        ----------
        loss_fn:
            Scalar loss function taking a 1-D parameter vector.
        x0:
            Initial parameter vector.

        Returns
        -------
        scipy.optimize.OptimizeResult
        """
        return minimize(
            loss_fn,
            x0,
            method=self.method,
            bounds=self.bounds,
            tol=self.tol,
            options={"maxiter": self.max_iter},
        )


class EnsembleKalmanInversion:
    """Derivative-free ensemble Kalman inversion (EKI) for parameter estimation.

    Parameters
    ----------
    n_ensemble:
        Number of ensemble members.
    n_iterations:
        Number of EKI outer iterations.
    inflation:
        Covariance inflation factor (>1 prevents collapse).
    """

    def __init__(
        self,
        n_ensemble: int = 50,
        n_iterations: int = 20,
        inflation: float = 1.0,
    ) -> None:
        self.n_ensemble = n_ensemble
        self.n_iterations = n_iterations
        self.inflation = inflation

    def run(
        self,
        forward_model: Callable[[np.ndarray], np.ndarray],
        observations: np.ndarray,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        noise_cov: np.ndarray,
    ) -> np.ndarray:
        """Run EKI and return the posterior ensemble mean.

        Parameters
        ----------
        forward_model:
            Maps parameter vector → observable vector.
        observations:
            Observed data vector.
        prior_mean:
            Prior mean of the parameter vector.
        prior_cov:
            Prior covariance matrix.
        noise_cov:
            Observation noise covariance matrix.

        Returns
        -------
        np.ndarray
            Estimated parameter vector (ensemble mean at final iteration).
        """
        raise NotImplementedError

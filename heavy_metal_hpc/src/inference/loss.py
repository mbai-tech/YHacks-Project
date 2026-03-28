"""Loss / misfit functions for comparing model output to observations."""

from __future__ import annotations

import numpy as np


def mse_loss(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Mean squared error between predicted and observed concentrations.

    Parameters
    ----------
    predicted:
        (N,) or (T, nx, ny) model output array.
    observed:
        Array of the same shape as *predicted*.

    Returns
    -------
    float
        Scalar MSE value.
    """
    return float(np.mean((predicted - observed) ** 2))


def weighted_mse_loss(
    predicted: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted MSE; useful for heteroscedastic observation errors.

    Parameters
    ----------
    predicted, observed:
        Arrays of the same shape.
    weights:
        Non-negative weight array of the same shape (e.g. 1/σ²).

    Returns
    -------
    float
        Weighted MSE.
    """
    return float(np.sum(weights * (predicted - observed) ** 2) / np.sum(weights))


def log_likelihood(
    predicted: np.ndarray,
    observed: np.ndarray,
    sigma: float | np.ndarray,
) -> float:
    """Gaussian log-likelihood of observations given model predictions.

    Parameters
    ----------
    predicted:
        Model-predicted concentrations.
    observed:
        Measured concentrations.
    sigma:
        Observation noise standard deviation (scalar or array).

    Returns
    -------
    float
        Log-likelihood value (higher is better).
    """
    residual = observed - predicted
    return float(-0.5 * np.sum((residual / sigma) ** 2 + np.log(2 * np.pi * sigma**2)))


class CompositeLoss:
    """Weighted sum of multiple loss terms.

    Parameters
    ----------
    terms:
        Mapping of loss-name → (callable, weight) pairs.
    """

    def __init__(self, terms: dict[str, tuple]) -> None:
        self.terms = terms

    def __call__(self, predicted: np.ndarray, observed: np.ndarray) -> float:
        """Evaluate the composite loss.

        Returns
        -------
        float
            Weighted sum of individual loss values.
        """
        total = 0.0
        for name, (fn, weight) in self.terms.items():
            total += weight * fn(predicted, observed)
        return total

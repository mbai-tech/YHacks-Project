"""Remediation objective functions."""

from __future__ import annotations

from typing import Callable

import numpy as np


class RemediationObjective:
    """Objective function for heavy-metal remediation optimisation.

    The objective minimises a weighted combination of:
    1. Residual contamination mass in the water column.
    2. Total remediation cost.

    Parameters
    ----------
    simulator_fn:
        Callable that takes a remediation action vector and returns the
        final-state concentration field (nx, ny).
    cost_fn:
        Callable that takes the action vector and returns a scalar cost (CNY).
    mass_weight:
        Penalty weight for residual contamination.
    cost_weight:
        Penalty weight for total remediation cost.
    target_concentration:
        Maximum acceptable concentration threshold (µg L⁻¹).
    """

    def __init__(
        self,
        simulator_fn: Callable[[np.ndarray], np.ndarray],
        cost_fn: Callable[[np.ndarray], float],
        mass_weight: float = 1.0,
        cost_weight: float = 1e-6,
        target_concentration: float = 10.0,
    ) -> None:
        self.simulator_fn = simulator_fn
        self.cost_fn = cost_fn
        self.mass_weight = mass_weight
        self.cost_weight = cost_weight
        self.target = target_concentration

    def __call__(self, action: np.ndarray) -> float:
        """Evaluate the objective for a given remediation action vector.

        Parameters
        ----------
        action:
            Decision variable vector (e.g. treatment intensities at each site).

        Returns
        -------
        float
            Scalar objective value to minimise.
        """
        final_c = self.simulator_fn(action)
        excess = np.maximum(final_c - self.target, 0.0)
        mass_penalty = float(np.sum(excess**2))
        cost_penalty = self.cost_fn(action)
        return self.mass_weight * mass_penalty + self.cost_weight * cost_penalty

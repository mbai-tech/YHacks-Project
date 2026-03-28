"""Constraint definitions for the remediation optimisation problem."""

from __future__ import annotations

import numpy as np


class BudgetConstraint:
    """Total remediation budget constraint: sum(cost(a)) ≤ budget.

    Parameters
    ----------
    cost_per_unit:
        Per-site unit cost vector (CNY / unit action).
    budget:
        Maximum allowable total cost (CNY).
    """

    def __init__(self, cost_per_unit: np.ndarray, budget: float) -> None:
        self.cost_per_unit = cost_per_unit
        self.budget = budget

    def residual(self, action: np.ndarray) -> float:
        """Return budget - total_cost; ≥ 0 means feasible."""
        return self.budget - float(np.dot(self.cost_per_unit, action))

    def to_scipy_dict(self) -> dict:
        """Return a SciPy-compatible constraint dictionary."""
        return {"type": "ineq", "fun": self.residual}


class ConcentrationConstraint:
    """Enforce that peak post-remediation concentration stays below a threshold.

    Parameters
    ----------
    simulator_fn:
        Callable that maps action vector → (nx, ny) final concentration field.
    threshold:
        Maximum allowable concentration (µg L⁻¹).
    """

    def __init__(self, simulator_fn, threshold: float = 10.0) -> None:
        self.simulator_fn = simulator_fn
        self.threshold = threshold

    def residual(self, action: np.ndarray) -> float:
        """Return threshold - max_concentration; ≥ 0 means feasible."""
        c = self.simulator_fn(action)
        return self.threshold - float(c.max())

    def to_scipy_dict(self) -> dict:
        """Return a SciPy-compatible constraint dictionary."""
        return {"type": "ineq", "fun": self.residual}


class ActionBounds:
    """Box constraints for individual action components.

    Parameters
    ----------
    lower:
        Lower bound per action variable.
    upper:
        Upper bound per action variable.
    """

    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
        self.lower = lower
        self.upper = upper

    def to_scipy_bounds(self) -> list[tuple[float, float]]:
        """Return a list of (low, high) tuples for ``scipy.optimize.minimize``."""
        return list(zip(self.lower.tolist(), self.upper.tolist()))

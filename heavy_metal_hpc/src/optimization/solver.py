"""Remediation optimisation solver that assembles objective + constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from .objective import RemediationObjective
from .constraints import BudgetConstraint, ConcentrationConstraint, ActionBounds


@dataclass
class RemediationPlan:
    """Output of a successful optimisation run.

    Attributes
    ----------
    action:
        Optimal remediation action vector.
    objective_value:
        Final objective value.
    total_cost:
        Total estimated remediation cost (CNY).
    converged:
        Whether the solver reported convergence.
    message:
        Solver status message.
    """

    action: np.ndarray
    objective_value: float
    total_cost: float
    converged: bool
    message: str


class RemediationSolver:
    """Assembles and solves the remediation optimisation problem.

    Parameters
    ----------
    objective:
        Remediation objective function.
    budget_constraint:
        Budget feasibility constraint.
    concentration_constraint:
        Peak-concentration feasibility constraint.
    action_bounds:
        Box bounds on the decision variables.
    method:
        SciPy solver (default: ``"SLSQP"`` for constrained NLP).
    """

    def __init__(
        self,
        objective: RemediationObjective,
        budget_constraint: BudgetConstraint,
        concentration_constraint: ConcentrationConstraint,
        action_bounds: ActionBounds,
        method: str = "SLSQP",
    ) -> None:
        self.objective = objective
        self.budget_constraint = budget_constraint
        self.concentration_constraint = concentration_constraint
        self.action_bounds = action_bounds
        self.method = method

    def solve(self, x0: np.ndarray) -> RemediationPlan:
        """Run the constrained optimisation.

        Parameters
        ----------
        x0:
            Initial action vector.

        Returns
        -------
        RemediationPlan
        """
        constraints = [
            self.budget_constraint.to_scipy_dict(),
            self.concentration_constraint.to_scipy_dict(),
        ]
        bounds = self.action_bounds.to_scipy_bounds()

        result: OptimizeResult = minimize(
            self.objective,
            x0,
            method=self.method,
            bounds=bounds,
            constraints=constraints,
        )

        total_cost = float(
            np.dot(self.budget_constraint.cost_per_unit, result.x)
        )

        return RemediationPlan(
            action=result.x,
            objective_value=float(result.fun),
            total_cost=total_cost,
            converged=bool(result.success),
            message=result.message,
        )

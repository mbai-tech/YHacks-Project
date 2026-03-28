"""Unit tests for the optimization module."""

from __future__ import annotations

import numpy as np
import pytest

from src.optimization.objective import RemediationObjective
from src.optimization.constraints import BudgetConstraint, ActionBounds
from src.optimization.solver import RemediationSolver, RemediationPlan
from src.optimization.constraints import ConcentrationConstraint


def _dummy_simulator(action: np.ndarray) -> np.ndarray:
    """Toy simulator: returns a 5×5 concentration field proportional to action sum."""
    level = max(0.0, 20.0 - float(action.sum()))
    return np.full((5, 5), level)


def _dummy_cost(action: np.ndarray) -> float:
    return float(action.sum() * 1000.0)


@pytest.fixture
def n_sites() -> int:
    return 3


@pytest.fixture
def objective(n_sites: int) -> RemediationObjective:
    return RemediationObjective(
        simulator_fn=_dummy_simulator,
        cost_fn=_dummy_cost,
        mass_weight=1.0,
        cost_weight=1e-5,
        target_concentration=10.0,
    )


@pytest.fixture
def budget_constraint(n_sites: int) -> BudgetConstraint:
    cost_per_unit = np.ones(n_sites) * 100.0
    return BudgetConstraint(cost_per_unit=cost_per_unit, budget=5_000.0)


class TestRemediationObjective:
    """Tests for the remediation objective function."""

    def test_zero_action_positive_objective(self, objective: RemediationObjective, n_sites: int) -> None:
        val = objective(np.zeros(n_sites))
        assert val > 0

    def test_high_action_reduces_objective(self, objective: RemediationObjective, n_sites: int) -> None:
        low = objective(np.zeros(n_sites))
        high_action = np.ones(n_sites) * 10.0
        val_high = objective(high_action)
        assert val_high < low


class TestBudgetConstraint:
    """Tests for the BudgetConstraint."""

    def test_feasible_action(self, budget_constraint: BudgetConstraint, n_sites: int) -> None:
        action = np.ones(n_sites) * 5.0  # cost = 3*5*100 = 1500 < 5000
        assert budget_constraint.residual(action) > 0

    def test_infeasible_action(self, budget_constraint: BudgetConstraint, n_sites: int) -> None:
        action = np.ones(n_sites) * 100.0  # cost = 3*100*100 = 30000 > 5000
        assert budget_constraint.residual(action) < 0

    def test_scipy_dict_format(self, budget_constraint: BudgetConstraint) -> None:
        d = budget_constraint.to_scipy_dict()
        assert d["type"] == "ineq"
        assert callable(d["fun"])


class TestActionBounds:
    """Tests for the ActionBounds box-constraint helper."""

    def test_to_scipy_bounds_length(self, n_sites: int) -> None:
        bounds = ActionBounds(lower=np.zeros(n_sites), upper=np.ones(n_sites) * 50.0)
        sb = bounds.to_scipy_bounds()
        assert len(sb) == n_sites
        assert sb[0] == (0.0, 50.0)


class TestRemediationSolver:
    """Integration test: solver finds a feasible plan."""

    def test_solver_returns_plan(self, objective, budget_constraint, n_sites) -> None:
        conc_con = ConcentrationConstraint(simulator_fn=_dummy_simulator, threshold=15.0)
        bounds = ActionBounds(lower=np.zeros(n_sites), upper=np.ones(n_sites) * 50.0)
        solver = RemediationSolver(objective, budget_constraint, conc_con, bounds)
        plan = solver.solve(x0=np.ones(n_sites) * 5.0)
        assert isinstance(plan, RemediationPlan)
        assert plan.action.shape == (n_sites,)
        assert plan.total_cost >= 0

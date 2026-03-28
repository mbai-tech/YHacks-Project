#!/usr/bin/env python
"""Solve the heavy-metal remediation optimisation problem.

Usage
-----
    python scripts/run_optimization.py --config config/default.yaml --budget 1e6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.grid.mesh import StructuredMesh
from src.model.simulator import Simulator
from src.model.state import SimulationState
from src.optimization.objective import RemediationObjective
from src.optimization.constraints import BudgetConstraint, ConcentrationConstraint, ActionBounds
from src.optimization.solver import RemediationSolver
from src.utils.logging import logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Remediation optimisation.")
    p.add_argument("--config", required=True)
    p.add_argument("--budget", type=float, default=1_000_000.0, help="Budget (CNY).")
    p.add_argument("--n-sites", type=int, default=10, help="Number of candidate remediation sites.")
    p.add_argument("--output", default="data/processed/remediation_plan.npz")
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    cfg = load_config(args.config)
    logger.info(f"Optimisation | budget={args.budget:.0f} CNY | sites={args.n_sites}")

    mesh = StructuredMesh(
        x_min=cfg.domain.x_min, x_max=cfg.domain.x_max,
        y_min=cfg.domain.y_min, y_max=cfg.domain.y_max,
        nx=cfg.domain.nx, ny=cfg.domain.ny,
    )
    simulator = Simulator(mesh=mesh, phys=cfg.physics, num=cfg.numerics)

    initial_state = SimulationState(
        concentration=np.zeros(mesh.shape),
        sediment_concentration=np.zeros(mesh.shape),
        u=np.zeros(mesh.shape),
        v=np.zeros(mesh.shape),
        depth=np.ones(mesh.shape) * 2.0,
    )
    n_steps = cfg.numerics.n_steps
    forcing = {"u": np.zeros((n_steps, *mesh.shape)), "v": np.zeros((n_steps, *mesh.shape))}

    def simulator_fn(action: np.ndarray) -> np.ndarray:
        """Run the model and return the final concentration field."""
        history = simulator.run(initial_state, forcing)
        return history.snapshots[-1].concentration

    cost_per_unit = np.ones(args.n_sites) * (args.budget / args.n_sites / 10)

    objective = RemediationObjective(
        simulator_fn=simulator_fn,
        cost_fn=lambda a: float(np.dot(cost_per_unit, a)),
        target_concentration=10.0,
    )
    budget_con = BudgetConstraint(cost_per_unit=cost_per_unit, budget=args.budget)
    conc_con = ConcentrationConstraint(simulator_fn=simulator_fn, threshold=10.0)
    bounds = ActionBounds(
        lower=np.zeros(args.n_sites),
        upper=np.ones(args.n_sites) * 100.0,
    )

    solver = RemediationSolver(objective, budget_con, conc_con, bounds)
    x0 = np.ones(args.n_sites) * 10.0
    plan = solver.solve(x0)

    np.savez(
        args.output,
        action=plan.action,
        objective_value=plan.objective_value,
        total_cost=plan.total_cost,
    )
    logger.success(
        f"Optimisation {'converged' if plan.converged else 'did not converge'} | "
        f"cost={plan.total_cost:.0f} CNY | obj={plan.objective_value:.4f} → {args.output}"
    )


if __name__ == "__main__":
    main()

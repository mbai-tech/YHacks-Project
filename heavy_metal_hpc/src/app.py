"""CLI entry-point for the heavy_metal_hpc package."""

from __future__ import annotations

import click

from .utils.config import load_config, RunConfig
from .grid.mesh import StructuredMesh
from .model.simulator import Simulator
from .model.state import SimulationState
import numpy as np


@click.group()
def cli() -> None:
    """heavy_metal_hpc — HPC simulation of heavy metal transport in Baiyangdian Lake."""


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to YAML config file.")
@click.option("--output", "-o", default="data/processed/simulation.h5", show_default=True, help="Output HDF5 path.")
def simulate(config: str, output: str) -> None:
    """Run a forward simulation."""
    cfg: RunConfig = load_config(config)
    click.echo(f"Running simulation '{cfg.name}'...")

    mesh = StructuredMesh(
        x_min=cfg.domain.x_min,
        x_max=cfg.domain.x_max,
        y_min=cfg.domain.y_min,
        y_max=cfg.domain.y_max,
        nx=cfg.domain.nx,
        ny=cfg.domain.ny,
    )
    simulator = Simulator(mesh=mesh, phys=cfg.physics, num=cfg.numerics)

    shape = (cfg.numerics.n_steps, cfg.domain.nx, cfg.domain.ny)
    initial_state = SimulationState(
        concentration=np.zeros((cfg.domain.nx, cfg.domain.ny)),
        sediment_concentration=np.zeros((cfg.domain.nx, cfg.domain.ny)),
        u=np.zeros((cfg.domain.nx, cfg.domain.ny)),
        v=np.zeros((cfg.domain.nx, cfg.domain.ny)),
        depth=np.ones((cfg.domain.nx, cfg.domain.ny)) * 2.0,
    )
    forcing = {
        "u": np.zeros(shape),
        "v": np.zeros(shape),
    }

    history = simulator.run(initial_state, forcing)
    click.echo(f"Simulation complete. {len(history.snapshots)} snapshots saved to {output}.")


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--n-members", "-n", default=50, show_default=True, type=int)
def ensemble(config: str, n_members: int) -> None:
    """Run a Monte-Carlo ensemble."""
    click.echo(f"Launching ensemble with {n_members} members...")
    raise NotImplementedError("Wire up EnsembleRunner here.")


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--budget", default=1_000_000.0, show_default=True, type=float, help="Remediation budget (CNY).")
def optimize(config: str, budget: float) -> None:
    """Solve the remediation optimisation problem."""
    click.echo(f"Solving remediation optimisation (budget={budget:.0f} CNY)...")
    raise NotImplementedError("Wire up RemediationSolver here.")


if __name__ == "__main__":
    cli()

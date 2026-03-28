#!/usr/bin/env python
"""Run a single forward simulation.

Usage
-----
    python scripts/run_simulation.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging import logger
from src.grid.mesh import StructuredMesh
from src.model.simulator import Simulator
from src.model.state import SimulationState
from src.utils.io import save_hdf5


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Run forward heavy-metal transport simulation.")
    p.add_argument("--config", required=True, help="Path to YAML run configuration.")
    p.add_argument("--output", default="data/processed/simulation.h5", help="Output HDF5 file.")
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    cfg = load_config(args.config)
    logger.info(f"Starting run '{cfg.name}' | grid {cfg.domain.nx}×{cfg.domain.ny}")

    mesh = StructuredMesh(
        x_min=cfg.domain.x_min,
        x_max=cfg.domain.x_max,
        y_min=cfg.domain.y_min,
        y_max=cfg.domain.y_max,
        nx=cfg.domain.nx,
        ny=cfg.domain.ny,
    )

    initial_state = SimulationState(
        concentration=np.zeros(mesh.shape),
        sediment_concentration=np.zeros(mesh.shape),
        u=np.zeros(mesh.shape),
        v=np.zeros(mesh.shape),
        depth=np.ones(mesh.shape) * 2.0,
    )

    n_steps = cfg.numerics.n_steps
    forcing = {
        "u": np.zeros((n_steps, *mesh.shape)),
        "v": np.zeros((n_steps, *mesh.shape)),
    }

    simulator = Simulator(mesh=mesh, phys=cfg.physics, num=cfg.numerics)
    history = simulator.run(initial_state, forcing)

    save_hdf5(
        {"concentration": history.concentration_stack(), "times": history.times()},
        args.output,
    )
    logger.success(f"Saved {len(history.snapshots)} snapshots → {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run a Monte-Carlo ensemble of forward simulations.

Usage
-----
    python scripts/run_ensemble.py --config config/default.yaml --n-members 64
    mpirun -n 8 python scripts/run_ensemble.py --config config/default.yaml --n-members 64
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
from src.model.parameters import PhysicalParameters
from src.ensemble.sampler import MonteCarloSampler
from src.ensemble.runner import EnsembleRunner
from src.utils.io import save_hdf5
from src.utils.logging import logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Run Monte-Carlo ensemble.")
    p.add_argument("--config", required=True)
    p.add_argument("--n-members", type=int, default=50)
    p.add_argument("--std-frac", type=float, default=0.2, help="Parameter perturbation fraction.")
    p.add_argument("--backend", default="serial", choices=["serial", "dask", "ray"])
    p.add_argument("--output", default="data/processed/ensemble.h5")
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    cfg = load_config(args.config)
    logger.info(f"Ensemble run '{cfg.name}' | {args.n_members} members | backend={args.backend}")

    mesh = StructuredMesh(
        x_min=cfg.domain.x_min, x_max=cfg.domain.x_max,
        y_min=cfg.domain.y_min, y_max=cfg.domain.y_max,
        nx=cfg.domain.nx, ny=cfg.domain.ny,
    )
    initial_state = SimulationState(
        concentration=np.zeros(mesh.shape),
        sediment_concentration=np.zeros(mesh.shape),
        u=np.zeros(mesh.shape),
        v=np.zeros(mesh.shape),
        depth=np.ones(mesh.shape) * 2.0,
    )
    n_steps = cfg.numerics.n_steps
    forcing = {"u": np.zeros((n_steps, *mesh.shape)), "v": np.zeros((n_steps, *mesh.shape))}

    sampler = MonteCarloSampler(
        mean=cfg.physics,
        std_frac=args.std_frac,
        n_samples=args.n_members,
        seed=cfg.random_seed,
    )
    param_ensemble = sampler.sample()

    def factory(p: PhysicalParameters) -> Simulator:
        return Simulator(mesh=mesh, phys=p, num=cfg.numerics)

    runner = EnsembleRunner(factory, backend=args.backend)
    result = runner.run(param_ensemble, initial_state, forcing)

    save_hdf5(
        {
            "mean": result.mean(),
            "std": result.std(),
            "p10": result.percentile(10),
            "p90": result.percentile(90),
        },
        args.output,
    )
    logger.success(f"Ensemble done → {args.output}")


if __name__ == "__main__":
    main()

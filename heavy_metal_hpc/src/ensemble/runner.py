"""Parallel ensemble runner using Dask or multiprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.parameters import PhysicalParameters
from ..model.simulator import Simulator
from ..model.state import SimulationState, StateHistory


@dataclass
class EnsembleResult:
    """Aggregated output from an ensemble run.

    Attributes
    ----------
    members:
        List of per-member :class:`StateHistory` objects.
    parameters:
        List of :class:`PhysicalParameters` used for each member.
    """

    members: list[StateHistory]
    parameters: list[PhysicalParameters]

    def concentration_ensemble(self) -> np.ndarray:
        """Return (M, T, nx, ny) concentration array over all members and snapshots."""
        return np.stack([m.concentration_stack() for m in self.members], axis=0)

    def mean(self) -> np.ndarray:
        """Ensemble mean concentration (T, nx, ny)."""
        return self.concentration_ensemble().mean(axis=0)

    def std(self) -> np.ndarray:
        """Ensemble standard deviation (T, nx, ny)."""
        return self.concentration_ensemble().std(axis=0)

    def percentile(self, q: float) -> np.ndarray:
        """Return the q-th percentile concentration (T, nx, ny)."""
        return np.percentile(self.concentration_ensemble(), q, axis=0)


class EnsembleRunner:
    """Runs multiple forward simulations in parallel.

    Parameters
    ----------
    simulator_factory:
        Callable that takes :class:`PhysicalParameters` and returns a
        configured :class:`Simulator` instance.
    backend:
        Parallelism backend: ``"dask"``, ``"ray"``, or ``"serial"``.
    n_workers:
        Number of parallel workers (ignored for ``"serial"``).
    """

    def __init__(
        self,
        simulator_factory,
        backend: str = "serial",
        n_workers: int = 4,
    ) -> None:
        self.simulator_factory = simulator_factory
        self.backend = backend
        self.n_workers = n_workers

    def run(
        self,
        parameter_ensemble: list[PhysicalParameters],
        initial_state: SimulationState,
        forcing: dict,
    ) -> EnsembleResult:
        """Execute all ensemble members and collect results.

        Parameters
        ----------
        parameter_ensemble:
            List of parameter sets (one per member).
        initial_state:
            Shared initial state for all members.
        forcing:
            Shared forcing arrays.

        Returns
        -------
        EnsembleResult
        """
        if self.backend == "serial":
            return self._run_serial(parameter_ensemble, initial_state, forcing)
        elif self.backend == "dask":
            return self._run_dask(parameter_ensemble, initial_state, forcing)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

    def _run_serial(
        self,
        params: list[PhysicalParameters],
        state: SimulationState,
        forcing: dict,
    ) -> EnsembleResult:
        """Serial (single-process) fallback."""
        histories = []
        for p in params:
            sim = self.simulator_factory(p)
            histories.append(sim.run(state, forcing))
        return EnsembleResult(members=histories, parameters=params)

    def _run_dask(
        self,
        params: list[PhysicalParameters],
        state: SimulationState,
        forcing: dict,
    ) -> EnsembleResult:
        """Dask-distributed parallel execution."""
        raise NotImplementedError("Wire up a Dask client and submit tasks here.")

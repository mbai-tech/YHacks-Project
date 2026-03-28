"""High-level parameter estimation driver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.parameters import PhysicalParameters
from ..model.simulator import Simulator
from ..model.state import SimulationState
from .loss import CompositeLoss
from .optimizer import GradientOptimizer


@dataclass
class EstimationResult:
    """Output of a parameter estimation run.

    Attributes
    ----------
    parameters:
        Estimated :class:`PhysicalParameters`.
    loss_history:
        Loss value at each optimisation iteration.
    converged:
        Whether the optimiser reported convergence.
    message:
        Optimiser status message.
    """

    parameters: PhysicalParameters
    loss_history: list[float]
    converged: bool
    message: str


class ParameterEstimator:
    """Calibrates physical parameters by minimising model-observation misfit.

    Parameters
    ----------
    simulator:
        Configured forward-model instance (parameters will be overwritten
        at each optimisation step).
    loss_fn:
        Composite loss function to minimise.
    optimizer:
        Underlying optimisation algorithm.
    """

    def __init__(
        self,
        simulator: Simulator,
        loss_fn: CompositeLoss,
        optimizer: GradientOptimizer,
    ) -> None:
        self.simulator = simulator
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self._loss_history: list[float] = []

    def fit(
        self,
        initial_state: SimulationState,
        forcing: dict,
        observations: np.ndarray,
        x0: PhysicalParameters,
    ) -> EstimationResult:
        """Run the full estimation loop.

        Parameters
        ----------
        initial_state:
            Starting model state.
        forcing:
            Velocity / depth forcing arrays.
        observations:
            Observed concentration data to fit.
        x0:
            Initial parameter guess.

        Returns
        -------
        EstimationResult
        """
        self._loss_history = []

        def objective(v: np.ndarray) -> float:
            params = PhysicalParameters.from_vector(v.tolist())
            self.simulator.phys = params
            history = self.simulator.run(initial_state, forcing)
            predicted = history.concentration_stack()
            loss = self.loss_fn(predicted, observations)
            self._loss_history.append(loss)
            return loss

        result = self.optimizer.minimize(objective, np.array(x0.to_vector()))

        best_params = PhysicalParameters.from_vector(result.x.tolist())
        return EstimationResult(
            parameters=best_params,
            loss_history=self._loss_history,
            converged=bool(result.success),
            message=result.message,
        )

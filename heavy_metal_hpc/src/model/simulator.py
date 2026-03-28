"""Top-level forward simulator that wires together physics and forcing."""

from __future__ import annotations

from loguru import logger

from ..grid.mesh import StructuredMesh
from ..physics.transport import TransportModel
from ..physics.sediment import SedimentExchange
from .parameters import PhysicalParameters, NumericalParameters
from .state import SimulationState, StateHistory


class Simulator:
    """Forward-model runner for heavy-metal transport in Baiyangdian Lake.

    The simulator integrates:
    - Advection-diffusion-reaction of dissolved metal
    - Sediment-water partitioning exchange
    - External velocity and depth forcing

    Parameters
    ----------
    mesh:
        Computational grid.
    phys:
        Calibratable physical parameters.
    num:
        Numerical scheme settings.
    """

    def __init__(
        self,
        mesh: StructuredMesh,
        phys: PhysicalParameters,
        num: NumericalParameters,
    ) -> None:
        self.mesh = mesh
        self.phys = phys
        self.num = num
        self._transport = TransportModel(mesh, diffusivity=phys.diffusivity)
        self._sediment = SedimentExchange(
            k_deposition=phys.k_deposition,
            k_resuspension=phys.k_resuspension,
        )

    def run(
        self,
        initial_state: SimulationState,
        forcing: dict,
    ) -> StateHistory:
        """Execute a full forward simulation.

        Parameters
        ----------
        initial_state:
            Starting concentration and sediment fields.
        forcing:
            Dictionary with keys ``"u"``, ``"v"`` (velocity arrays of shape
            (n_steps, nx, ny)) and optionally ``"depth"`` (same shape).

        Returns
        -------
        StateHistory
            All saved snapshots.
        """
        state = initial_state.copy()
        history = StateHistory()
        history.append(state)

        for step in range(self.num.n_steps):
            state = self._advance(state, forcing, step)
            if (step + 1) % self.num.output_interval == 0:
                history.append(state)
                logger.debug(
                    f"Step {step + 1}/{self.num.n_steps} | "
                    f"max_C={state.concentration.max():.4f} µg/L"
                )

        return history

    def _advance(
        self,
        state: SimulationState,
        forcing: dict,
        step: int,
    ) -> SimulationState:
        """Advance the model state by one time step.

        Parameters
        ----------
        state:
            Current state.
        forcing:
            Full forcing arrays (see :meth:`run`).
        step:
            Current step index (0-based).

        Returns
        -------
        SimulationState
            Updated state at time + dt.
        """
        dt = self.num.dt
        u = forcing["u"][step]
        v = forcing["v"][step]

        sed_source = self._sediment.flux(
            state.concentration,
            state.sediment_concentration,
            state.depth,
        )
        new_c = self._transport.step(state.concentration, u, v, sed_source, dt)
        new_sed = self._sediment.update_sediment(
            state.sediment_concentration, state.concentration, dt
        )

        return SimulationState(
            concentration=new_c,
            sediment_concentration=new_sed,
            u=u,
            v=v,
            depth=state.depth,
            time=state.time + dt,
        )

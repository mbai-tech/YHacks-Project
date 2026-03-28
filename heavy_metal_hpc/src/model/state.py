"""Simulation state vector and snapshot I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SimulationState:
    """Full model state at a single time instant.

    Parameters
    ----------
    concentration:
        (nx, ny) dissolved heavy-metal concentration (µg L⁻¹).
    sediment_concentration:
        (nx, ny) sediment-bound heavy-metal concentration (µg kg⁻¹).
    u, v:
        (nx, ny) depth-averaged velocity components (m s⁻¹).
    depth:
        (nx, ny) water depth (m).
    time:
        Simulation time (s since epoch).
    """

    concentration: np.ndarray
    sediment_concentration: np.ndarray
    u: np.ndarray
    v: np.ndarray
    depth: np.ndarray
    time: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial grid shape (nx, ny)."""
        return self.concentration.shape

    def copy(self) -> "SimulationState":
        """Return a deep copy of the current state."""
        return SimulationState(
            concentration=self.concentration.copy(),
            sediment_concentration=self.sediment_concentration.copy(),
            u=self.u.copy(),
            v=self.v.copy(),
            depth=self.depth.copy(),
            time=self.time,
        )

    def to_dict(self) -> dict[str, np.ndarray | float]:
        """Serialise state to a plain dictionary for HDF5 / zarr storage."""
        return {
            "concentration": self.concentration,
            "sediment_concentration": self.sediment_concentration,
            "u": self.u,
            "v": self.v,
            "depth": self.depth,
            "time": np.float64(self.time),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationState":
        """Deserialise from a dictionary produced by :meth:`to_dict`."""
        return cls(
            concentration=np.asarray(d["concentration"]),
            sediment_concentration=np.asarray(d["sediment_concentration"]),
            u=np.asarray(d["u"]),
            v=np.asarray(d["v"]),
            depth=np.asarray(d["depth"]),
            time=float(d["time"]),
        )


@dataclass
class StateHistory:
    """Ordered collection of state snapshots written during a model run."""

    snapshots: list[SimulationState] = field(default_factory=list)

    def append(self, state: SimulationState) -> None:
        """Append a (deep-copied) snapshot."""
        self.snapshots.append(state.copy())

    def concentration_stack(self) -> np.ndarray:
        """Return (T, nx, ny) concentration array for all saved snapshots."""
        return np.stack([s.concentration for s in self.snapshots], axis=0)

    def times(self) -> np.ndarray:
        """Return 1-D array of snapshot times (s)."""
        return np.array([s.time for s in self.snapshots])

    def save(self, path: str | Path) -> None:
        """Persist history to an HDF5 file via h5py."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "StateHistory":
        """Load history from an HDF5 file."""
        raise NotImplementedError

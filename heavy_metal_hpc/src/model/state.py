"""Simulation state data model for heavy-metal transport in Baiyangdian Lake.

All arrays use float64 unless noted.  Axis conventions:
  - Spatial arrays are (nx, ny)  — x is the first (row) axis, y the second.
  - Multi-metal arrays are (N_METALS, nx, ny), ordered by METALS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Metal registry — single source of truth for ordering across the project
# ---------------------------------------------------------------------------

MetalName = Literal["Pb", "Cu", "Cd", "Cr"]
METALS: tuple[MetalName, ...] = ("Pb", "Cu", "Cd", "Cr")
N_METALS: int = len(METALS)  # 4
METAL_INDEX: dict[MetalName, int] = {m: i for i, m in enumerate(METALS)}


# ---------------------------------------------------------------------------
# Grid dimensions
# ---------------------------------------------------------------------------

@dataclass
class GridDimensions:
    """Describes the 2-D structured Cartesian grid.

    Parameters
    ----------
    nx, ny:
        Number of cells along the x (east) and y (north) axes.
    dx, dy:
        Cell width and height (m).
    x_origin, y_origin:
        Coordinates of the south-west corner of cell (0, 0) (m or decimal
        degrees — must be consistent with dx/dy).
    """

    nx: int
    ny: int
    dx: float          # m
    dy: float          # m
    x_origin: float = 0.0
    y_origin: float = 0.0

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError(f"Grid dimensions must be positive, got nx={self.nx}, ny={self.ny}")
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError(f"Cell sizes must be positive, got dx={self.dx}, dy={self.dy}")

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial shape (nx, ny)."""
        return (self.nx, self.ny)

    @property
    def n_cells(self) -> int:
        """Total number of grid cells."""
        return self.nx * self.ny

    @property
    def cell_area(self) -> float:
        """Area of a single cell (m²)."""
        return self.dx * self.dy

    def x_centers(self) -> np.ndarray:
        """1-D array of cell-centre x coordinates (m), shape (nx,)."""
        return self.x_origin + (np.arange(self.nx) + 0.5) * self.dx

    def y_centers(self) -> np.ndarray:
        """1-D array of cell-centre y coordinates (m), shape (ny,)."""
        return self.y_origin + (np.arange(self.ny) + 0.5) * self.dy

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, Y) coordinate arrays, each of shape (nx, ny)."""
        return np.meshgrid(self.x_centers(), self.y_centers(), indexing="ij")


# ---------------------------------------------------------------------------
# Per-field concentration containers
# ---------------------------------------------------------------------------

@dataclass
class WaterConcentrations:
    """Dissolved heavy-metal concentrations in the water column.

    Each field holds the concentration for one metal species on the model
    grid.  Units: µg L⁻¹.

    Parameters
    ----------
    Pb, Cu, Cd, Cr:
        (nx, ny) float64 arrays.
    """

    Pb: np.ndarray   # µg L⁻¹
    Cu: np.ndarray
    Cd: np.ndarray
    Cr: np.ndarray

    def validate_shapes(self, expected_shape: tuple[int, int]) -> None:
        """Raise ValueError if any array does not match *expected_shape*."""
        for name in METALS:
            arr = getattr(self, name)
            if arr.shape != expected_shape:
                raise ValueError(
                    f"WaterConcentrations.{name}: expected shape {expected_shape}, "
                    f"got {arr.shape}"
                )

    def as_array(self) -> np.ndarray:
        """Return a (N_METALS, nx, ny) view ordered by METALS."""
        return np.stack([getattr(self, m) for m in METALS], axis=0)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "WaterConcentrations":
        """Construct from a (N_METALS, nx, ny) array ordered by METALS.

        Parameters
        ----------
        arr:
            Array with first axis length == N_METALS (4).
        """
        if arr.shape[0] != N_METALS:
            raise ValueError(
                f"Expected first dimension {N_METALS} (one per metal), got {arr.shape[0]}"
            )
        return cls(**{m: arr[i] for i, m in enumerate(METALS)})

    @classmethod
    def zeros(cls, shape: tuple[int, int]) -> "WaterConcentrations":
        """Create a zero-concentration field on a grid of *shape* = (nx, ny)."""
        return cls(**{m: np.zeros(shape, dtype=np.float64) for m in METALS})

    def copy(self) -> "WaterConcentrations":
        """Return an independent deep copy."""
        return WaterConcentrations(**{m: getattr(self, m).copy() for m in METALS})

    def total(self) -> np.ndarray:
        """Sum concentration across all metals, shape (nx, ny)."""
        return self.as_array().sum(axis=0)


@dataclass
class SedimentConcentrations:
    """Heavy-metal concentrations sorbed to bed sediment.

    Each field holds the metal content per unit mass of dry sediment.
    Units: µg kg⁻¹.

    Parameters
    ----------
    Pb, Cu, Cd, Cr:
        (nx, ny) float64 arrays.
    """

    Pb: np.ndarray   # µg kg⁻¹
    Cu: np.ndarray
    Cd: np.ndarray
    Cr: np.ndarray

    def validate_shapes(self, expected_shape: tuple[int, int]) -> None:
        """Raise ValueError if any array does not match *expected_shape*."""
        for name in METALS:
            arr = getattr(self, name)
            if arr.shape != expected_shape:
                raise ValueError(
                    f"SedimentConcentrations.{name}: expected shape {expected_shape}, "
                    f"got {arr.shape}"
                )

    def as_array(self) -> np.ndarray:
        """Return a (N_METALS, nx, ny) view ordered by METALS."""
        return np.stack([getattr(self, m) for m in METALS], axis=0)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SedimentConcentrations":
        """Construct from a (N_METALS, nx, ny) array ordered by METALS."""
        if arr.shape[0] != N_METALS:
            raise ValueError(
                f"Expected first dimension {N_METALS} (one per metal), got {arr.shape[0]}"
            )
        return cls(**{m: arr[i] for i, m in enumerate(METALS)})

    @classmethod
    def zeros(cls, shape: tuple[int, int]) -> "SedimentConcentrations":
        """Create a zero-content field on a grid of *shape* = (nx, ny)."""
        return cls(**{m: np.zeros(shape, dtype=np.float64) for m in METALS})

    def copy(self) -> "SedimentConcentrations":
        """Return an independent deep copy."""
        return SedimentConcentrations(**{m: getattr(self, m).copy() for m in METALS})


# ---------------------------------------------------------------------------
# Velocity and depth
# ---------------------------------------------------------------------------

@dataclass
class VelocityField:
    """Depth-averaged flow field.

    Parameters
    ----------
    u:
        x-component of depth-averaged velocity (m s⁻¹), shape (nx, ny).
    v:
        y-component of depth-averaged velocity (m s⁻¹), shape (nx, ny).
    depth:
        Water depth (m), shape (nx, ny).  Zero indicates dry / land cells.
    """

    u: np.ndarray      # m s⁻¹
    v: np.ndarray      # m s⁻¹
    depth: np.ndarray  # m

    def __post_init__(self) -> None:
        if not (self.u.shape == self.v.shape == self.depth.shape):
            raise ValueError(
                f"u, v, depth must share the same shape; got "
                f"u={self.u.shape}, v={self.v.shape}, depth={self.depth.shape}"
            )

    def validate_shapes(self, expected_shape: tuple[int, int]) -> None:
        """Raise ValueError if any array does not match *expected_shape*."""
        for name, arr in (("u", self.u), ("v", self.v), ("depth", self.depth)):
            if arr.shape != expected_shape:
                raise ValueError(
                    f"VelocityField.{name}: expected shape {expected_shape}, "
                    f"got {arr.shape}"
                )

    @property
    def speed(self) -> np.ndarray:
        """Scalar flow speed (m s⁻¹), shape (nx, ny)."""
        return np.sqrt(self.u**2 + self.v**2)

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial shape (nx, ny)."""
        return self.u.shape

    @classmethod
    def zeros(cls, shape: tuple[int, int], depth: float = 2.0) -> "VelocityField":
        """Quiescent flow with uniform *depth* (m).

        Parameters
        ----------
        shape:
            (nx, ny) grid shape.
        depth:
            Uniform initial water depth (m).
        """
        return cls(
            u=np.zeros(shape, dtype=np.float64),
            v=np.zeros(shape, dtype=np.float64),
            depth=np.full(shape, depth, dtype=np.float64),
        )

    @classmethod
    def uniform(
        cls,
        shape: tuple[int, int],
        u: float,
        v: float,
        depth: float,
    ) -> "VelocityField":
        """Spatially uniform flow with constant depth.

        Parameters
        ----------
        shape:
            (nx, ny) grid shape.
        u, v:
            Uniform velocity components (m s⁻¹).
        depth:
            Uniform water depth (m).
        """
        return cls(
            u=np.full(shape, u, dtype=np.float64),
            v=np.full(shape, v, dtype=np.float64),
            depth=np.full(shape, depth, dtype=np.float64),
        )

    def copy(self) -> "VelocityField":
        """Return an independent deep copy."""
        return VelocityField(
            u=self.u.copy(),
            v=self.v.copy(),
            depth=self.depth.copy(),
        )


# ---------------------------------------------------------------------------
# Remediation mask
# ---------------------------------------------------------------------------

@dataclass
class RemediationMask:
    """Identifies cells where active remediation treatment is applied.

    Parameters
    ----------
    active:
        Boolean (nx, ny) array; True marks a cell receiving treatment.
    intensity:
        Float (nx, ny) array in [0, 1] giving fractional treatment intensity
        for each cell.  Zero in cells where *active* is False.
    """

    active: np.ndarray    # bool, (nx, ny)
    intensity: np.ndarray # float64, (nx, ny), values in [0, 1]

    def __post_init__(self) -> None:
        if self.active.shape != self.intensity.shape:
            raise ValueError(
                f"active and intensity must share shape; got "
                f"active={self.active.shape}, intensity={self.intensity.shape}"
            )
        if self.intensity.min() < 0.0 or self.intensity.max() > 1.0:
            raise ValueError(
                "intensity values must lie in [0, 1]; "
                f"got range [{self.intensity.min():.4f}, {self.intensity.max():.4f}]"
            )

    def validate_shapes(self, expected_shape: tuple[int, int]) -> None:
        """Raise ValueError if arrays do not match *expected_shape*."""
        if self.active.shape != expected_shape:
            raise ValueError(
                f"RemediationMask: expected shape {expected_shape}, "
                f"got {self.active.shape}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial shape (nx, ny)."""
        return self.active.shape

    @property
    def n_active_cells(self) -> int:
        """Number of cells currently receiving treatment."""
        return int(self.active.sum())

    @classmethod
    def inactive(cls, shape: tuple[int, int]) -> "RemediationMask":
        """No remediation anywhere — all cells inactive, zero intensity."""
        return cls(
            active=np.zeros(shape, dtype=bool),
            intensity=np.zeros(shape, dtype=np.float64),
        )

    @classmethod
    def from_indices(
        cls,
        shape: tuple[int, int],
        i_cells: np.ndarray,
        j_cells: np.ndarray,
        intensity: float | np.ndarray = 1.0,
    ) -> "RemediationMask":
        """Activate specific cells by index.

        Parameters
        ----------
        shape:
            (nx, ny) grid shape.
        i_cells, j_cells:
            Integer arrays of i/j indices for cells to activate.
        intensity:
            Scalar or per-cell array of treatment intensities in [0, 1].
        """
        active = np.zeros(shape, dtype=bool)
        inten = np.zeros(shape, dtype=np.float64)
        active[i_cells, j_cells] = True
        inten[i_cells, j_cells] = intensity
        return cls(active=active, intensity=inten)

    def copy(self) -> "RemediationMask":
        """Return an independent deep copy."""
        return RemediationMask(
            active=self.active.copy(),
            intensity=self.intensity.copy(),
        )


# ---------------------------------------------------------------------------
# Top-level simulation state
# ---------------------------------------------------------------------------

@dataclass
class SimulationState:
    """Complete model state at a single point in time.

    All spatial arrays are consistent with ``grid.shape``.

    Parameters
    ----------
    grid:
        Grid dimensions (does not change during a run).
    water:
        Dissolved metal concentrations (µg L⁻¹).
    sediment:
        Sediment-bound metal concentrations (µg kg⁻¹).
    velocity:
        Depth-averaged velocity and water depth.
    remediation:
        Treatment mask and per-cell intensity.
    time:
        Simulation time (s since the start of the run).
    step:
        Integer time-step counter (0-based).
    """

    grid: GridDimensions
    water: WaterConcentrations
    sediment: SedimentConcentrations
    velocity: VelocityField
    remediation: RemediationMask
    time: float = 0.0
    step: int = 0

    def __post_init__(self) -> None:
        shape = self.grid.shape
        self.water.validate_shapes(shape)
        self.sediment.validate_shapes(shape)
        self.velocity.validate_shapes(shape)
        self.remediation.validate_shapes(shape)

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial grid shape (nx, ny)."""
        return self.grid.shape

    def copy(self) -> "SimulationState":
        """Return an independent deep copy of the entire state."""
        return SimulationState(
            grid=self.grid,                    # GridDimensions is immutable — shared ref is fine
            water=self.water.copy(),
            sediment=self.sediment.copy(),
            velocity=self.velocity.copy(),
            remediation=self.remediation.copy(),
            time=self.time,
            step=self.step,
        )

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary suitable for HDF5 / zarr storage.

        Multi-metal arrays are stored as (N_METALS, nx, ny) float64 arrays
        under the keys ``"water"`` and ``"sediment"``.
        """
        return {
            "water": self.water.as_array(),
            "sediment": self.sediment.as_array(),
            "u": self.velocity.u,
            "v": self.velocity.v,
            "depth": self.velocity.depth,
            "remediation_active": self.remediation.active.astype(np.uint8),
            "remediation_intensity": self.remediation.intensity,
            "time": np.float64(self.time),
            "step": np.int64(self.step),
            # grid metadata as scalar attributes
            "nx": self.grid.nx,
            "ny": self.grid.ny,
            "dx": self.grid.dx,
            "dy": self.grid.dy,
            "x_origin": self.grid.x_origin,
            "y_origin": self.grid.y_origin,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationState":
        """Reconstruct from a dictionary produced by :meth:`to_dict`."""
        grid = GridDimensions(
            nx=int(d["nx"]),
            ny=int(d["ny"]),
            dx=float(d["dx"]),
            dy=float(d["dy"]),
            x_origin=float(d["x_origin"]),
            y_origin=float(d["y_origin"]),
        )
        return cls(
            grid=grid,
            water=WaterConcentrations.from_array(np.asarray(d["water"])),
            sediment=SedimentConcentrations.from_array(np.asarray(d["sediment"])),
            velocity=VelocityField(
                u=np.asarray(d["u"]),
                v=np.asarray(d["v"]),
                depth=np.asarray(d["depth"]),
            ),
            remediation=RemediationMask(
                active=np.asarray(d["remediation_active"]).astype(bool),
                intensity=np.asarray(d["remediation_intensity"]),
            ),
            time=float(d["time"]),
            step=int(d["step"]),
        )


# ---------------------------------------------------------------------------
# Helper constructor
# ---------------------------------------------------------------------------

def create_empty_state(
    grid: GridDimensions,
    initial_depth: float = 2.0,
    initial_water: dict[MetalName, float] | None = None,
    initial_sediment: dict[MetalName, float] | None = None,
) -> SimulationState:
    """Build a zeroed-out :class:`SimulationState` on *grid*.

    Parameters
    ----------
    grid:
        Target grid dimensions.
    initial_depth:
        Uniform starting water depth (m).  Default: 2.0 m (typical for
        Baiyangdian Lake).
    initial_water:
        Optional mapping of metal name → uniform background concentration
        (µg L⁻¹).  Metals not listed default to 0.
    initial_sediment:
        Optional mapping of metal name → uniform background sediment
        concentration (µg kg⁻¹).  Metals not listed default to 0.

    Returns
    -------
    SimulationState
        Fresh state with quiescent flow and no active remediation.
    """
    shape = grid.shape
    water_arrays: dict[str, np.ndarray] = {}
    sediment_arrays: dict[str, np.ndarray] = {}

    for metal in METALS:
        w_bg = (initial_water or {}).get(metal, 0.0)
        s_bg = (initial_sediment or {}).get(metal, 0.0)
        water_arrays[metal] = np.full(shape, w_bg, dtype=np.float64)
        sediment_arrays[metal] = np.full(shape, s_bg, dtype=np.float64)

    return SimulationState(
        grid=grid,
        water=WaterConcentrations(**water_arrays),
        sediment=SedimentConcentrations(**sediment_arrays),
        velocity=VelocityField.zeros(shape, depth=initial_depth),
        remediation=RemediationMask.inactive(shape),
        time=0.0,
        step=0,
    )


# ---------------------------------------------------------------------------
# State history
# ---------------------------------------------------------------------------

@dataclass
class StateHistory:
    """Ordered collection of :class:`SimulationState` snapshots.

    Snapshots are deep-copied on append so that the history is immutable
    with respect to subsequent solver steps.
    """

    snapshots: list[SimulationState] = field(default_factory=list)

    def append(self, state: SimulationState) -> None:
        """Deep-copy *state* and add it to the history."""
        self.snapshots.append(state.copy())

    def __len__(self) -> int:
        return len(self.snapshots)

    def times(self) -> np.ndarray:
        """1-D float64 array of snapshot times (s), shape (T,)."""
        return np.array([s.time for s in self.snapshots], dtype=np.float64)

    def steps(self) -> np.ndarray:
        """1-D int array of snapshot step indices, shape (T,)."""
        return np.array([s.step for s in self.snapshots], dtype=np.int64)

    def water_stack(self, metal: MetalName) -> np.ndarray:
        """Concentration time series for one metal in the water column.

        Parameters
        ----------
        metal:
            One of the METALS names: ``"Pb"``, ``"Cu"``, ``"Cd"``, ``"Cr"``.

        Returns
        -------
        np.ndarray
            Shape (T, nx, ny).
        """
        if metal not in METAL_INDEX:
            raise ValueError(f"Unknown metal {metal!r}. Choose from {METALS}.")
        return np.stack([getattr(s.water, metal) for s in self.snapshots], axis=0)

    def sediment_stack(self, metal: MetalName) -> np.ndarray:
        """Sediment concentration time series for one metal.

        Returns
        -------
        np.ndarray
            Shape (T, nx, ny).
        """
        if metal not in METAL_INDEX:
            raise ValueError(f"Unknown metal {metal!r}. Choose from {METALS}.")
        return np.stack([getattr(s.sediment, metal) for s in self.snapshots], axis=0)

    def all_water_stack(self) -> np.ndarray:
        """Full water concentration history.

        Returns
        -------
        np.ndarray
            Shape (T, N_METALS, nx, ny).
        """
        return np.stack([s.water.as_array() for s in self.snapshots], axis=0)

    def save(self, path: str | Path) -> None:
        """Persist snapshot history to an HDF5 file via h5py.

        Layout: one group per snapshot named ``step_XXXXXXXX``, each
        containing the datasets from :meth:`SimulationState.to_dict`.
        """
        import h5py

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["n_snapshots"] = len(self.snapshots)
            f.attrs["metals"] = list(METALS)
            for snap in self.snapshots:
                grp = f.create_group(f"step_{snap.step:08d}")
                for key, val in snap.to_dict().items():
                    if isinstance(val, np.ndarray):
                        grp.create_dataset(key, data=val, compression="gzip")
                    else:
                        grp.attrs[key] = val

    @classmethod
    def load(cls, path: str | Path) -> "StateHistory":
        """Load a history previously written by :meth:`save`.

        Parameters
        ----------
        path:
            Path to the ``.h5`` file.

        Returns
        -------
        StateHistory
        """
        import h5py

        history = cls()
        with h5py.File(path, "r") as f:
            group_names = sorted(f.keys())
            for name in group_names:
                grp = f[name]
                d = {k: grp[k][()] for k in grp.keys()}
                d.update({k: grp.attrs[k] for k in grp.attrs.keys()})
                history.snapshots.append(SimulationState.from_dict(d))
        return history

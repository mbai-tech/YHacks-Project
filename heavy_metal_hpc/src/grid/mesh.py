"""Structured 2-D rectangular grid for the heavy_metal_hpc transport model.

Design philosophy
-----------------
* ``GridSpec`` is a **plain value object** — it holds only the numbers that
  define the grid's geometry (cell count, cell size, origin).  No arrays live
  inside it, so it is cheap to copy, hash, and serialise.

* Free functions (``x_centers``, ``y_centers``, ``meshgrid``, ``cell_edges``)
  derive coordinate arrays on demand, keeping ``GridSpec`` lightweight.

* ``BoundaryRegion`` records a named rectangular patch of grid cells that
  represents an inflow channel, outflow point, or any other zone that needs
  special treatment at the boundary.

* ``Grid`` is the operational object that a solver actually uses: it bundles
  a ``GridSpec``, the derived coordinate arrays, and a collection of named
  ``BoundaryRegion``s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# GridSpec — lightweight value object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridSpec:
    """Defines the geometry of a uniform 2-D Cartesian grid.

    The grid covers the rectangular domain::

        x in [x_origin,  x_origin + nx * dx]
        y in [y_origin,  y_origin + ny * dy]

    with ``nx * ny`` square-ish cells of size ``dx`` × ``dy``.

    Parameters
    ----------
    nx, ny:
        Number of cells along the x (east) and y (north) axes.
    dx, dy:
        Cell width and height (m).  Both must be strictly positive.
    x_origin, y_origin:
        Coordinates of the **south-west corner** of cell (0, 0).
        Defaults to (0, 0) for synthetic / test grids.
    """

    nx: int
    ny: int
    dx: float
    dy: float
    x_origin: float = 0.0
    y_origin: float = 0.0

    # frozen dataclasses run __post_init__ via object.__setattr__
    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError(
                f"Cell counts must be positive integers, got nx={self.nx}, ny={self.ny}"
            )
        if self.dx <= 0.0 or self.dy <= 0.0:
            raise ValueError(
                f"Cell sizes must be strictly positive, got dx={self.dx}, dy={self.dy}"
            )

    # ------------------------------------------------------------------
    # Derived geometry (computed, not stored)
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial array shape ``(nx, ny)`` used throughout the solver."""
        return (self.nx, self.ny)

    @property
    def n_cells(self) -> int:
        """Total number of grid cells."""
        return self.nx * self.ny

    @property
    def cell_area(self) -> float:
        """Area of one cell (m²)."""
        return self.dx * self.dy

    @property
    def x_extent(self) -> float:
        """Total domain width in x (m)."""
        return self.nx * self.dx

    @property
    def y_extent(self) -> float:
        """Total domain height in y (m)."""
        return self.ny * self.dy

    @property
    def x_max(self) -> float:
        """East boundary coordinate (m)."""
        return self.x_origin + self.x_extent

    @property
    def y_max(self) -> float:
        """North boundary coordinate (m)."""
        return self.y_origin + self.y_extent

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for YAML / JSON serialisation."""
        return {
            "nx": self.nx,
            "ny": self.ny,
            "dx": self.dx,
            "dy": self.dy,
            "x_origin": self.x_origin,
            "y_origin": self.y_origin,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GridSpec":
        """Construct from a dict produced by :meth:`to_dict`."""
        return cls(
            nx=int(d["nx"]),
            ny=int(d["ny"]),
            dx=float(d["dx"]),
            dy=float(d["dy"]),
            x_origin=float(d.get("x_origin", 0.0)),
            y_origin=float(d.get("y_origin", 0.0)),
        )


# ---------------------------------------------------------------------------
# Coordinate array constructors
# ---------------------------------------------------------------------------

def x_centers(spec: GridSpec) -> np.ndarray:
    """1-D array of cell-centre x coordinates (m), shape ``(nx,)``.

    Cell centre ``i`` sits at ``x_origin + (i + 0.5) * dx``.
    """
    return spec.x_origin + (np.arange(spec.nx, dtype=np.float64) + 0.5) * spec.dx


def y_centers(spec: GridSpec) -> np.ndarray:
    """1-D array of cell-centre y coordinates (m), shape ``(ny,)``.

    Cell centre ``j`` sits at ``y_origin + (j + 0.5) * dy``.
    """
    return spec.y_origin + (np.arange(spec.ny, dtype=np.float64) + 0.5) * spec.dy


def x_edges(spec: GridSpec) -> np.ndarray:
    """1-D array of cell-edge x coordinates (m), shape ``(nx + 1,)``.

    Includes both the west boundary (``x_origin``) and the east boundary
    (``x_origin + nx * dx``).
    """
    return spec.x_origin + np.arange(spec.nx + 1, dtype=np.float64) * spec.dx


def y_edges(spec: GridSpec) -> np.ndarray:
    """1-D array of cell-edge y coordinates (m), shape ``(ny + 1,)``.

    Includes both the south boundary (``y_origin``) and the north boundary.
    """
    return spec.y_origin + np.arange(spec.ny + 1, dtype=np.float64) * spec.dy


def meshgrid(spec: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """2-D cell-centre coordinate arrays, each of shape ``(nx, ny)``.

    Returns
    -------
    X, Y : np.ndarray
        ``X[i, j]`` is the x coordinate of cell (i, j);
        ``Y[i, j]`` is the y coordinate of cell (i, j).

    Uses ``indexing="ij"`` so axis 0 is x and axis 1 is y — consistent with
    all spatial arrays in the model.
    """
    return np.meshgrid(x_centers(spec), y_centers(spec), indexing="ij")


def cell_distances(spec: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Distance between neighbouring cell centres in each direction.

    Returns
    -------
    dist_x, dist_y : np.ndarray
        Both are scalar-valued floats wrapped in 0-D arrays so they can be
        broadcast against spatial fields:  ``dx_arr * field`` works cleanly.
        For a uniform grid these are just ``spec.dx`` and ``spec.dy``.
    """
    return np.float64(spec.dx), np.float64(spec.dy)


# ---------------------------------------------------------------------------
# Boundary region
# ---------------------------------------------------------------------------

@dataclass
class BoundaryRegion:
    """A named rectangular patch of cells on or near the grid boundary.

    Used to represent inflow channels, outflow sections, and other zones
    that require special treatment in the solver (e.g. prescribed
    concentration or flux boundary conditions).

    Parameters
    ----------
    name:
        Human-readable label, e.g. ``"FuRiver_N"``.
    i_start, i_end:
        Inclusive range of i (x-axis) cell indices.
    j_start, j_end:
        Inclusive range of j (y-axis) cell indices.
    kind:
        Semantic type of the region.  Recognised values:
        ``"inflow"``, ``"outflow"``, ``"remediation"``, ``"monitoring"``.
    metadata:
        Optional free-form dictionary for extra attributes (e.g. expected
        discharge, metal concentrations) that the solver or post-processor
        may use.
    """

    name: str
    i_start: int
    i_end: int
    j_start: int
    j_end: int
    kind: str = "inflow"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.i_start > self.i_end:
            raise ValueError(
                f"i_start ({self.i_start}) must be <= i_end ({self.i_end})"
            )
        if self.j_start > self.j_end:
            raise ValueError(
                f"j_start ({self.j_start}) must be <= j_end ({self.j_end})"
            )
        recognised = {"inflow", "outflow", "remediation", "monitoring"}
        if self.kind not in recognised:
            raise ValueError(
                f"kind={self.kind!r} is not recognised.  Choose from {recognised}."
            )

    @property
    def n_cells(self) -> int:
        """Number of cells covered by this region."""
        return (self.i_end - self.i_start + 1) * (self.j_end - self.j_start + 1)

    def as_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Boolean ``(nx, ny)`` mask that is ``True`` inside this region.

        Parameters
        ----------
        shape:
            ``(nx, ny)`` of the target grid.

        Returns
        -------
        np.ndarray
            Boolean array with ``True`` at every cell inside the region.
        """
        mask = np.zeros(shape, dtype=bool)
        mask[self.i_start : self.i_end + 1, self.j_start : self.j_end + 1] = True
        return mask

    def cell_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(i_indices, j_indices)`` arrays for all cells in the region.

        Suitable for fancy-indexing into any ``(nx, ny)`` array.
        """
        ii = np.arange(self.i_start, self.i_end + 1)
        jj = np.arange(self.j_start, self.j_end + 1)
        return np.repeat(ii, len(jj)), np.tile(jj, len(ii))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "name": self.name,
            "i_start": self.i_start,
            "i_end": self.i_end,
            "j_start": self.j_start,
            "j_end": self.j_end,
            "kind": self.kind,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BoundaryRegion":
        """Construct from a dict produced by :meth:`to_dict`."""
        return cls(
            name=d["name"],
            i_start=int(d["i_start"]),
            i_end=int(d["i_end"]),
            j_start=int(d["j_start"]),
            j_end=int(d["j_end"]),
            kind=d.get("kind", "inflow"),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Grid — operational object combining spec + coordinates + regions
# ---------------------------------------------------------------------------

@dataclass
class Grid:
    """Operational grid object used by the solver and I/O layers.

    Bundles a :class:`GridSpec`, the derived coordinate arrays, and a
    registry of named :class:`BoundaryRegion`s.

    Parameters
    ----------
    spec:
        Immutable grid geometry.
    regions:
        Named boundary / zone regions (inflow, outflow, remediation …).
        Duplicate names are not allowed.
    """

    spec: GridSpec
    regions: list[BoundaryRegion] = field(default_factory=list)

    # Coordinate arrays — built once and stored
    _X: np.ndarray = field(init=False, repr=False)
    _Y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._X, self._Y = meshgrid(self.spec)
        self._validate_regions()

    def _validate_regions(self) -> None:
        """Raise if any region name is duplicated or indices are out of range."""
        names: list[str] = []
        nx, ny = self.spec.shape
        for r in self.regions:
            if r.name in names:
                raise ValueError(f"Duplicate boundary region name: {r.name!r}")
            names.append(r.name)
            if r.i_start < 0 or r.i_end >= nx:
                raise ValueError(
                    f"Region {r.name!r}: i indices [{r.i_start}, {r.i_end}] out of "
                    f"range [0, {nx - 1}]"
                )
            if r.j_start < 0 or r.j_end >= ny:
                raise ValueError(
                    f"Region {r.name!r}: j indices [{r.j_start}, {r.j_end}] out of "
                    f"range [0, {ny - 1}]"
                )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial shape ``(nx, ny)``."""
        return self.spec.shape

    @property
    def X(self) -> np.ndarray:
        """2-D x cell-centre array, shape ``(nx, ny)``."""
        return self._X

    @property
    def Y(self) -> np.ndarray:
        """2-D y cell-centre array, shape ``(nx, ny)``."""
        return self._Y

    def region(self, name: str) -> BoundaryRegion:
        """Look up a :class:`BoundaryRegion` by name.

        Raises
        ------
        KeyError
            If no region with that name exists.
        """
        for r in self.regions:
            if r.name == name:
                return r
        raise KeyError(f"No boundary region named {name!r}")

    def regions_of_kind(self, kind: str) -> list[BoundaryRegion]:
        """Return all regions whose :attr:`~BoundaryRegion.kind` equals *kind*."""
        return [r for r in self.regions if r.kind == kind]

    def region_mask(self, name: str) -> np.ndarray:
        """Boolean ``(nx, ny)`` mask for the named region."""
        return self.region(name).as_mask(self.shape)

    def combined_mask(self, kind: str) -> np.ndarray:
        """Boolean ``(nx, ny)`` mask covering **all** regions of *kind*.

        Useful for applying the same boundary condition to every inflow or
        every remediation zone at once.
        """
        mask = np.zeros(self.shape, dtype=bool)
        for r in self.regions_of_kind(kind):
            mask |= r.as_mask(self.shape)
        return mask

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (YAML / JSON compatible)."""
        return {
            "spec": self.spec.to_dict(),
            "regions": [r.to_dict() for r in self.regions],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Grid":
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        return cls(
            spec=GridSpec.from_dict(d["spec"]),
            regions=[BoundaryRegion.from_dict(r) for r in d.get("regions", [])],
        )

    def save(self, path: str | Path) -> None:
        """Save the grid spec and regions to a NumPy ``.npz`` archive.

        The archive contains:
        - scalar grid spec values as 0-D arrays
        - one boolean mask per region, named ``mask_<region_name>``
        - region metadata as a JSON blob under ``regions_json``
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {
            "nx": np.int64(self.spec.nx),
            "ny": np.int64(self.spec.ny),
            "dx": np.float64(self.spec.dx),
            "dy": np.float64(self.spec.dy),
            "x_origin": np.float64(self.spec.x_origin),
            "y_origin": np.float64(self.spec.y_origin),
        }
        for r in self.regions:
            arrays[f"mask_{r.name}"] = r.as_mask(self.shape)

        # Encode region metadata separately so .npz stays array-only
        region_json = json.dumps([r.to_dict() for r in self.regions])
        arrays["regions_json"] = np.bytes_(region_json.encode())

        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Grid":
        """Load a :class:`Grid` previously saved with :meth:`save`.

        Parameters
        ----------
        path:
            Path to the ``.npz`` file (the ``.npz`` extension may be omitted).
        """
        import json

        data = np.load(Path(path), allow_pickle=False)
        spec = GridSpec(
            nx=int(data["nx"]),
            ny=int(data["ny"]),
            dx=float(data["dx"]),
            dy=float(data["dy"]),
            x_origin=float(data["x_origin"]),
            y_origin=float(data["y_origin"]),
        )
        regions_raw = json.loads(data["regions_json"].item().decode())
        regions = [BoundaryRegion.from_dict(r) for r in regions_raw]
        return cls(spec=spec, regions=regions)

"""Synthetic geometry helpers for the Baiyangdian Lake model domain.

This module provides:

1. **Mask constructors** — functions that build boolean ``(nx, ny)`` arrays
   marking water cells, land cells, and remediation zones.  All masks use
   the convention ``True`` = the feature is *present* in that cell.

2. **Baiyangdian placeholder geometry** — a realistic but entirely synthetic
   representation of the lake basin with three named inflow channels (Fu
   River from the north, Tang River from the west, and a southern drainage
   canal) and one outflow section.  It is designed to be runnable immediately
   and swapped out for a GIS-derived geometry later without touching any
   other module.

3. **Mask I/O helpers** — ``save_masks`` / ``load_masks`` for persisting the
   mask set to a compressed NumPy archive.

Coordinate convention
---------------------
All spatial arrays use shape ``(nx, ny)`` with axis 0 along x (east) and
axis 1 along y (north), consistent with :mod:`src.grid.mesh` and the
:mod:`src.model.state` arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .mesh import Grid, GridSpec, BoundaryRegion


# ---------------------------------------------------------------------------
# Mask constructors
# ---------------------------------------------------------------------------

def water_mask_from_depth(
    depth: np.ndarray,
    min_depth: float = 0.1,
) -> np.ndarray:
    """Mark cells with water depth >= *min_depth* as open water.

    Parameters
    ----------
    depth:
        ``(nx, ny)`` float array of water depths (m).
    min_depth:
        Cells shallower than this threshold are treated as land / dry (m).

    Returns
    -------
    np.ndarray
        Boolean ``(nx, ny)`` array; ``True`` means open water.
    """
    return depth >= min_depth


def land_mask_from_depth(
    depth: np.ndarray,
    min_depth: float = 0.1,
) -> np.ndarray:
    """Complement of :func:`water_mask_from_depth`.

    Returns
    -------
    np.ndarray
        Boolean ``(nx, ny)`` array; ``True`` means land / dry cell.
    """
    return depth < min_depth


def rectangular_mask(
    shape: tuple[int, int],
    i_start: int,
    i_end: int,
    j_start: int,
    j_end: int,
) -> np.ndarray:
    """Boolean mask for an axis-aligned rectangular patch of cells.

    Parameters
    ----------
    shape:
        ``(nx, ny)`` of the full grid.
    i_start, i_end:
        Inclusive i (x-axis) index range.
    j_start, j_end:
        Inclusive j (y-axis) index range.

    Returns
    -------
    np.ndarray
        Boolean ``(nx, ny)`` array.
    """
    mask = np.zeros(shape, dtype=bool)
    mask[i_start : i_end + 1, j_start : j_end + 1] = True
    return mask


def elliptical_mask(
    shape: tuple[int, int],
    cx: float,
    cy: float,
    rx: float,
    ry: float,
) -> np.ndarray:
    """Boolean mask for an elliptical region in fractional grid coordinates.

    Parameters
    ----------
    shape:
        ``(nx, ny)`` of the full grid.
    cx, cy:
        Centre of the ellipse in fractional coordinates (0–1 across the grid).
    rx, ry:
        Semi-axis lengths in fractional coordinates.

    Returns
    -------
    np.ndarray
        Boolean ``(nx, ny)`` array; ``True`` inside or on the ellipse.

    Examples
    --------
    A circle covering the central 60 % of the domain::

        mask = elliptical_mask((100, 80), cx=0.5, cy=0.5, rx=0.3, ry=0.3)
    """
    nx, ny = shape
    # Normalised cell-centre coordinates in [0, 1]
    xi = (np.arange(nx) + 0.5) / nx   # shape (nx,)
    yi = (np.arange(ny) + 0.5) / ny   # shape (ny,)
    XI, YI = np.meshgrid(xi, yi, indexing="ij")
    return ((XI - cx) / rx) ** 2 + ((YI - cy) / ry) ** 2 <= 1.0


def combine_masks(*masks: np.ndarray, mode: str = "union") -> np.ndarray:
    """Combine two or more boolean masks element-wise.

    Parameters
    ----------
    *masks:
        Two or more boolean arrays of identical shape.
    mode:
        ``"union"``  — logical OR  (cell is marked if *any* mask flags it).
        ``"intersection"`` — logical AND (cell is marked only if *all* masks flag it).

    Returns
    -------
    np.ndarray
        Combined boolean array.
    """
    if not masks:
        raise ValueError("At least one mask is required.")
    result = masks[0].copy()
    for m in masks[1:]:
        if mode == "union":
            result = result | m
        elif mode == "intersection":
            result = result & m
        else:
            raise ValueError(f"Unknown mode {mode!r}.  Use 'union' or 'intersection'.")
    return result


# ---------------------------------------------------------------------------
# MaskSet — collected masks for one geometry
# ---------------------------------------------------------------------------

@dataclass
class MaskSet:
    """All boolean masks that describe the lake geometry.

    Attributes
    ----------
    water:
        ``True`` in open-water cells.
    land:
        ``True`` in land / dry cells (complement of *water*).
    inflow:
        ``True`` in cells belonging to any inflow boundary region.
    outflow:
        ``True`` in cells belonging to any outflow boundary region.
    remediation_zone:
        ``True`` in cells designated as candidate remediation sites.
    shape:
        ``(nx, ny)`` grid shape.
    """

    water: np.ndarray
    land: np.ndarray
    inflow: np.ndarray
    outflow: np.ndarray
    remediation_zone: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            "water": self.water.shape,
            "land": self.land.shape,
            "inflow": self.inflow.shape,
            "outflow": self.outflow.shape,
            "remediation_zone": self.remediation_zone.shape,
        }
        unique_shapes = set(shapes.values())
        if len(unique_shapes) != 1:
            raise ValueError(
                f"All masks must share the same shape.  Got: {shapes}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape ``(nx, ny)``."""
        return self.water.shape

    @property
    def n_water_cells(self) -> int:
        """Number of open-water cells."""
        return int(self.water.sum())

    @property
    def n_land_cells(self) -> int:
        """Number of land / dry cells."""
        return int(self.land.sum())

    @property
    def n_remediation_cells(self) -> int:
        """Number of candidate remediation cells."""
        return int(self.remediation_zone.sum())

    def summary(self) -> dict[str, Any]:
        """Return a plain dict of cell-count statistics."""
        nx, ny = self.shape
        return {
            "shape": self.shape,
            "total_cells": nx * ny,
            "water_cells": self.n_water_cells,
            "land_cells": self.n_land_cells,
            "inflow_cells": int(self.inflow.sum()),
            "outflow_cells": int(self.outflow.sum()),
            "remediation_cells": self.n_remediation_cells,
            "water_fraction": round(self.n_water_cells / (nx * ny), 4),
        }


# ---------------------------------------------------------------------------
# Mask I/O
# ---------------------------------------------------------------------------

_MASK_KEYS = ("water", "land", "inflow", "outflow", "remediation_zone")


def save_masks(masks: MaskSet, path: str | Path) -> None:
    """Persist a :class:`MaskSet` to a compressed NumPy ``.npz`` archive.

    Parameters
    ----------
    masks:
        The mask set to save.
    path:
        Output file path.  The ``.npz`` extension is added automatically if
        absent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        water=masks.water,
        land=masks.land,
        inflow=masks.inflow,
        outflow=masks.outflow,
        remediation_zone=masks.remediation_zone,
    )


def load_masks(path: str | Path) -> MaskSet:
    """Load a :class:`MaskSet` from an archive written by :func:`save_masks`.

    Parameters
    ----------
    path:
        Path to the ``.npz`` file (the extension may be omitted).

    Returns
    -------
    MaskSet
    """
    data = np.load(Path(path), allow_pickle=False)
    return MaskSet(
        water=data["water"].astype(bool),
        land=data["land"].astype(bool),
        inflow=data["inflow"].astype(bool),
        outflow=data["outflow"].astype(bool),
        remediation_zone=data["remediation_zone"].astype(bool),
    )


# ---------------------------------------------------------------------------
# Baiyangdian placeholder geometry
# ---------------------------------------------------------------------------

#: Default grid specification for the synthetic Baiyangdian domain.
#:
#: The real lake covers roughly 366 km².  This synthetic grid uses
#: 100 × 80 cells of 500 m × 500 m ≈ 2 500 m² each, giving a domain
#: of 50 km × 40 km.
BAIYANGDIAN_SPEC = GridSpec(
    nx=100,
    ny=80,
    dx=500.0,   # m per cell
    dy=500.0,   # m per cell
    x_origin=0.0,
    y_origin=0.0,
)

#: Named inflow / outflow regions for the synthetic Baiyangdian geometry.
#:
#: Three inflow channels are loosely inspired by the real lake's main
#: tributaries:
#:   - Fu River  : enters from the north-west (top-left in grid coords)
#:   - Tang River: enters from the west (left edge, mid-height)
#:   - South Canal: enters from the south (bottom edge, right-of-centre)
#:
#: One outflow section sits on the east (right) edge representing the
#: Baiyangdian outflow to the Daqing River.
BAIYANGDIAN_REGIONS: list[BoundaryRegion] = [
    BoundaryRegion(
        name="FuRiver",
        i_start=10, i_end=15,   # x = 5 000–7 500 m (north-west)
        j_start=72, j_end=79,   # j at top edge (north boundary)
        kind="inflow",
        metadata={
            "description": "Fu River — primary northern inflow",
            "typical_discharge_m3s": 25.0,
        },
    ),
    BoundaryRegion(
        name="TangRiver",
        i_start=0, i_end=0,     # i at left edge (west boundary)
        j_start=35, j_end=45,   # y = 17 500–22 500 m (mid-height)
        kind="inflow",
        metadata={
            "description": "Tang River — western inflow channel",
            "typical_discharge_m3s": 10.0,
        },
    ),
    BoundaryRegion(
        name="SouthCanal",
        i_start=55, i_end=65,   # x = 27 500–32 500 m (right-of-centre)
        j_start=0, j_end=0,     # j at bottom edge (south boundary)
        kind="inflow",
        metadata={
            "description": "Southern drainage canal — intermittent inflow",
            "typical_discharge_m3s": 5.0,
        },
    ),
    BoundaryRegion(
        name="DaqingOutflow",
        i_start=99, i_end=99,   # i at right edge (east boundary)
        j_start=25, j_end=55,   # y = 12 500–27 500 m (central east)
        kind="outflow",
        metadata={
            "description": "Baiyangdian outflow to Daqing River",
        },
    ),
    BoundaryRegion(
        name="CentralReedbedRemediation",
        i_start=40, i_end=60,   # central patch
        j_start=30, j_end=50,
        kind="remediation",
        metadata={
            "description": "High-contamination reed-bed zone — priority remediation area",
        },
    ),
    BoundaryRegion(
        name="NorthMonitoring",
        i_start=8, i_end=20,
        j_start=60, j_end=79,
        kind="monitoring",
        metadata={
            "description": "Northern monitoring transect near Fu River inflow",
        },
    ),
]


def make_baiyangdian_depth(spec: GridSpec) -> np.ndarray:
    """Synthetic bathymetry for the Baiyangdian domain.

    Generates a smooth depth field (m) that captures the key features of a
    shallow lake:

    * A broad central basin (~2.5 m deep) represented by a 2-D Gaussian.
    * Shallow northern and southern margins (~0.5 m) where reed beds develop.
    * A narrow shallow sill along the western edge representing the inlet
      transition.
    * Dry land cells (depth = 0) outside the lake ellipse.

    All values are entirely synthetic.

    Parameters
    ----------
    spec:
        Grid specification.  Designed for :data:`BAIYANGDIAN_SPEC` but
        works on any ``GridSpec``.

    Returns
    -------
    np.ndarray
        ``(nx, ny)`` float64 depth array (m).  Dry cells have depth 0.
    """
    nx, ny = spec.shape
    # Normalised [0, 1] cell-centre coordinates
    xi = (np.arange(nx) + 0.5) / nx
    yi = (np.arange(ny) + 0.5) / ny
    XI, YI = np.meshgrid(xi, yi, indexing="ij")

    # Elliptical lake boundary — cells outside are dry land
    lake_mask = elliptical_mask(spec.shape, cx=0.50, cy=0.50, rx=0.45, ry=0.40)

    # Central basin: deep in the middle, shallow towards edges
    basin_depth = 2.5 * np.exp(
        -(((XI - 0.50) / 0.30) ** 2 + ((YI - 0.50) / 0.25) ** 2)
    )

    # Reed-bed shallows along the northern shore
    north_shallow = 0.6 * np.exp(-((YI - 0.90) / 0.10) ** 2)

    # Reed-bed shallows along the southern shore
    south_shallow = 0.5 * np.exp(-((YI - 0.08) / 0.10) ** 2)

    # Combine: base depth 0.4 m everywhere in the lake + Gaussian features
    depth = np.where(
        lake_mask,
        np.clip(0.4 + basin_depth - north_shallow - south_shallow, 0.0, None),
        0.0,   # land / dry
    )
    return depth.astype(np.float64)


def make_baiyangdian_grid() -> Grid:
    """Construct the default synthetic Baiyangdian :class:`~mesh.Grid`.

    Returns a :class:`~mesh.Grid` built from :data:`BAIYANGDIAN_SPEC`
    and :data:`BAIYANGDIAN_REGIONS`.  Ready to use immediately — no file
    I/O required.

    Returns
    -------
    Grid
    """
    return Grid(spec=BAIYANGDIAN_SPEC, regions=list(BAIYANGDIAN_REGIONS))


def make_baiyangdian_masks(grid: Grid | None = None) -> MaskSet:
    """Build the :class:`MaskSet` for the synthetic Baiyangdian geometry.

    Parameters
    ----------
    grid:
        A :class:`~mesh.Grid` to derive masks from.  Defaults to the output
        of :func:`make_baiyangdian_grid`.

    Returns
    -------
    MaskSet
        All five boolean masks populated from the synthetic depth field and
        the named boundary regions.
    """
    if grid is None:
        grid = make_baiyangdian_grid()

    depth = make_baiyangdian_depth(grid.spec)
    water = water_mask_from_depth(depth, min_depth=0.1)
    land = ~water

    inflow = grid.combined_mask("inflow")
    outflow = grid.combined_mask("outflow")
    remediation_zone = grid.combined_mask("remediation")

    # Inflow and outflow cells that happen to fall on dry land are kept as-is;
    # the solver is responsible for enforcing boundary conditions only on wet
    # cells.  No further filtering is applied here.

    return MaskSet(
        water=water,
        land=land,
        inflow=inflow,
        outflow=outflow,
        remediation_zone=remediation_zone,
    )

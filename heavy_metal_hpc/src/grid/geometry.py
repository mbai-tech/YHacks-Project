"""Lake boundary, bathymetry, and land-mask utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Bathymetry:
    """Water-depth field interpolated from survey data onto a model mesh.

    Parameters
    ----------
    survey_path:
        Path to a CSV / netCDF file containing (lon, lat, depth_m) columns.
    """

    def __init__(self, survey_path: str | Path) -> None:
        self.survey_path = Path(survey_path)
        self._depth: np.ndarray | None = None

    def load(self) -> "Bathymetry":
        """Read the survey file and cache the raw point cloud.  Returns self."""
        raise NotImplementedError

    def interpolate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of depth onto arbitrary (x, y) query points.

        Parameters
        ----------
        x, y:
            1-D arrays of query coordinates (same units as survey data).

        Returns
        -------
        np.ndarray
            Depth (m) at each query point; land points return 0.
        """
        raise NotImplementedError


class LandMask:
    """Boolean mask separating open-water cells from land / reed-bed cells.

    Parameters
    ----------
    shape:
        (nx, ny) grid shape.
    """

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape
        self.mask: np.ndarray = np.ones(shape, dtype=bool)  # True = water

    def from_shapefile(self, path: str | Path) -> "LandMask":
        """Rasterise a polygon shapefile onto the grid.  Returns self."""
        raise NotImplementedError

    def from_depth(self, depth: np.ndarray, min_depth: float = 0.1) -> "LandMask":
        """Mark cells with depth < *min_depth* as land.  Returns self."""
        self.mask = depth >= min_depth
        return self

    @property
    def water_fraction(self) -> float:
        """Fraction of grid cells classified as open water."""
        return float(self.mask.mean())

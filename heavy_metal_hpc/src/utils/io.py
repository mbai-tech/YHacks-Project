"""Generic I/O helpers for HDF5, netCDF, and zarr formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_hdf5(data: dict[str, np.ndarray], path: str | Path) -> None:
    """Write a dictionary of arrays to an HDF5 file.

    Parameters
    ----------
    data:
        Mapping of dataset name → numpy array.
    path:
        Output ``.h5`` file path.
    """
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, arr in data.items():
            f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)


def load_hdf5(path: str | Path, keys: list[str] | None = None) -> dict[str, np.ndarray]:
    """Load datasets from an HDF5 file.

    Parameters
    ----------
    path:
        Path to the ``.h5`` file.
    keys:
        List of dataset names to load.  ``None`` loads all datasets.

    Returns
    -------
    dict[str, np.ndarray]
    """
    import h5py

    with h5py.File(path, "r") as f:
        names = list(f.keys()) if keys is None else keys
        return {k: f[k][()] for k in names}


def save_netcdf(
    data: dict[str, np.ndarray],
    coords: dict[str, np.ndarray],
    path: str | Path,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Write model output to a netCDF4 file using xarray.

    Parameters
    ----------
    data:
        Variable name → (T, nx, ny) array mapping.
    coords:
        Coordinate arrays (e.g. ``{"time": ..., "x": ..., "y": ...}``).
    path:
        Output ``.nc`` file path.
    attrs:
        Global attributes to attach to the dataset.
    """
    import xarray as xr

    ds = xr.Dataset(
        {k: (list(coords.keys()), v) for k, v in data.items()},
        coords=coords,
        attrs=attrs or {},
    )
    ds.to_netcdf(path)


def load_netcdf(path: str | Path) -> Any:
    """Open a netCDF4 file as an :class:`xarray.Dataset`.

    Parameters
    ----------
    path:
        Path to the ``.nc`` file.

    Returns
    -------
    xarray.Dataset
    """
    import xarray as xr

    return xr.open_dataset(path)

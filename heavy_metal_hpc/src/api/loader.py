"""High-level data-loading facade that orchestrates weather and hydrology APIs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from .weather import WeatherAPI, WeatherRecord
from .hydrology import HydrologyAPI, HydrologyRecord


class DataLoader:
    """Unified loader that fetches, validates, and caches all forcing data.

    Parameters
    ----------
    weather_api:
        Configured :class:`WeatherAPI` instance.
    hydrology_api:
        Configured :class:`HydrologyAPI` instance.
    cache_dir:
        Root directory for on-disk caching (netCDF / zarr).
    """

    def __init__(
        self,
        weather_api: WeatherAPI,
        hydrology_api: HydrologyAPI,
        cache_dir: str | Path = "data/cache",
    ) -> None:
        self.weather_api = weather_api
        self.hydrology_api = hydrology_api
        self.cache_dir = Path(cache_dir)

    def load(self, start: date, end: date) -> dict[str, Any]:
        """Load all forcing data required for a simulation run.

        Returns a dictionary with keys ``"weather"`` and ``"hydrology"``.
        Results are served from cache when available.

        Parameters
        ----------
        start, end:
            Simulation date range.

        Returns
        -------
        dict[str, Any]
            ``{"weather": np.ndarray (T×6), "hydrology": dict[station_id, np.ndarray]}``
        """
        raise NotImplementedError

    def _cache_path(self, key: str, start: date, end: date) -> Path:
        """Return the canonical cache file path for a given data key and date range."""
        tag = f"{start.isoformat()}_{end.isoformat()}"
        return self.cache_dir / f"{key}_{tag}.nc"

    def _is_cached(self, key: str, start: date, end: date) -> bool:
        """Return True if a valid cache file exists for *key* and the given date range."""
        return self._cache_path(key, start, end).exists()

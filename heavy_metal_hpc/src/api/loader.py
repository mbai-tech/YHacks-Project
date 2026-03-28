"""High-level data-loading facade that orchestrates weather and hydrology APIs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from ..ai.gemini import GeminiClient
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
        gemini_client: GeminiClient | None = None,
    ) -> None:
        self.weather_api = weather_api
        self.hydrology_api = hydrology_api
        self.cache_dir = Path(cache_dir)
        self.gemini_client = gemini_client

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
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        weather_records = self.weather_api.fetch(start, end, lat=23.6850, lon=90.3563)
        hydrology_records = self.hydrology_api.fetch_all_inflows(start, end)
        result: dict[str, Any] = {
            "weather": self.weather_api.to_numpy(weather_records),
            "hydrology": {
                station_id: self.hydrology_api.discharge_array(records)
                for station_id, records in hydrology_records.items()
            },
        }

        if self.gemini_client is not None:
            result["forcing_summary"] = self._summarize_forcing(weather_records, hydrology_records)

        return result

    def _cache_path(self, key: str, start: date, end: date) -> Path:
        """Return the canonical cache file path for a given data key and date range."""
        tag = f"{start.isoformat()}_{end.isoformat()}"
        return self.cache_dir / f"{key}_{tag}.nc"

    def _is_cached(self, key: str, start: date, end: date) -> bool:
        """Return True if a valid cache file exists for *key* and the given date range."""
        return self._cache_path(key, start, end).exists()

    def _summarize_forcing(
        self,
        weather: list[WeatherRecord],
        hydrology: dict[str, list[HydrologyRecord]],
    ) -> dict[str, Any]:
        """Use Gemini to derive a compact operational forcing summary."""
        precip_total = float(sum(item.precipitation_mm for item in weather))
        discharge_peaks = {
            station_id: max(record.discharge_m3s for record in records)
            for station_id, records in hydrology.items()
        }
        prompt = (
            "You are supporting an arsenic-transport digital twin for Bangladesh. "
            "Return JSON with keys monsoon_risk, likely_hotspot_driver, and operator_note. "
            f"Total precipitation_mm={precip_total:.2f}. "
            f"Peak discharges={discharge_peaks}."
        )
        return self.gemini_client.generate_json(prompt)

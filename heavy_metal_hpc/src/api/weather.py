"""Fetch and cache meteorological forcing data for Baiyangdian Lake simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np


@dataclass
class WeatherRecord:
    """Single time-step meteorological record."""

    timestamp: str
    wind_speed_ms: float          # m s⁻¹
    wind_direction_deg: float     # meteorological convention (0 = N)
    air_temp_c: float             # °C
    precipitation_mm: float       # mm hr⁻¹
    solar_radiation_wm2: float    # W m⁻²
    relative_humidity: float      # 0–1


class WeatherAPI:
    """Client for a meteorological data service.

    Parameters
    ----------
    base_url:
        Root URL of the weather REST API.
    api_key:
        Authentication token.
    cache_dir:
        Local directory for caching downloaded responses.
    """

    def __init__(self, base_url: str, api_key: str, cache_dir: str = "data/cache") -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.cache_dir = cache_dir

    def fetch(self, start: date, end: date, lat: float, lon: float) -> list[WeatherRecord]:
        """Download weather records for a lat/lon bounding box and date range.

        Parameters
        ----------
        start, end:
            Inclusive date range.
        lat, lon:
            Centroid of the domain (decimal degrees).

        Returns
        -------
        list[WeatherRecord]
            Time-ordered list of meteorological records.
        """
        raise NotImplementedError("Implement HTTP fetch against your chosen API endpoint.")

    def to_numpy(self, records: list[WeatherRecord]) -> np.ndarray:
        """Stack records into a (T, 6) float64 array ordered as the WeatherRecord fields."""
        return np.array(
            [
                [
                    r.wind_speed_ms,
                    r.wind_direction_deg,
                    r.air_temp_c,
                    r.precipitation_mm,
                    r.solar_radiation_wm2,
                    r.relative_humidity,
                ]
                for r in records
            ],
            dtype=np.float64,
        )

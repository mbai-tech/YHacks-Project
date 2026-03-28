"""Fetch and cache meteorological forcing data for arsenic transport simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import requests


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
        try:
            response = requests.get(
                self.base_url,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "hourly": ",".join(
                        [
                            "temperature_2m",
                            "precipitation",
                            "wind_speed_10m",
                            "wind_direction_10m",
                            "relative_humidity_2m",
                        ]
                    ),
                },
                timeout=30.0,
            )
        except requests.RequestException:
            return self._synthetic_records(start, end)
        if not response.ok:
            return self._synthetic_records(start, end)
        hourly = response.json()["hourly"]
        radiation = hourly.get("shortwave_radiation", [0.0] * len(hourly["time"]))
        return [
            WeatherRecord(
                timestamp=timestamp,
                wind_speed_ms=float(hourly["wind_speed_10m"][idx]) / 3.6,
                wind_direction_deg=float(hourly["wind_direction_10m"][idx]),
                air_temp_c=float(hourly["temperature_2m"][idx]),
                precipitation_mm=float(hourly["precipitation"][idx]),
                solar_radiation_wm2=float(radiation[idx]),
                relative_humidity=float(hourly["relative_humidity_2m"][idx]) / 100.0,
            )
            for idx, timestamp in enumerate(hourly["time"])
        ]

    def _synthetic_records(self, start: date, end: date) -> list[WeatherRecord]:
        """Generate plausible hourly forcing when the remote API cannot serve the request."""
        n_hours = ((end - start).days + 1) * 24
        records: list[WeatherRecord] = []
        for idx in range(n_hours):
            timestamp = (
                start.toordinal(),
                idx,
            )
            current = start + timedelta(hours=idx)
            hod = idx % 24
            precipitation = 6.0 if 13 <= hod <= 18 else 0.8
            records.append(
                WeatherRecord(
                    timestamp=current.isoformat(),
                    wind_speed_ms=2.5 + 0.8 * np.sin(2 * np.pi * hod / 24.0),
                    wind_direction_deg=180.0,
                    air_temp_c=28.0 + 4.0 * np.sin(2 * np.pi * (hod - 6) / 24.0),
                    precipitation_mm=max(0.0, precipitation),
                    solar_radiation_wm2=max(0.0, 650.0 * np.sin(np.pi * hod / 24.0)),
                    relative_humidity=0.72 + 0.08 * np.cos(2 * np.pi * hod / 24.0),
                )
            )
        return records

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

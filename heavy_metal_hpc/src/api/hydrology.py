"""Fetch river inflow / outflow boundary conditions from hydrological data sources."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json

import numpy as np


@dataclass
class HydrologyRecord:
    """Instantaneous flow record at a monitoring station."""

    station_id: str
    timestamp: str
    discharge_m3s: float      # m³ s⁻¹
    water_level_m: float      # m above datum
    suspended_sediment_gl: float  # g L⁻¹
    heavy_metal_ppb: dict[str, float]  # element → µg L⁻¹


class HydrologyAPI:
    """Client for a hydrological / water-quality monitoring service.

    Parameters
    ----------
    base_url:
        Root URL of the hydrological data REST API.
    api_key:
        Authentication token.
    """

    def __init__(self, base_url: str, api_key: str, cache_dir: str = "data/cache") -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)

    def fetch_station(
        self, station_id: str, start: date, end: date
    ) -> list[HydrologyRecord]:
        """Retrieve time-series records for a single monitoring station.

        Parameters
        ----------
        station_id:
            Unique identifier of the gauging station.
        start, end:
            Inclusive date range.

        Returns
        -------
        list[HydrologyRecord]
        """
        if self.base_url == "synthetic":
            return self._synthetic_station(station_id, start, end)

        path = self.cache_dir / f"{station_id}_{start.isoformat()}_{end.isoformat()}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No hydrology fixture found for station {station_id} at {path}. "
                "Use base_url='synthetic' for generated flows or populate cached JSON files."
            )

        raw = json.loads(path.read_text())
        return [
            HydrologyRecord(
                station_id=item["station_id"],
                timestamp=item["timestamp"],
                discharge_m3s=float(item["discharge_m3s"]),
                water_level_m=float(item["water_level_m"]),
                suspended_sediment_gl=float(item["suspended_sediment_gl"]),
                heavy_metal_ppb={k: float(v) for k, v in item["heavy_metal_ppb"].items()},
            )
            for item in raw
        ]

    def fetch_all_inflows(self, start: date, end: date) -> dict[str, list[HydrologyRecord]]:
        """Retrieve inflow records for all boundary stations.

        Returns
        -------
        dict mapping station_id → list of records.
        """
        return {
            station_id: self.fetch_station(station_id, start, end)
            for station_id in ("north_inflow", "south_inflow")
        }

    def discharge_array(self, records: list[HydrologyRecord]) -> np.ndarray:
        """Return a 1-D float64 array of discharge values (m³ s⁻¹)."""
        return np.array([r.discharge_m3s for r in records], dtype=np.float64)

    def _synthetic_station(
        self,
        station_id: str,
        start: date,
        end: date,
    ) -> list[HydrologyRecord]:
        """Generate a simple monsoon-sensitive hydrograph for demos and tests."""
        n_days = (end - start).days + 1
        t = np.arange(n_days, dtype=float)
        discharge = 150.0 + 80.0 * np.sin(2 * np.pi * t / max(n_days, 2))
        water_level = 2.0 + 0.5 * np.sin(2 * np.pi * t / max(n_days, 2))
        sediment = 0.08 + 0.03 * np.cos(2 * np.pi * t / max(n_days, 2))
        arsenic = 25.0 + 6.0 * np.sin(2 * np.pi * t / max(n_days, 2))
        return [
            HydrologyRecord(
                station_id=station_id,
                timestamp=start.fromordinal(start.toordinal() + idx).isoformat(),
                discharge_m3s=float(discharge[idx]),
                water_level_m=float(water_level[idx]),
                suspended_sediment_gl=float(sediment[idx]),
                heavy_metal_ppb={"arsenic": float(arsenic[idx])},
            )
            for idx in range(n_days)
        ]

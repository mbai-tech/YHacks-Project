"""Fetch river inflow / outflow boundary conditions from hydrological data sources."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

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

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key

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
        raise NotImplementedError

    def fetch_all_inflows(self, start: date, end: date) -> dict[str, list[HydrologyRecord]]:
        """Retrieve inflow records for all boundary stations.

        Returns
        -------
        dict mapping station_id → list of records.
        """
        raise NotImplementedError

    def discharge_array(self, records: list[HydrologyRecord]) -> np.ndarray:
        """Return a 1-D float64 array of discharge values (m³ s⁻¹)."""
        return np.array([r.discharge_m3s for r in records], dtype=np.float64)

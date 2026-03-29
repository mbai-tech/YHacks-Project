"""Spatial index over RSEI facility scores.

Usage:
    from scripts.spatial_index import FacilityIndex

    idx = FacilityIndex()
    results = idx.get_nearby_facilities(41.76, -72.69, radius_km=50)
    risk = idx.compute_location_risk(41.76, -72.69)
"""

import math
from dataclasses import dataclass

import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent.parent
FACILITY_SCORES_PATH    = ROOT / "data" / "processed" / "facility_scores.parquet"
FACILITY_PATH           = ROOT / "data" / "processed" / "facility.parquet"
STREAM_SCORES_PATH      = ROOT / "data" / "processed" / "stream_scores.parquet"
WASTEWATER_PLANTS_PATH  = ROOT / "data" / "processed" / "wastewater_plants.parquet"
SDWA_SCORES_PATH        = ROOT / "data" / "processed" / "sdwa_scores.parquet"

# Earth radius used for degree→km conversion
_EARTH_RADIUS_KM = 6371.0


@dataclass
class LocationRisk:
    score: float                        # 0–100, blended final score
    facility_score: float               # 0–100, proximity component
    downstream_score: float             # 0–100, stream segment component
    wastewater_score: float             # 0–100, wastewater proximity component
    sdwa_score: float                   # 0–100, drinking water violations component
    raw_facility_score: float           # sum(TotalWaterImpactScore / (1 + d^2))
    raw_wastewater_score: float         # sum(population_served / distance_km)
    raw_sdwa_score: float               # sum(sdwa_system_risk / distance_km)
    facility_count: int
    segment_count: int
    wastewater_plant_count: int
    sdwa_system_count: int
    top_contributors: pd.DataFrame      # FacilityName, TotalWaterImpactScore, distance_km, contribution


class FacilityIndex:
    def __init__(
        self,
        facility_scores_path: Path = FACILITY_SCORES_PATH,
        facility_path: Path = FACILITY_PATH,
        stream_scores_path: Path = STREAM_SCORES_PATH,
        wastewater_plants_path: Path = WASTEWATER_PLANTS_PATH,
        sdwa_scores_path: Path = SDWA_SCORES_PATH,
    ):
        scores = pd.read_parquet(facility_scores_path)

        # Bring in FinalCOMID from the full facility table
        comids = pd.read_parquet(facility_path, columns=["FacilityNumber", "FinalCOMID"])
        comids = comids.dropna(subset=["FinalCOMID"])
        comids["FinalCOMID"] = comids["FinalCOMID"].astype("int64")
        scores = scores.merge(comids, on="FacilityNumber", how="left")

        self.gdf = gpd.GeoDataFrame(
            scores,
            geometry=gpd.points_from_xy(scores["Longitude"], scores["Latitude"]),
            crs="EPSG:4326",
        )
        # Project to metres for accurate distance queries
        self._gdf_m = self.gdf.to_crs("EPSG:3857")
        self._sindex = self._gdf_m.sindex

        # Stream segment scores keyed by ComID
        self._stream = (
            pd.read_parquet(stream_scores_path, columns=["FinalCOMID", "segment_score"])
            .set_index("FinalCOMID")["segment_score"]
        )
        self._max_segment_score = self._stream.max()

        # SDWA drinking water violation scores
        sdwa = pd.read_parquet(sdwa_scores_path)
        self._sdwa_gdf = gpd.GeoDataFrame(
            sdwa,
            geometry=gpd.points_from_xy(sdwa["LONGITUDE"], sdwa["LATITUDE"]),
            crs="EPSG:4326",
        )
        self._sdwa_gdf_m = self._sdwa_gdf.to_crs("EPSG:3857")
        self._sdwa_sindex = self._sdwa_gdf_m.sindex
        self._max_sdwa_risk = self._sdwa_gdf["sdwa_system_risk"].max()

        # Wastewater treatment plants
        ww = pd.read_parquet(wastewater_plants_path)
        self._ww_gdf = gpd.GeoDataFrame(
            ww,
            geometry=gpd.points_from_xy(ww["LONGITUDE"], ww["LATITUDE"]),
            crs="EPSG:4326",
        )
        self._ww_gdf_m = self._ww_gdf.to_crs("EPSG:3857")
        self._ww_sindex = self._ww_gdf_m.sindex
        self._max_ww_pop = self._ww_gdf["POPULATION_WASTEWATER_CONFIRMED"].max()

    def get_nearby_facilities(
        self, lat: float, lon: float, radius_km: float
    ) -> gpd.GeoDataFrame:
        """Return facilities within *radius_km* of (lat, lon).

        Parameters
        ----------
        lat, lon    : WGS-84 coordinates of the query point
        radius_km   : search radius in kilometres

        Returns
        -------
        GeoDataFrame with all original columns plus a `distance_km` column,
        sorted by distance ascending.
        """
        radius_m = radius_km * 1000.0
        query_point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

        candidate_idx = list(self._sindex.query(query_point.buffer(radius_m)))
        if not candidate_idx:
            return self.gdf.iloc[0:0].copy().assign(distance_km=pd.Series(dtype="float64"))

        candidates = self._gdf_m.iloc[candidate_idx].copy()
        candidates["distance_km"] = candidates.geometry.distance(query_point) / 1000.0

        result = candidates[candidates["distance_km"] <= radius_km].copy()
        result = result.drop(columns="geometry").join(self.gdf["geometry"].iloc[candidate_idx])

        return (
            gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")
            .sort_values("distance_km")
            .reset_index(drop=True)
        )

    def _get_nearby_sdwa(
        self, lat: float, lon: float, radius_km: float
    ) -> gpd.GeoDataFrame:
        """Return SDWA water systems within *radius_km* of (lat, lon)."""
        radius_m = radius_km * 1000.0
        query_point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

        candidate_idx = list(self._sdwa_sindex.query(query_point.buffer(radius_m)))
        if not candidate_idx:
            return self._sdwa_gdf.iloc[0:0].copy().assign(distance_km=pd.Series(dtype="float64"))

        candidates = self._sdwa_gdf_m.iloc[candidate_idx].copy()
        candidates["distance_km"] = candidates.geometry.distance(query_point) / 1000.0
        result = candidates[candidates["distance_km"] <= radius_km].copy()

        return result.sort_values("distance_km").reset_index(drop=True)

    def _get_nearby_wastewater(
        self, lat: float, lon: float, radius_km: float
    ) -> gpd.GeoDataFrame:
        """Return wastewater plants within *radius_km* of (lat, lon)."""
        radius_m = radius_km * 1000.0
        query_point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

        candidate_idx = list(self._ww_sindex.query(query_point.buffer(radius_m)))
        if not candidate_idx:
            return self._ww_gdf.iloc[0:0].copy().assign(distance_km=pd.Series(dtype="float64"))

        candidates = self._ww_gdf_m.iloc[candidate_idx].copy()
        candidates["distance_km"] = candidates.geometry.distance(query_point) / 1000.0
        result = candidates[candidates["distance_km"] <= radius_km].copy()

        return result.sort_values("distance_km").reset_index(drop=True)

    def compute_location_risk(self, lat: float, lon: float, radius_km: float = 50.0) -> LocationRisk:
        """Compute a 0–100 water contamination risk score for a location.

        facility_score  (0–100): proximity-weighted sum of facility impact scores.
            contribution = TotalWaterImpactScore / (1 + distance_km^2)
            normalised via log1p against the worst-case single facility at distance 0.

        downstream_score (0–100): aggregate stream segment risk for ComIDs
            associated with nearby facilities.
            normalised via log1p against the dataset's maximum segment score.

        final_score = 0.6 * facility_score + 0.4 * downstream_score
        """
        nearby = self.get_nearby_facilities(lat, lon, radius_km)

        empty_top3 = pd.DataFrame(
            columns=["FacilityName", "TotalWaterImpactScore", "distance_km", "contribution"]
        )

        # Wastewater risk: sum(population_served / distance_km)
        ww_nearby = self._get_nearby_wastewater(lat, lon, radius_km)
        if ww_nearby.empty:
            raw_wastewater = 0.0
            ww_plant_count = 0
        else:
            # Clamp min distance to 0.1 km to avoid division by zero
            ww_nearby = ww_nearby.copy()
            ww_nearby["distance_km"] = ww_nearby["distance_km"].clip(lower=0.1)
            ww_nearby["contribution"] = (
                ww_nearby["POPULATION_WASTEWATER_CONFIRMED"] / ww_nearby["distance_km"]
            )
            raw_wastewater = ww_nearby["contribution"].sum()
            ww_plant_count = len(ww_nearby)

        # Normalise wastewater via log1p
        log_max_ww = math.log1p(self._max_ww_pop)
        wastewater_score = min((math.log1p(raw_wastewater) / log_max_ww) * 100.0, 100.0) if log_max_ww > 0 else 0.0

        # SDWA risk: sum(sdwa_system_risk / distance_km)
        sdwa_nearby = self._get_nearby_sdwa(lat, lon, radius_km)
        if sdwa_nearby.empty:
            raw_sdwa = 0.0
            sdwa_system_count = 0
        else:
            sdwa_nearby = sdwa_nearby.copy()
            sdwa_nearby["distance_km"] = sdwa_nearby["distance_km"].clip(lower=0.1)
            sdwa_nearby["contribution"] = (
                sdwa_nearby["sdwa_system_risk"] / sdwa_nearby["distance_km"]
            )
            raw_sdwa = sdwa_nearby["contribution"].sum()
            sdwa_system_count = len(sdwa_nearby)

        # Normalise SDWA via log1p
        log_max_sdwa = math.log1p(self._max_sdwa_risk)
        sdwa_score = min((math.log1p(raw_sdwa) / log_max_sdwa) * 100.0, 100.0) if log_max_sdwa > 0 else 0.0

        if nearby.empty:
            final = round(0.25 * wastewater_score + 0.25 * sdwa_score, 2)
            return LocationRisk(
                score=final,
                facility_score=0.0, downstream_score=0.0,
                wastewater_score=round(wastewater_score, 2),
                sdwa_score=round(sdwa_score, 2),
                raw_facility_score=0.0, raw_wastewater_score=round(raw_wastewater, 4),
                raw_sdwa_score=round(raw_sdwa, 4),
                facility_count=0, segment_count=0,
                wastewater_plant_count=ww_plant_count,
                sdwa_system_count=sdwa_system_count,
                top_contributors=empty_top3,
            )

        nearby = nearby.copy()

        # ------------------------------------------------------------------
        # Facility score (proximity-weighted)
        # ------------------------------------------------------------------
        nearby["contribution"] = nearby["TotalWaterImpactScore"] / (1.0 + nearby["distance_km"] ** 2)
        raw_facility = nearby["contribution"].sum()

        max_single = self.gdf["TotalWaterImpactScore"].max()
        log_max = math.log1p(max_single)
        facility_score = min((math.log1p(raw_facility) / log_max) * 100.0, 100.0) if log_max > 0 else 0.0

        # ------------------------------------------------------------------
        # Downstream score (stream segment lookup)
        # ------------------------------------------------------------------
        nearby_comids = nearby["FinalCOMID"].dropna().astype("int64").unique()
        matched = self._stream.reindex(nearby_comids).dropna()
        segment_count = len(matched)

        if segment_count > 0:
            raw_downstream = matched.sum()
            downstream_score = min(
                (math.log1p(raw_downstream) / math.log1p(self._max_segment_score)) * 100.0,
                100.0,
            )
        else:
            downstream_score = 0.0

        # ------------------------------------------------------------------
        # Blended final score
        # ------------------------------------------------------------------
        final_score = round(
            0.3 * facility_score
            + 0.2 * downstream_score
            + 0.25 * wastewater_score
            + 0.25 * sdwa_score,
            2,
        )

        top3 = (
            nearby[["FacilityName", "TotalWaterImpactScore", "distance_km", "contribution"]]
            .sort_values("contribution", ascending=False)
            .head(3)
            .reset_index(drop=True)
        )

        return LocationRisk(
            score=final_score,
            facility_score=round(facility_score, 2),
            downstream_score=round(downstream_score, 2),
            wastewater_score=round(wastewater_score, 2),
            sdwa_score=round(sdwa_score, 2),
            raw_facility_score=round(raw_facility, 4),
            raw_wastewater_score=round(raw_wastewater, 4),
            raw_sdwa_score=round(raw_sdwa, 4),
            facility_count=len(nearby),
            segment_count=segment_count,
            wastewater_plant_count=ww_plant_count,
            sdwa_system_count=sdwa_system_count,
            top_contributors=top3,
        )

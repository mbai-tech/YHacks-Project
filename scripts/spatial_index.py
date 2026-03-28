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
FACILITY_SCORES_PATH = ROOT / "data" / "processed" / "facility_scores.parquet"
FACILITY_PATH        = ROOT / "data" / "processed" / "facility.parquet"
STREAM_SCORES_PATH   = ROOT / "data" / "processed" / "stream_scores.parquet"

# Earth radius used for degree→km conversion
_EARTH_RADIUS_KM = 6371.0


@dataclass
class LocationRisk:
    score: float                        # 0–100, blended final score
    facility_score: float               # 0–100, proximity component
    downstream_score: float             # 0–100, stream segment component
    raw_facility_score: float           # sum(TotalWaterImpactScore / (1 + d^2))
    facility_count: int
    segment_count: int
    top_contributors: pd.DataFrame      # FacilityName, TotalWaterImpactScore, distance_km, contribution


class FacilityIndex:
    def __init__(
        self,
        facility_scores_path: Path = FACILITY_SCORES_PATH,
        facility_path: Path = FACILITY_PATH,
        stream_scores_path: Path = STREAM_SCORES_PATH,
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
        if nearby.empty:
            return LocationRisk(
                score=0.0, facility_score=0.0, downstream_score=0.0,
                raw_facility_score=0.0, facility_count=0, segment_count=0,
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
        final_score = round(0.6 * facility_score + 0.4 * downstream_score, 2)

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
            raw_facility_score=round(raw_facility, 4),
            facility_count=len(nearby),
            segment_count=segment_count,
            top_contributors=top3,
        )

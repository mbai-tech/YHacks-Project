"""Data-backed risk service for the Water Risk Explorer."""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONTIGUOUS_BOUNDS = {
    "min_lat": 24.5,
    "max_lat": 49.7,
    "min_lon": -124.8,
    "max_lon": -66.7,
}

DEFAULT_SCENARIO = {
    "industrial_reduction": 20,
    "wastewater_improvement": 30,
    "plastic_reduction": 15,
    "household_filter_adoption": 40,
    "microfiber_filter_adoption": 50,
}

COMPONENT_WEIGHTS = {
    "industrial": 0.40,
    "wastewater": 0.25,
    "plastic_waste": 0.20,
    "microfiber": 0.15,
}

MICROFIBER_EFFECTIVENESS = 0.35
HOUSEHOLD_FILTER_EFFECTIVENESS = 0.45


@dataclass(frozen=True)
class SearchCandidate:
    """Normalized search candidate used by the frontend."""

    id: str
    name: str
    kind: str
    latitude: float
    longitude: float
    subtitle: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "latitude": round(self.latitude, 5),
            "longitude": round(self.longitude, 5),
            "subtitle": self.subtitle,
        }


class WaterRiskService:
    """Loads local datasets and exposes risk/search/map operations."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.industrial = self._load_industrial()
        self.wastewater = self._load_wastewater()
        self.search_index = self._build_search_index()
        self.component_scales = self._calibrate_component_scales()
        self.map_component_cache = self._build_map_component_cache()
        self.state_outlines = self._load_state_outlines()

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _parquet_facilities_path(self) -> Path:
        return self._project_root() / "data" / "interim" / "facilities_clean.parquet"

    def _parquet_stream_scores_path(self) -> Path:
        return self._project_root() / "data" / "processed" / "stream_scores.parquet"

    def _has_parquet_inputs(self) -> bool:
        return self._parquet_facilities_path().exists() and self._parquet_stream_scores_path().exists()

    def _load_parquet_facility_frame(self) -> pd.DataFrame:
        facilities = pd.read_parquet(self._parquet_facilities_path()).copy()
        stream_scores = pd.read_parquet(self._parquet_stream_scores_path()).copy()

        facilities["Latitude"] = pd.to_numeric(facilities.get("Latitude"), errors="coerce")
        facilities["Longitude"] = pd.to_numeric(facilities.get("Longitude"), errors="coerce")
        facilities["FacilityID"] = pd.to_numeric(facilities.get("FacilityNumber"), errors="coerce")

        if "stream_score_raw" not in stream_scores.columns and "segment_score" in stream_scores.columns:
            stream_scores = stream_scores.rename(columns={"segment_score": "stream_score_raw"})
        if "COMID" not in stream_scores.columns and "FinalCOMID" in stream_scores.columns:
            stream_scores = stream_scores.rename(columns={"FinalCOMID": "COMID"})
        if "COMID" not in facilities.columns and "FinalCOMID" in facilities.columns:
            facilities = facilities.rename(columns={"FinalCOMID": "COMID"})

        if "WaterReleases" in facilities.columns:
            water_releases = facilities["WaterReleases"]
            if pd.api.types.is_bool_dtype(water_releases):
                facilities["is_water_release"] = water_releases.fillna(False).astype(bool)
            else:
                facilities["is_water_release"] = water_releases.astype("string").fillna("").str.strip().ne("")
        else:
            facilities["is_water_release"] = False

        facilities = facilities.dropna(subset=["Latitude", "Longitude"]).copy()

        if "COMID" in facilities.columns and "COMID" in stream_scores.columns:
            facilities = facilities.merge(
                stream_scores[[column for column in ["COMID", "stream_score_raw"] if column in stream_scores.columns]].drop_duplicates("COMID"),
                on="COMID",
                how="left",
                suffixes=("", "_score"),
            )

        facilities["stream_score_raw"] = pd.to_numeric(facilities.get("stream_score_raw"), errors="coerce").fillna(0.0)
        facilities["normalized_risk"] = (self._minmax(facilities["stream_score_raw"]).fillna(0.0) * 100.0).round(1)
        facilities["population_proxy"] = 1000.0 + facilities["normalized_risk"] * 25.0
        facilities["facility_weight"] = 0.35 + self._minmax(facilities["stream_score_raw"]).fillna(0.0) * 0.65
        return facilities.reset_index(drop=True)

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Search by coordinates, ZIP, city, or county using local data."""
        query = (query or "").strip()
        if not query:
            return [candidate.to_dict() for candidate in self.search_index[:limit]]

        coords = self._parse_coordinates(query)
        if coords is not None:
            lat, lon = coords
            return [
                SearchCandidate(
                    id=f"coords:{lat:.4f},{lon:.4f}",
                    name=f"Coordinates ({lat:.4f}, {lon:.4f})",
                    kind="coordinates",
                    latitude=lat,
                    longitude=lon,
                    subtitle="Direct coordinate search",
                ).to_dict()
            ]

        normalized = self._normalize_text(query)
        zip_match = re.fullmatch(r"\d{5}", query)

        exact_matches: list[SearchCandidate] = []
        fuzzy_matches: list[SearchCandidate] = []
        for candidate in self.search_index:
            haystack = self._normalize_text(candidate.name)
            if zip_match and candidate.kind == "zip" and query in candidate.name:
                exact_matches.append(candidate)
                continue
            if haystack == normalized:
                exact_matches.append(candidate)
                continue
            if normalized in haystack:
                fuzzy_matches.append(candidate)

        ranked = exact_matches + fuzzy_matches
        ranked.sort(key=lambda candidate: (self._search_priority(candidate.kind), candidate.name))
        deduped: list[SearchCandidate] = []
        seen: set[str] = set()
        for candidate in ranked:
            if candidate.id in seen:
                continue
            seen.add(candidate.id)
            deduped.append(candidate)
            if len(deduped) >= limit:
                break
        return [candidate.to_dict() for candidate in deduped]

    def get_risk(self, latitude: float, longitude: float, *, name: str | None = None) -> dict[str, Any]:
        """Return baseline risk, factor breakdown, nearby sources, and text."""
        baseline = self._baseline_factors(latitude, longitude)
        scenario = self.apply_scenario(latitude, longitude, DEFAULT_SCENARIO, baseline=baseline, name=name)
        place_name = name or self._nearest_named_place(latitude, longitude)
        return {
            "location": {
                "name": place_name,
                "latitude": round(latitude, 5),
                "longitude": round(longitude, 5),
                "subtitle": self._build_subtitle(baseline),
            },
            "baseline": scenario["baseline"],
            "scenario": scenario["scenario"],
            "nearby_sources": baseline["nearby_sources"],
            "explanation": baseline["explanation"],
            "recommendations": baseline["recommendations"],
            "context": baseline["context"],
        }

    def apply_scenario(
        self,
        latitude: float,
        longitude: float,
        scenario: dict[str, Any],
        *,
        baseline: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Apply the intervention sliders to a location."""
        scenario_state = self._normalize_scenario(scenario)
        baseline = baseline or self._baseline_factors(latitude, longitude)

        industrial = baseline["factor_scores"]["industrial_score"]
        wastewater = baseline["factor_scores"]["wastewater_score"]
        plastic = baseline["factor_scores"]["plastic_waste_score"]
        microfiber = baseline["factor_scores"]["microfiber_score"]

        adjusted = {
            "industrial_score": industrial * (1 - scenario_state["industrial_reduction"] / 100.0),
            "wastewater_score": wastewater * (1 - scenario_state["wastewater_improvement"] / 100.0),
            "plastic_waste_score": plastic * (1 - scenario_state["plastic_reduction"] / 100.0),
            "microfiber_score": microfiber
            * (1 - scenario_state["microfiber_filter_adoption"] / 100.0 * MICROFIBER_EFFECTIVENESS),
        }
        adjusted_total = self._weighted_total(adjusted)
        exposure = adjusted_total * (
            1 - scenario_state["household_filter_adoption"] / 100.0 * HOUSEHOLD_FILTER_EFFECTIVENESS
        )
        baseline_total = baseline["factor_scores"]["total_score"]
        improvement_pct = max(0.0, (baseline_total - adjusted_total) / max(baseline_total, 1e-6) * 100.0)

        intervention_impacts = {
            "Industrial controls": industrial - adjusted["industrial_score"],
            "Wastewater upgrades": wastewater - adjusted["wastewater_score"],
            "Plastic waste reduction": plastic - adjusted["plastic_waste_score"],
            "Household filtration": baseline_total - exposure,
            "Microfiber filters": microfiber - adjusted["microfiber_score"],
        }
        best_intervention = max(intervention_impacts.items(), key=lambda item: item[1])[0]

        location_name = name or self._nearest_named_place(latitude, longitude)
        return {
            "location": {
                "name": location_name,
                "latitude": round(latitude, 5),
                "longitude": round(longitude, 5),
            },
            "baseline": {
                **baseline["factor_scores"],
                "risk_level": self._risk_level(baseline_total),
            },
            "scenario": {
                **{k: round(v, 1) for k, v in adjusted.items()},
                "total_score": round(adjusted_total, 1),
                "exposure_score": round(exposure, 1),
                "risk_level": self._risk_level(adjusted_total),
                "improvement_pct": round(improvement_pct, 1),
                "best_intervention": best_intervention,
                "summary": self._scenario_summary(location_name, adjusted_total, improvement_pct, best_intervention),
            },
            "scenario_state": scenario_state,
        }

    def compare(
        self,
        location_a: dict[str, Any],
        location_b: dict[str, Any],
        scenario: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare two locations using the same scenario state."""
        risk_a = self.get_risk(float(location_a["latitude"]), float(location_a["longitude"]), name=location_a.get("name"))
        risk_b = self.get_risk(float(location_b["latitude"]), float(location_b["longitude"]), name=location_b.get("name"))
        scenario_a = self.apply_scenario(
            float(location_a["latitude"]),
            float(location_a["longitude"]),
            scenario,
            baseline={
                "factor_scores": risk_a["baseline"],
                "nearby_sources": risk_a["nearby_sources"],
                "context": risk_a["context"],
                "recommendations": risk_a["recommendations"],
                "explanation": risk_a["explanation"],
            },
            name=location_a.get("name"),
        )
        scenario_b = self.apply_scenario(
            float(location_b["latitude"]),
            float(location_b["longitude"]),
            scenario,
            baseline={
                "factor_scores": risk_b["baseline"],
                "nearby_sources": risk_b["nearby_sources"],
                "context": risk_b["context"],
                "recommendations": risk_b["recommendations"],
                "explanation": risk_b["explanation"],
            },
            name=location_b.get("name"),
        )

        total_a = scenario_a["scenario"]["total_score"]
        total_b = scenario_b["scenario"]["total_score"]
        key_difference = self._comparison_difference(risk_a["baseline"], risk_b["baseline"])
        more_vulnerable = location_a["name"] if total_a >= total_b else location_b["name"]

        return {
            "location_a": {
                **risk_a,
                "scenario": scenario_a["scenario"],
            },
            "location_b": {
                **risk_b,
                "scenario": scenario_b["scenario"],
            },
            "comparison": {
                "more_vulnerable": more_vulnerable,
                "score_gap": round(abs(total_a - total_b), 1),
                "key_difference": key_difference,
                "insight": (
                    f"{more_vulnerable} remains more vulnerable after the current intervention mix. "
                    f"{key_difference}"
                ),
            },
        }

    def build_map_payload(self, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return the heatmap grid and marker set used by the interactive map."""
        if self._has_parquet_inputs():
            return self._build_parquet_map_payload()

        scenario_state = self._normalize_scenario(scenario or DEFAULT_SCENARIO)
        baseline = self.map_component_cache
        adjusted = self._apply_scenario_to_components(
            baseline["industrial"],
            baseline["wastewater"],
            baseline["plastic_waste"],
            baseline["microfiber"],
            scenario_state,
        )
        total = self._weighted_total_grid(adjusted)
        hotspots = self._hotspot_markers()

        return {
            "bounds": CONTIGUOUS_BOUNDS,
            "latitudes": baseline["latitudes"].round(4).tolist(),
            "longitudes": baseline["longitudes"].round(4).tolist(),
            "baseline_grid": baseline["total"].round(2).tolist(),
            "scenario_grid": total.round(2).tolist(),
            "markers": hotspots,
            "state_outlines": self.state_outlines,
            "top_places": [candidate.to_dict() for candidate in self.search_index[:12]],
            "stats": {
                "baseline_peak": round(float(np.max(baseline["total"])), 2),
                "scenario_peak": round(float(np.max(total)), 2),
                "hotspot_count": len(hotspots),
            },
        }

    def _build_parquet_map_payload(self) -> dict[str, Any]:
        facilities = self.industrial.copy()
        facilities["Latitude"] = pd.to_numeric(facilities["Latitude"], errors="coerce")
        facilities["Longitude"] = pd.to_numeric(facilities["Longitude"], errors="coerce")
        facilities["stream_score_raw"] = pd.to_numeric(facilities["stream_score_raw"], errors="coerce").fillna(0.0)
        facilities["normalized_risk"] = pd.to_numeric(facilities["normalized_risk"], errors="coerce").fillna(0.0)
        facilities = facilities.dropna(subset=["Latitude", "Longitude"]).copy()

        lat_pad = 0.18
        lon_pad = 0.24
        bounds = {
            "min_lat": round(float(facilities["Latitude"].min() - lat_pad), 4),
            "max_lat": round(float(facilities["Latitude"].max() + lat_pad), 4),
            "min_lon": round(float(facilities["Longitude"].min() - lon_pad), 4),
            "max_lon": round(float(facilities["Longitude"].max() + lon_pad), 4),
        }

        top_places_df = facilities.sort_values("stream_score_raw", ascending=False).head(12)
        top_places = [
            SearchCandidate(
                id=f"facility:{int(row['FacilityID'])}" if pd.notna(row["FacilityID"]) else f"facility:{idx}",
                name=str(row["FacilityName"]),
                kind="facility",
                latitude=float(row["Latitude"]),
                longitude=float(row["Longitude"]),
                subtitle=f"{row.get('County', '')}, {row.get('State', 'CT')} · score {float(row['stream_score_raw']):.2f}",
            ).to_dict()
            for idx, (_, row) in enumerate(top_places_df.iterrows())
        ]

        heat_source = facilities.sort_values("stream_score_raw", ascending=False).copy()
        if not heat_source.empty:
            threshold = float(heat_source["stream_score_raw"].quantile(0.55))
            heat_source = heat_source[heat_source["stream_score_raw"] >= threshold].copy()
            heat_source = heat_source.head(260)

        heat_points = [
            {
                "latitude": round(float(row["Latitude"]), 5),
                "longitude": round(float(row["Longitude"]), 5),
                "intensity": round(float(max(row["stream_score_raw"], 0.0)), 4),
            }
            for _, row in heat_source.iterrows()
        ]

        hotspots = self._hotspot_markers()
        peak_score = round(float(facilities["stream_score_raw"].max()), 2) if not facilities.empty else 0.0

        return {
            "bounds": bounds,
            "heat_points": heat_points,
            "markers": hotspots,
            "state_outlines": [],
            "top_places": top_places,
            "stats": {
                "baseline_peak": peak_score,
                "scenario_peak": peak_score,
                "hotspot_count": len(hotspots),
            },
        }

    def build_brief(self, latitude: float, longitude: float, scenario: dict[str, Any], name: str | None = None) -> str:
        """Generate a lightweight markdown brief from the current selection."""
        risk = self.get_risk(latitude, longitude, name=name)
        scenario_result = self.apply_scenario(latitude, longitude, scenario, baseline={
            "factor_scores": risk["baseline"],
            "nearby_sources": risk["nearby_sources"],
            "context": risk["context"],
            "recommendations": risk["recommendations"],
            "explanation": risk["explanation"],
        }, name=name)
        lines = [
            f"## {risk['location']['name']}",
            "",
            f"- Baseline contamination risk: **{risk['baseline']['total_score']:.1f} / 100** ({risk['baseline']['risk_level']})",
            f"- Scenario-adjusted risk: **{scenario_result['scenario']['total_score']:.1f} / 100**",
            f"- Household exposure risk: **{scenario_result['scenario']['exposure_score']:.1f} / 100**",
            f"- Estimated improvement: **{scenario_result['scenario']['improvement_pct']:.1f}%**",
            "",
            "### Main drivers",
            *[f"- {item}" for item in risk["explanation"]],
            "",
            "### Nearby sources",
            *[
                f"- {source['name']} ({source['type']}, {source['distance_km']:.1f} km): {source['summary']}"
                for source in risk["nearby_sources"][:5]
            ],
            "",
            "### Recommended next move",
            f"- Highest-impact intervention right now: **{scenario_result['scenario']['best_intervention']}**",
            *[f"- {item}" for item in risk["recommendations"][:3]],
        ]
        return "\n".join(lines)

    def _load_industrial(self) -> pd.DataFrame:
        if self._has_parquet_inputs():
            facilities = self._load_parquet_facility_frame()
            renamed = facilities.rename(
                columns={
                    "CountyName": "County",
                    "StateCode": "State",
                    "ZipCode": "ZIPCode",
                }
            ).copy()
            for column, default in {
                "FacilityID": 0,
                "FacilityName": "Selected facility",
                "City": "",
                "County": "",
                "State": "CT",
                "ZIPCode": "",
            }.items():
                if column not in renamed.columns:
                    renamed[column] = default
            renamed["release_proxy"] = renamed["stream_score_raw"].fillna(0.0)
            renamed["tox_proxy"] = renamed["normalized_risk"].fillna(0.0)
            renamed["pop_proxy"] = renamed["population_proxy"].fillna(0.0)
            return renamed.reset_index(drop=True)

        facility_path = self.data_dir / "Pollution" / "facility_data_rsei_v2312.csv"
        grid_metrics = self._load_grid_metrics()
        df = pd.read_csv(
            facility_path,
            usecols=[
                "FacilityID",
                "FacilityName",
                "Latitude",
                "Longitude",
                "City",
                "County",
                "State",
                "ZIPCode",
                "GridCode",
            ],
            low_memory=False,
        )
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        df["GridCode"] = pd.to_numeric(df["GridCode"], errors="coerce")
        df = df.dropna(subset=["Latitude", "Longitude"])
        df = df[
            df["Latitude"].between(CONTIGUOUS_BOUNDS["min_lat"], CONTIGUOUS_BOUNDS["max_lat"])
            & df["Longitude"].between(CONTIGUOUS_BOUNDS["min_lon"], CONTIGUOUS_BOUNDS["max_lon"])
        ].copy()
        df = df.merge(grid_metrics, on="GridCode", how="left")
        df["release_proxy"] = pd.to_numeric(df["num_releases"], errors="coerce").fillna(df["num_releases"].median())
        df["tox_proxy"] = pd.to_numeric(df["tox_conc"], errors="coerce").fillna(df["tox_conc"].median())
        df["pop_proxy"] = pd.to_numeric(df["pop_proxy"], errors="coerce").fillna(df["pop_proxy"].median())
        df["facility_weight"] = (
            0.35
            + self._minmax(df["release_proxy"]).fillna(0.0) * 0.35
            + self._minmax(np.log1p(df["tox_proxy"])).fillna(0.0) * 0.30
        )
        return df.reset_index(drop=True)

    def _load_grid_metrics(self) -> pd.DataFrame:
        if self._has_parquet_inputs():
            facilities = self._load_parquet_facility_frame()
            grid = facilities.reset_index().rename(columns={"index": "GridCode"}).copy()
            return grid[["GridCode", "stream_score_raw", "normalized_risk", "population_proxy"]].rename(
                columns={
                    "stream_score_raw": "num_releases",
                    "normalized_risk": "tox_conc",
                    "population_proxy": "pop_proxy",
                }
            )

        gridcodes = (
            pd.read_csv(
                self.data_dir / "Pollution" / "facility_data_rsei_v2312.csv",
                usecols=["GridCode"],
                low_memory=False,
            )["GridCode"]
            .dropna()
            .astype(float)
            .unique()
            .tolist()
        )
        target_gridcodes = set(gridcodes)
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            self.data_dir / "Pollution" / "aggmicro2022_2022.csv",
            usecols=["GridCode", "NumReleases", "ToxConc", "Pop"],
            chunksize=250_000,
            low_memory=False,
        ):
            chunk["GridCode"] = pd.to_numeric(chunk["GridCode"], errors="coerce")
            filtered = chunk[chunk["GridCode"].isin(target_gridcodes)]
            if not filtered.empty:
                chunks.append(filtered)

        if not chunks:
            return pd.DataFrame(columns=["GridCode", "num_releases", "tox_conc", "pop_proxy"])

        metrics = pd.concat(chunks, ignore_index=True)
        metrics["NumReleases"] = pd.to_numeric(metrics["NumReleases"], errors="coerce")
        metrics["ToxConc"] = pd.to_numeric(metrics["ToxConc"], errors="coerce")
        metrics["Pop"] = pd.to_numeric(metrics["Pop"], errors="coerce")
        return (
            metrics.groupby("GridCode", as_index=False)
            .agg(num_releases=("NumReleases", "mean"), tox_conc=("ToxConc", "mean"), pop_proxy=("Pop", "mean"))
        )

    def _load_wastewater(self) -> pd.DataFrame:
        if self._has_parquet_inputs():
            facilities = self._load_parquet_facility_frame()
            wastewater = facilities[facilities["is_water_release"]].copy()
            if wastewater.empty:
                wastewater = facilities.copy()
            wastewater["FACILITY_ID"] = wastewater["FacilityID"].fillna(0).astype(int)
            wastewater["LATITUDE"] = wastewater["Latitude"]
            wastewater["LONGITUDE"] = wastewater["Longitude"]
            wastewater["FACILITY_NAME"] = wastewater["FacilityName"].fillna("Water-linked facility")
            wastewater["COUNTY_NAME"] = wastewater["County"].fillna("")
            wastewater["STATE_CODE_x"] = wastewater["State"].fillna("CT")
            wastewater["CITY"] = wastewater["City"].fillna("") if "City" in wastewater.columns else ""
            wastewater["ZIP_CODE"] = wastewater["ZipCode"].fillna("") if "ZipCode" in wastewater.columns else ""
            wastewater["population_proxy"] = 1500.0 + wastewater["normalized_risk"].fillna(0.0) * 30.0
            wastewater["wastewater_weight"] = 0.30 + self._minmax(wastewater["stream_score_raw"]).fillna(0.0) * 0.70
            return wastewater.reset_index(drop=True)

        facilities = pd.read_csv(
            self.data_dir / "WasteWaterFacilities" / "FACILITIES.csv",
            usecols=["CWNS_ID", "FACILITY_ID", "FACILITY_NAME", "STATE_CODE", "INFRASTRUCTURE_TYPE"],
            low_memory=False,
        )
        locations = pd.read_csv(
            self.data_dir / "WasteWaterFacilities" / "PHYSICAL_LOCATION.csv",
            usecols=["CWNS_ID", "FACILITY_ID", "LATITUDE", "LONGITUDE", "CITY", "STATE_CODE", "ZIP_CODE", "COUNTY_NAME"],
            low_memory=False,
        )
        populations = pd.read_csv(
            self.data_dir / "WasteWaterFacilities" / "POPULATION_WASTEWATER.csv",
            usecols=[
                "CWNS_ID",
                "FACILITY_ID",
                "TOTAL_RES_POPULATION_2022",
                "NET_TREATED_POPULATION_2022",
                "TREATED_DISCHARGE_PERCENTAGE_2022",
            ],
            low_memory=False,
        )

        merged = locations.merge(facilities, on=["CWNS_ID", "FACILITY_ID"], how="left").merge(
            populations, on=["CWNS_ID", "FACILITY_ID"], how="left"
        )
        merged["LATITUDE"] = pd.to_numeric(merged["LATITUDE"], errors="coerce")
        merged["LONGITUDE"] = pd.to_numeric(merged["LONGITUDE"], errors="coerce")
        merged["TOTAL_RES_POPULATION_2022"] = pd.to_numeric(merged["TOTAL_RES_POPULATION_2022"], errors="coerce")
        merged["NET_TREATED_POPULATION_2022"] = pd.to_numeric(merged["NET_TREATED_POPULATION_2022"], errors="coerce")
        merged["TREATED_DISCHARGE_PERCENTAGE_2022"] = pd.to_numeric(
            merged["TREATED_DISCHARGE_PERCENTAGE_2022"], errors="coerce"
        )
        merged = merged.dropna(subset=["LATITUDE", "LONGITUDE"])
        merged = merged[
            merged["LATITUDE"].between(CONTIGUOUS_BOUNDS["min_lat"], CONTIGUOUS_BOUNDS["max_lat"])
            & merged["LONGITUDE"].between(CONTIGUOUS_BOUNDS["min_lon"], CONTIGUOUS_BOUNDS["max_lon"])
        ].copy()
        pop = merged["NET_TREATED_POPULATION_2022"].fillna(merged["TOTAL_RES_POPULATION_2022"]).fillna(0.0)
        discharge_pct = merged["TREATED_DISCHARGE_PERCENTAGE_2022"].fillna(0.75)
        merged["wastewater_weight"] = 0.30 + self._minmax(np.log1p(pop)).fillna(0.0) * 0.45 + discharge_pct.clip(0, 1) * 0.25
        merged["population_proxy"] = pop
        return merged.reset_index(drop=True)

    def _build_search_index(self) -> list[SearchCandidate]:
        candidates: list[SearchCandidate] = []
        industrial = self.industrial.copy()
        wastewater = self.wastewater.copy()

        city_groups = []
        for df, city_col, state_col, lat_col, lon_col in [
            (industrial, "City", "State", "Latitude", "Longitude"),
            (wastewater, "CITY", "STATE_CODE_x", "LATITUDE", "LONGITUDE"),
        ]:
            group = (
                df.dropna(subset=[city_col, state_col])
                .assign(city=df[city_col].astype(str).str.strip(), state=df[state_col].astype(str).str.strip())
                .groupby(["city", "state"], as_index=False)
                .agg(latitude=(lat_col, "mean"), longitude=(lon_col, "mean"), count=(lat_col, "size"))
            )
            city_groups.append(group)

        all_cities = (
            pd.concat(city_groups, ignore_index=True)
            .groupby(["city", "state"], as_index=False)
            .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"), count=("count", "sum"))
            .sort_values(["count", "city"], ascending=[False, True])
        )
        for _, row in all_cities.head(120).iterrows():
            candidates.append(
                SearchCandidate(
                    id=f"city:{self._slug(row['city'])}-{row['state']}",
                    name=f"{row['city']}, {row['state']}",
                    kind="city",
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    subtitle=f"{int(row['count'])} local data points aggregated here",
                )
            )

        county_groups = (
            industrial.dropna(subset=["County", "State"])
            .assign(county=industrial["County"].astype(str).str.strip(), state=industrial["State"].astype(str).str.strip())
            .groupby(["county", "state"], as_index=False)
            .agg(latitude=("Latitude", "mean"), longitude=("Longitude", "mean"), count=("Latitude", "size"))
            .sort_values(["count", "county"], ascending=[False, True])
            .head(120)
        )
        for _, row in county_groups.iterrows():
            candidates.append(
                SearchCandidate(
                    id=f"county:{self._slug(row['county'])}-{row['state']}",
                    name=f"{row['county']}, {row['state']}",
                    kind="county",
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    subtitle=f"{int(row['count'])} industrial facilities in this county cluster",
                )
            )

        zip_groups = (
            industrial.dropna(subset=["ZIPCode"])
            .assign(zip_code=industrial["ZIPCode"].astype(str).str[:5])
            .query("zip_code.str.len() == 5", engine="python")
            .groupby("zip_code", as_index=False)
            .agg(latitude=("Latitude", "mean"), longitude=("Longitude", "mean"), count=("Latitude", "size"))
            .sort_values("count", ascending=False)
            .head(120)
        )
        for _, row in zip_groups.iterrows():
            candidates.append(
                SearchCandidate(
                    id=f"zip:{row['zip_code']}",
                    name=f"ZIP {row['zip_code']}",
                    kind="zip",
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    subtitle=f"{int(row['count'])} facilities contribute to this ZIP centroid",
                )
            )

        deduped: list[SearchCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.id in seen:
                continue
            seen.add(candidate.id)
            deduped.append(candidate)
        return deduped

    def _build_map_component_cache(self) -> dict[str, Any]:
        longitudes = np.linspace(CONTIGUOUS_BOUNDS["min_lon"], CONTIGUOUS_BOUNDS["max_lon"], 40)
        latitudes = np.linspace(CONTIGUOUS_BOUNDS["min_lat"], CONTIGUOUS_BOUNDS["max_lat"], 22)
        industrial = np.zeros((len(latitudes), len(longitudes)), dtype=float)
        wastewater = np.zeros_like(industrial)
        plastic = np.zeros_like(industrial)
        microfiber = np.zeros_like(industrial)

        for i, lat in enumerate(latitudes):
            for j, lon in enumerate(longitudes):
                factors = self._baseline_factors(lat, lon, include_details=False)["factor_scores"]
                industrial[i, j] = factors["industrial_score"]
                wastewater[i, j] = factors["wastewater_score"]
                plastic[i, j] = factors["plastic_waste_score"]
                microfiber[i, j] = factors["microfiber_score"]

        total = self._weighted_total_grid(
            {
                "industrial": industrial,
                "wastewater": wastewater,
                "plastic_waste": plastic,
                "microfiber": microfiber,
            }
        )
        return {
            "latitudes": latitudes,
            "longitudes": longitudes,
            "industrial": industrial,
            "wastewater": wastewater,
            "plastic_waste": plastic,
            "microfiber": microfiber,
            "total": total,
        }

    def _load_state_outlines(self) -> list[list[list[float]]]:
        shp_path = self.data_dir / "GeographyShapes" / "tl_2022_16_tract" / "tl_2022_us_state" / "tl_2022_us_state.shp"
        if not shp_path.exists():
            return []

        outlines: list[list[list[float]]] = []
        with shp_path.open("rb") as handle:
            handle.seek(100)
            while True:
                header = handle.read(8)
                if len(header) < 8:
                    break
                _, content_length_words = struct.unpack(">2i", header)
                content = handle.read(content_length_words * 2)
                if len(content) < 44:
                    continue
                shape_type = struct.unpack("<i", content[:4])[0]
                if shape_type not in (5, 15, 25):
                    continue
                xmin, ymin, xmax, ymax = struct.unpack("<4d", content[4:36])
                if not self._bbox_intersects_contiguous(xmin, ymin, xmax, ymax):
                    continue
                num_parts, num_points = struct.unpack("<2i", content[36:44])
                parts_offset = 44
                parts = list(struct.unpack(f"<{num_parts}i", content[parts_offset : parts_offset + 4 * num_parts]))
                points_offset = parts_offset + 4 * num_parts
                points = [
                    struct.unpack("<2d", content[points_offset + idx * 16 : points_offset + (idx + 1) * 16])
                    for idx in range(num_points)
                ]
                part_starts = parts + [num_points]
                for part_index in range(num_parts):
                    ring = points[part_starts[part_index] : part_starts[part_index + 1]]
                    simplified = self._simplify_ring(ring)
                    if len(simplified) >= 2:
                        outlines.append([[round(lon, 4), round(lat, 4)] for lon, lat in simplified])
        return outlines

    def _bbox_intersects_contiguous(self, xmin: float, ymin: float, xmax: float, ymax: float) -> bool:
        return not (
            xmax < CONTIGUOUS_BOUNDS["min_lon"]
            or xmin > CONTIGUOUS_BOUNDS["max_lon"]
            or ymax < CONTIGUOUS_BOUNDS["min_lat"]
            or ymin > CONTIGUOUS_BOUNDS["max_lat"]
        )

    def _simplify_ring(self, ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(ring) < 2:
            return ring
        simplified = [ring[0]]
        step = 6 if len(ring) > 120 else 3
        for index in range(step, len(ring) - 1, step):
            lon, lat = ring[index]
            if self._point_in_contiguous_bounds(lat, lon):
                simplified.append((lon, lat))
        if self._point_in_contiguous_bounds(ring[-1][1], ring[-1][0]):
            simplified.append(ring[-1])
        return simplified

    def _point_in_contiguous_bounds(self, lat: float, lon: float) -> bool:
        return (
            CONTIGUOUS_BOUNDS["min_lat"] <= lat <= CONTIGUOUS_BOUNDS["max_lat"]
            and CONTIGUOUS_BOUNDS["min_lon"] <= lon <= CONTIGUOUS_BOUNDS["max_lon"]
        )

    def _calibrate_component_scales(self) -> dict[str, float]:
        sample_candidates = self.search_index[: min(80, len(self.search_index))]
        industrial_raw: list[float] = []
        wastewater_raw: list[float] = []
        plastic_raw: list[float] = []
        microfiber_raw: list[float] = []
        for candidate in sample_candidates:
            raw = self._raw_components(candidate.latitude, candidate.longitude)
            industrial_raw.append(raw["industrial_raw"])
            wastewater_raw.append(raw["wastewater_raw"])
            plastic_raw.append(raw["plastic_raw"])
            microfiber_raw.append(raw["microfiber_raw"])

        def scale(values: list[float], floor: float) -> float:
            if not values:
                return floor
            return max(float(np.percentile(values, 90)), floor)

        return {
            "industrial": scale(industrial_raw, 3.0),
            "wastewater": scale(wastewater_raw, 2.4),
            "plastic_waste": scale(plastic_raw, 3.5),
            "microfiber": scale(microfiber_raw, 3.5),
        }

    def _baseline_factors(self, latitude: float, longitude: float, *, include_details: bool = True) -> dict[str, Any]:
        raw = self._raw_components(latitude, longitude)
        industrial_dist = raw["industrial_dist"]
        wastewater_dist = raw["wastewater_dist"]
        industrial_influence = raw["industrial_influence"]
        wastewater_influence = raw["wastewater_influence"]

        industrial_score = self._bounded_score(raw["industrial_raw"], scale=self.component_scales["industrial"])
        wastewater_score = self._bounded_score(raw["wastewater_raw"], scale=self.component_scales["wastewater"])
        plastic_score = self._bounded_score(raw["plastic_raw"], scale=self.component_scales["plastic_waste"])
        microfiber_score = self._bounded_score(raw["microfiber_raw"], scale=self.component_scales["microfiber"])

        factors = {
            "industrial_score": round(industrial_score, 1),
            "wastewater_score": round(wastewater_score, 1),
            "plastic_waste_score": round(plastic_score, 1),
            "microfiber_score": round(microfiber_score, 1),
        }
        total_score = round(self._weighted_total(factors), 1)
        factor_scores = {
            **factors,
            "total_score": total_score,
            "risk_level": self._risk_level(total_score),
        }

        if not include_details:
            return {"factor_scores": factor_scores}

        nearby_sources = self._nearby_sources(industrial_dist, wastewater_dist, industrial_influence, wastewater_influence)
        top_contributors = self._top_contributors(factor_scores)
        explanation = self._build_explanation(top_contributors, factor_scores, nearby_sources)
        recommendations = self._build_recommendations(top_contributors)
        context = {
            "nearby_industrial_facilities": int((industrial_dist <= 50).sum()),
            "nearby_wastewater_plants": int((wastewater_dist <= 30).sum()),
            "nearest_wastewater_distance_km": round(float(np.nanmin(wastewater_dist)), 1) if len(wastewater_dist) else None,
            "nearest_industrial_distance_km": round(float(np.nanmin(industrial_dist)), 1) if len(industrial_dist) else None,
            "population_density_category": self._population_category(raw["urban_raw"]),
        }
        return {
            "factor_scores": factor_scores,
            "nearby_sources": nearby_sources,
            "explanation": explanation,
            "recommendations": recommendations,
            "context": context,
        }

    def _raw_components(self, latitude: float, longitude: float) -> dict[str, Any]:
        industrial_dist = self._haversine_km(
            latitude,
            longitude,
            self.industrial["Latitude"].to_numpy(),
            self.industrial["Longitude"].to_numpy(),
        )
        wastewater_dist = self._haversine_km(
            latitude,
            longitude,
            self.wastewater["LATITUDE"].to_numpy(),
            self.wastewater["LONGITUDE"].to_numpy(),
        )

        industrial_influence = self.industrial["facility_weight"].to_numpy() * np.exp(-industrial_dist / 25.0)
        industrial_influence[industrial_dist > 80.0] = 0.0
        wastewater_influence = self.wastewater["wastewater_weight"].to_numpy() * np.exp(-wastewater_dist / 20.0)
        wastewater_influence[wastewater_dist > 60.0] = 0.0

        industrial_raw = float(industrial_influence.sum())
        wastewater_raw = float(wastewater_influence.sum())
        industrial_pop_proxy = self.industrial["pop_proxy"].fillna(0.0).to_numpy()
        wastewater_pop_proxy = self.wastewater["population_proxy"].fillna(0.0).to_numpy()
        urban_raw = float(
            (np.log1p(industrial_pop_proxy) / 12.0 * np.exp(-industrial_dist / 35.0)).sum()
            + (np.log1p(wastewater_pop_proxy) / 12.0 * np.exp(-wastewater_dist / 35.0)).sum()
        )
        plastic_raw = 0.65 * urban_raw + 0.35 * industrial_raw + 0.20 * wastewater_raw
        microfiber_raw = urban_raw * 1.15
        return {
            "industrial_dist": industrial_dist,
            "wastewater_dist": wastewater_dist,
            "industrial_influence": industrial_influence,
            "wastewater_influence": wastewater_influence,
            "industrial_raw": industrial_raw,
            "wastewater_raw": wastewater_raw,
            "urban_raw": urban_raw,
            "plastic_raw": plastic_raw,
            "microfiber_raw": microfiber_raw,
        }

    def _nearby_sources(
        self,
        industrial_dist: np.ndarray,
        wastewater_dist: np.ndarray,
        industrial_influence: np.ndarray,
        wastewater_influence: np.ndarray,
    ) -> list[dict[str, Any]]:
        industrial_top_idx = np.argsort(industrial_influence)[-3:][::-1]
        wastewater_top_idx = np.argsort(wastewater_influence)[-3:][::-1]
        sources: list[dict[str, Any]] = []
        for idx in industrial_top_idx:
            if industrial_influence[idx] <= 0:
                continue
            row = self.industrial.iloc[int(idx)]
            sources.append(
                {
                    "id": str(int(row["FacilityID"])) if pd.notna(row["FacilityID"]) else row["FacilityName"],
                    "type": "industrial",
                    "name": row["FacilityName"],
                    "latitude": round(float(row["Latitude"]), 5),
                    "longitude": round(float(row["Longitude"]), 5),
                    "distance_km": round(float(industrial_dist[idx]), 1),
                    "source_weight": round(float(industrial_influence[idx]), 3),
                    "summary": f"{row['City']}, {row['State']} industrial release proxy",
                }
            )
        for idx in wastewater_top_idx:
            if wastewater_influence[idx] <= 0:
                continue
            row = self.wastewater.iloc[int(idx)]
            sources.append(
                {
                    "id": f"ww-{row['FACILITY_ID']}",
                    "type": "wastewater",
                    "name": row["FACILITY_NAME"],
                    "latitude": round(float(row["LATITUDE"]), 5),
                    "longitude": round(float(row["LONGITUDE"]), 5),
                    "distance_km": round(float(wastewater_dist[idx]), 1),
                    "source_weight": round(float(wastewater_influence[idx]), 3),
                    "summary": f"{row['CITY']}, {row['STATE_CODE_x']} wastewater treatment footprint",
                }
            )
        return sorted(sources, key=lambda item: item["source_weight"], reverse=True)[:6]

    def _hotspot_markers(self) -> list[dict[str, Any]]:
        hotspots: list[dict[str, Any]] = []
        top_industrial = self.industrial.sort_values("facility_weight", ascending=False).head(18)
        for _, row in top_industrial.iterrows():
            hotspots.append(
                {
                    "id": str(int(row["FacilityID"])) if pd.notna(row["FacilityID"]) else row["FacilityName"],
                    "type": "industrial",
                    "name": row["FacilityName"],
                    "latitude": round(float(row["Latitude"]), 5),
                    "longitude": round(float(row["Longitude"]), 5),
                    "label": f"{row['FacilityName']} ({row['City']}, {row['State']})",
                }
            )
        top_wastewater = self.wastewater.sort_values("wastewater_weight", ascending=False).head(18)
        for _, row in top_wastewater.iterrows():
            hotspots.append(
                {
                    "id": f"ww-{row['FACILITY_ID']}",
                    "type": "wastewater",
                    "name": row["FACILITY_NAME"],
                    "latitude": round(float(row["LATITUDE"]), 5),
                    "longitude": round(float(row["LONGITUDE"]), 5),
                    "label": f"{row['FACILITY_NAME']} ({row['CITY']}, {row['STATE_CODE_x']})",
                }
            )
        return hotspots

    def _build_subtitle(self, baseline: dict[str, Any]) -> str:
        context = baseline["context"]
        return (
            f"{context['population_density_category']} population pressure with "
            f"{context['nearby_industrial_facilities']} nearby industrial facilities and "
            f"{context['nearby_wastewater_plants']} wastewater plants inside the scoring radius."
        )

    def _scenario_summary(self, name: str, total: float, improvement_pct: float, best_intervention: str) -> str:
        return (
            f"{name} falls to {total:.1f}/100 under the current scenario, a {improvement_pct:.1f}% improvement. "
            f"{best_intervention} is the strongest lever here."
        )

    def _comparison_difference(self, risk_a: dict[str, Any], risk_b: dict[str, Any]) -> str:
        diffs = {
            "industrial releases": risk_a["industrial_score"] - risk_b["industrial_score"],
            "wastewater pressure": risk_a["wastewater_score"] - risk_b["wastewater_score"],
            "plastic waste pressure": risk_a["plastic_waste_score"] - risk_b["plastic_waste_score"],
            "microfiber pressure": risk_a["microfiber_score"] - risk_b["microfiber_score"],
        }
        label, delta = max(diffs.items(), key=lambda item: abs(item[1]))
        if delta > 0:
            return f"Location A is more driven by {label}."
        return f"Location B is more driven by {label}."

    def _top_contributors(self, factor_scores: dict[str, float]) -> list[tuple[str, float]]:
        items = [
            ("industrial releases", factor_scores["industrial_score"]),
            ("wastewater pressure", factor_scores["wastewater_score"]),
            ("plastic waste pressure", factor_scores["plastic_waste_score"]),
            ("microfiber exposure", factor_scores["microfiber_score"]),
        ]
        return sorted(items, key=lambda item: item[1], reverse=True)

    def _build_explanation(
        self,
        top_contributors: list[tuple[str, float]],
        factor_scores: dict[str, float],
        nearby_sources: list[dict[str, Any]],
    ) -> list[str]:
        lead = top_contributors[0][0]
        second = top_contributors[1][0]
        source_text = nearby_sources[0]["name"] if nearby_sources else "nearby infrastructure"
        return [
            f"This area is elevated mainly by {lead} and {second}.",
            f"The strongest nearby source in the local model is {source_text}.",
            f"Overall proxy contamination risk is {factor_scores['total_score']:.1f}/100, which lands in the {factor_scores['risk_level'].lower()} band.",
        ]

    def _build_recommendations(self, top_contributors: list[tuple[str, float]]) -> list[str]:
        mapping = {
            "industrial releases": "Target industrial discharge controls and permit enforcement around nearby facilities.",
            "wastewater pressure": "Prioritize wastewater capture, overflow reduction, and treatment upgrades.",
            "plastic waste pressure": "Reduce plastic leakage through stormwater capture and waste-diversion programs.",
            "microfiber exposure": "Promote microfiber filters and household filtration in dense residential blocks.",
        }
        recommendations = [mapping[label] for label, _ in top_contributors[:3]]
        recommendations.append("Use the scenario sliders to confirm which intervention gives the biggest local reduction.")
        return recommendations

    def _population_category(self, urban_raw: float) -> str:
        if urban_raw >= 4.0:
            return "High"
        if urban_raw >= 2.3:
            return "Medium"
        return "Lower"

    def _weighted_total(self, factors: dict[str, float]) -> float:
        return (
            COMPONENT_WEIGHTS["industrial"] * factors["industrial_score"]
            + COMPONENT_WEIGHTS["wastewater"] * factors["wastewater_score"]
            + COMPONENT_WEIGHTS["plastic_waste"] * factors["plastic_waste_score"]
            + COMPONENT_WEIGHTS["microfiber"] * factors["microfiber_score"]
        )

    def _weighted_total_grid(self, factors: dict[str, np.ndarray]) -> np.ndarray:
        return (
            COMPONENT_WEIGHTS["industrial"] * factors["industrial"]
            + COMPONENT_WEIGHTS["wastewater"] * factors["wastewater"]
            + COMPONENT_WEIGHTS["plastic_waste"] * factors["plastic_waste"]
            + COMPONENT_WEIGHTS["microfiber"] * factors["microfiber"]
        )

    def _apply_scenario_to_components(
        self,
        industrial: np.ndarray,
        wastewater: np.ndarray,
        plastic_waste: np.ndarray,
        microfiber: np.ndarray,
        scenario: dict[str, float],
    ) -> dict[str, np.ndarray]:
        return {
            "industrial": industrial * (1 - scenario["industrial_reduction"] / 100.0),
            "wastewater": wastewater * (1 - scenario["wastewater_improvement"] / 100.0),
            "plastic_waste": plastic_waste * (1 - scenario["plastic_reduction"] / 100.0),
            "microfiber": microfiber
            * (1 - scenario["microfiber_filter_adoption"] / 100.0 * MICROFIBER_EFFECTIVENESS),
        }

    def _risk_level(self, score: float) -> str:
        if score >= 70:
            return "High"
        if score >= 40:
            return "Medium"
        return "Low"

    def _normalize_scenario(self, payload: dict[str, Any]) -> dict[str, float]:
        merged = {**DEFAULT_SCENARIO, **(payload or {})}
        return {key: float(min(100.0, max(0.0, merged[key]))) for key in DEFAULT_SCENARIO}

    def _nearest_named_place(self, latitude: float, longitude: float) -> str:
        if not self.search_index:
            return f"Selected Point ({latitude:.3f}, {longitude:.3f})"
        distances = np.array(
            [self._haversine_scalar(latitude, longitude, candidate.latitude, candidate.longitude) for candidate in self.search_index]
        )
        candidate = self.search_index[int(np.argmin(distances))]
        return candidate.name

    def _bounded_score(self, raw_value: float, *, scale: float) -> float:
        return 100.0 * (1.0 - math.exp(-max(raw_value, 0.0) / scale))

    def _parse_coordinates(self, query: str) -> tuple[float, float] | None:
        match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*", query)
        if not match:
            return None
        lat = float(match.group(1))
        lon = float(match.group(2))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
        return lat, lon

    def _haversine_km(
        self,
        latitude: float,
        longitude: float,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> np.ndarray:
        lat1 = np.radians(latitude)
        lon1 = np.radians(longitude)
        lat2 = np.radians(latitudes.astype(float))
        lon2 = np.radians(longitudes.astype(float))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.sqrt(a))
        return 6371.0 * c

    def _haversine_scalar(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        return float(self._haversine_km(lat1, lon1, np.array([lat2]), np.array([lon2]))[0])

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _slug(self, value: str) -> str:
        return self._normalize_text(value).replace(" ", "-")

    def _search_priority(self, kind: str) -> int:
        priorities = {"coordinates": 0, "city": 1, "zip": 2, "county": 3}
        return priorities.get(kind, 9)

    def _minmax(self, values: Any) -> pd.Series:
        series = pd.Series(values, dtype=float)
        spread = series.max() - series.min()
        if spread <= 1e-9:
            return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
        return (series - series.min()) / spread

"""Build wastewater plant dataset with population served and location.

Joins CWNS population, facilities, and location tables to produce
a single parquet with one row per wastewater plant, including:
    - FACILITY_ID, FACILITY_NAME, LATITUDE, LONGITUDE
    - POPULATION_WASTEWATER_CONFIRMED (total served population)

Used by spatial_index.py to compute:
    wastewater_risk = sum(population_served / distance)

Output: data/processed/wastewater_plants.parquet

Usage:
    python scripts/build_wastewater_scores.py
"""

import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
POP_PATH = ROOT / "data" / "processed" / "cwns_population.parquet"
LOC_PATH = ROOT / "data" / "processed" / "cwns_locations.parquet"
FAC_PATH = ROOT / "data" / "processed" / "cwns_facilities.parquet"
OUT_PATH = ROOT / "data" / "processed" / "wastewater_plants.parquet"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print("Loading population data...")
pop = pd.read_parquet(POP_PATH)

print("Loading location data...")
loc = pd.read_parquet(LOC_PATH, columns=["FACILITY_ID", "LATITUDE", "LONGITUDE", "STATE_CODE"])

print("Loading facility names...")
fac = pd.read_parquet(FAC_PATH, columns=["FACILITY_ID", "FACILITY_NAME"])

# ---------------------------------------------------------------------------
# Filter to end facilities (actual treatment plants) and sum population
# ---------------------------------------------------------------------------

end = pop[pop["END_FACILITY"] == "Y"]
print(f"  {len(end):,} end-facility population rows")

by_fac = (
    end.groupby("FACILITY_ID")["NET_TREATED_POPULATION_2022"]
    .sum()
    .reset_index()
    .rename(columns={"NET_TREATED_POPULATION_2022": "POPULATION_WASTEWATER_CONFIRMED"})
)
print(f"  {len(by_fac):,} facilities with population data")

# ---------------------------------------------------------------------------
# Join with location and facility name
# ---------------------------------------------------------------------------

plants = by_fac.merge(loc, on="FACILITY_ID", how="inner")
plants = plants.merge(fac, on="FACILITY_ID", how="left")
plants = plants.dropna(subset=["LATITUDE", "LONGITUDE"])
plants = plants[plants["POPULATION_WASTEWATER_CONFIRMED"] > 0]

# Filter to Connecticut
plants_ct = plants[plants["STATE_CODE"] == "CT"].copy()
print(f"  {len(plants_ct):,} CT wastewater plants with population > 0")

result = plants_ct[[
    "FACILITY_ID", "FACILITY_NAME", "LATITUDE", "LONGITUDE",
    "POPULATION_WASTEWATER_CONFIRMED",
]].sort_values("POPULATION_WASTEWATER_CONFIRMED", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUT_PATH, index=False)

print(f"\nSaved {len(result):,} plants → {OUT_PATH}")
print(result.head(10).to_string(index=False))

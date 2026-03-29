"""Precompute blended risk scores for each facility location.

Uses spatial_index.FacilityIndex.compute_location_risk() to produce a
per-facility score that blends industrial releases, stream contamination,
wastewater plant proximity, and SDWA drinking water violations.

Output: data/processed/facility_risk_scores.parquet
    FacilityNumber, risk_score, facility_score, downstream_score,
    wastewater_score, sdwa_score

Usage:
    python scripts/build_risk_scores.py
"""

import pandas as pd
from pathlib import Path
from scripts.spatial_index import FacilityIndex

ROOT = Path(__file__).resolve().parent.parent
FAC_PATH = ROOT / "data" / "processed" / "facilities_clean.parquet"
OUT_PATH = ROOT / "data" / "processed" / "facility_risk_scores.parquet"

print("Loading facility index...")
idx = FacilityIndex()

print("Loading facilities...")
fac = pd.read_parquet(FAC_PATH, columns=["FacilityNumber", "Latitude", "Longitude"])
fac = fac.dropna(subset=["Latitude", "Longitude"])

print(f"Computing risk for {len(fac):,} facilities...")
rows = []
for i, (_, row) in enumerate(fac.iterrows()):
    r = idx.compute_location_risk(row["Latitude"], row["Longitude"], radius_km=25)
    rows.append({
        "FacilityNumber": row["FacilityNumber"],
        "risk_score": r.score,
        "facility_score": r.facility_score,
        "downstream_score": r.downstream_score,
        "wastewater_score": r.wastewater_score,
        "sdwa_score": r.sdwa_score,
    })
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(fac)}")

result = pd.DataFrame(rows)
result = result.sort_values("risk_score", ascending=False).reset_index(drop=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUT_PATH, index=False)

print(f"\nSaved {len(result):,} → {OUT_PATH}")
print(result.head(10).to_string(index=False))
print(f"\nScore distribution:")
print(result["risk_score"].describe().round(2).to_string())

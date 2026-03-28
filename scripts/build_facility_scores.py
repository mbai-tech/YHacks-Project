"""Join facility data with release-element data and compute water impact scores.

Assumptions:
    - data/interim/release_elements.parquet contains a FacilityNumber column
      (standard in RSEI release_data — links releases back to facilities).
    - Score is a numeric column from the elements join.
    - Media values for water follow RSEI conventions (MediaCode in {5,6,7,8}
      or MediaText containing 'Water'); adjust WATER_MEDIA_CODES if needed.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FACILITY_PATH     = ROOT / "data" / "processed"  / "facility.parquet"
RELEASES_PATH     = ROOT / "data" / "interim"    / "release_elements.parquet"
OUT_PATH          = ROOT / "data" / "processed"  / "facility_scores.parquet"

# RSEI media codes that represent water pathways
WATER_MEDIA_CODES = {5, 6, 7, 8}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

facility = pd.read_parquet(
    FACILITY_PATH,
    columns=["FacilityNumber", "FacilityName", "Latitude", "Longitude"],
)

releases = pd.read_parquet(RELEASES_PATH)  # must include FacilityNumber, Media, Score

# ---------------------------------------------------------------------------
# Filter to water releases
# ---------------------------------------------------------------------------

if pd.api.types.is_integer_dtype(releases["Media"]):
    water = releases[releases["Media"].isin(WATER_MEDIA_CODES)]
else:
    # fallback: text match
    water = releases[releases["Media"].str.contains("water", case=False, na=False)]

print(f"Water releases: {len(water):,} of {len(releases):,} total")

# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------

merged = water.merge(facility, on="FacilityNumber", how="inner")

result = merged[["FacilityNumber", "FacilityName", "Latitude", "Longitude", "Score", "Media"]]

# ---------------------------------------------------------------------------
# Compute TotalWaterImpactScore per facility
# ---------------------------------------------------------------------------

scores = (
    result.groupby(["FacilityNumber", "FacilityName", "Latitude", "Longitude"], as_index=False)
    ["Score"].sum()
    .rename(columns={"Score": "TotalWaterImpactScore"})
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
scores.to_parquet(OUT_PATH, index=False)

print(f"Saved {len(scores):,} facilities → {OUT_PATH}")
print(scores.sort_values("TotalWaterImpactScore", ascending=False).head(10).to_string(index=False))

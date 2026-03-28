"""Build stream segment scores by grouping facilities by FinalCOMID.

Approach:
    - Join facility.parquet to aggmicro.parquet on shared (X, Y) grid coordinates
      to obtain toxicity concentrations per facility location.
    - Filter to facilities flagged as having water releases.
    - Group by FinalCOMID and sum ToxConc across all facilities on that segment.
    - Apply segment_score = log(1 + total_concentration).

Output: data/processed/stream_scores.parquet
    ComID, facility_count, total_concentration, segment_score
"""

import math
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
FAC_PATH = ROOT / "data" / "processed" / "facility.parquet"
AGG_PATH = ROOT / "data" / "processed" / "aggmicro.parquet"
OUT_PATH = ROOT / "data" / "processed" / "stream_scores.parquet"

FAC_COLS = ["FacilityNumber", "FacilityName", "FinalCOMID", "WaterReleases", "X", "Y", "State"]
AGG_COLS = ["X", "Y", "ToxConc", "CTConc", "NCTConc"]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print("Loading facility data...")
fac = pd.read_parquet(FAC_PATH, columns=FAC_COLS)

print("Loading aggmicro data...")
agg = pd.read_parquet(AGG_PATH, columns=AGG_COLS)

# ---------------------------------------------------------------------------
# Join facility → aggmicro on grid coordinates
# ---------------------------------------------------------------------------

fac = fac[fac["State"] == "CT"]
print(f"  {len(fac):,} Connecticut facilities")

print("Joining on (X, Y)...")
merged = fac.merge(agg, on=["X", "Y"], how="inner")
print(f"  {len(merged):,} rows after join")

# ---------------------------------------------------------------------------
# Filter: water-releasing facilities with a valid ComID
# ---------------------------------------------------------------------------

water = merged[merged["WaterReleases"] == True].copy()
water = water.dropna(subset=["FinalCOMID"])
water["FinalCOMID"] = water["FinalCOMID"].astype("int64")
print(f"  {len(water):,} water-releasing facilities with a valid ComID")

# ---------------------------------------------------------------------------
# Aggregate by ComID
# ---------------------------------------------------------------------------

stream = (
    water.groupby("FinalCOMID")
    .agg(
        facility_count   = ("FacilityNumber", "nunique"),
        total_tox_conc   = ("ToxConc",  "sum"),
        total_ct_conc    = ("CTConc",   "sum"),
        total_nct_conc   = ("NCTConc",  "sum"),
    )
    .reset_index()
)

# Total concentration = sum of all toxicity pathways
stream["total_concentration"] = stream["total_tox_conc"]

# segment_score = log(1 + total_concentration)
stream["segment_score"] = stream["total_concentration"].apply(lambda x: math.log1p(x))

stream = stream.sort_values("segment_score", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
stream.to_parquet(OUT_PATH, index=False)

print(f"\nSaved {len(stream):,} stream segments → {OUT_PATH}")
print()
print("Score distribution:")
print(stream["segment_score"].describe().round(3).to_string())
print()
print("Top 10 segments by score:")
print(stream.head(10).to_string(index=False))

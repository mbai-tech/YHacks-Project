"""Filter and clean RSEI facility data for Connecticut."""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "processed" / "facility.parquet"
OUT_PATH = ROOT / "data" / "interim" / "facilities_clean.parquet"

COLS = [
    "FacilityNumber",
    "FacilityName",
    "Latitude",
    "Longitude",
    "State",
    "County",
    "ModeledNAICS",
    "WaterReleases",
    "FinalCOMID",
]

df = pd.read_parquet(IN_PATH, columns=COLS)

df = df[df["State"] == "CT"]
df = df.dropna(subset=["Latitude", "Longitude"])

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PATH, index=False)

print(f"Saved {len(df):,} facilities → {OUT_PATH}")
print(df.dtypes)
print(df.head())

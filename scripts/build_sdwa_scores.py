"""Build per-water-system SDWA violation risk scores.

Formula (from hackathon spec):
    violation_score = 3 * health_based + 1 * monitoring_reporting
                    + 1.5 * public_notice + 0.5 * other
    population_factor = log(1 + population_served)
    sdwa_system_risk = violation_score * population_factor

Violation code categories (EPA SDWIS convention):
    01-19  → Health-based (MCL / treatment technique)
    20-29  → Monitoring & reporting
    30-39  → Public notification
    40+    → Other

Output: data/processed/sdwa_scores.parquet
    PWSID, PWS_NAME, CITY_NAME, ZIP_CODE, POPULATION_SERVED_COUNT,
    violation_score, sdwa_system_risk

Usage:
    python scripts/build_sdwa_scores.py
"""

import math
import pandas as pd
import pgeocode
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
PWS_PATH = ROOT / "data" / "processed" / "sdwa_water_systems.parquet"
VIO_PATH = ROOT / "data" / "processed" / "sdwa_violations.parquet"
OUT_PATH = ROOT / "data" / "processed" / "sdwa_scores.parquet"

# Severity weights by violation category
WEIGHTS = {
    "health_based": 3.0,
    "monitoring_reporting": 1.0,
    "public_notice": 1.5,
    "other": 0.5,
}


def _categorise_code(code: str) -> str:
    """Map SDWA violation code to a category."""
    # Strip letters, get leading numeric part
    num = ""
    for ch in str(code):
        if ch.isdigit():
            num += ch
        else:
            break
    if not num:
        return "other"
    n = int(num)
    if 1 <= n <= 19:
        return "health_based"
    elif 20 <= n <= 29:
        return "monitoring_reporting"
    elif 30 <= n <= 39:
        return "public_notice"
    else:
        return "other"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print("Loading public water systems...")
pws = pd.read_parquet(
    PWS_PATH,
    columns=["PWSID", "PWS_NAME", "CITY_NAME", "ZIP_CODE",
             "STATE_CODE", "POPULATION_SERVED_COUNT", "PWS_ACTIVITY_CODE"],
)

print("Loading violations...")
vio = pd.read_parquet(VIO_PATH, columns=["PWSID", "VIOLATION_CODE"])

# ---------------------------------------------------------------------------
# Filter to CT active systems
# ---------------------------------------------------------------------------

pws = pws[pws["STATE_CODE"] == "CT"]
pws = pws[pws["PWS_ACTIVITY_CODE"] == "A"]
pws = pws.drop(columns=["STATE_CODE", "PWS_ACTIVITY_CODE"])
print(f"  {len(pws):,} active CT water systems")

# ---------------------------------------------------------------------------
# Categorise and weight violations
# ---------------------------------------------------------------------------

vio = vio[vio["PWSID"].isin(pws["PWSID"])].copy()
print(f"  {len(vio):,} violations for CT systems")

vio["category"] = vio["VIOLATION_CODE"].apply(_categorise_code)
vio["weight"] = vio["category"].map(WEIGHTS)

# Count per category per system
counts = (
    vio.groupby(["PWSID", "category"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["health_based", "monitoring_reporting", "public_notice", "other"], fill_value=0)
)

# Weighted violation score
counts["violation_score"] = (
    WEIGHTS["health_based"] * counts["health_based"]
    + WEIGHTS["monitoring_reporting"] * counts["monitoring_reporting"]
    + WEIGHTS["public_notice"] * counts["public_notice"]
    + WEIGHTS["other"] * counts["other"]
)
counts = counts.reset_index()

# ---------------------------------------------------------------------------
# Join with PWS metadata and compute system risk
# ---------------------------------------------------------------------------

scores = pws.merge(counts[["PWSID", "violation_score"]], on="PWSID", how="left")
scores["violation_score"] = scores["violation_score"].fillna(0.0)
scores["POPULATION_SERVED_COUNT"] = pd.to_numeric(scores["POPULATION_SERVED_COUNT"], errors="coerce").fillna(0)

scores["population_factor"] = scores["POPULATION_SERVED_COUNT"].apply(lambda x: math.log1p(x))
scores["sdwa_system_risk"] = scores["violation_score"] * scores["population_factor"]

# Keep only systems with violations
scores = scores[scores["violation_score"] > 0].copy()

# ---------------------------------------------------------------------------
# Geocode via ZIP code centroid
# ---------------------------------------------------------------------------

print("Geocoding by ZIP code...")
nomi = pgeocode.Nominatim("us")
scores["zip5"] = scores["ZIP_CODE"].str[:5]
unique_zips = scores["zip5"].dropna().unique()
geo = nomi.query_postal_code(unique_zips.tolist())
zip_coords = geo[["postal_code", "latitude", "longitude"]].dropna()
zip_coords = zip_coords.rename(columns={"postal_code": "zip5"})

scores = scores.merge(zip_coords, on="zip5", how="left")
scores = scores.drop(columns=["zip5"])
scores = scores.rename(columns={"latitude": "LATITUDE", "longitude": "LONGITUDE"})

geocoded = scores["LATITUDE"].notna().sum()
print(f"  Geocoded {geocoded}/{len(scores)} systems")

scores = scores.dropna(subset=["LATITUDE", "LONGITUDE"])
scores = scores.sort_values("sdwa_system_risk", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
scores.to_parquet(OUT_PATH, index=False)

print(f"\nSaved {len(scores):,} systems → {OUT_PATH}")
print(scores.head(10).to_string(index=False))

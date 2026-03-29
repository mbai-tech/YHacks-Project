"""Generate future-prediction fine-tuning pairs by extrapolating 2020→2022 trends."""

import json
import math
import random
import pandas as pd
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "data" / "mlx_train"
random.seed(99)

FUTURE_YEARS = [2026, 2028, 2030, 2035, 2040, 2045, 2050]
BASE_YEAR    = 2022
TREND_SPAN   = 2   # years between 2020 and 2022

AGG_COLS = ["X", "Y", "ToxConc", "CTConc", "NCTConc"]

def load_aggmicro(path):
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path, columns=AGG_COLS)
    return pd.read_csv(path, skiprows=2, header=None,
        names=["GridCode","X","Y","NumFacs","NumReleases","NumChems",
               "ToxConc","Score","Pop","CTConc","NCTConc"],
        dtype={"GridCode":"Int16","X":"Int16","Y":"Int16",
               "ToxConc":"float64","CTConc":"float64","NCTConc":"float64"},
        usecols=["X","Y","ToxConc","CTConc","NCTConc"])

def risk_level(score):
    if score > 10: return "high"
    if score > 7:  return "moderate"
    return "low"

def uncertainty_qualifier(years_out):
    """Return a hedge phrase that grows with prediction horizon."""
    if years_out <= 4:
        return "Based on the 2020–2022 trend, projections suggest"
    if years_out <= 10:
        return "Extrapolating the observed trend with moderate uncertainty,"
    if years_out <= 20:
        return "Under a business-as-usual scenario (high uncertainty),"
    return "As a long-range scenario with substantial uncertainty,"

def uncertainty_note(years_out):
    if years_out <= 4:
        return ("Projections assume continued operations at the recent rate. "
                "Regulatory changes or facility closures could alter this.")
    if years_out <= 10:
        return ("Medium-term projections carry moderate uncertainty. "
                "Policy interventions, economic shifts, or technology upgrades "
                "could significantly change actual outcomes.")
    return ("Long-range projections are highly speculative. "
            "Climate policy, industrial transitions, and regulatory tightening "
            "under the Clean Water Act could substantially change actual risk levels.")

def project(tox_base, annual_rate, years_out):
    """Compound projection with a floor at 0 and a reasonable cap."""
    raw = tox_base * ((1 + annual_rate) ** years_out)
    return max(0.0, raw)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading facility and aggmicro data...")

fac = pd.read_parquet(ROOT / "data" / "processed" / "facility.parquet",
    columns=["FacilityNumber","FacilityName","State","FinalCOMID",
             "WaterReleases","X","Y","ModeledNAICS","County"])
ct = fac[fac["State"] == "CT"].copy()

agg_2020 = load_aggmicro(ROOT / "data" / "raw" / "aggmicro2022_2020.csv")
agg_2022 = load_aggmicro(ROOT / "data" / "processed" / "aggmicro.parquet")

fac_2020 = ct.merge(agg_2020, on=["X","Y"], how="inner")
fac_2020 = fac_2020[fac_2020["WaterReleases"] == True].dropna(subset=["FinalCOMID"])
fac_2020["ToxConc"] = fac_2020["ToxConc"].fillna(0)

fac_2022 = ct.merge(agg_2022, on=["X","Y"], how="inner")
fac_2022 = fac_2022[fac_2022["WaterReleases"] == True].dropna(subset=["FinalCOMID"])
fac_2022["ToxConc"] = fac_2022["ToxConc"].fillna(0)

fac_2020 = fac_2020.set_index("FacilityNumber")
fac_2022 = fac_2022.set_index("FacilityNumber")
common_facs = fac_2020.index.intersection(fac_2022.index)
print(f"  Facilities with 2020+2022 data: {len(common_facs)}")

# ---------------------------------------------------------------------------
# Build per-facility trend table
# ---------------------------------------------------------------------------

records = []
for fnum in common_facs:
    r20 = fac_2020.loc[fnum]
    r22 = fac_2022.loc[fnum]
    tox20 = float(r20["ToxConc"]) if not pd.isna(r20["ToxConc"]) else 0
    tox22 = float(r22["ToxConc"]) if not pd.isna(r22["ToxConc"]) else 0
    if tox20 <= 0:
        continue
    # Annual rate of change (geometric)
    annual_rate = (tox22 / tox20) ** (1 / TREND_SPAN) - 1
    # Clamp extreme rates: cap at ±30% per year to avoid absurd extrapolation
    annual_rate = max(-0.30, min(0.30, annual_rate))
    records.append({
        "FacilityNumber": fnum,
        "FacilityName":   r22["FacilityName"],
        "County":         r22["County"],
        "FinalCOMID":     r22["FinalCOMID"],
        "tox20": tox20,
        "tox22": tox22,
        "annual_rate": annual_rate,
    })

trends = pd.DataFrame(records)
print(f"  Facilities with valid trends: {len(trends)}")

# ---------------------------------------------------------------------------
# Build per-segment trend table
# ---------------------------------------------------------------------------

def build_seg(fac_df):
    seg = fac_df.reset_index().groupby("FinalCOMID").agg(
        total_tox=("ToxConc","sum"),
        facility_count=("FacilityName","count"),
    ).reset_index()
    seg["score"] = seg["total_tox"].apply(lambda x: math.log10(1 + x) if x > 0 else 0)
    return seg.set_index("FinalCOMID")

seg_2020 = build_seg(fac_2020)
seg_2022 = build_seg(fac_2022)
common_segs = seg_2020.index.intersection(seg_2022.index)

seg_trends = []
for comid in common_segs:
    t20 = seg_2020.loc[comid, "total_tox"]
    t22 = seg_2022.loc[comid, "total_tox"]
    if t20 <= 0:
        continue
    annual_rate = (t22 / t20) ** (1 / TREND_SPAN) - 1
    annual_rate = max(-0.30, min(0.30, annual_rate))
    # Sample facility names on this segment
    facs_on = fac_2022.reset_index()
    facs_on = facs_on[facs_on["FinalCOMID"] == comid]["FacilityName"].tolist()[:3]
    seg_trends.append({
        "FinalCOMID":    comid,
        "tox22":         t22,
        "score22":       seg_2022.loc[comid, "score"],
        "annual_rate":   annual_rate,
        "fac_str":       ", ".join(facs_on) if facs_on else "multiple facilities",
    })

seg_trends = pd.DataFrame(seg_trends)
print(f"  Segments with valid trends: {len(seg_trends)}")

# ---------------------------------------------------------------------------
# Generate Q&A pairs
# ---------------------------------------------------------------------------

pairs = []

def add(prompt, completion):
    pairs.append({
        "text": f"<|user|>\n{prompt.strip()}\n<|assistant|>\n{completion.strip()}"
    })

# ── Per-facility future projections ─────────────────────────────────────────

for _, row in trends.iterrows():
    name   = row["FacilityName"]
    county = row["County"]
    rate   = row["annual_rate"]
    tox22  = row["tox22"]
    rate_pct = rate * 100

    for year in FUTURE_YEARS:
        years_out = year - BASE_YEAR
        proj = project(tox22, rate, years_out)
        qualifier = uncertainty_qualifier(years_out)
        note = uncertainty_note(years_out)
        direction = "increase" if rate > 0.01 else "decrease" if rate < -0.01 else "remain stable"

        add(
            f"What is the projected water contamination risk for {name} in {county} County in {year}?",
            f"{qualifier} {name} ({county} County) will {direction} its toxicity concentration "
            f"to approximately {proj:,.0f} units by {year}, compared to {tox22:,.0f} in 2022. "
            f"This projection assumes the observed annual rate of {rate_pct:+.1f}% continues from 2022 to {year}. "
            f"{note}"
        )

    # "When will X exceed threshold" pairs for facilities on an upward trend
    if rate > 0.02 and tox22 > 100:
        thresholds = [tox22 * 2, tox22 * 5]
        for thresh in thresholds:
            # Solve: thresh = tox22 * (1+rate)^n → n = log(thresh/tox22)/log(1+rate)
            n = math.log(thresh / tox22) / math.log(1 + rate)
            reach_year = int(BASE_YEAR + n)
            if reach_year > 2050:
                continue
            multiplier = thresh / tox22
            add(
                f"At its current trend, when will {name} reach {thresh:,.0f} toxicity units?",
                f"If {name} continues its current annual growth rate of {rate_pct:.1f}%, "
                f"it would reach {thresh:,.0f} toxicity units (approximately {multiplier:.0f}× the 2022 level) "
                f"around {reach_year}. This assumes no regulatory intervention, facility changes, or "
                f"chemical substitution — all of which could alter this trajectory."
            )

# ── Per-segment future projections ──────────────────────────────────────────

for _, row in seg_trends.sample(min(len(seg_trends), 200), random_state=42).iterrows():
    comid    = row["FinalCOMID"]
    tox22    = row["tox22"]
    score22  = row["score22"]
    rate     = row["annual_rate"]
    fac_str  = row["fac_str"]

    for year in random.sample(FUTURE_YEARS, min(3, len(FUTURE_YEARS))):
        years_out = year - BASE_YEAR
        proj_tox  = project(tox22, rate, years_out)
        proj_score = math.log10(1 + proj_tox) if proj_tox > 0 else 0
        qualifier  = uncertainty_qualifier(years_out)
        note       = uncertainty_note(years_out)
        direction  = "increase" if rate > 0.01 else "decrease" if rate < -0.01 else "remain stable"

        add(
            f"What is the projected risk score for stream segment {comid} in {year}?",
            f"{qualifier} stream segment {comid} ({fac_str}) will {direction} "
            f"to a toxicity concentration of {proj_tox:,.0f} and a risk score of {proj_score:.2f} by {year}. "
            f"In 2022 the score was {score22:.2f}. "
            f"The projected {year} risk level is {risk_level(proj_score)}. "
            f"{note}"
        )

# ── County-level future projections ─────────────────────────────────────────

for county in trends["County"].dropna().unique():
    c_rows = trends[trends["County"] == county]
    if len(c_rows) == 0:
        continue
    tox22_total = c_rows["tox22"].sum()
    # Weighted average annual rate
    avg_rate = (c_rows["annual_rate"] * c_rows["tox22"]).sum() / tox22_total if tox22_total > 0 else 0
    avg_rate = max(-0.30, min(0.30, avg_rate))
    n_facs = len(c_rows)

    for year in FUTURE_YEARS:
        years_out = year - BASE_YEAR
        proj = project(tox22_total, avg_rate, years_out)
        qualifier = uncertainty_qualifier(years_out)
        note = uncertainty_note(years_out)
        direction = "worsen" if avg_rate > 0.01 else "improve" if avg_rate < -0.01 else "remain relatively stable"

        add(
            f"What is the projected water contamination risk for {county} County in {year}?",
            f"{qualifier} {county} County's total water contamination toxicity will {direction}, "
            f"reaching approximately {proj:,.0f} units by {year} across {n_facs} tracked facilities. "
            f"The 2022 baseline was {tox22_total:,.0f} units, with a weighted annual trend of {avg_rate*100:+.1f}%. "
            f"{note}"
        )

# ── Statewide future summary ─────────────────────────────────────────────────

ct_tox22 = trends["tox22"].sum()
ct_avg_rate = (trends["annual_rate"] * trends["tox22"]).sum() / ct_tox22 if ct_tox22 > 0 else 0
ct_avg_rate = max(-0.30, min(0.30, ct_avg_rate))

for year in FUTURE_YEARS:
    years_out = year - BASE_YEAR
    proj = project(ct_tox22, ct_avg_rate, years_out)
    qualifier = uncertainty_qualifier(years_out)
    note = uncertainty_note(years_out)
    direction = "increase" if ct_avg_rate > 0.01 else "decrease" if ct_avg_rate < -0.01 else "remain stable"

    add(
        f"What is the projected overall water contamination risk for Connecticut in {year}?",
        f"{qualifier} Connecticut's statewide industrial water contamination toxicity will {direction} "
        f"to approximately {proj:,.0f} units by {year}. "
        f"The 2022 baseline across all tracked CT facilities was {ct_tox22:,.0f} units, "
        f"with a weighted annual trend of {ct_avg_rate*100:+.1f}%. "
        f"{note}"
    )

# ── Highest-risk future facilities ──────────────────────────────────────────

for year in [2030, 2040, 2050]:
    years_out = year - BASE_YEAR
    temp = trends.copy()
    temp[f"proj_{year}"] = temp.apply(
        lambda r: project(r["tox22"], r["annual_rate"], years_out), axis=1
    )
    top5 = temp.nlargest(5, f"proj_{year}")
    top5_str = "; ".join(
        f"{r['FacilityName']} ({r['County']}, projected {r[f'proj_{year}']:,.0f})"
        for _, r in top5.iterrows()
    )
    qualifier = uncertainty_qualifier(years_out)
    add(
        f"Which Connecticut facilities are projected to have the highest water contamination risk in {year}?",
        f"{qualifier} the highest-risk CT facilities in {year} will be: {top5_str}. "
        f"These projections are based on each facility's 2020–2022 annual trend rate. "
        f"{uncertainty_note(years_out)}"
    )

# ---------------------------------------------------------------------------
# Write output — merge with existing splits
# ---------------------------------------------------------------------------

print(f"\nNew future-prediction pairs: {len(pairs)}")

existing = []
for split in ["train", "valid", "test"]:
    path = OUTDIR / f"{split}.jsonl"
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append((split, json.loads(line)))

print(f"Existing pairs: {len(existing)}")

random.shuffle(pairs)
n_val  = max(30, int(len(pairs) * 0.1))
n_test = max(15, int(len(pairs) * 0.05))

new_splits = {
    "valid": pairs[:n_val],
    "test":  pairs[n_val:n_val + n_test],
    "train": pairs[n_val + n_test:],
}

for split in ["train", "valid", "test"]:
    combined = [r for s, r in existing if s == split] + new_splits[split]
    random.shuffle(combined)
    with open(OUTDIR / f"{split}.jsonl", "w") as f:
        for r in combined:
            f.write(json.dumps(r) + "\n")
    print(f"{split}: {len(combined)} total records")

print(f"\nDone. Files updated in {OUTDIR}")

"""Generate temporal fine-tuning pairs from multi-year aggmicro data."""

import json
import math
import random
import pandas as pd
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "data" / "mlx_train"
random.seed(42)

# ---------------------------------------------------------------------------
# Load base data
# ---------------------------------------------------------------------------

fac = pd.read_parquet(ROOT / "data" / "processed" / "facility.parquet",
    columns=["FacilityNumber","FacilityName","State","FinalCOMID",
             "WaterReleases","X","Y","ModeledNAICS","County"])
ct = fac[fac["State"] == "CT"].copy()

AGG_FILES = {
    2020: ROOT / "data" / "raw" / "aggmicro2022_2020.csv",
    2022: ROOT / "data" / "processed" / "aggmicro.parquet",
}

AGG_COLS = ["X","Y","ToxConc","CTConc","NCTConc"]

def load_aggmicro(path):
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path, columns=AGG_COLS)
    return pd.read_csv(path, skiprows=2, header=None,
        names=["GridCode","X","Y","NumFacs","NumReleases","NumChems",
               "ToxConc","Score","Pop","CTConc","NCTConc"],
        dtype={"GridCode":"Int16","X":"Int16","Y":"Int16",
               "ToxConc":"float64","CTConc":"float64","NCTConc":"float64"},
        usecols=["X","Y","ToxConc","CTConc","NCTConc"])

def build_scores(agg_df):
    merged = ct.merge(agg_df, on=["X","Y"], how="inner")
    merged = merged[merged["WaterReleases"] == True].dropna(subset=["FinalCOMID"])
    merged["ToxConc"] = merged["ToxConc"].fillna(0)

    seg = merged.groupby("FinalCOMID").agg(
        facility_count=("FacilityName","count"),
        total_tox=("ToxConc","sum"),
        total_ct=("CTConc","sum"),
        total_nct=("NCTConc","sum"),
    ).reset_index()
    seg["score"] = seg["total_tox"].apply(lambda x: math.log10(1 + x) if x > 0 else 0)

    max_tox = seg["total_tox"].max()
    seg["weight"] = seg["total_tox"] / max_tox if max_tox > 0 else 0
    return merged, seg

print("Loading aggmicro files...")
year_data = {}
for year, path in AGG_FILES.items():
    print(f"  {year}...")
    agg = load_aggmicro(path)
    fac_merged, seg_scores = build_scores(agg)
    year_data[year] = {"fac": fac_merged, "seg": seg_scores}
    print(f"    {len(seg_scores)} segments, {len(fac_merged)} facility rows")

years = sorted(year_data.keys())

# ---------------------------------------------------------------------------
# Build per-facility and per-segment temporal pairs
# ---------------------------------------------------------------------------

pairs = []

def add(prompt, completion):
    pairs.append({
        "text": f"<|user|>\n{prompt.strip()}\n<|assistant|>\n{completion.strip()}"
    })

def trend_word(delta_pct):
    if delta_pct > 20:   return "significantly increased"
    if delta_pct > 5:    return "moderately increased"
    if delta_pct > -5:   return "remained relatively stable"
    if delta_pct > -20:  return "moderately decreased"
    return "significantly decreased"

def risk_level(score):
    if score > 10: return "high"
    if score > 7:  return "moderate"
    return "low"

# ── Per-segment year-over-year ────────────────────────────────────────────────

seg_2020 = year_data[2020]["seg"].set_index("FinalCOMID")
seg_2022 = year_data[2022]["seg"].set_index("FinalCOMID")

common_segs = seg_2020.index.intersection(seg_2022.index)
print(f"\nSegments in both years: {len(common_segs)}")

for comid in common_segs:
    s20 = seg_2020.loc[comid]
    s22 = seg_2022.loc[comid]

    score20 = s20["score"]
    score22 = s22["score"]
    tox20   = s20["total_tox"]
    tox22   = s22["total_tox"]

    if tox20 == 0:
        continue

    delta_pct = (tox22 - tox20) / tox20 * 100
    trend = trend_word(delta_pct)

    # Facilities on this segment
    facs_on = year_data[2022]["fac"]
    facs_on = facs_on[facs_on["FinalCOMID"] == comid]["FacilityName"].tolist()[:3]
    fac_str = ", ".join(facs_on) if facs_on else "unknown facilities"

    add(
        f"How did water contamination risk change on stream segment {comid} between 2020 and 2022?",
        f"Stream segment {comid} ({fac_str}) saw risk {trend} from 2020 to 2022. "
        f"The toxicity concentration changed from {tox20:,.0f} to {tox22:,.0f} "
        f"({delta_pct:+.1f}%), and the segment score moved from {score20:.2f} to {score22:.2f}. "
        f"The 2022 risk level is {risk_level(score22)}."
    )

    add(
        f"Given that stream segment {comid} had a score of {score20:.2f} in 2020, "
        f"what would you predict for 2022?",
        f"Based on observed trends, the score for segment {comid} in 2022 was {score22:.2f} "
        f"({'higher' if score22 > score20 else 'lower'} than 2020). "
        f"The toxicity concentration {trend} by {abs(delta_pct):.1f}%, "
        f"from {tox20:,.0f} to {tox22:,.0f} units. "
        f"{'Industrial activity on this reach intensified.' if delta_pct > 5 else 'Releases on this reach declined or stabilized.' if delta_pct < -5 else 'Conditions on this reach were largely unchanged.'}"
    )


# ── Per-facility year-over-year ───────────────────────────────────────────────

fac_2020 = year_data[2020]["fac"].set_index("FacilityNumber")
fac_2022 = year_data[2022]["fac"].set_index("FacilityNumber")

common_facs = fac_2020.index.intersection(fac_2022.index)
print(f"Facilities in both years: {len(common_facs)}")

for fnum in common_facs:
    r20 = fac_2020.loc[fnum]
    r22 = fac_2022.loc[fnum]

    tox20 = float(r20["ToxConc"]) if not pd.isna(r20["ToxConc"]) else 0
    tox22 = float(r22["ToxConc"]) if not pd.isna(r22["ToxConc"]) else 0

    if tox20 == 0:
        continue

    name   = r22["FacilityName"]
    county = r22["County"]
    delta_pct = (tox22 - tox20) / tox20 * 100
    trend = trend_word(delta_pct)

    add(
        f"How did {name} in {county} County change its water contamination impact between 2020 and 2022?",
        f"{name} ({county} County) {trend} its water contamination impact between 2020 and 2022. "
        f"Toxicity concentration at its grid cell went from {tox20:,.0f} to {tox22:,.0f} "
        f"({delta_pct:+.1f}%). "
        f"{'This represents a meaningful worsening — the facility may have increased production or changed chemical use.' if delta_pct > 10 else 'This represents a meaningful improvement, possibly due to process changes or reduced production.' if delta_pct < -10 else 'The change was minor, suggesting stable operations.'}"
    )

    add(
        f"If {name} had a toxicity concentration of {tox20:,.0f} in 2020, "
        f"what was its 2022 level?",
        f"In 2022, {name}'s toxicity concentration was {tox22:,.0f} — "
        f"a {abs(delta_pct):.1f}% {'increase' if delta_pct > 0 else 'decrease'} from the 2020 level of {tox20:,.0f}."
    )


# ── County-level trend summary ────────────────────────────────────────────────

for county in fac_2022["County"].dropna().unique():
    c20 = fac_2020[fac_2020["County"] == county]["ToxConc"].sum()
    c22 = fac_2022[fac_2022["County"] == county]["ToxConc"].sum()
    if c20 == 0:
        continue
    delta_pct = (c22 - c20) / c20 * 100
    n_facs = len(fac_2022[fac_2022["County"] == county])

    add(
        f"How has water contamination risk in {county} County changed between 2020 and 2022?",
        f"{county} County's total toxicity concentration across {n_facs} facilities "
        f"{trend_word(delta_pct)} from 2020 to 2022, changing from {c20:,.0f} to {c22:,.0f} "
        f"({delta_pct:+.1f}%). "
        f"{'This county shows a worsening trend that warrants increased regulatory attention.' if delta_pct > 10 else 'This county shows improvement in industrial water releases.' if delta_pct < -10 else 'Risk levels in this county remained broadly stable over this period.'}"
    )


# ── Most improved / most worsened ─────────────────────────────────────────────

deltas = []
for fnum in common_facs:
    r20 = fac_2020.loc[fnum]
    r22 = fac_2022.loc[fnum]
    tox20 = float(r20["ToxConc"]) if not pd.isna(r20["ToxConc"]) else 0
    tox22 = float(r22["ToxConc"]) if not pd.isna(r22["ToxConc"]) else 0
    if tox20 == 0:
        continue
    delta_pct = (tox22 - tox20) / tox20 * 100
    deltas.append({
        "name": r22["FacilityName"],
        "county": r22["County"],
        "tox20": tox20,
        "tox22": tox22,
        "delta_pct": delta_pct,
    })

deltas_df = pd.DataFrame(deltas).dropna()
top_worse  = deltas_df.nlargest(5, "delta_pct")
top_better = deltas_df.nsmallest(5, "delta_pct")

worse_list = "; ".join(
    f"{r['name']} ({r['county']}, +{r['delta_pct']:.0f}%)"
    for _, r in top_worse.iterrows()
)
better_list = "; ".join(
    f"{r['name']} ({r['county']}, {r['delta_pct']:.0f}%)"
    for _, r in top_better.iterrows()
)

add(
    "Which Connecticut facilities showed the greatest increase in water contamination risk from 2020 to 2022?",
    f"The facilities with the largest increase in toxicity concentration from 2020 to 2022 were: {worse_list}. "
    "These facilities may have increased production, expanded operations, or changed chemical inputs."
)

add(
    "Which Connecticut facilities showed the greatest improvement in water contamination risk from 2020 to 2022?",
    f"The facilities with the largest reduction in toxicity concentration from 2020 to 2022 were: {better_list}. "
    "Improvements may reflect process changes, reduced production, or pollution control upgrades."
)

# ---------------------------------------------------------------------------
# Merge with existing train data and write splits
# ---------------------------------------------------------------------------

print(f"\nNew temporal pairs: {len(pairs)}")

existing = []
for split in ["train", "valid", "test"]:
    with open(OUTDIR / f"{split}.jsonl") as f:
        for line in f:
            existing.append((split, json.loads(line)))

print(f"Existing pairs: {len(existing)}")

random.shuffle(pairs)
n_val  = max(20, int(len(pairs) * 0.1))
n_test = max(10, int(len(pairs) * 0.05))

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

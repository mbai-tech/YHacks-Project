"""
Generate fine-tuning JSONL from parquet data.
Each record: {"prompt": "...", "completion": "..."}
"""

import json
import random
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "data" / "finetune.jsonl"

random.seed(42)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

fac     = pd.read_parquet(ROOT / "data" / "interim" / "facilities_clean.parquet")
stream  = pd.read_parquet(ROOT / "data" / "processed" / "stream_scores.parquet")
chem_df = pd.read_parquet(ROOT / "data" / "processed" / "chemical.parquet")[[
    "Chemical", "CASNumber", "ToxicityCategory", "MCL",
    "PFASFlag", "HAPFlag", "CERCLAFlag", "SDWAFlag",
    "OTW", "POTWPartitionRemoval", "WaterSolubility", "BCF",
    "RFDOral", "QSTAROral",
]]
releases = pd.read_parquet(ROOT / "data" / "processed" / "releases.parquet")
media_df = pd.read_parquet(ROOT / "data" / "processed" / "media.parquet")[[
    "Media", "AggDescription", "Otw",
]]

# NAICS industry descriptions (partial — covers codes present in CT data)
NAICS_DESC = {
    "331318": "copper rolling, drawing, extruding, and alloying",
    "332618": "other fabricated wire product manufacturing",
    "331521": "aluminum die-casting foundries",
    "339112": "surgical and medical instrument manufacturing",
    "326291": "rubber product manufacturing for mechanical use",
    "332912": "fluid power valve and fitting manufacturing",
    "337110": "wood kitchen cabinet and countertop manufacturing",
    "332117": "powder metallurgy part manufacturing",
    "325211": "plastics material and resin manufacturing",
    "331110": "iron and steel mills and ferroalloy manufacturing",
    "322121": "paper (except newsprint) mills",
    "325998": "specialty chemical manufacturing",
    "332999": "other miscellaneous fabricated metal product mfg",
    "336111": "automobile manufacturing",
    "334516": "analytical laboratory instrument manufacturing",
    "325180": "other basic inorganic chemical manufacturing",
    "339114": "dental equipment and supplies manufacturing",
    "322130": "paperboard mills",
    "311611": "animal (except poultry) slaughtering",
    "322219": "other paperboard container manufacturing",
    "327331": "concrete block and brick manufacturing",
    "339910": "jewelry and silverware manufacturing",
    "335311": "power, distribution, and specialty transformer mfg",
    "336412": "aircraft engine and engine parts manufacturing",
    "331491": "nonferrous metal (except copper and aluminum) rolling",
    "928110": "national security",
}

def naics_label(code):
    if pd.isna(code):
        return "industrial facility"
    code = str(code)
    return NAICS_DESC.get(code, f"NAICS {code} manufacturing")

# ---------------------------------------------------------------------------
# Build merged dataset
# ---------------------------------------------------------------------------

merged = fac.merge(stream, on="FinalCOMID", how="left")
merged["segment_score"] = merged["segment_score"].fillna(0)
merged["total_tox_conc"] = merged["total_tox_conc"].fillna(0)
scored = merged[merged["segment_score"] > 0].copy()

county_stats = scored.groupby("County").agg(
    n=("FacilityName", "count"),
    avg_score=("segment_score", "mean"),
    max_score=("segment_score", "max"),
    total_tox=("total_tox_conc", "sum"),
).sort_values("max_score", ascending=False)

water_chems = chem_df[chem_df["OTW"] > 0].sort_values("OTW", ascending=False)

# Water releases by media
water_rel = releases.merge(media_df, on="Media")
direct_total = float(water_rel[water_rel["Media"] == 3]["PoundsReleased"].sum())
potw_total   = float(water_rel[water_rel["Media"] == 6]["PoundsReleased"].sum())
water_total  = float(water_rel[water_rel["Otw"] == 1]["PoundsReleased"].sum())

# ---------------------------------------------------------------------------
# Pair generators
# ---------------------------------------------------------------------------

pairs = []

def add(prompt, completion):
    pairs.append({"prompt": prompt.strip(), "completion": completion.strip()})


# ── 1. Per-facility risk assessment ──────────────────────────────────────────

for _, row in scored.sort_values("segment_score", ascending=False).iterrows():
    name    = row["FacilityName"]
    county  = row["County"]
    score   = row["segment_score"]
    tox     = row["total_tox_conc"]
    naics   = naics_label(row["ModeledNAICS"])
    water   = "and is a confirmed water-releasing facility" if row["WaterReleases"] is True else ""
    fcount  = int(row["facility_count"]) if not pd.isna(row["facility_count"]) else 1

    level = "high" if score > 10 else "moderate" if score > 7 else "low"
    tox_note = (
        f"The stream segment receives a total toxicity concentration of {tox:,.0f} "
        f"toxicity-weighted units from {fcount} facilit{'y' if fcount == 1 else 'ies'} "
        "on the same reach."
    )

    add(
        f"Assess the water contamination risk posed by {name} in {county} County, CT.",
        f"{name} is a {naics} facility located in {county} County, Connecticut {water}. "
        f"It has a stream segment risk score of {score:.2f}, indicating {level} water contamination risk. "
        f"{tox_note} "
        f"{'This is one of the highest-risk stream segments in the state.' if score > 11 else ''} "
        f"Facilities in this industry sector typically release metals, solvents, or process chemicals "
        f"that can bioaccumulate in aquatic organisms and affect downstream drinking water quality."
    )

    add(
        f"What is the segment score for {name} and what does it mean?",
        f"{name}'s associated stream segment has a score of {score:.2f}. "
        f"Segment scores are derived from RSEI toxicity-weighted concentrations — "
        f"higher values indicate greater modeled risk to downstream water quality. "
        f"A score of {score:.2f} places this facility in the {level}-risk tier. "
        f"The total toxicity concentration on this reach is {tox:,.0f} units."
    )


# ── 2. County comparisons ─────────────────────────────────────────────────────

for county, row in county_stats.iterrows():
    top_fac = scored[scored["County"] == county].nlargest(1, "segment_score").iloc[0]
    add(
        f"Which county in Connecticut has the highest water contamination risk?",
        f"Based on RSEI stream segment scores, Hartford County and New Haven County carry the highest "
        f"water contamination risk. Hartford County has {county_stats.loc['HARTFORD', 'n']:.0f} scored facilities "
        f"with an average segment score of {county_stats.loc['HARTFORD', 'avg_score']:.2f} and a total toxicity "
        f"concentration of {county_stats.loc['HARTFORD', 'total_tox']:.2e}. "
        f"New Haven County has {county_stats.loc['NEW HAVEN', 'n']:.0f} facilities with a max score of "
        f"{county_stats.loc['NEW HAVEN', 'max_score']:.2f}. Both counties host dense concentrations of "
        f"metal manufacturing, plastics, and specialty chemical facilities."
    )

    add(
        f"What is the highest-risk facility in {county} County, CT?",
        f"The highest-risk facility in {county} County is {top_fac['FacilityName']}, "
        f"a {naics_label(top_fac['ModeledNAICS'])} operation, with a stream segment score of "
        f"{top_fac['segment_score']:.2f} and a toxicity concentration of "
        f"{top_fac['total_tox_conc']:,.0f} on its associated stream reach."
    )
    break  # county comparison prompt is the same for all — generate once, vary below


# County-specific top 3
for county in county_stats.index:
    top3 = scored[scored["County"] == county].nlargest(3, "segment_score")
    if len(top3) < 2:
        continue
    entries = "; ".join(
        f"{r['FacilityName']} (score {r['segment_score']:.2f})"
        for _, r in top3.iterrows()
    )
    add(
        f"List the top facilities by water risk in {county} County, Connecticut.",
        f"The highest water contamination risk facilities in {county} County, CT, by stream segment score are: "
        f"{entries}. These scores reflect RSEI-modeled toxicity-weighted concentrations on their respective "
        f"NHDPlus stream reaches."
    )

    add(
        f"How many facilities contribute to water risk in {county} County?",
        f"{county} County has {county_stats.loc[county, 'n']:.0f} facilities with a non-zero stream segment "
        f"score. The average score across these facilities is {county_stats.loc[county, 'avg_score']:.2f}, "
        f"the maximum is {county_stats.loc[county, 'max_score']:.2f}, and the combined toxicity "
        f"concentration across all reaches is {county_stats.loc[county, 'total_tox']:.2e}."
    )


# ── 3. Chemical-specific questions ───────────────────────────────────────────

for _, c in water_chems.head(20).iterrows():
    name = c["Chemical"]
    cat  = c["ToxicityCategory"] if not pd.isna(c["ToxicityCategory"]) else "unknown toxicity"
    otw  = c["OTW"]
    mcl  = f"EPA MCL of {c['MCL']:.2e} mg/L" if not pd.isna(c["MCL"]) else "no established federal MCL"
    potw = f"{c['POTWPartitionRemoval']:.0f}%" if not pd.isna(c["POTWPartitionRemoval"]) else "unknown"
    pfas = " It is classified as a PFAS compound." if c["PFASFlag"] else ""
    hap  = " It is listed as a Hazardous Air Pollutant." if c["HAPFlag"] else ""
    cercla = " It is a CERCLA priority substance." if c["CERCLAFlag"] else ""

    add(
        f"What are the water toxicity risks of {name}?",
        f"{name} is classified as a {cat} under RSEI. It has an oral toxicity weight (OTW) of {otw:.2e}, "
        f"indicating {'extremely high' if otw > 1e8 else 'high' if otw > 1e6 else 'moderate'} water pathway toxicity. "
        f"It has {mcl}. When routed through a POTW, approximately {potw} is removed by treatment.{pfas}{hap}{cercla}"
    )

    add(
        f"Should {name} be a priority concern for water quality monitoring?",
        f"{'Yes — ' + name + ' should be a high-priority monitoring target.' if otw > 1e6 else name + ' warrants monitoring but is lower priority than some other RSEI chemicals.'} "
        f"It has a water toxicity weight of {otw:.2e} and is categorized as {cat}. "
        f"{'Its PFAS classification means it is persistent and bioaccumulative.' if c['PFASFlag'] else ''} "
        f"{'CERCLA listing means it may be present at Superfund sites.' if c['CERCLAFlag'] else ''} "
        f"POTW removal efficiency is approximately {potw}."
    )


# ── 4. Stream segment analysis ───────────────────────────────────────────────

for _, seg in stream.iterrows():
    comid     = seg["FinalCOMID"]
    score     = seg["segment_score"]
    tox       = seg["total_tox_conc"]
    fcount    = int(seg["facility_count"])
    ct_conc   = seg["total_ct_conc"]
    nct_conc  = seg["total_nct_conc"]

    ct_pct  = ct_conc / tox * 100  if tox > 0 else 0
    nct_pct = nct_conc / tox * 100 if tox > 0 else 0

    facs_on_seg = scored[scored["FinalCOMID"] == comid]
    fac_names = ", ".join(facs_on_seg["FacilityName"].tolist()[:3])

    add(
        f"Describe the risk profile of NHDPlus stream segment {comid}.",
        f"Stream segment {comid} has a risk score of {score:.2f} and a total toxicity concentration of "
        f"{tox:,.0f}. It receives discharges from {fcount} facilit{'y' if fcount == 1 else 'ies'}"
        f"{': ' + fac_names if fac_names else ''}. "
        f"Of the total toxicity load, {ct_pct:.1f}% comes from carcinogenic compounds and "
        f"{nct_pct:.1f}% from non-carcinogenic compounds. "
        f"{'This segment is among the highest-risk reaches in Connecticut.' if score > 11 else ''}"
    )


# ── 5. Release pathway questions ─────────────────────────────────────────────

add(
    "What is the difference between Direct Water and POTW Transfer releases in RSEI?",
    "In RSEI, Direct Water releases (Media code 3) represent chemical discharges that go directly "
    "from a facility into a surface water body — a river, lake, or estuary — without any treatment. "
    "POTW Transfer (Media code 6) means the facility sends its wastewater to a publicly owned "
    "treatment works (municipal sewage plant) before it reaches a water body. "
    "Treatment plants remove a portion of contaminants — removal efficiency varies by chemical — "
    "so POTW releases generally pose lower direct surface water risk than equivalent Direct Water releases. "
    f"In the full RSEI dataset, Direct Water releases total approximately {direct_total/1e9:.1f} billion pounds "
    f"and POTW transfers total approximately {potw_total/1e9:.1f} billion pounds."
)

add(
    "How does rainfall affect water contamination risk from industrial facilities?",
    "Rainfall increases water contamination risk through several mechanisms. Heavy rain drives surface "
    "runoff that can carry released chemicals from facility grounds into nearby streams before they can "
    "be treated or contained. High flow events also scour stream sediments where chemicals may have "
    "accumulated. Conversely, rainfall increases stream volume and can dilute contamination. "
    "The net effect depends on soil permeability (sandy soils absorb more; clay soils generate more runoff), "
    "facility containment practices, and whether releases are direct or through POTWs. "
    "In the Connecticut context, spring snowmelt combined with rainfall represents the highest "
    "runoff risk period for facilities along the Connecticut River watershed."
)

add(
    "How does streamflow level affect drinking water risk from upstream industrial discharges?",
    "Low streamflow concentrates contaminants. When a river runs low — common in Connecticut summers — "
    "the same mass of discharged chemical occupies less water volume, raising the effective concentration "
    "downstream. High flow dilutes the discharge but may mobilize sediment-bound chemicals. "
    "RSEI models use a harmonic mean flow to estimate typical in-stream concentrations, but real-time "
    "USGS gauge data provides a more accurate current dilution factor. A facility with a moderate "
    "segment score under average flow may exceed safe thresholds under a summer low-flow condition."
)

add(
    "What role does soil permeability play in industrial water contamination?",
    "Soil permeability determines how much of a surface chemical release infiltrates groundwater versus "
    "running off into surface water. Clay soils are nearly impermeable — most spilled or runoff-carried "
    "chemicals stay on the surface and flow into streams. Sandy soils absorb liquid rapidly, potentially "
    "contaminating shallow aquifers and groundwater that may feed drinking water wells. "
    "Loam is intermediate. Connecticut's geology varies significantly by county: the Connecticut River "
    "valley has sandy glacial outwash deposits, while upland areas tend toward rocky or clay-rich soils."
)


# ── 6. Comparative and policy questions ──────────────────────────────────────

top5 = scored.nlargest(5, "segment_score")
top5_list = "\n".join(
    f"  {i+1}. {r['FacilityName']} ({r['County']} County) — score {r['segment_score']:.2f}"
    for i, (_, r) in enumerate(top5.iterrows())
)
add(
    "Which five facilities in Connecticut pose the greatest water contamination risk?",
    f"Based on RSEI stream segment scores, the five highest-risk facilities in Connecticut are:\n"
    f"{top5_list}\n"
    "These scores reflect modeled toxicity-weighted concentrations on their downstream NHDPlus stream "
    "reaches and incorporate the toxicity profiles of each facility's reported chemical releases."
)

add(
    "What industry sectors contribute most to water contamination risk in Connecticut?",
    "In Connecticut, the highest water contamination risk comes from metal manufacturing and finishing "
    "sectors. Facilities in copper rolling, aluminum die-casting, iron/steel mills, powder metallurgy, "
    "and fabricated metal products dominate the top segment scores. These sectors release metals "
    "(copper, chromium, lead, nickel) and solvents with high water toxicity weights. "
    "Specialty chemical and plastics manufacturing are secondary contributors. "
    f"Hartford and New Haven counties have the highest combined toxicity concentrations — "
    f"{county_stats.loc['HARTFORD', 'total_tox']:.2e} and "
    f"{county_stats.loc['NEW HAVEN', 'total_tox']:.2e} respectively."
)

add(
    "If regulators could target three facilities to reduce water contamination risk most efficiently, which should they prioritize?",
    f"Based on stream segment scores, regulators should prioritize: "
    f"(1) {top5.iloc[0]['FacilityName']} in {top5.iloc[0]['County']} County (score {top5.iloc[0]['segment_score']:.2f}), "
    f"(2) {top5.iloc[2]['FacilityName']} in {top5.iloc[2]['County']} County (score {top5.iloc[2]['segment_score']:.2f}), and "
    f"(3) {top5.iloc[4]['FacilityName']} in {top5.iloc[4]['County']} County (score {top5.iloc[4]['segment_score']:.2f}). "
    "These selections spread across different stream reaches to maximize geographic impact. "
    "Reducing releases at the highest-scoring facilities has a non-linear benefit because toxicity "
    "concentration compounds when multiple facilities share the same stream reach."
)

add(
    "How many CT facilities have confirmed water releases in the RSEI dataset?",
    f"Of the {len(fac):,} Connecticut facilities in the RSEI dataset, "
    f"{fac['WaterReleases'].sum():,.0f} are flagged as confirmed water-releasing facilities. "
    f"{len(scored):,} facilities have a non-zero stream segment risk score, meaning their releases "
    "have been modeled to reach a downstream NHDPlus stream reach with a calculated toxicity concentration."
)

add(
    "What does a RSEI segment score represent and how is it calculated?",
    "A RSEI stream segment score is a log-transformed, toxicity-weighted measure of modeled chemical "
    "concentration in a stream reach. It is derived by: (1) estimating the pounds of each chemical "
    "released to water by upstream facilities, (2) routing releases downstream through the NHDPlus "
    "stream network using dilution factors based on harmonic mean flow, (3) multiplying the resulting "
    "concentration by the chemical's oral toxicity weight (OTW) to produce a toxicity-weighted "
    "concentration, and (4) summing contributions from all facilities on the reach. "
    "The final score is log10 of that sum, so each unit increase represents a 10× increase in modeled risk."
)

add(
    "What is the total toxicity concentration in Connecticut's most contaminated stream segment?",
    f"The highest total toxicity concentration in Connecticut belongs to the stream segment shared by "
    f"{top5.iloc[0]['FacilityName']} and {top5.iloc[1]['FacilityName']} in New Haven County, with a "
    f"total toxicity concentration of {scored.iloc[0]['total_tox_conc']:,.0f} toxicity-weighted units "
    f"and a segment score of {scored.iloc[0]['segment_score']:.2f}."
)


# ── 7. Environmental scenario Q&A ─────────────────────────────────────────────

scenarios = [
    ("heavy rainfall (8 in/month)", "Heavy rainfall significantly increases contamination risk. Elevated runoff "
     "mobilizes chemicals from facility grounds and increases the rate of direct water releases reaching streams. "
     "Heatmap spread radius and blur increase by approximately 50–60%, and the effective weight of each facility's "
     "contribution rises by ~30% due to runoff amplification. The combined environmental multiplier under these "
     "conditions can reach 2–3× baseline."),
    ("summer low-flow conditions", "Summer is the highest-risk season under RSEI assumptions. Low streamflow "
     "reduces dilution capacity — the same released mass produces higher in-stream concentrations. "
     "The season multiplier in the risk model is 1.5× (vs. 1.0× in fall). Combined with high temperatures "
     "that accelerate chemical reactions, summer conditions can push risk multipliers to 1.8–2.5× baseline."),
    ("winter (frozen ground)", "Winter suppresses surface water risk significantly. Frozen ground prevents "
     "infiltration and reduces runoff. The season multiplier is 0.5×. However, freeze–thaw cycles in early "
     "spring can cause sudden releases of accumulated contaminants — so risk rebounds sharply in March–April."),
    ("a 100-year flood event", "A flood event amplifies risk catastrophically for facilities in FEMA flood zones. "
     "Containment infrastructure is overwhelmed, chemicals stored on-site are carried directly into floodwaters, "
     "and the normal stream-routing model breaks down. Facilities within the floodplain of the Connecticut, "
     "Housatonic, or Thames rivers would have their effective release volumes multiplied many times over."),
]

for condition, explanation in scenarios:
    add(
        f"How does {condition} affect water contamination risk from CT industrial facilities?",
        explanation
    )


# ── 8. PFAS-specific questions ────────────────────────────────────────────────

pfas_chems = chem_df[chem_df["PFASFlag"] == True]
add(
    "Which PFAS compounds in the RSEI chemical database have the highest water risk?",
    f"The RSEI database includes {len(pfas_chems)} PFAS-flagged chemicals. "
    "PFAS compounds are of particular concern because they are persistent (resist biodegradation), "
    "bioaccumulative (concentrate in living tissue), and have low POTW removal efficiency — "
    "treatment plants remove only a small fraction. "
    "Connecticut has conducted active PFAS monitoring in drinking water sources, particularly near "
    "military installations and facilities that have used AFFF firefighting foam. "
    "Even facilities with moderate RSEI scores may be high-priority PFAS risks if they release "
    "PFAS-listed compounds."
)

add(
    "Does POTW treatment effectively remove PFAS from industrial wastewater?",
    "No — POTW treatment is largely ineffective at removing PFAS compounds. Unlike many organic "
    "pollutants, PFAS resists biodegradation and does not volatilize easily. Municipal wastewater "
    "treatment typically removes less than 20% of PFAS mass, and some treatment processes "
    "(particularly those that concentrate sludge) can actually increase PFAS concentrations in "
    "plant effluent relative to influent. This means POTW-pathway PFAS releases carry a higher "
    "effective downstream risk than the POTW removal fraction in RSEI would suggest."
)


# ── 9. Dataset overview Q&A ──────────────────────────────────────────────────

add(
    "What data sources does this Connecticut water contamination tool use?",
    "The tool uses EPA RSEI (Risk-Screening Environmental Indicators) version 23.12 data. "
    "This includes: (1) facility location and industry data for all TRI-reporting facilities in Connecticut, "
    "(2) chemical-level release quantities by environmental medium (air, water, land, POTW), "
    "(3) chemical toxicity weights derived from EPA IRIS and other authoritative sources, "
    "(4) NHDPlus stream network routing to estimate downstream toxicity-weighted concentrations, and "
    "(5) media-specific transport and fate factors. The heatmap weights are further adjusted by "
    "user-specified environmental conditions (rainfall, season, temperature, soil type)."
)

add(
    "How many chemicals are tracked in the RSEI water contamination model for Connecticut?",
    f"RSEI tracks {len(chem_df):,} chemicals in total. Of these, {len(water_chems):,} have a non-zero "
    "oral/water toxicity weight (OTW), meaning they are modeled to pose a water pathway risk. "
    f"The highest water-toxicity chemical is dioxin and dioxin-like compounds (OTW ~1.4×10⁹), "
    "followed by thorium dioxide, benzidine, and bis(chloromethyl) ether. "
    f"{len(pfas_chems):,} PFAS compounds are included in the database."
)

add(
    "What is the total volume of chemicals released to water by RSEI facilities?",
    f"Across all RSEI-reporting facilities, approximately {water_total/1e9:.1f} billion pounds of chemicals "
    f"were released via water pathways in the dataset period. Of this, approximately "
    f"{direct_total/1e9:.1f} billion pounds ({direct_total/water_total*100:.0f}%) went via direct "
    f"surface water discharge, and {potw_total/1e9:.1f} billion pounds ({potw_total/water_total*100:.0f}%) "
    "were transferred to publicly owned treatment works (POTWs). "
    "Volume alone does not determine risk — the toxicity weight of each chemical is the primary driver "
    "of modeled downstream impact."
)

# ---------------------------------------------------------------------------
# Write JSONL
# ---------------------------------------------------------------------------

print(f"Total pairs generated: {len(pairs)}")

with open(OUT, "w") as f:
    for pair in pairs:
        f.write(json.dumps(pair) + "\n")

print(f"Written to {OUT}")

# Preview first 3
for p in pairs[:3]:
    print("\n--- PROMPT ---")
    print(p["prompt"])
    print("--- COMPLETION ---")
    print(p["completion"])

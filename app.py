"""Streamlit app — Connecticut water contamination risk heatmap."""

import math
import re
import numpy as np
import pandas as pd
import folium
import streamlit as st
from folium.plugins import HeatMap
from pathlib import Path
from streamlit_folium import st_folium
from openai import OpenAI
from scripts.spatial_index import FacilityIndex

ROOT = Path(__file__).resolve().parent

# Local LLaMA server (Ollama default port — change to 1234 if using LM Studio)
llm_client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="not-needed",
)

st.set_page_config(page_title="CT Water Risk", layout="wide", page_icon="💧")

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono:wght@400;500&display=swap');

  /* ── global ── */
  html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    background: #dbe7ef;
    color: #000000;
  }
  [data-testid="stHeader"] { background: transparent; }

  /* ── sidebar ── */
  [data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #d0d0d0;
  }
  [data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
    color: #000000 !important;
  }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: #000000 !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
  }

  /* ── slider accent ── */
  [data-testid="stSlider"] > div > div > div > div {
    background: #0ea5e9;
  }

  /* ── metric cards ── */
  [data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #90c4e8;
    border-radius: 8px;
    padding: 12px 16px;
  }
  [data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    color: #000000 !important;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  [data-testid="stMetricValue"] {
    font-family: 'DM Sans', sans-serif !important;
    color: #000000 !important;
    font-size: 1.5rem;
    font-weight: 700;
  }

  /* ── dataframe ── */
  [data-testid="stDataFrame"] { border: 1px solid #90c4e8; border-radius: 8px; overflow: hidden; background: #ffffff !important; }
  .stDataFrame thead th {
    font-family: 'DM Mono', monospace !important;
    background: #ffffff !important;
    color: #000000 !important;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .stDataFrame tbody tr:nth-child(even) { background: #f5f9fc !important; }
  .stDataFrame tbody tr:hover { background: #e8f2f8 !important; }

  /* ── section divider ── */
  hr { border-color: #90c4e8; }

  /* ── hide default streamlit chrome ── */
  #MainMenu, footer { visibility: hidden; }

  /* ── map container ── */
  iframe { border-radius: 10px; border: 1px solid #90c4e8; background: #ffffff !important; }

  /* ── risk badge ── */
  .risk-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .risk-low    { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
  .risk-medium { background: #fef9c3; color: #92400e; border: 1px solid #fde047; }
  .risk-high   { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div style="padding: 1.2rem 0 0.4rem 0;">
  <span style="font-size:0.75rem; letter-spacing:0.15em; color:#555555; text-transform:uppercase;">
    EPA RSEI v23.12 · Industrial Facility Water Releases
  </span>
  <h1 style="margin:0.2rem 0 0 0; font-size:2rem; font-weight:800; color:#000000; line-height:1.1; font-family:'DM Sans', sans-serif;">
    Connecticut Water<br>Contamination Risk
  </h1>
</div>
<hr style="margin: 0.8rem 0 1.2rem 0;">
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    fac = pd.read_parquet(ROOT / "data" / "processed" / "facilities_clean.parquet")
    stream = pd.read_parquet(ROOT / "data" / "processed" / "stream_scores.parquet")
    merged = fac.merge(stream[["FinalCOMID", "segment_score"]], on="FinalCOMID", how="left")
    merged["segment_score"] = merged["segment_score"].fillna(0)

    # Load precomputed blended risk scores
    risk_path = ROOT / "data" / "processed" / "facility_risk_scores.parquet"
    if risk_path.exists():
        risk_df = pd.read_parquet(risk_path)
        merged = merged.merge(risk_df, on="FacilityNumber", how="left")
        merged["risk_score"] = merged["risk_score"].fillna(0)
    else:
        merged["risk_score"] = merged["segment_score"]

    max_risk = merged["risk_score"].max()
    merged["weight"] = merged["risk_score"] / max_risk if max_risk > 0 else 0
    return merged

@st.cache_data
def load_releases():
    """Aggregate releases.parquet into per-pathway stats for display and heatmap scaling."""
    rel = pd.read_parquet(ROOT / "data" / "processed" / "releases.parquet")
    media = pd.read_parquet(ROOT / "data" / "processed" / "media.parquet",
                            columns=["Media", "AggDescription", "Otw"])

    rel = rel.merge(media, on="Media", how="left")

    # Water pathways we care about
    # Media=3: Direct Water (on-site surface water discharge)
    # Media=6: POTW Transfer (goes to municipal wastewater treatment first)
    direct = rel[rel["Media"] == 3]
    potw   = rel[rel["Media"] == 6]
    water  = rel[rel["Otw"] == 1]

    direct_lbs = direct["PoundsReleased"].sum()
    potw_lbs   = potw["PoundsReleased"].sum()
    water_lbs  = water["PoundsReleased"].sum()

    # Proportional contribution of each pathway to total water releases
    direct_share = direct_lbs / water_lbs if water_lbs > 0 else 0.5
    potw_share   = potw_lbs   / water_lbs if water_lbs > 0 else 0.5

    # TEF-weighted toxic load (sparse — only ~6k rows have TEF)
    water_tef = water.dropna(subset=["TEF"])
    toxic_load = (water_tef["PoundsReleased"] * water_tef["TEF"]).sum()

    # Breakdown by AggDescription for chart
    breakdown = (
        rel[rel["Otw"] == 1]
        .groupby("AggDescription")["PoundsReleased"]
        .sum()
        .sort_values(ascending=False)
    )

    return {
        "direct_lbs":    direct_lbs,
        "potw_lbs":      potw_lbs,
        "water_lbs":     water_lbs,
        "direct_share":  direct_share,
        "potw_share":    potw_share,
        "toxic_load":    toxic_load,
        "total_releases": len(rel),
        "water_releases": int(rel["Otw"].sum()),
        "breakdown":     breakdown,
    }

@st.cache_data
def load_trends():
    """Compute per-facility and per-county CAGR from 2020→2022 aggmicro data."""
    AGG_COLS = ["X", "Y", "ToxConc"]
    fac = pd.read_parquet(ROOT / "data" / "processed" / "facility.parquet",
        columns=["FacilityNumber", "FacilityName", "State", "FinalCOMID",
                 "WaterReleases", "X", "Y", "County"])
    ct = fac[fac["State"] == "CT"].copy()

    raw_2020 = ROOT / "data" / "raw" / "aggmicro2022_2020.csv"
    agg_2020 = pd.read_csv(raw_2020, skiprows=2, header=None,
        names=["GridCode","X","Y","NumFacs","NumReleases","NumChems",
               "ToxConc","Score","Pop","CTConc","NCTConc"],
        dtype={"X":"Int16","Y":"Int16","ToxConc":"float64"},
        usecols=["X","Y","ToxConc"])
    agg_2022 = pd.read_parquet(ROOT / "data" / "processed" / "aggmicro.parquet",
        columns=["X","Y","ToxConc"])

    def merge(agg):
        m = ct.merge(agg, on=["X","Y"], how="inner")
        m = m[m["WaterReleases"] == True]
        m["ToxConc"] = m["ToxConc"].fillna(0)
        return m.set_index("FacilityNumber")

    f20 = merge(agg_2020)
    f22 = merge(agg_2022)
    common = f20.index.intersection(f22.index)

    rows = []
    for fnum in common:
        t20 = float(f20.loc[fnum, "ToxConc"]) if not pd.isna(f20.loc[fnum, "ToxConc"]) else 0
        t22 = float(f22.loc[fnum, "ToxConc"]) if not pd.isna(f22.loc[fnum, "ToxConc"]) else 0
        if t20 <= 0:
            continue
        rate = max(-0.30, min(0.30, (t22 / t20) ** 0.5 - 1))
        rows.append({
            "FacilityName": f22.loc[fnum, "FacilityName"],
            "County":       f22.loc[fnum, "County"],
            "tox22":        t22,
            "annual_rate":  rate,
        })
    return pd.DataFrame(rows)


@st.cache_resource
def load_facility_index():
    return FacilityIndex()


def project_tox(tox22, annual_rate, target_year, base_year=2022):
    years_out = target_year - base_year
    return max(0.0, tox22 * ((1 + annual_rate) ** years_out))


def build_future_context(query, trends):
    """If the query mentions a future year, compute projections and return context string."""
    years = [int(y) for y in re.findall(r'\b(202[6-9]|20[3-4]\d|2050)\b', query)]
    if not years:
        return ""

    year = years[0]
    lines = [f"Computed projections for {year} (extrapolated from 2020–2022 trends, ±30%/yr cap):"]

    # County match
    county_matches = [c for c in trends["County"].dropna().unique()
                      if c.lower() in query.lower()]
    for county in county_matches:
        rows = trends[trends["County"] == county]
        total_tox22 = rows["tox22"].sum()
        avg_rate = ((rows["annual_rate"] * rows["tox22"]).sum() / total_tox22
                    if total_tox22 > 0 else 0)
        proj = project_tox(total_tox22, avg_rate, year)
        lines.append(f"- {county} County: {proj:,.0f} toxicity units projected in {year} "
                     f"(2022 baseline: {total_tox22:,.0f}, trend: {avg_rate*100:+.1f}%/yr)")

    # Facility name match (fuzzy: any word overlap)
    query_words = set(re.sub(r'[^a-z ]', '', query.lower()).split())
    for _, row in trends.iterrows():
        name_words = set(row["FacilityName"].lower().split())
        if len(name_words & query_words) >= 2:
            proj = project_tox(row["tox22"], row["annual_rate"], year)
            lines.append(f"- {row['FacilityName']}: {proj:,.0f} units projected in {year} "
                         f"(2022: {row['tox22']:,.0f}, trend: {row['annual_rate']*100:+.1f}%/yr)")

    # Statewide fallback if no specific entity matched
    if len(lines) == 1:
        total = trends["tox22"].sum()
        avg_rate = ((trends["annual_rate"] * trends["tox22"]).sum() / total if total > 0 else 0)
        proj = project_tox(total, avg_rate, year)
        lines.append(f"- Connecticut statewide: {proj:,.0f} units projected in {year} "
                     f"(2022 baseline: {total:,.0f}, trend: {avg_rate*100:+.1f}%/yr)")

    horizon = year - 2022
    if horizon <= 6:
        lines.append("Uncertainty: low-moderate (near-term projection).")
    elif horizon <= 15:
        lines.append("Uncertainty: moderate. Policy or operational changes may alter outcomes.")
    else:
        lines.append("Uncertainty: high. Long-range projections assume business-as-usual.")

    return "\n".join(lines)


def get_llm_multipliers(year, trends_df, client):
    """Ask the LLM to predict risk multipliers for each facility in a given year.
    Returns a dict {FacilityName: multiplier} or empty dict on failure."""
    years_out = year - 2022
    top = trends_df.nlargest(10, "tox22")

    lines = []
    for _, r in top.iterrows():
        lines.append(
            f"- {r['FacilityName']} ({r['County']}): tox2022={r['tox22']:.0f}, "
            f"annual_trend={r['annual_rate']*100:+.1f}%"
        )
    facility_list = "\n".join(lines)

    prompt = f"""You are a water contamination risk analyst.
Below are Connecticut industrial facilities with their 2022 toxicity levels and observed 2020-2022 annual trend rates.
Predict a risk multiplier for each facility in {year} (i.e., projected_tox / tox2022).
Consider: trend trajectory, regulatory likelihood for this industry, typical plateau/decline patterns.
Respond ONLY with a JSON object mapping facility name to multiplier, e.g.:
{{"FACILITY A": 1.4, "FACILITY B": 0.7}}

Facilities:
{facility_list}

JSON:"""

    try:
        resp = client.chat.completions.create(
            model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        # Extract JSON from response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            import json
            return json.loads(text[start:end])
    except Exception:
        pass
    return {}


df = load_data()
rel_stats = load_releases()
trends = load_trends()
facility_idx = load_facility_index()

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Facility Filters")

show_water_only = st.sidebar.checkbox("Water-releasing facilities only", value=True)
min_score = st.sidebar.slider("Minimum risk score", 0.0, float(df["risk_score"].max()), 0.0, 1.0)

filtered = df.copy()
if show_water_only:
    filtered = filtered[filtered["WaterReleases"] == True]
filtered = filtered[filtered["risk_score"] >= min_score]

st.sidebar.markdown(f"<span style='color:#5bc8f5; font-size:0.8rem;'>▸ **{len(filtered):,}** facilities shown</span>", unsafe_allow_html=True)

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — release pathway filter
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Release Pathway")
st.sidebar.caption("Which water discharge pathway to weight the map by.")

pathway = st.sidebar.radio(
    "Pathway",
    ["Both", "Direct Water", "POTW Transfer"],
    horizontal=True,
    label_visibility="collapsed",
    help=(
        "Direct Water: on-site discharge straight to surface water. "
        "POTW: transferred to a municipal wastewater treatment plant."
    ),
)

pathway_lbs = {
    "Both":            rel_stats["water_lbs"],
    "Direct Water":    rel_stats["direct_lbs"],
    "POTW Transfer":   rel_stats["potw_lbs"],
}[pathway]

# Per-pathway heatmap behavior:
#
# Direct Water — raw on-site discharge straight to a water body:
#   tight, hot, high-opacity point sources; red/orange gradient.
#
# POTW Transfer — routed through a treatment plant first:
#   broader, cooler spread (effluent disperses from plant outfall);
#   treatment removes ~60-70% of load; blue/teal gradient.
#
# Both — combined view, neutral gradient.

PATHWAY_PARAMS = {
    "Direct Water": dict(
        weight_mult  = 1.4,        # full raw load, no treatment removal
        radius_scale = 0.65,       # tight point source
        blur_scale   = 0.55,       # sharp edges
        min_opacity  = 0.5,
        gradient     = {"0.0": "#ff6600", "0.4": "#ff3300",
                        "0.7": "#cc0000", "1.0": "#800000"},
    ),
    "POTW Transfer": dict(
        weight_mult  = 0.45,       # ~55% load removed by treatment
        radius_scale = 1.7,        # effluent spreads broadly from outfall
        blur_scale   = 1.6,        # diffuse plume
        min_opacity  = 0.2,
        gradient     = {"0.0": "#003366", "0.35": "#0066cc",
                        "0.65": "#00aacc", "1.0": "#00ddff"},
    ),
    "Both": dict(
        weight_mult  = 1.0,
        radius_scale = 1.0,
        blur_scale   = 1.0,
        min_opacity  = 0.3,
        gradient     = {"0.0": "#0000ff", "0.25": "#00aaff",
                        "0.5": "#00ffaa", "0.75": "#ffaa00", "1.0": "#ff0000"},
    ),
}

pw = PATHWAY_PARAMS[pathway]

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — environmental parameters
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Environmental Conditions")
st.sidebar.caption("Adjust how contamination spreads.")

rainfall = st.sidebar.slider(
    "Monthly Rainfall (in)", 0.0, 10.0, 3.5, 0.5,
    help="Higher rainfall increases runoff, spreading contaminants further downstream.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Future Projection")
st.sidebar.caption("Scales the heatmap by projected facility toxicity.")
projection_year = st.sidebar.slider("Projection Year", 2022, 2050, 2022, 2)
is_future = projection_year > 2022

# --- Compute environmental modifiers ---
rainfall_weight_mult = 1.0 + (rainfall / 10.0) * 0.4
rainfall_radius_mult = 1.0 + (rainfall / 10.0) * 0.6
rainfall_blur_mult   = 1.0 + (rainfall / 10.0) * 0.5

env_weight_mult = rainfall_weight_mult
env_radius      = int(20 * rainfall_radius_mult)
env_blur        = int(25 * rainfall_blur_mult)

# Combined with pathway params (needed for sidebar display below)
combined_mult   = env_weight_mult * pw["weight_mult"]
combined_radius = max(8, int(env_radius * pw["radius_scale"]))
combined_blur   = max(6, int(env_blur   * pw["blur_scale"]))

# Risk level badge
if env_weight_mult < 0.8:
    badge = '<span class="risk-badge risk-low">Low Dispersion</span>'
elif env_weight_mult < 1.4:
    badge = '<span class="risk-badge risk-medium">Moderate Dispersion</span>'
else:
    badge = '<span class="risk-badge risk-high">High Dispersion</span>'

st.sidebar.markdown(f"""
<div style="margin-top:0.6rem;">
  {badge}
  <table style="margin-top:0.6rem; font-size:0.78rem; border-collapse:collapse; width:100%;">
    <tr><td style="color:#38bdf8; padding:2px 0;">Weight mult</td><td style="text-align:right; color:#e0eaff; font-weight:600;">{combined_mult:.2f}×</td></tr>
    <tr><td style="color:#38bdf8; padding:2px 0;">Spread radius</td><td style="text-align:right; color:#e0eaff; font-weight:600;">{combined_radius} px</td></tr>
    <tr><td style="color:#38bdf8; padding:2px 0;">Blur</td><td style="text-align:right; color:#e0eaff; font-weight:600;">{combined_blur} px</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Facilities", f"{len(filtered):,}")
m2.metric("Avg Risk", f"{filtered['risk_score'].mean():.1f}")
m3.metric("Max Risk", f"{filtered['risk_score'].max():.1f}")
m4.metric("Risk Multiplier", f"{env_weight_mult:.2f}×")
m5.metric(
    "Lbs to Water" if pathway == "Both" else f"Lbs ({pathway})",
    f"{pathway_lbs/1e9:.1f}B",
    help="Total pounds released via selected water pathway across all RSEI facilities.",
)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Build map
# ---------------------------------------------------------------------------

CT_CENTER = [41.6, -72.7]
m = folium.Map(location=CT_CENTER, zoom_start=9, tiles="OpenStreetMap")

if is_future:
    cache_key = f"llm_mult_{projection_year}"
    if cache_key not in st.session_state:
        # Show CAGR heatmap immediately; fetch LLM multipliers for top 10 in background
        st.session_state[cache_key] = {}
        st.toast(f"Fetching LLM risk adjustments for {projection_year}…")
        st.session_state[cache_key] = get_llm_multipliers(projection_year, trends, llm_client)
        st.rerun()
    llm_mults = st.session_state[cache_key]

    proj_lookup = trends.groupby("FacilityName").first()[["tox22", "annual_rate"]]
    filt_ext = filtered.join(proj_lookup, on="FacilityName", how="left").copy()
    filt_ext["tox22"] = filt_ext["tox22"].fillna(0).values.astype(float)
    filt_ext["annual_rate"] = filt_ext["annual_rate"].fillna(0).values.astype(float)
    years_out = projection_year - 2022

    # Use LLM multiplier where available, fall back to CAGR
    t22 = filt_ext["tox22"].values
    rate = filt_ext["annual_rate"].values
    cagr_scale = np.where(t22 > 0, ((1 + rate) ** years_out), 1.0)
    llm_scale = np.array([
        llm_mults.get(name, 0) for name in filt_ext["FacilityName"].values
    ])
    scale = np.where(llm_scale > 0, llm_scale, cagr_scale)

    weights = filt_ext["weight"].values * scale
    weights = np.clip(weights, 0, None)
    max_w = weights.max()
    if max_w > 0:
        weights = weights / max_w
    lats = filt_ext["Latitude"].values
    lons = filt_ext["Longitude"].values
    heat_data = [
        [float(lats[i]), float(lons[i]), min(float(weights[i]) * combined_mult, 1.0)]
        for i in range(len(filt_ext)) if weights[i] > 0
    ]
else:
    heat_data = [
        [row["Latitude"], row["Longitude"], min(row["weight"] * combined_mult, 1.0)]
        for _, row in filtered.iterrows()
        if row["weight"] > 0
    ]

if heat_data:
    HeatMap(
        heat_data,
        radius=combined_radius,
        blur=combined_blur,
        min_opacity=pw["min_opacity"],
        gradient=pw["gradient"],
    ).add_to(m)

top = filtered.nlargest(20, "risk_score")
for _, row in top.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=6,
        color="#ff2200",
        fill=True,
        fill_color="#ff5533",
        fill_opacity=0.85,
        tooltip=folium.Tooltip(
            f"<div style='font-family:monospace; font-size:12px;'>"
            f"<b style='color:#ff5533'>{row['FacilityName']}</b><br>"
            f"<span style='color:#888'>County:</span> {row['County']}<br>"
            f"<span style='color:#888'>Risk:</span> {row['risk_score']:.1f}<br>"
            f"<span style='color:#888'>NAICS:</span> {row['ModeledNAICS']}"
            f"</div>"
        ),
    ).add_to(m)

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

col1, col2 = st.columns([3, 1])

with col1:
    map_data = st_folium(m, width=None, height=580)

with col2:
    # --- Location risk panel (click on map) ---
    clicked = None
    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]

    if clicked:
        clat, clon = clicked["lat"], clicked["lng"]
        risk = facility_idx.compute_location_risk(clat, clon)

        # Color by score
        if risk.score < 30:
            score_color, label = "#4ade80", "Low"
        elif risk.score < 60:
            score_color, label = "#fbbf24", "Moderate"
        else:
            score_color, label = "#f87171", "High"

        st.markdown(f"""
        <div style="background:#ffffff; border:1px solid #90c4e8; border-radius:8px; padding:0.8rem; margin-bottom:0.8rem;">
          <p style="font-size:0.65rem; letter-spacing:0.12em; text-transform:uppercase; color:#000000; margin:0 0 0.3rem 0;">
            Location Risk · {clat:.4f}, {clon:.4f}
          </p>
          <div style="display:flex; align-items:baseline; gap:0.5rem; margin-bottom:0.6rem;">
            <span style="font-size:1.8rem; font-weight:800; color:{score_color};">{risk.score}</span>
            <span style="font-size:0.75rem; color:{score_color}; font-weight:600;">{label}</span>
            <span style="font-size:0.65rem; color:#555555;">/ 100</span>
          </div>
          <table style="font-size:0.72rem; border-collapse:collapse; width:100%;">
            <tr><td style="color:#000000; padding:2px 0;">Industrial Facilities</td>
                <td style="text-align:right; color:#000000; font-weight:600;">{risk.facility_score}</td>
                <td style="text-align:right; color:#555555; font-size:0.6rem; padding-left:4px;">30%</td></tr>
            <tr><td style="color:#000000; padding:2px 0;">Stream Contamination</td>
                <td style="text-align:right; color:#000000; font-weight:600;">{risk.downstream_score}</td>
                <td style="text-align:right; color:#555555; font-size:0.6rem; padding-left:4px;">20%</td></tr>
            <tr><td style="color:#000000; padding:2px 0;">Wastewater Plants</td>
                <td style="text-align:right; color:#000000; font-weight:600;">{risk.wastewater_score}</td>
                <td style="text-align:right; color:#555555; font-size:0.6rem; padding-left:4px;">25%</td></tr>
            <tr><td style="color:#000000; padding:2px 0;">Drinking Water (SDWA)</td>
                <td style="text-align:right; color:#000000; font-weight:600;">{risk.sdwa_score}</td>
                <td style="text-align:right; color:#555555; font-size:0.6rem; padding-left:4px;">25%</td></tr>
          </table>
          <div style="margin-top:0.5rem; font-size:0.65rem; color:#555555; border-top:1px solid #90c4e8; padding-top:0.4rem;">
            {risk.facility_count} facilities · {risk.wastewater_plant_count} WW plants · {risk.sdwa_system_count} water systems nearby
          </div>
        </div>
        """, unsafe_allow_html=True)

        if not risk.top_contributors.empty:
            st.markdown("<p style='font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; color:#38bdf8; margin:0 0 0.3rem 0;'>Top Contributors</p>", unsafe_allow_html=True)
            st.dataframe(
                risk.top_contributors.rename(columns={
                    "FacilityName": "Facility",
                    "TotalWaterImpactScore": "Impact",
                    "distance_km": "Dist (km)",
                    "contribution": "Score",
                }).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.markdown(
            "<div style='background:#ffffff; border:1px solid #90c4e8; border-radius:8px; padding:0.8rem; margin-bottom:0.8rem;'>"
            "<p style='font-size:0.75rem; color:#000000; margin:0;'>Click anywhere on the map to compute a location risk score.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<p style='font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:#38bdf8; margin-bottom:0.4rem;'>Top 10 Facilities</p>", unsafe_allow_html=True)
    st.dataframe(
        filtered.nlargest(10, "risk_score")[["FacilityName", "County", "risk_score"]]
        .rename(columns={"FacilityName": "Facility", "risk_score": "Score"})
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<p style='font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:#38bdf8; margin:1rem 0 0.2rem;'>Water Release Pathways</p>", unsafe_allow_html=True)

    direct_pct = rel_stats["direct_lbs"] / rel_stats["water_lbs"] * 100
    potw_pct   = rel_stats["potw_lbs"]   / rel_stats["water_lbs"] * 100

    # Highlight selected pathway
    direct_color = "#0ea5e9" if pathway in ("Both", "Direct Water")  else "#1f2d3d"
    potw_color   = "#0ea5e9" if pathway in ("Both", "POTW Transfer") else "#1f2d3d"

    st.markdown(f"""
    <div style="font-size:0.78rem; margin-bottom:0.8rem;">
      <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
        <span style="color:#000000;">Direct Water</span>
        <span style="color:#000000; font-weight:600;">{rel_stats['direct_lbs']/1e9:.1f}B lbs ({direct_pct:.0f}%)</span>
      </div>
      <div style="background:#d0dde6; border-radius:3px; height:6px; margin-bottom:10px;">
        <div style="background:{direct_color}; width:{direct_pct:.0f}%; height:6px; border-radius:3px;"></div>
      </div>
      <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
        <span style="color:#000000;">POTW Transfer</span>
        <span style="color:#000000; font-weight:600;">{rel_stats['potw_lbs']/1e9:.1f}B lbs ({potw_pct:.0f}%)</span>
      </div>
      <div style="background:#d0dde6; border-radius:3px; height:6px;">
        <div style="background:{potw_color}; width:{potw_pct:.0f}%; height:6px; border-radius:3px;"></div>
      </div>
    </div>
    <div style="font-size:0.68rem; color:#555555; border-top:1px solid #90c4e8; padding-top:0.5rem;">
      Pathway weight mult: <b style="color:#000000;">{pw["weight_mult"]:.2f}×</b><br>
      TEF toxic load: <b style="color:#000000;">{rel_stats['toxic_load']:.3f}</b> (lb·TEF units)
    </div>
    """, unsafe_allow_html=True)


top3 = filtered.nlargest(3, "risk_score")[["FacilityName", "County", "risk_score"]]
top3_text = "; ".join(
    f"{r['FacilityName']} ({r['County']}, risk {r['risk_score']:.1f})"
    for _, r in top3.iterrows()
)

# ---------------------------------------------------------------------------
# Future risk summary + year input
# ---------------------------------------------------------------------------

def llm_year_summary(year, trends, top3_text, client):
    future_ctx = build_future_context(
        f"What is the projected water contamination risk for Connecticut in {year}?",
        trends,
    )
    prompt = f"""You are a water contamination risk analyst for Connecticut.
{future_ctx}
Top facilities by current risk: {top3_text}

In 2-3 sentences, summarize the projected water contamination outlook for Connecticut in {year}. Mention the most concerning counties or facilities and what the trend implies."""
    try:
        resp = client.chat.completions.create(
            model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Could not generate summary: {e}"


st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
st.markdown("<p style='font-size:0.7rem; letter-spacing:0.15em; text-transform:uppercase; color:#38bdf8;'>Risk Projection</p>", unsafe_allow_html=True)

proj_col1, proj_col2 = st.columns([1, 3])
with proj_col1:
    custom_year = st.number_input("Enter any year", min_value=2023, max_value=2100, value=int(projection_year) if is_future else 2030, step=1)
    run_proj = st.button("Predict")

with proj_col2:
    summary_key = f"proj_summary_{custom_year}"
    if run_proj or (is_future and projection_year == custom_year and summary_key not in st.session_state):
        with st.spinner(f"Generating outlook for {custom_year}…"):
            st.session_state[summary_key] = llm_year_summary(custom_year, trends, top3_text, llm_client)
    if summary_key in st.session_state:
        st.markdown(
            f"<div style='background:#ffffff; border:1px solid #90c4e8; border-radius:8px; "
            f"padding:1rem; font-size:0.85rem; color:#000000;'>"
            f"<b style='color:#000000;'>{custom_year} outlook</b><br><br>"
            f"{st.session_state[summary_key]}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<p style='color:#4a6080; font-size:0.85rem;'>Enter a year and click Predict to generate an LLM risk outlook.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# LLaMA risk assistant
# ---------------------------------------------------------------------------

st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
st.markdown("""
<p style='font-size:0.7rem; letter-spacing:0.15em; text-transform:uppercase; color:#38bdf8;'>
  Risk Assistant
</p>
<p style='font-size:0.85rem; color:#8fadc7; margin-top:0.2rem;'>
  Ask anything about the current map, facilities, or conditions.
</p>
""", unsafe_allow_html=True)

location_ctx = ""
if clicked:
    location_ctx = f"""
Selected location: {clat:.4f}, {clon:.4f}
- Overall risk score: {risk.score}/100
- Industrial facility score: {risk.facility_score}/100 ({risk.facility_count} facilities)
- Stream contamination score: {risk.downstream_score}/100
- Wastewater plant score: {risk.wastewater_score}/100 ({risk.wastewater_plant_count} plants)
- Drinking water violations (SDWA): {risk.sdwa_score}/100 ({risk.sdwa_system_count} systems)
- Top contributors: {', '.join(f"{r['Facility']} ({r['Dist (km)']:.1f} km)" for _, r in risk.top_contributors.rename(columns={"FacilityName":"Facility","distance_km":"Dist (km)"}).iterrows()) if not risk.top_contributors.empty else 'none'}"""

system_prompt = f"""You are a water contamination risk analyst for Connecticut.
Current map state:
- Viewing year: {projection_year} {"(projected)" if is_future else "(current data)"}
- Pathway: {pathway}
- Rainfall: {rainfall} in/month
- Risk multiplier: {combined_mult:.2f}x
- Facilities shown: {len(filtered)}
- Top facilities by risk: {top3_text}
{location_ctx}
Answer concisely and ground your response in the data above."""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    role_color = "#e0eaff" if msg["role"] == "user" else "#38bdf8"
    role_label = "You" if msg["role"] == "user" else "LLaMA"
    st.markdown(
        f"<div style='margin:0.4rem 0; font-size:0.85rem;'>"
        f"<span style='color:{role_color}; font-weight:600;'>{role_label}:</span> "
        f"<span style='color:#a8c4e0;'>{msg['content']}</span></div>",
        unsafe_allow_html=True,
    )

user_input = st.chat_input("Ask about risk, facilities, or conditions...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    future_ctx = build_future_context(user_input, trends)
    active_system = (system_prompt + "\n\n" + future_ctx) if future_ctx else system_prompt
    messages = [{"role": "system", "content": active_system}] + st.session_state.chat_history

    try:
        response = llm_client.chat.completions.create(
            model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            messages=messages,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"Could not reach local model: {e}"

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.rerun()

"""Streamlit app — Connecticut water contamination risk heatmap."""

import numpy as np
import pandas as pd
import folium
import streamlit as st
from folium.plugins import HeatMap
from pathlib import Path
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent

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
    background: #0e1117;
    color: #dce8f0;
  }
  [data-testid="stHeader"] { background: transparent; }

  /* ── sidebar ── */
  [data-testid="stSidebar"] {
    background: #13181f;
    border-right: 1px solid #1f2d3d;
  }
  [data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
    color: #8fadc7 !important;
  }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: #38bdf8 !important;
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
    background: #13181f;
    border: 1px solid #1f2d3d;
    border-radius: 8px;
    padding: 12px 16px;
  }
  [data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    color: #38bdf8 !important;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  [data-testid="stMetricValue"] {
    font-family: 'DM Sans', sans-serif !important;
    color: #dce8f0 !important;
    font-size: 1.5rem;
    font-weight: 700;
  }

  /* ── dataframe ── */
  [data-testid="stDataFrame"] { border: 1px solid #1f2d3d; border-radius: 8px; overflow: hidden; }
  .stDataFrame thead th {
    font-family: 'DM Mono', monospace !important;
    background: #13181f !important;
    color: #38bdf8 !important;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .stDataFrame tbody tr:nth-child(even) { background: #0e1117 !important; }
  .stDataFrame tbody tr:hover { background: #1a2535 !important; }

  /* ── section divider ── */
  hr { border-color: #1f2d3d; }

  /* ── hide default streamlit chrome ── */
  #MainMenu, footer { visibility: hidden; }

  /* ── map container ── */
  iframe { border-radius: 10px; border: 1px solid #1f2d3d; }

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
  .risk-low    { background: #052e16; color: #4ade80; border: 1px solid #166534; }
  .risk-medium { background: #1c1400; color: #fbbf24; border: 1px solid #92400e; }
  .risk-high   { background: #2d0a0a; color: #f87171; border: 1px solid #7f1d1d; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div style="padding: 1.2rem 0 0.4rem 0;">
  <span style="font-size:0.75rem; letter-spacing:0.15em; color:#5bc8f5; text-transform:uppercase;">
    EPA RSEI v23.12 · Industrial Facility Water Releases
  </span>
  <h1 style="margin:0.2rem 0 0 0; font-size:2rem; font-weight:800; color:#e0eaff; line-height:1.1;">
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
    fac = pd.read_parquet(ROOT / "data" / "interim" / "facilities_clean.parquet")
    stream = pd.read_parquet(ROOT / "data" / "processed" / "stream_scores.parquet")
    merged = fac.merge(stream[["FinalCOMID", "segment_score"]], on="FinalCOMID", how="left")
    merged["segment_score"] = merged["segment_score"].fillna(0)
    max_score = merged["segment_score"].max()
    merged["weight"] = merged["segment_score"] / max_score if max_score > 0 else 0
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

df = load_data()
rel_stats = load_releases()

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Facility Filters")

show_water_only = st.sidebar.checkbox("Water-releasing facilities only", value=True)
min_score = st.sidebar.slider("Minimum segment score", 0.0, float(df["segment_score"].max()), 0.0, 0.1)

filtered = df.copy()
if show_water_only:
    filtered = filtered[filtered["WaterReleases"] == True]
filtered = filtered[filtered["segment_score"] >= min_score]

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
        gradient     = {"0.0": "#00aaff", "0.4": "#00ffcc",
                        "0.65": "#aaff00", "0.8": "#ffcc00", "1.0": "#ff2200"},
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
season = st.sidebar.selectbox(
    "Season", ["Spring", "Summer", "Fall", "Winter"],
    help="Season affects streamflow volume and contaminant dilution.",
)
temp_f = st.sidebar.slider(
    "Temperature (°F)", 20, 95, 60, 5,
    help="Higher temperatures accelerate chemical reactions and reduce dilution via evaporation.",
)
soil_type = st.sidebar.selectbox(
    "Soil Permeability", ["Clay (low)", "Loam (medium)", "Sandy (high)"],
    help="Controls how much surface runoff vs. groundwater infiltration occurs.",
)

# --- Compute environmental modifiers ---
rainfall_weight_mult = 1.0 + (rainfall / 10.0) * 0.4
rainfall_radius_mult = 1.0 + (rainfall / 10.0) * 0.6
rainfall_blur_mult   = 1.0 + (rainfall / 10.0) * 0.5
season_mult = {"Spring": 1.2, "Summer": 1.5, "Fall": 1.0, "Winter": 0.5}[season]
temp_mult = 0.7 + (temp_f - 20) / (95 - 20) * 0.6
soil_weight_mult = {"Clay (low)": 1.3,  "Loam (medium)": 1.0, "Sandy (high)": 0.65}[soil_type]
soil_radius_mult = {"Clay (low)": 1.15, "Loam (medium)": 1.0, "Sandy (high)": 0.85}[soil_type]

env_weight_mult = rainfall_weight_mult * season_mult * temp_mult * soil_weight_mult
env_radius      = int(20 * rainfall_radius_mult * soil_radius_mult)
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
m2.metric("Avg Score", f"{filtered['segment_score'].mean():.1f}")
m3.metric("Max Score", f"{filtered['segment_score'].max():.1f}")
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
m = folium.Map(location=CT_CENTER, zoom_start=9, tiles="CartoDB dark_matter")

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

top = filtered.nlargest(20, "segment_score")
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
            f"<span style='color:#888'>Score:</span> {row['segment_score']:.2f}<br>"
            f"<span style='color:#888'>NAICS:</span> {row['ModeledNAICS']}"
            f"</div>"
        ),
    ).add_to(m)

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

col1, col2 = st.columns([3, 1])

with col1:
    st_folium(m, width=None, height=580)

with col2:
    st.markdown("<p style='font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:#38bdf8; margin-bottom:0.4rem;'>Top 10 Facilities</p>", unsafe_allow_html=True)
    st.dataframe(
        filtered.nlargest(10, "segment_score")[["FacilityName", "County", "segment_score"]]
        .rename(columns={"FacilityName": "Facility", "segment_score": "Score"})
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
        <span style="color:#a8c4e0;">Direct Water</span>
        <span style="color:#e0eaff; font-weight:600;">{rel_stats['direct_lbs']/1e9:.1f}B lbs ({direct_pct:.0f}%)</span>
      </div>
      <div style="background:#1f2d3d; border-radius:3px; height:6px; margin-bottom:10px;">
        <div style="background:{direct_color}; width:{direct_pct:.0f}%; height:6px; border-radius:3px;"></div>
      </div>
      <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
        <span style="color:#a8c4e0;">POTW Transfer</span>
        <span style="color:#e0eaff; font-weight:600;">{rel_stats['potw_lbs']/1e9:.1f}B lbs ({potw_pct:.0f}%)</span>
      </div>
      <div style="background:#1f2d3d; border-radius:3px; height:6px;">
        <div style="background:{potw_color}; width:{potw_pct:.0f}%; height:6px; border-radius:3px;"></div>
      </div>
    </div>
    <div style="font-size:0.68rem; color:#4a6080; border-top:1px solid #1f2d3d; padding-top:0.5rem;">
      Pathway weight mult: <b style="color:#38bdf8;">{pw["weight_mult"]:.2f}×</b><br>
      TEF toxic load: <b style="color:#38bdf8;">{rel_stats['toxic_load']:.3f}</b> (lb·TEF units)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:#38bdf8; margin:1rem 0 0.4rem;'>Score Distribution</p>", unsafe_allow_html=True)
    st.bar_chart(
        filtered[filtered["segment_score"] > 0]["segment_score"]
        .value_counts(bins=10)
        .sort_index(),
        color="#0ea5e9",
    )

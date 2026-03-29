"""Streamlit app — Connecticut water contamination risk heatmap."""

import numpy as np
import pandas as pd
import folium
import streamlit as st
from folium.plugins import HeatMap
from pathlib import Path
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="CT Water Risk", layout="wide")
st.title("Connecticut Water Contamination Risk")
st.caption("Based on EPA RSEI v23.12 — industrial facility water releases")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    fac = pd.read_parquet(ROOT / "data" / "interim" / "facilities_clean.parquet")
    stream = pd.read_parquet(ROOT / "data" / "processed" / "stream_scores.parquet")
    merged = fac.merge(stream[["FinalCOMID", "segment_score"]], on="FinalCOMID", how="left")
    merged["segment_score"] = merged["segment_score"].fillna(0)
    # Normalise to 0–1 for heatmap weight
    max_score = merged["segment_score"].max()
    merged["weight"] = merged["segment_score"] / max_score if max_score > 0 else 0
    return merged

df = load_data()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")

show_water_only = st.sidebar.checkbox("Water-releasing facilities only", value=True)
min_score = st.sidebar.slider("Minimum segment score", 0.0, float(df["segment_score"].max()), 0.0, 0.1)

filtered = df.copy()
if show_water_only:
    filtered = filtered[filtered["WaterReleases"] == True]
filtered = filtered[filtered["segment_score"] >= min_score]

st.sidebar.markdown(f"**{len(filtered):,}** facilities shown")

# ---------------------------------------------------------------------------
# Build map
# ---------------------------------------------------------------------------

CT_CENTER = [41.6, -72.7]

m = folium.Map(location=CT_CENTER, zoom_start=9, tiles="CartoDB positron")

# Heatmap layer — weighted by segment_score
heat_data = [
    [row["Latitude"], row["Longitude"], row["weight"]]
    for _, row in filtered.iterrows()
    if row["weight"] > 0
]

if heat_data:
    HeatMap(
        heat_data,
        radius=20,
        blur=25,
        min_opacity=0.3,
        gradient={
            "0.0": "blue",
            "0.4": "cyan",
            "0.65": "lime",
            "0.8": "yellow",
            "1.0": "red",
        },
    ).add_to(m)

# Facility markers for top 20 by score
top = filtered.nlargest(20, "segment_score")
for _, row in top.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=6,
        color="darkred",
        fill=True,
        fill_color="red",
        fill_opacity=0.8,
        tooltip=folium.Tooltip(
            f"<b>{row['FacilityName']}</b><br>"
            f"County: {row['County']}<br>"
            f"Segment score: {row['segment_score']:.2f}<br>"
            f"NAICS: {row['ModeledNAICS']}"
        ),
    ).add_to(m)

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

col1, col2 = st.columns([3, 1])

with col1:
    st_folium(m, width=900, height=600)

with col2:
    st.subheader("Top 10 Facilities")
    st.dataframe(
        filtered.nlargest(10, "segment_score")[["FacilityName", "County", "segment_score"]]
        .rename(columns={"segment_score": "Score"})
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Score Distribution")
    st.bar_chart(
        filtered[filtered["segment_score"] > 0]["segment_score"]
        .value_counts(bins=10)
        .sort_index()
    )

"""Prompt builders for agent-facing remediation reports."""

from __future__ import annotations

import json


def build_brief_prompt(user_profile: dict, forcing: dict, config_name: str) -> str:
    """Create the Gemini prompt for the remediation brief."""
    hydrology_summary = {
        station_id: {
            "peak_discharge_m3s": float(values.max()) if len(values) else 0.0,
            "mean_discharge_m3s": float(values.mean()) if len(values) else 0.0,
        }
        for station_id, values in forcing["hydrology"].items()
    }
    weather = forcing["weather"]
    forcing_summary = forcing.get("forcing_summary", {})
    return (
        "You are an environmental operations copilot for a Bangladesh arsenic digital twin. "
        "Write a concise markdown remediation brief with sections: Situation, Risk Drivers, "
        "Recommended Actions, and Authenticated Context. "
        f"Run name: {config_name}. "
        f"Authenticated user: {user_profile.get('name') or user_profile.get('email')}. "
        f"Weather stats: total_precipitation_mm={float(weather[:, 3].sum()):.2f}, "
        f"max_temperature_c={float(weather[:, 2].max()):.2f}, "
        f"mean_wind_speed_ms={float(weather[:, 0].mean()):.2f}. "
        f"Hydrology stats: {json.dumps(hydrology_summary)}. "
        f"Gemini forcing summary: {json.dumps(forcing_summary)}. "
        "Focus on monsoon-sensitive arsenic spread, near-term mitigation, and user-scoped operational actions."
    )

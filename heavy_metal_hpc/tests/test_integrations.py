"""Tests for Gemini/Auth0/data-loading integrations."""

from __future__ import annotations

from datetime import date

import pytest

import numpy as np
from flask import Flask

from src.agents.auth0 import Auth0AgentContext, Auth0ConfigurationError
from src.agents.reporting import build_brief_prompt
from src.ai.gemini import GeminiClient
from src.api.hydrology import HydrologyAPI
from src.api.loader import DataLoader
from src.api.weather import WeatherAPI, WeatherRecord
from src.utils.config import load_config
from src.web.app import create_app


def test_load_config_parses_ai_sections(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
name: test
domain:
  x_min: 0
  x_max: 1
  y_min: 0
  y_max: 1
  nx: 2
  ny: 2
time:
  start_date: 2026-01-01
  end_date: 2026-01-02
gemini:
  enabled: true
auth0:
  enabled: true
  domain: example.us.auth0.com
  audience: https://example/api
"""
    )
    cfg = load_config(path)
    assert cfg.gemini.enabled is True
    assert cfg.auth0.domain == "example.us.auth0.com"


def test_auth0_context_requires_env(monkeypatch) -> None:
    monkeypatch.delenv("AUTH0_CLIENT_ID", raising=False)
    monkeypatch.delenv("AUTH0_CLIENT_SECRET", raising=False)
    with pytest.raises(Auth0ConfigurationError):
        Auth0AgentContext.from_env("tenant.auth0.com", "https://api")


def test_auth0_headers(monkeypatch) -> None:
    monkeypatch.setenv("AUTH0_CLIENT_ID", "abc")
    monkeypatch.setenv("AUTH0_CLIENT_SECRET", "def")
    ctx = Auth0AgentContext.from_env("tenant.auth0.com", "https://api")
    assert ctx.token_endpoint() == "https://tenant.auth0.com/oauth/token"
    assert ctx.device_authorization_endpoint() == "https://tenant.auth0.com/oauth/device/code"
    assert ctx.user_api_headers("token")["Authorization"] == "Bearer token"


def test_gemini_generate_json(monkeypatch) -> None:
    client = GeminiClient(api_key="secret")

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": '{"monsoon_risk":"high","likely_hotspot_driver":"runoff"}'}
                            ]
                        }
                    }
                ]
            }

    def fake_post(*args, **kwargs):
        return DummyResponse()

    monkeypatch.setattr("src.ai.gemini.requests.post", fake_post)
    data = client.generate_json("hello")
    assert data["monsoon_risk"] == "high"


def test_data_loader_uses_gemini_summary() -> None:
    class FakeWeatherAPI(WeatherAPI):
        def __init__(self) -> None:
            super().__init__(base_url="unused", api_key="")

        def fetch(self, start, end, lat, lon):
            return [
                WeatherRecord(
                    timestamp="2026-01-01T00:00",
                    wind_speed_ms=1.0,
                    wind_direction_deg=180.0,
                    air_temp_c=25.0,
                    precipitation_mm=12.0,
                    solar_radiation_wm2=200.0,
                    relative_humidity=0.8,
                )
            ]

    class FakeGemini:
        def generate_json(self, prompt: str):
            assert "precipitation_mm" in prompt
            return {"monsoon_risk": "moderate"}

    loader = DataLoader(
        weather_api=FakeWeatherAPI(),
        hydrology_api=HydrologyAPI(base_url="synthetic", api_key=""),
        gemini_client=FakeGemini(),
    )
    result = loader.load(date(2026, 1, 1), date(2026, 1, 2))
    assert result["weather"].shape == (1, 6)
    assert result["forcing_summary"]["monsoon_risk"] == "moderate"


def test_build_brief_prompt_contains_user_and_forcing() -> None:
    prompt = build_brief_prompt(
        {"email": "user@example.com"},
        {
            "weather": np.array([[1.0, 180.0, 31.0, 12.0, 0.0, 0.8]]),
            "hydrology": {"north_inflow": np.array([120.0, 160.0])},
            "forcing_summary": {"monsoon_risk": "high"},
        },
        "demo-run",
    )
    assert "user@example.com" in prompt
    assert "demo-run" in prompt
    assert "monsoon_risk" in prompt


def test_create_app_exposes_index_route(tmp_path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
name: web-demo
domain:
  x_min: 0
  x_max: 1
  y_min: 0
  y_max: 1
  nx: 2
  ny: 2
time:
  start_date: 2026-01-01
  end_date: 2026-01-02
auth0:
  enabled: true
  domain: example.us.auth0.com
  audience: https://example/api
"""
    )
    app = create_app(cfg)
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Bangladesh Arsenic Operations Copilot" in response.data


def test_start_login_redirects_to_auth0(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
name: web-demo
domain:
  x_min: 0
  x_max: 1
  y_min: 0
  y_max: 1
  nx: 2
  ny: 2
time:
  start_date: 2026-01-01
  end_date: 2026-01-02
auth0:
  enabled: true
  domain: example.us.auth0.com
  audience: https://example/api
"""
    )
    monkeypatch.setenv("AUTH0_CLIENT_ID", "cid")
    monkeypatch.setenv("AUTH0_CLIENT_SECRET", "secret")
    app = create_app(cfg)
    client = app.test_client()
    response = client.get("/login")
    assert response.status_code == 302
    assert "example.us.auth0.com/authorize" in response.headers["Location"]


def test_map_simulation_endpoint_returns_frames(tmp_path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
name: web-demo
domain:
  x_min: 0
  x_max: 1
  y_min: 0
  y_max: 1
  nx: 6
  ny: 5
time:
  start_date: 2026-01-01
  end_date: 2026-01-02
numerics:
  dt: 300
  n_steps: 12
  output_interval: 3
auth0:
  enabled: true
  domain: example.us.auth0.com
  audience: https://example/api
"""
    )
    app = create_app(cfg)
    client = app.test_client()
    response = client.get("/api/map-simulation")
    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["frames"]) == payload["stats"]["snapshot_count"]
    assert len(payload["frames"][0]) == 6
    assert len(payload["frames"][0][0]) == 5
    assert payload["stats"]["max_concentration_ugL"] > 0.0

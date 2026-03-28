#!/usr/bin/env python
"""Authenticate with Auth0, load forcing, call Gemini, and write a remediation brief."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.auth0 import Auth0AgentContext
from src.agents.reporting import build_brief_prompt
from src.ai.gemini import GeminiClient
from src.api.hydrology import HydrologyAPI
from src.api.loader import DataLoader
from src.api.weather import WeatherAPI
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-end CLI demo for Auth0-authenticated forcing analysis with Gemini."
    )
    parser.add_argument("--config", required=True, help="Path to YAML run configuration.")
    parser.add_argument(
        "--output",
        default="data/processed/remediation_brief.md",
        help="Where to write the generated remediation brief.",
    )
    parser.add_argument(
        "--scope",
        default="openid profile email",
        help="OAuth scopes requested during Auth0 device login.",
    )
    return parser.parse_args()

def main() -> None:
    """Entry point."""
    args = parse_args()
    cfg = load_config(args.config)

    auth0 = Auth0AgentContext.from_env(
        domain=cfg.auth0.domain,
        audience=cfg.auth0.audience,
        client_id_env=cfg.auth0.client_id_env,
        client_secret_env=cfg.auth0.client_secret_env,
        token_vault_audience=cfg.auth0.token_vault_audience,
    )
    gemini = GeminiClient.from_env(
        api_key_env=cfg.gemini.api_key_env,
        model=cfg.gemini.model,
        base_url=cfg.gemini.base_url,
    )

    flow = auth0.start_device_flow(scope=args.scope)
    print(f"Open this URL in your browser: {flow['verification_uri_complete']}")
    print(f"Or visit {flow['verification_uri']} and enter code: {flow['user_code']}")
    token = auth0.poll_device_token(
        device_code=flow["device_code"],
        interval=int(flow.get("interval", 5)),
    )
    user_profile = auth0.fetch_user_profile(token["access_token"])

    weather_api = WeatherAPI(
        base_url=cfg.data_sources.weather_base_url,
        api_key="",
        cache_dir=cfg.data_sources.cache_dir,
    )
    hydrology_api = HydrologyAPI(
        base_url=cfg.data_sources.hydrology_base_url,
        api_key="",
        cache_dir=cfg.data_sources.cache_dir,
    )
    loader = DataLoader(
        weather_api=weather_api,
        hydrology_api=hydrology_api,
        cache_dir=cfg.data_sources.cache_dir,
        gemini_client=gemini,
    )
    if cfg.time is None:
        raise ValueError("Config must define a time window for the end-to-end demo.")
    forcing = loader.load(cfg.time.start_date, cfg.time.end_date)

    prompt = build_brief_prompt(user_profile, forcing, cfg.name)
    brief = gemini.generate_text(
        prompt=prompt,
        system_instruction=(
            "Return practical, safety-aware markdown for environmental remediation planning. "
            "Keep the tone executive and action-oriented."
        ),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(brief)

    print(f"Authenticated as: {user_profile.get('email', user_profile.get('sub', 'unknown user'))}")
    print(f"Saved remediation brief to {output_path}")


if __name__ == "__main__":
    main()

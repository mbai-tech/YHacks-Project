"""YAML configuration loader with Pydantic schema validation."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..model.parameters import PhysicalParameters, NumericalParameters


class DomainConfig(BaseModel):
    """Spatial domain specification."""

    x_min: float = Field(..., description="West boundary (m or decimal degrees)")
    x_max: float = Field(..., description="East boundary")
    y_min: float = Field(..., description="South boundary")
    y_max: float = Field(..., description="North boundary")
    nx: int = Field(..., gt=0)
    ny: int = Field(..., gt=0)


class TimeConfig(BaseModel):
    """Simulation window."""

    start_date: date
    end_date: date


class GeminiConfig(BaseModel):
    """Gemini API configuration."""

    enabled: bool = False
    api_key_env: str = "GEMINI_API_KEY"
    model: str = "gemini-2.5-flash"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"


class Auth0Config(BaseModel):
    """Auth0 for AI Agents configuration."""

    enabled: bool = False
    domain: str = ""
    audience: str = ""
    client_id_env: str = "AUTH0_CLIENT_ID"
    client_secret_env: str = "AUTH0_CLIENT_SECRET"
    token_vault_audience: str | None = None


class DataSourceConfig(BaseModel):
    """External data source configuration."""

    weather_base_url: str = "https://api.open-meteo.com/v1/forecast"
    hydrology_base_url: str = "synthetic"
    cache_dir: str = "data/cache"


class RunConfig(BaseModel):
    """Top-level simulation run configuration."""

    name: str = "default"
    domain: DomainConfig
    time: TimeConfig | None = None
    physics: PhysicalParameters = Field(default_factory=PhysicalParameters)
    numerics: NumericalParameters = Field(default_factory=NumericalParameters)
    data_sources: DataSourceConfig = Field(default_factory=DataSourceConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    auth0: Auth0Config = Field(default_factory=Auth0Config)
    output_dir: str = "data/processed"
    random_seed: int = 42


def load_config(path: str | Path) -> RunConfig:
    """Parse a YAML file into a validated :class:`RunConfig`.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    RunConfig

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    pydantic.ValidationError
        If any required field is missing or has an invalid value.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    if hasattr(RunConfig, "model_validate"):
        return RunConfig.model_validate(raw)
    return RunConfig.parse_obj(raw)


def dump_config(config: RunConfig, path: str | Path) -> None:
    """Serialise a :class:`RunConfig` to a YAML file.

    Parameters
    ----------
    config:
        Configuration to write.
    path:
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        data = config.model_dump() if hasattr(config, "model_dump") else config.dict()
        yaml.safe_dump(data, fh, default_flow_style=False)

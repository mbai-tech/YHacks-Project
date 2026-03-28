"""YAML configuration loader with Pydantic schema validation."""

from __future__ import annotations

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


class RunConfig(BaseModel):
    """Top-level simulation run configuration."""

    name: str = "default"
    domain: DomainConfig
    physics: PhysicalParameters = Field(default_factory=PhysicalParameters)
    numerics: NumericalParameters = Field(default_factory=NumericalParameters)
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
    return RunConfig.model_validate(raw)


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
        yaml.safe_dump(config.model_dump(), fh, default_flow_style=False)

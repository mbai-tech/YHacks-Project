"""Physical and numerical parameter containers with Pydantic validation."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PhysicalParameters(BaseModel):
    """Calibratable physical parameters of the heavy-metal transport model."""

    diffusivity: float = Field(1e-3, gt=0, description="Effective diffusivity (m² s⁻¹)")
    k_deposition: float = Field(1e-5, gt=0, description="Sediment deposition rate (m s⁻¹)")
    k_resuspension: float = Field(1e-6, gt=0, description="Sediment resuspension rate (s⁻¹)")
    partition_coefficient: float = Field(
        1e3, gt=0, description="Kd: sediment-water partition coefficient (L kg⁻¹)"
    )
    decay_rate: float = Field(0.0, ge=0, description="First-order decay / uptake rate (s⁻¹)")

    def to_vector(self) -> list[float]:
        """Return parameters as an ordered list for optimisation routines."""
        return [
            self.diffusivity,
            self.k_deposition,
            self.k_resuspension,
            self.partition_coefficient,
            self.decay_rate,
        ]

    @classmethod
    def from_vector(cls, v: list[float]) -> "PhysicalParameters":
        """Construct from an ordered list produced by :meth:`to_vector`."""
        keys = ["diffusivity", "k_deposition", "k_resuspension", "partition_coefficient", "decay_rate"]
        return cls(**dict(zip(keys, v)))


class NumericalParameters(BaseModel):
    """Numerical scheme settings."""

    dt: float = Field(60.0, gt=0, description="Time step (s)")
    n_steps: int = Field(1440, gt=0, description="Number of time steps per run")
    output_interval: int = Field(60, gt=0, description="Steps between saved snapshots")
    cfl_target: float = Field(0.8, gt=0, le=1.0, description="Target CFL number")

    @field_validator("output_interval")
    @classmethod
    def _output_divides_steps(cls, v: int, info) -> int:
        n = info.data.get("n_steps")
        if n and n % v != 0:
            raise ValueError("output_interval must divide n_steps evenly")
        return v

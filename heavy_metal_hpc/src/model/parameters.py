"""Parameter data model for heavy-metal transport in Baiyangdian Lake.

All rate constants use SI base units (s, m, kg).  Literature default values
are drawn from published studies on shallow Chinese lakes with comparable
sediment and hydrological characteristics.

References (abbreviated):
  - Fan et al. (2016) Environ. Sci. Pollut. Res. — Baiyangdian heavy metal loading
  - Li et al. (2020) Water Research — sediment-water exchange rates
  - Thomann & Mueller (1987) — turbulent diffusivity range for lakes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .state import METALS, METAL_INDEX, MetalName, N_METALS


# ---------------------------------------------------------------------------
# Per-metal transport parameters
# ---------------------------------------------------------------------------

@dataclass
class MetalParams:
    """All transport rate constants for a single heavy-metal species.

    Parameters
    ----------
    name:
        Metal symbol — one of ``"Pb"``, ``"Cu"``, ``"Cd"``, ``"Cr"``.
    diffusivity:
        Turbulent (eddy) diffusivity (m² s⁻¹).  Governs lateral spreading;
        molecular diffusivity (~10⁻⁹ m² s⁻¹) is negligible compared with
        turbulent values (10⁻³–10⁻¹ m² s⁻¹) in a shallow lake.
    settling_rate:
        Particle settling / deposition velocity (m s⁻¹).  Positive values
        transfer dissolved metal from the water column to the bed.
    resuspension_rate:
        First-order resuspension rate constant (s⁻¹).  Transfers metal from
        bed sediment back into the water column.
    decay_rate:
        First-order removal rate (s⁻¹) lumping biological uptake, chemical
        precipitation, and other irreversible loss processes.  Set to 0 if
        not applicable.
    """

    name: MetalName
    diffusivity: float       # m² s⁻¹
    settling_rate: float     # m s⁻¹
    resuspension_rate: float # s⁻¹
    decay_rate: float        # s⁻¹

    def __post_init__(self) -> None:
        if self.name not in METALS:
            raise ValueError(f"Unknown metal {self.name!r}. Must be one of {METALS}.")
        if self.diffusivity <= 0:
            raise ValueError(f"diffusivity must be > 0, got {self.diffusivity}")
        if self.settling_rate < 0:
            raise ValueError(f"settling_rate must be >= 0, got {self.settling_rate}")
        if self.resuspension_rate < 0:
            raise ValueError(f"resuspension_rate must be >= 0, got {self.resuspension_rate}")
        if self.decay_rate < 0:
            raise ValueError(f"decay_rate must be >= 0, got {self.decay_rate}")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of parameter name → value."""
        return {
            "name": self.name,
            "diffusivity": self.diffusivity,
            "settling_rate": self.settling_rate,
            "resuspension_rate": self.resuspension_rate,
            "decay_rate": self.decay_rate,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MetalParams":
        """Construct from a dict produced by :meth:`to_dict`."""
        return cls(
            name=d["name"],
            diffusivity=float(d["diffusivity"]),
            settling_rate=float(d["settling_rate"]),
            resuspension_rate=float(d["resuspension_rate"]),
            decay_rate=float(d["decay_rate"]),
        )


@dataclass
class MetalTransportParameters:
    """Per-metal transport parameters for all four heavy-metal species.

    Attributes are named after each metal for direct attribute access
    (``params.Pb``, ``params.Cu``, etc.) as well as vectorised access via
    :meth:`as_arrays`.

    Parameters
    ----------
    Pb, Cu, Cd, Cr:
        :class:`MetalParams` instances for each species.
    """

    Pb: MetalParams
    Cu: MetalParams
    Cd: MetalParams
    Cr: MetalParams

    def __post_init__(self) -> None:
        # Ensure each MetalParams carries the correct name tag
        for metal in METALS:
            p: MetalParams = getattr(self, metal)
            if p.name != metal:
                raise ValueError(
                    f"MetalParams at attribute {metal!r} has name={p.name!r}. "
                    f"They must match."
                )

    def __getitem__(self, metal: MetalName) -> MetalParams:
        """Subscript access: ``params["Cd"]``."""
        if metal not in METAL_INDEX:
            raise KeyError(f"Unknown metal {metal!r}. Choose from {METALS}.")
        return getattr(self, metal)

    def as_dict(self) -> dict[MetalName, MetalParams]:
        """Return an ordered dict of metal → MetalParams."""
        return {m: getattr(self, m) for m in METALS}

    # ------------------------------------------------------------------
    # Vectorised accessors — return 1-D arrays ordered by METALS
    # ------------------------------------------------------------------

    def diffusivities(self) -> np.ndarray:
        """Diffusivity for each metal, shape (N_METALS,), ordered by METALS."""
        return np.array([getattr(self, m).diffusivity for m in METALS])

    def settling_rates(self) -> np.ndarray:
        """Settling rate for each metal, shape (N_METALS,), ordered by METALS."""
        return np.array([getattr(self, m).settling_rate for m in METALS])

    def resuspension_rates(self) -> np.ndarray:
        """Resuspension rate for each metal, shape (N_METALS,), ordered by METALS."""
        return np.array([getattr(self, m).resuspension_rate for m in METALS])

    def decay_rates(self) -> np.ndarray:
        """Decay rate for each metal, shape (N_METALS,), ordered by METALS."""
        return np.array([getattr(self, m).decay_rate for m in METALS])

    @classmethod
    def defaults(cls) -> "MetalTransportParameters":
        """Literature-based default parameters for Baiyangdian Lake conditions.

        Values represent order-of-magnitude estimates appropriate for a
        shallow (~2 m), eutrophic lake with fine-grained sediment.  All
        rates should be calibrated against field observations before use in
        production runs.

        Metal-specific notes
        --------------------
        Pb  : Strong sorption to organic matter; low resuspension mobility.
        Cu  : Moderately mobile; elevated agricultural runoff in Baiyangdian.
        Cd  : Highly mobile in water column; low decay rate.
        Cr  : Mixed-valence; lower diffusivity reflects faster precipitation of Cr(III).
        """
        return cls(
            Pb=MetalParams(
                name="Pb",
                diffusivity=5.0e-3,    # m² s⁻¹
                settling_rate=2.0e-5,  # m s⁻¹
                resuspension_rate=1.5e-7,  # s⁻¹
                decay_rate=5.0e-8,         # s⁻¹
            ),
            Cu=MetalParams(
                name="Cu",
                diffusivity=8.0e-3,
                settling_rate=1.5e-5,
                resuspension_rate=2.0e-7,
                decay_rate=3.0e-8,
            ),
            Cd=MetalParams(
                name="Cd",
                diffusivity=1.0e-2,
                settling_rate=1.0e-5,
                resuspension_rate=3.0e-7,
                decay_rate=1.0e-8,
            ),
            Cr=MetalParams(
                name="Cr",
                diffusivity=3.0e-3,
                settling_rate=2.5e-5,
                resuspension_rate=1.0e-7,
                decay_rate=8.0e-8,
            ),
        )


# ---------------------------------------------------------------------------
# Inflow boundary sources
# ---------------------------------------------------------------------------

@dataclass
class InflowSource:
    """A single river / channel inflow boundary entering the lake grid.

    Parameters
    ----------
    name:
        Human-readable label (e.g. ``"FuRiver_N"``).
    i, j:
        Grid-cell indices (0-based) of the inflow boundary cell.
    discharge:
        Volumetric flow rate (m³ s⁻¹).  Must be >= 0.
    concentrations:
        Mapping of metal name → dissolved concentration in the inflow
        (µg L⁻¹).  Missing metals are treated as zero.
    """

    name: str
    i: int
    j: int
    discharge: float                     # m³ s⁻¹
    concentrations: dict[MetalName, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.discharge < 0:
            raise ValueError(f"discharge must be >= 0, got {self.discharge}")
        unknown = set(self.concentrations) - set(METALS)
        if unknown:
            raise ValueError(f"Unknown metal keys in concentrations: {unknown}")

    def concentration_array(self) -> np.ndarray:
        """Return inflow concentrations as a (N_METALS,) array ordered by METALS.

        Metals absent from :attr:`concentrations` contribute zero.
        """
        return np.array(
            [self.concentrations.get(m, 0.0) for m in METALS],
            dtype=np.float64,
        )

    def mass_flux(self) -> np.ndarray:
        """Return metal mass flux (µg s⁻¹) for each metal, shape (N_METALS,).

        Calculated as discharge (m³ s⁻¹) × concentration (µg L⁻¹) × 1000 (L m⁻³).
        """
        return self.discharge * self.concentration_array() * 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "name": self.name,
            "i": self.i,
            "j": self.j,
            "discharge": self.discharge,
            "concentrations": dict(self.concentrations),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InflowSource":
        """Construct from a dict produced by :meth:`to_dict`."""
        return cls(
            name=d["name"],
            i=int(d["i"]),
            j=int(d["j"]),
            discharge=float(d["discharge"]),
            concentrations={k: float(v) for k, v in d.get("concentrations", {}).items()},
        )


# ---------------------------------------------------------------------------
# Time integration parameters
# ---------------------------------------------------------------------------

@dataclass
class TimeParameters:
    """Numerical time-stepping configuration.

    Parameters
    ----------
    dt:
        Length of a single time step (s).  Must satisfy the CFL condition
        for the chosen spatial resolution and flow speed.
    total_duration:
        Total simulation duration (s).
    output_interval:
        Number of time steps between consecutive snapshot saves.  Must
        divide ``n_steps`` evenly.
    """

    dt: float            # s
    total_duration: float  # s
    output_interval: int   # steps

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.total_duration <= self.dt:
            raise ValueError(
                f"total_duration ({self.total_duration} s) must exceed dt ({self.dt} s)"
            )
        if self.output_interval <= 0:
            raise ValueError(f"output_interval must be > 0, got {self.output_interval}")
        if self.n_steps % self.output_interval != 0:
            raise ValueError(
                f"output_interval ({self.output_interval}) must divide "
                f"n_steps ({self.n_steps}) evenly"
            )

    @property
    def n_steps(self) -> int:
        """Total number of time steps (rounded down to nearest integer)."""
        return int(self.total_duration / self.dt)

    @property
    def total_duration_hours(self) -> float:
        """Total simulation duration in hours."""
        return self.total_duration / 3600.0

    @property
    def n_outputs(self) -> int:
        """Number of snapshot saves that will occur during the run."""
        return self.n_steps // self.output_interval

    @property
    def output_times(self) -> np.ndarray:
        """1-D array of times (s) at which snapshots are saved."""
        return np.arange(1, self.n_outputs + 1) * self.output_interval * self.dt

    @classmethod
    def from_hours(
        cls,
        dt_seconds: float,
        duration_hours: float,
        output_interval: int = 60,
    ) -> "TimeParameters":
        """Convenience constructor specifying duration in hours.

        Parameters
        ----------
        dt_seconds:
            Time step in seconds.
        duration_hours:
            Total run duration in hours.
        output_interval:
            Steps between snapshot saves.
        """
        return cls(
            dt=dt_seconds,
            total_duration=duration_hours * 3600.0,
            output_interval=output_interval,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "dt": self.dt,
            "total_duration": self.total_duration,
            "output_interval": self.output_interval,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TimeParameters":
        """Construct from a dict produced by :meth:`to_dict`."""
        return cls(
            dt=float(d["dt"]),
            total_duration=float(d["total_duration"]),
            output_interval=int(d["output_interval"]),
        )


# ---------------------------------------------------------------------------
# Remediation parameters
# ---------------------------------------------------------------------------

@dataclass
class RemediationParameters:
    """Configuration for active remediation treatment.

    Parameters
    ----------
    active:
        Whether remediation is enabled for this run.
    intensity:
        Baseline treatment intensity applied uniformly to all active cells
        (dimensionless, 0–1).  Cell-level overrides are stored in the
        :class:`~state.RemediationMask`.
    removal_rate:
        Additional first-order removal rate (s⁻¹) applied to dissolved
        metal in treated cells on top of the natural decay rate.
    cost_per_unit_area:
        Economic cost per unit area per unit time of treatment
        (CNY m⁻² s⁻¹).  Used by the optimisation module.
    target_metals:
        Metals targeted by this remediation strategy.  Defaults to all four.
    """

    active: bool = False
    intensity: float = 0.0          # dimensionless, [0, 1]
    removal_rate: float = 0.0       # s⁻¹
    cost_per_unit_area: float = 0.0 # CNY m⁻² s⁻¹
    target_metals: tuple[MetalName, ...] = field(default_factory=lambda: tuple(METALS))

    def __post_init__(self) -> None:
        if not (0.0 <= self.intensity <= 1.0):
            raise ValueError(f"intensity must be in [0, 1], got {self.intensity}")
        if self.removal_rate < 0:
            raise ValueError(f"removal_rate must be >= 0, got {self.removal_rate}")
        if self.cost_per_unit_area < 0:
            raise ValueError(f"cost_per_unit_area must be >= 0, got {self.cost_per_unit_area}")
        unknown = set(self.target_metals) - set(METALS)
        if unknown:
            raise ValueError(f"Unknown metals in target_metals: {unknown}")

    def targets_metal(self, metal: MetalName) -> bool:
        """Return True if *metal* is listed in :attr:`target_metals`."""
        return metal in self.target_metals

    def effective_removal_rate(self, metal: MetalName) -> float:
        """Effective additional removal rate for *metal*.

        Returns :attr:`removal_rate` × :attr:`intensity` if the metal is
        targeted and remediation is active; otherwise 0.
        """
        if self.active and self.targets_metal(metal):
            return self.removal_rate * self.intensity
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "active": self.active,
            "intensity": self.intensity,
            "removal_rate": self.removal_rate,
            "cost_per_unit_area": self.cost_per_unit_area,
            "target_metals": list(self.target_metals),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RemediationParameters":
        """Construct from a dict produced by :meth:`to_dict`."""
        return cls(
            active=bool(d["active"]),
            intensity=float(d["intensity"]),
            removal_rate=float(d["removal_rate"]),
            cost_per_unit_area=float(d.get("cost_per_unit_area", 0.0)),
            target_metals=tuple(d.get("target_metals", METALS)),
        )


# ---------------------------------------------------------------------------
# Top-level parameter container
# ---------------------------------------------------------------------------

@dataclass
class ModelParameters:
    """Top-level container that bundles all parameter groups for one run.

    Parameters
    ----------
    transport:
        Per-metal transport rate constants.
    inflow_sources:
        List of river / channel inflow boundaries.
    time:
        Time-stepping configuration.
    remediation:
        Remediation treatment settings (optional; defaults to inactive).
    """

    transport: MetalTransportParameters
    inflow_sources: list[InflowSource]
    time: TimeParameters
    remediation: RemediationParameters = field(
        default_factory=RemediationParameters
    )

    def __post_init__(self) -> None:
        # Guard against duplicate inflow source names
        names = [src.name for src in self.inflow_sources]
        if len(names) != len(set(names)):
            duplicates = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate inflow source names: {duplicates}")

    def source_by_name(self, name: str) -> InflowSource:
        """Look up an inflow source by its :attr:`~InflowSource.name`.

        Raises
        ------
        KeyError
            If no source with that name exists.
        """
        for src in self.inflow_sources:
            if src.name == name:
                return src
        raise KeyError(f"No inflow source named {name!r}")

    def total_discharge(self) -> float:
        """Sum of discharge across all inflow sources (m³ s⁻¹)."""
        return sum(src.discharge for src in self.inflow_sources)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain, JSON-compatible dictionary."""
        return {
            "transport": {m: getattr(self.transport, m).to_dict() for m in METALS},
            "inflow_sources": [src.to_dict() for src in self.inflow_sources],
            "time": self.time.to_dict(),
            "remediation": self.remediation.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelParameters":
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        transport = MetalTransportParameters(
            **{m: MetalParams.from_dict(d["transport"][m]) for m in METALS}
        )
        inflow_sources = [InflowSource.from_dict(s) for s in d["inflow_sources"]]
        return cls(
            transport=transport,
            inflow_sources=inflow_sources,
            time=TimeParameters.from_dict(d["time"]),
            remediation=RemediationParameters.from_dict(d["remediation"]),
        )

    @classmethod
    def defaults(
        cls,
        dt: float = 60.0,
        duration_hours: float = 24.0,
        output_interval: int = 60,
    ) -> "ModelParameters":
        """Construct a ready-to-use parameter set with literature defaults.

        Parameters
        ----------
        dt:
            Time step (s).  Default: 60 s.
        duration_hours:
            Total run duration in hours.  Default: 24 h.
        output_interval:
            Steps between snapshot saves.  Default: 60 (saves every hour
            for dt=60 s).

        Returns
        -------
        ModelParameters
            Baiyangdian-relevant defaults with no inflow sources and
            remediation inactive.  Add inflow sources before running.
        """
        return cls(
            transport=MetalTransportParameters.defaults(),
            inflow_sources=[],
            time=TimeParameters.from_hours(dt, duration_hours, output_interval),
            remediation=RemediationParameters(active=False),
        )

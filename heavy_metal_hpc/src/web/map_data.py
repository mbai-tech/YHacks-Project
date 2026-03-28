"""Build a synthetic Bangladesh arsenic-spread scenario for the web map."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..api.loader import DataLoader
from ..grid.mesh import StructuredMesh
from ..model.simulator import Simulator
from ..model.state import SimulationState
from ..utils.config import RunConfig


def build_map_payload(config: RunConfig, loader: DataLoader) -> dict[str, Any]:
    """Run a Bangladesh-inspired arsenic spread scenario and serialize it for the UI."""
    if config.time is None:
        raise ValueError("Config must define a time window.")

    forcing_inputs = loader.load(config.time.start_date, config.time.end_date)
    mesh = StructuredMesh(
        x_min=0.0,
        x_max=240_000.0,
        y_min=0.0,
        y_max=320_000.0,
        nx=config.domain.nx,
        ny=config.domain.ny,
    )
    map_x = np.linspace(88.0, 92.8, config.domain.nx)
    map_y = np.linspace(20.6, 26.7, config.domain.ny)
    scenario = _build_scenario(mesh, config, forcing_inputs)
    simulator = Simulator(mesh=mesh, phys=config.physics, num=config.numerics)
    history = simulator.run(scenario["initial_state"], scenario["forcing"])
    stack = history.concentration_stack()
    vmax = float(np.nanpercentile(stack, 99.0))

    return {
        "x": np.round(map_x, 4).tolist(),
        "y": np.round(map_y, 4).tolist(),
        "times": [round(float(hours), 2) for hours in history.times() / 3600.0],
        "frames": [np.round(frame, 3).tolist() for frame in stack],
        "mask": scenario["mask"].astype(int).tolist(),
        "source_zones": {
            "groundwater": np.round(scenario["groundwater_map"], 3).tolist(),
            "runoff": np.round(scenario["runoff_map"], 3).tolist(),
            "river": np.round(scenario["river_map"], 3).tolist(),
            "remediation": np.round(scenario["remediation_map"], 3).tolist(),
        },
        "forcing_summary": forcing_inputs.get("forcing_summary", {}),
        "stats": {
            "max_concentration_ugL": round(float(np.nanmax(stack)), 3),
            "mean_concentration_ugL": round(float(np.nanmean(stack[-1])), 3),
            "hotspot_threshold_ugL": round(vmax * 0.7, 3),
            "snapshot_count": int(stack.shape[0]),
        },
    }


def _build_scenario(mesh: StructuredMesh, config: RunConfig, forcing_inputs: dict[str, Any]) -> dict[str, Any]:
    """Construct initial conditions and time-varying forcing fields."""
    weather = np.asarray(forcing_inputs["weather"], dtype=float)
    hydrology = forcing_inputs["hydrology"]

    n_steps = config.numerics.n_steps
    weather_steps = _resample_signal(weather[:, 3], n_steps)
    north_discharge = _resample_signal(np.asarray(hydrology["north_inflow"], dtype=float), n_steps)
    south_discharge = _resample_signal(np.asarray(hydrology["south_inflow"], dtype=float), n_steps)

    Xn = (mesh.X - mesh.x_min) / max(mesh.x_max - mesh.x_min, 1e-9)
    Yn = (mesh.Y - mesh.y_min) / max(mesh.y_max - mesh.y_min, 1e-9)
    river_centerline = 0.6 - 0.14 * np.sin(2.6 * (Yn - 0.1)) + 0.03 * np.cos(7.0 * Yn)
    river_map = np.exp(-((Xn - river_centerline) ** 2) / 0.008)
    delta_mask = (
        ((Xn - 0.53) ** 2) / 0.16 + ((Yn - 0.48) ** 2) / 0.32 < 1.0
    ) | (
        ((Xn - 0.48) ** 2) / 0.2 + ((Yn - 0.62) ** 2) / 0.18 < 1.0
    )
    delta_mask |= river_map > 0.16
    delta_mask = delta_mask.astype(float)

    groundwater_map = (
        1.2 * _gaussian(Xn, Yn, 0.28, 0.71, 0.08, 0.10)
        + 0.8 * _gaussian(Xn, Yn, 0.38, 0.53, 0.12, 0.09)
    ) * delta_mask
    runoff_map = (
        0.9 * _gaussian(Xn, Yn, 0.62, 0.66, 0.18, 0.08)
        + 0.6 * _gaussian(Xn, Yn, 0.72, 0.42, 0.12, 0.10)
    ) * delta_mask
    remediation_map = (
        1.1 * _gaussian(Xn, Yn, 0.55, 0.35, 0.08, 0.06)
        + 0.7 * _gaussian(Xn, Yn, 0.60, 0.22, 0.07, 0.05)
    ) * delta_mask

    monsoon_signal = np.clip(weather_steps / max(weather_steps.max(), 1.0), 0.0, None)
    north_norm = north_discharge / max(north_discharge.max(), 1.0)
    south_norm = south_discharge / max(south_discharge.max(), 1.0)

    base_speed = 0.012 + 0.03 * north_norm
    lateral_pulse = 0.006 * np.sin(np.linspace(0.0, 4.0 * np.pi, n_steps))
    u = np.zeros((n_steps, *mesh.shape), dtype=float)
    v = np.zeros_like(u)
    source = np.zeros_like(u)
    remediation = np.zeros_like(u)
    depth = np.zeros_like(u)

    for step in range(n_steps):
        floodplain = delta_mask * (1.3 + 0.8 * river_map + 0.45 * monsoon_signal[step] * (0.4 + Yn))
        depth[step] = np.maximum(floodplain, 0.15)
        u[step] = delta_mask * river_map * lateral_pulse[step] * (0.4 + 0.6 * Yn)
        v[step] = delta_mask * river_map * (base_speed[step] + 0.015 * monsoon_signal[step]) * (
            0.7 + 0.3 * (1.0 - Yn)
        )
        groundwater_source = (0.00018 + 0.00006 * south_norm[step]) * groundwater_map
        runoff_source = (0.00010 + 0.00018 * monsoon_signal[step]) * runoff_map
        river_source = (0.00012 + 0.00015 * north_norm[step]) * river_map * (0.55 + 0.45 * Yn)
        source[step] = (groundwater_source + runoff_source + river_source) * delta_mask
        remediation_activation = max(0.0, (step - 0.55 * n_steps) / max(0.45 * n_steps, 1.0))
        remediation[step] = remediation_activation * 0.00024 * remediation_map * (0.8 + 0.2 * monsoon_signal[step])

    initial_concentration = delta_mask * (
        12.0 * groundwater_map + 6.0 * river_map + 3.0 * runoff_map
    )
    initial_sediment = initial_concentration * (0.6 + 0.0003 * config.physics.partition_coefficient)

    return {
        "initial_state": SimulationState(
            concentration=initial_concentration,
            sediment_concentration=initial_sediment,
            u=u[0],
            v=v[0],
            depth=depth[0],
        ),
        "forcing": {"u": u, "v": v, "source": source, "remediation": remediation, "depth": depth},
        "mask": delta_mask > 0.0,
        "groundwater_map": groundwater_map,
        "runoff_map": runoff_map,
        "river_map": river_map * delta_mask,
        "remediation_map": remediation_map,
    }


def _gaussian(x: np.ndarray, y: np.ndarray, x0: float, y0: float, sx: float, sy: float) -> np.ndarray:
    """Return an anisotropic Gaussian plume."""
    return np.exp(-(((x - x0) ** 2) / (2.0 * sx**2) + ((y - y0) ** 2) / (2.0 * sy**2)))


def _resample_signal(values: np.ndarray, n_steps: int) -> np.ndarray:
    """Linearly resample a 1-D signal onto the simulation timeline."""
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.full(n_steps, float(values[0]), dtype=float)
    xp = np.linspace(0.0, 1.0, len(values))
    x = np.linspace(0.0, 1.0, n_steps)
    return np.interp(x, xp, values)

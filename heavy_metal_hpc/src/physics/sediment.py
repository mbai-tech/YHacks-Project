"""Sediment-water exchange model for heavy metal partitioning."""

from __future__ import annotations

import numpy as np


class SedimentExchange:
    """First-order linear partitioning between dissolved and sediment-bound metal.

    The exchange flux (µg m⁻² s⁻¹) is:
        F = k_d · C_dissolved - k_r · C_sediment

    where *k_d* is the deposition rate constant and *k_r* is the resuspension
    rate constant.

    Parameters
    ----------
    k_deposition:
        Deposition rate constant (m s⁻¹).
    k_resuspension:
        Resuspension rate constant (s⁻¹).
    """

    def __init__(self, k_deposition: float = 1e-5, k_resuspension: float = 1e-6) -> None:
        self.k_d = k_deposition
        self.k_r = k_resuspension

    def flux(
        self,
        c_dissolved: np.ndarray,
        c_sediment: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        """Compute volumetric exchange source term (µg L⁻¹ s⁻¹) for the water column.

        Parameters
        ----------
        c_dissolved:
            (nx, ny) dissolved concentration (µg L⁻¹).
        c_sediment:
            (nx, ny) sediment-bound concentration (µg kg⁻¹).
        depth:
            (nx, ny) water depth (m).

        Returns
        -------
        np.ndarray
            (nx, ny) volumetric source term (µg L⁻¹ s⁻¹).
        """
        safe_depth = np.maximum(depth, 1e-6)
        return (-self.k_d * c_dissolved + self.k_r * c_sediment) / safe_depth

    def update_sediment(
        self,
        c_sediment: np.ndarray,
        c_dissolved: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Advance the sediment-bound concentration by one time step.

        Parameters
        ----------
        c_sediment:
            Current sediment concentration (µg kg⁻¹).
        c_dissolved:
            Current dissolved concentration (µg L⁻¹).
        dt:
            Time step (s).

        Returns
        -------
        np.ndarray
            Updated sediment concentration (µg kg⁻¹).
        """
        updated = c_sediment + dt * (self.k_d * c_dissolved - self.k_r * c_sediment)
        return np.maximum(updated, 0.0)

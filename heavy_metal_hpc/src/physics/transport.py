"""Advection-diffusion-reaction transport model for dissolved heavy metals."""

from __future__ import annotations

import numpy as np

from ..grid.mesh import StructuredMesh
from .operators import apply_neumann_bc, laplacian


class TransportModel:
    """2-D finite-volume advection-diffusion-reaction solver.

    Solves:
        ∂C/∂t + u·∇C = ∇·(D∇C) + S

    where *C* is dissolved metal concentration (µg L⁻¹), *u* is the depth-
    averaged velocity field, *D* is the effective diffusivity tensor, and *S*
    represents source / sink terms (sediment exchange, remediation).

    Parameters
    ----------
    mesh:
        Computational grid.
    diffusivity:
        Isotropic diffusivity coefficient (m² s⁻¹).
    """

    def __init__(self, mesh: StructuredMesh, diffusivity: float = 1e-3) -> None:
        self.mesh = mesh
        self.diffusivity = diffusivity

    def advect(
        self,
        concentration: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """First-order upwind advection step.

        Parameters
        ----------
        concentration:
            (nx, ny) concentration field (µg L⁻¹).
        u, v:
            (nx, ny) depth-averaged velocity components (m s⁻¹).
        dt:
            Time step (s).

        Returns
        -------
        np.ndarray
            Updated (nx, ny) concentration field after advection.
        """
        c = concentration
        dx = self.mesh.dx
        dy = self.mesh.dy

        flux_x = np.where(
            u >= 0.0,
            u * (c - np.roll(c, 1, axis=0)) / dx,
            u * (np.roll(c, -1, axis=0) - c) / dx,
        )
        flux_y = np.where(
            v >= 0.0,
            v * (c - np.roll(c, 1, axis=1)) / dy,
            v * (np.roll(c, -1, axis=1) - c) / dy,
        )
        updated = c - dt * (flux_x + flux_y)
        return apply_neumann_bc(updated)

    def diffuse(self, concentration: np.ndarray, dt: float) -> np.ndarray:
        """Explicit central-difference diffusion step.

        Parameters
        ----------
        concentration:
            (nx, ny) field (µg L⁻¹).
        dt:
            Time step (s).

        Returns
        -------
        np.ndarray
            Updated (nx, ny) field after diffusion.
        """
        updated = concentration + dt * self.diffusivity * laplacian(
            concentration, dx=self.mesh.dx, dy=self.mesh.dy
        )
        return apply_neumann_bc(updated)

    def react(
        self,
        concentration: np.ndarray,
        source: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Apply source / sink terms (Euler forward).

        Parameters
        ----------
        concentration:
            (nx, ny) field (µg L⁻¹).
        source:
            (nx, ny) volumetric source rate (µg L⁻¹ s⁻¹).
        dt:
            Time step (s).

        Returns
        -------
        np.ndarray
            Updated concentration field.
        """
        return concentration + dt * source

    def step(
        self,
        concentration: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        source: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Full operator-split time step: advect → diffuse → react.

        Returns
        -------
        np.ndarray
            Updated (nx, ny) concentration field.
        """
        c = self.advect(concentration, u, v, dt)
        c = self.diffuse(c, dt)
        c = self.react(c, source, dt)
        return c

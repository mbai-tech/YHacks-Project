"""Structured and unstructured mesh representations for the Baiyangdian domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class StructuredMesh:
    """2-D regular (uniform Δx, Δy) Cartesian mesh.

    Parameters
    ----------
    x_min, x_max:
        West and east domain extents (m or decimal degrees).
    y_min, y_max:
        South and north domain extents.
    nx, ny:
        Number of cells in each direction.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int

    def __post_init__(self) -> None:
        self.dx = (self.x_max - self.x_min) / self.nx
        self.dy = (self.y_max - self.y_min) / self.ny
        self.x_centers: np.ndarray = np.linspace(
            self.x_min + 0.5 * self.dx, self.x_max - 0.5 * self.dx, self.nx
        )
        self.y_centers: np.ndarray = np.linspace(
            self.y_min + 0.5 * self.dy, self.y_max - 0.5 * self.dy, self.ny
        )
        self.X, self.Y = np.meshgrid(self.x_centers, self.y_centers, indexing="ij")

    @property
    def shape(self) -> tuple[int, int]:
        """Return (nx, ny) grid shape."""
        return (self.nx, self.ny)

    @property
    def n_cells(self) -> int:
        """Total number of active cells."""
        return self.nx * self.ny

    def cell_area(self) -> float:
        """Uniform cell area (m²)."""
        return self.dx * self.dy


@dataclass
class UnstructuredMesh:
    """Triangular unstructured mesh (e.g. from Gmsh / Triangle).

    Parameters
    ----------
    nodes:
        (N, 2) array of node coordinates.
    triangles:
        (M, 3) integer connectivity array (0-indexed).
    """

    nodes: np.ndarray      # (N, 2)
    triangles: np.ndarray  # (M, 3)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_cells(self) -> int:
        return len(self.triangles)

    def cell_centroids(self) -> np.ndarray:
        """Return (M, 2) array of triangle centroids."""
        return self.nodes[self.triangles].mean(axis=1)

    def cell_areas(self) -> np.ndarray:
        """Return (M,) array of triangle areas via the cross-product formula."""
        v0 = self.nodes[self.triangles[:, 0]]
        v1 = self.nodes[self.triangles[:, 1]]
        v2 = self.nodes[self.triangles[:, 2]]
        return 0.5 * np.abs(
            (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
            - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
        )

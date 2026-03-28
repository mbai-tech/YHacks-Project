"""Unit tests for the transport physics module."""

from __future__ import annotations

import numpy as np
import pytest

from src.grid.mesh import StructuredMesh
from src.model.parameters import NumericalParameters, PhysicalParameters
from src.model.simulator import Simulator
from src.model.state import SimulationState
from src.physics.transport import TransportModel
from src.physics.operators import laplacian, gradient_x, gradient_y, apply_neumann_bc


@pytest.fixture
def small_mesh() -> StructuredMesh:
    """Return a small 10×10 test mesh."""
    return StructuredMesh(x_min=0, x_max=1000, y_min=0, y_max=1000, nx=10, ny=10)


@pytest.fixture
def transport(small_mesh: StructuredMesh) -> TransportModel:
    """Return a TransportModel on the small mesh."""
    return TransportModel(small_mesh, diffusivity=1e-2)


class TestStructuredMesh:
    """Tests for StructuredMesh geometry."""

    def test_shape(self, small_mesh: StructuredMesh) -> None:
        assert small_mesh.shape == (10, 10)

    def test_dx_dy(self, small_mesh: StructuredMesh) -> None:
        assert small_mesh.dx == pytest.approx(100.0)
        assert small_mesh.dy == pytest.approx(100.0)

    def test_cell_area(self, small_mesh: StructuredMesh) -> None:
        assert small_mesh.cell_area() == pytest.approx(10_000.0)

    def test_n_cells(self, small_mesh: StructuredMesh) -> None:
        assert small_mesh.n_cells == 100


class TestOperators:
    """Tests for finite-difference stencil operators."""

    def test_gradient_x_linear(self) -> None:
        """Gradient of a linearly varying field should be constant."""
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 15)
        X, _ = np.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]
        grad = gradient_x(X, dx)
        np.testing.assert_allclose(grad, np.ones_like(X), atol=1e-10)

    def test_laplacian_constant_field(self) -> None:
        """Laplacian of a constant field must be zero."""
        field = np.ones((20, 15))
        lap = laplacian(field, dx=10.0, dy=10.0)
        np.testing.assert_allclose(lap, 0.0, atol=1e-12)

    def test_neumann_bc_zeroes_boundary_gradient(self) -> None:
        """After applying Neumann BC, boundary rows/cols equal their interior neighbours."""
        field = np.random.default_rng(0).random((8, 8))
        out = apply_neumann_bc(field)
        np.testing.assert_array_equal(out[0, :], out[1, :])
        np.testing.assert_array_equal(out[-1, :], out[-2, :])
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        np.testing.assert_array_equal(out[:, -1], out[:, -2])


class TestTransportModel:
    """Tests for the TransportModel."""

    def test_react_positive_source(self, transport: TransportModel, small_mesh: StructuredMesh) -> None:
        """A positive source should increase concentration."""
        c0 = np.zeros(small_mesh.shape)
        source = np.ones(small_mesh.shape) * 2.0
        c1 = transport.react(c0, source, dt=60.0)
        assert (c1 > 0).all()

    def test_react_mass_conservation(self, transport: TransportModel, small_mesh: StructuredMesh) -> None:
        """React step: total added mass equals source × dt × n_cells."""
        c0 = np.zeros(small_mesh.shape)
        source = np.ones(small_mesh.shape)
        dt = 30.0
        c1 = transport.react(c0, source, dt)
        assert c1.sum() == pytest.approx(small_mesh.n_cells * dt)


def test_simulator_applies_source_and_remediation_terms() -> None:
    """External source fields should increase concentration while remediation reduces it."""
    mesh = StructuredMesh(x_min=0, x_max=10, y_min=0, y_max=10, nx=4, ny=4)
    simulator = Simulator(
        mesh,
        phys=PhysicalParameters(diffusivity=1e-6, k_deposition=1e-6, k_resuspension=1e-6, decay_rate=0.0),
        num=NumericalParameters(dt=10.0, n_steps=2, output_interval=1),
    )
    initial = SimulationState(
        concentration=np.zeros(mesh.shape),
        sediment_concentration=np.zeros(mesh.shape),
        u=np.zeros(mesh.shape),
        v=np.zeros(mesh.shape),
        depth=np.ones(mesh.shape),
    )
    source = np.ones((2, *mesh.shape)) * 0.2
    remediation = np.zeros_like(source)
    remediation[1] = 0.1
    history = simulator.run(
        initial,
        forcing={
            "u": np.zeros((2, *mesh.shape)),
            "v": np.zeros((2, *mesh.shape)),
            "source": source,
            "remediation": remediation,
        },
    )

    first = history.snapshots[1].concentration.mean()
    second = history.snapshots[2].concentration.mean()
    assert first > 0.0
    assert second > first
    assert second < first + simulator.num.dt * 0.2

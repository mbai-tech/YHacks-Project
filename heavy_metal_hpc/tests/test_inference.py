"""Unit tests for the inference module."""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.loss import mse_loss, weighted_mse_loss, log_likelihood, CompositeLoss
from src.inference.optimizer import GradientOptimizer
from src.model.parameters import PhysicalParameters


class TestLossFunctions:
    """Tests for scalar loss / misfit functions."""

    def test_mse_zero_on_perfect_prediction(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        assert mse_loss(a, a) == pytest.approx(0.0)

    def test_mse_known_value(self) -> None:
        pred = np.array([0.0, 0.0])
        obs = np.array([1.0, 3.0])
        # MSE = (1 + 9) / 2 = 5.0
        assert mse_loss(pred, obs) == pytest.approx(5.0)

    def test_weighted_mse_uniform_weights(self) -> None:
        pred = np.array([0.0, 0.0])
        obs = np.array([1.0, 3.0])
        w = np.ones(2)
        assert weighted_mse_loss(pred, obs, w) == pytest.approx(mse_loss(pred, obs))

    def test_log_likelihood_peak_at_perfect_fit(self) -> None:
        obs = np.ones(10) * 5.0
        ll_perfect = log_likelihood(obs, obs, sigma=1.0)
        ll_offset = log_likelihood(obs + 1.0, obs, sigma=1.0)
        assert ll_perfect > ll_offset

    def test_composite_loss_sums_terms(self) -> None:
        pred = np.zeros(5)
        obs = np.ones(5)
        loss = CompositeLoss({"mse": (mse_loss, 1.0), "mse2": (mse_loss, 2.0)})
        # mse = 1.0, so composite = 1*1 + 2*1 = 3.0
        assert loss(pred, obs) == pytest.approx(3.0)


class TestPhysicalParameters:
    """Tests for PhysicalParameters serialisation round-trip."""

    def test_to_from_vector_roundtrip(self) -> None:
        p = PhysicalParameters(diffusivity=5e-4, k_deposition=2e-5)
        p2 = PhysicalParameters.from_vector(p.to_vector())
        assert p2.diffusivity == pytest.approx(p.diffusivity)
        assert p2.k_deposition == pytest.approx(p.k_deposition)

    def test_default_values_positive(self) -> None:
        p = PhysicalParameters()
        for val in p.to_vector():
            assert val >= 0


class TestGradientOptimizer:
    """Tests for GradientOptimizer on a trivial quadratic."""

    def test_minimises_quadratic(self) -> None:
        opt = GradientOptimizer(method="L-BFGS-B")
        result = opt.minimize(lambda x: float(np.sum(x**2)), np.array([3.0, -4.0]))
        assert result.success
        np.testing.assert_allclose(result.x, 0.0, atol=1e-5)

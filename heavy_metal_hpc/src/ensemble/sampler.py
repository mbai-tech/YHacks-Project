"""Parameter-space samplers for ensemble / uncertainty quantification runs."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ..model.parameters import PhysicalParameters


class LatinHypercubeSampler:
    """Generate LHS samples over a multi-dimensional parameter space.

    Parameters
    ----------
    bounds:
        Dict mapping parameter name → (low, high) range.
    n_samples:
        Number of ensemble members to generate.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        n_samples: int,
        seed: int = 42,
    ) -> None:
        self.bounds = bounds
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def sample(self) -> list[PhysicalParameters]:
        """Draw *n_samples* LHS parameter vectors.

        Returns
        -------
        list[PhysicalParameters]
            Sampled parameter sets.
        """
        keys = list(self.bounds.keys())
        ndim = len(keys)
        # Permute within each dimension to enforce LHS structure
        cuts = np.linspace(0, 1, self.n_samples + 1)
        samples = np.empty((self.n_samples, ndim))
        for d in range(ndim):
            perm = self.rng.permutation(self.n_samples)
            u = self.rng.uniform(cuts[:-1], cuts[1:])[perm]
            low, high = self.bounds[keys[d]]
            samples[:, d] = low + u * (high - low)

        result = []
        for row in samples:
            kwargs = dict(zip(keys, row))
            result.append(PhysicalParameters(**kwargs))
        return result


class MonteCarloSampler:
    """Simple Monte-Carlo sampler drawing from independent Gaussians.

    Parameters
    ----------
    mean:
        PhysicalParameters representing the prior mean.
    std_frac:
        Fractional standard deviation applied to each parameter (e.g. 0.2 → 20 %).
    n_samples:
        Number of ensemble members.
    seed:
        Random seed.
    """

    def __init__(
        self,
        mean: PhysicalParameters,
        std_frac: float = 0.2,
        n_samples: int = 100,
        seed: int = 0,
    ) -> None:
        self.mean = mean
        self.std_frac = std_frac
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def sample(self) -> list[PhysicalParameters]:
        """Return *n_samples* perturbed parameter sets."""
        mu = np.array(self.mean.to_vector())
        sigma = np.abs(mu) * self.std_frac
        draws = self.rng.normal(mu, sigma, size=(self.n_samples, len(mu)))
        # Clip to non-negative (all physical params must be > 0)
        draws = np.clip(draws, 1e-12, None)
        return [PhysicalParameters.from_vector(row.tolist()) for row in draws]

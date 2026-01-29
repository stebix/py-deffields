# ruff: noqa: F722
"""Gaussian noise generator for volumetric data augmentation."""

from __future__ import annotations

import numpy as np
from jaxtyping import Float
from numpy import ndarray as Array

from .base import NoiseGenerator
from .._types import jaxcheck


# =============================================================================
# Functional interface
# =============================================================================


@jaxcheck
def generate_gaussian(
    shape: tuple[int, int, int],
    mean: float = 0.0,
    std: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Float[Array, "d h w"]:
    """
    Generate Gaussian (normal) noise field.

    Creates a noise field where each voxel is independently sampled from
    a normal distribution. This is the simplest form of noise and serves
    as a baseline for texture augmentation.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).
    mean : float
        Mean of the Gaussian distribution.
    std : float
        Standard deviation of the Gaussian distribution.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    Float[Array, "d h w"]
        Gaussian noise field with values centered around `mean`.

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import generate_gaussian
    >>> rng = np.random.default_rng(42)
    >>> noise = generate_gaussian((32, 64, 64), mean=0, std=0.1, rng=rng)
    >>> noise.shape
    (32, 64, 64)
    """
    if rng is None:
        rng = np.random.default_rng()

    return rng.normal(mean, std, size=shape).astype(np.float64)


# =============================================================================
# Class-based interface
# =============================================================================


class GaussianNoise(NoiseGenerator):
    """
    Gaussian noise generator with configurable parameters.

    Generates independent and identically distributed (i.i.d.) Gaussian
    noise at each voxel. Useful for simulating sensor noise or as a
    baseline texture pattern.

    Parameters
    ----------
    mean : float
        Mean of the Gaussian distribution.
    std : float
        Standard deviation of the Gaussian distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import GaussianNoise
    >>> generator = GaussianNoise(mean=0, std=0.1)
    >>> rng = np.random.default_rng(42)
    >>> noise = generator((32, 64, 64), rng=rng)
    >>> noise.shape
    (32, 64, 64)
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    @jaxcheck
    def generate(
        self,
        shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> Float[Array, "d h w"]:
        """
        Generate Gaussian noise field.

        Parameters
        ----------
        shape : tuple[int, int, int]
            Volume dimensions (D, H, W).
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        Float[Array, "d h w"]
            Gaussian noise field.
        """
        return generate_gaussian(shape, self.mean, self.std, rng)

    def __repr__(self) -> str:
        return f"GaussianNoise(mean={self.mean}, std={self.std})"

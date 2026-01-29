# ruff: noqa: F722
"""Abstract base class for noise generators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from jaxtyping import Float
from numpy import ndarray as Array


class NoiseGenerator(ABC):
    """
    Abstract base class for noise generators.

    Noise generators produce scalar fields that can be used to augment
    volumetric data with texture patterns for digital volume correlation (DVC).

    Unlike deformation fields which produce vector displacements (d, h, w, 3),
    noise generators produce scalar intensity values (d, h, w) that modify
    voxel intensities.

    Subclasses must implement the `generate` method.
    """

    @abstractmethod
    def generate(
        self,
        shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> Float[Array, "d h w"]:
        """
        Generate scalar noise field.

        Parameters
        ----------
        shape : tuple[int, int, int]
            Volume dimensions (D, H, W).
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        Float[Array, "d h w"]
            Scalar noise field.
        """
        pass

    def __call__(
        self,
        shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> Float[Array, "d h w"]:
        """Generate noise field (convenience method)."""
        return self.generate(shape, rng)

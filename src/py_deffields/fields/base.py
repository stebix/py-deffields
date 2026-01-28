# ruff: noqa: F722
"""Abstract base class for deformation field generators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from jaxtyping import Float
from numpy import ndarray as Array


class DeformationField(ABC):
    """Abstract base class for deformation field generators."""

    @abstractmethod
    def generate(
        self,
        shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> Float[Array, "d h w 3"]:
        """
        Generate displacement field.

        Parameters
        ----------
        shape : tuple[int, int, int]
            Volume dimensions (D, H, W).
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        Float[Array, "d h w 3"]
            Displacement field.
        """
        pass

    def __call__(
        self,
        shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> Float[Array, "d h w 3"]:
        """Generate displacement field (convenience method)."""
        return self.generate(shape, rng)

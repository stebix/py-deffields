# ruff: noqa: F722
"""Elastic deformation field generator using Gaussian smoothing."""

from __future__ import annotations

import numpy as np
from jaxtyping import Float
from numpy import ndarray as Array
from scipy.ndimage import gaussian_filter

from .base import DeformationField
from .._types import jaxcheck


class ElasticDeformation(DeformationField):
    """
    Generate elastic deformation fields using Gaussian-smoothed random noise.

    This implements the elastic deformation approach commonly used in
    data augmentation, where random displacement vectors are smoothed
    with a Gaussian filter to create locally coherent deformations.

    Parameters
    ----------
    alpha : float
        Displacement magnitude scaling factor. Higher values produce
        larger deformations.
    sigma : float
        Gaussian smoothing standard deviation in voxels. Higher values
        produce smoother, more global deformations.
    """

    def __init__(self, alpha: float = 10.0, sigma: float = 5.0) -> None:
        self.alpha = alpha
        self.sigma = sigma

    @jaxcheck
    def generate(
        self,
        shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> Float[Array, "D H W 3"]:
        """
        Generate elastic displacement field.

        Parameters
        ----------
        shape : tuple[int, int, int]
            Volume dimensions (D, H, W).
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        Float[Array, "D H W 3"]
            Displacement field.
        """
        if rng is None:
            rng = np.random.default_rng()

        displacement = np.zeros((*shape, 3), dtype=np.float64)

        for i in range(3):
            # Generate random noise in [-1, 1]
            noise = rng.uniform(-1, 1, size=shape)
            # Smooth with Gaussian filter
            smoothed = gaussian_filter(noise, sigma=self.sigma, mode="constant")
            # Scale by alpha
            displacement[..., i] = smoothed * self.alpha

        return displacement

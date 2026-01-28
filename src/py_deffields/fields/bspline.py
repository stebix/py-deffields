# ruff: noqa: F722
"""B-spline Free-Form Deformation field generator."""

from __future__ import annotations

import numpy as np
from jaxtyping import Float
from numpy import ndarray as Array
from scipy.ndimage import map_coordinates

from .base import DeformationField
from .._types import jaxcheck


class BSplineDeformation(DeformationField):
    """
    Generate B-spline Free-Form Deformation (FFD) fields.

    This implements FFD by creating a coarse control point grid with
    random displacements and interpolating to the full resolution
    using cubic B-spline interpolation.

    Parameters
    ----------
    control_point_spacing : int or tuple[int, int, int]
        Spacing between control points in voxels. Can be a single
        value for isotropic spacing or a tuple for anisotropic.
    displacement_sigma : float
        Standard deviation of random control point displacements
        (in voxels). Controls the magnitude of deformation.
    """

    def __init__(
        self,
        control_point_spacing: int | tuple[int, int, int] = 16,
        displacement_sigma: float = 2.0,
    ) -> None:
        if isinstance(control_point_spacing, int):
            self.spacing = (control_point_spacing,) * 3
        else:
            self.spacing = control_point_spacing
        self.displacement_sigma = displacement_sigma

    @jaxcheck
    def generate(
        self,
        shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> Float[Array, "D H W 3"]:
        """
        Generate B-spline displacement field.

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

        # Calculate control point grid dimensions
        # Add padding to ensure coverage at boundaries
        cp_shape = tuple(
            (s + spacing - 1) // spacing + 3
            for s, spacing in zip(shape, self.spacing)
        )

        displacement = np.zeros((*shape, 3), dtype=np.float64)

        for i in range(3):
            # Generate random displacements at control points
            cp_displacements = rng.normal(
                0, self.displacement_sigma, size=cp_shape
            )

            # Create coordinate grid for interpolation
            # Map voxel coordinates to control point coordinates
            coords = np.meshgrid(
                np.arange(shape[0], dtype=np.float64) / self.spacing[0] + 1,
                np.arange(shape[1], dtype=np.float64) / self.spacing[1] + 1,
                np.arange(shape[2], dtype=np.float64) / self.spacing[2] + 1,
                indexing="ij",
            )

            # Interpolate using cubic B-spline (order=3)
            displacement[..., i] = map_coordinates(
                cp_displacements,
                coords,
                order=3,
                mode="nearest",
            )

        return displacement

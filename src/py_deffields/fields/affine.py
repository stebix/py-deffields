# ruff: noqa: F722
"""Affine transformation deformation field generator."""

from __future__ import annotations

import numpy as np
from jaxtyping import Float
from numpy import ndarray as Array

from .base import DeformationField
from ..grid import identity_grid
from .._types import jaxcheck


class AffineDeformation(DeformationField):
    """
    Generate affine transformation deformation fields.

    This creates deformation fields from affine transformations including
    rotation, scaling, shearing, and translation. Parameters can be
    randomized within specified ranges.

    Parameters
    ----------
    rotation_range : tuple[float, float]
        Range for random rotation angles in radians, for each axis.
    scale_range : tuple[float, float]
        Range for random scale factors, for each axis.
    shear_range : tuple[float, float]
        Range for random shear factors.
    translation_range : tuple[float, float]
        Range for random translation in voxels, for each axis.
    """

    def __init__(
        self,
        rotation_range: tuple[float, float] = (-0.1, 0.1),
        scale_range: tuple[float, float] = (0.9, 1.1),
        shear_range: tuple[float, float] = (-0.1, 0.1),
        translation_range: tuple[float, float] = (-10, 10),
    ) -> None:
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.translation_range = translation_range
        self._matrix: Float[Array, "4 4"] | None = None

    @classmethod
    def from_matrix(cls, matrix: Float[Array, "4 4"]) -> AffineDeformation:
        """
        Create from explicit 4x4 transformation matrix.

        Parameters
        ----------
        matrix : Float[Array, "4 4"]
            4x4 affine transformation matrix.

        Returns
        -------
        AffineDeformation
            Instance with the specified transformation matrix.
        """
        if matrix.shape != (4, 4):
            raise ValueError("Matrix must be 4x4")
        instance = cls(
            rotation_range=(0, 0),
            scale_range=(1, 1),
            shear_range=(0, 0),
            translation_range=(0, 0),
        )
        instance._matrix = matrix.copy()
        return instance

    def _build_rotation_matrix(
        self, angles: tuple[float, float, float]
    ) -> Float[Array, "3 3"]:
        """Build 3x3 rotation matrix from Euler angles (XYZ order)."""
        rx, ry, rz = angles
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        # Rotation matrices for each axis
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        return Rz @ Ry @ Rx

    def _build_affine_matrix(
        self,
        rotation: tuple[float, float, float],
        scale: tuple[float, float, float],
        shear: tuple[float, float, float],
        translation: tuple[float, float, float],
    ) -> Float[Array, "4 4"]:
        """Build 4x4 affine transformation matrix."""
        # Rotation matrix
        R = self._build_rotation_matrix(rotation)

        # Scale matrix
        S = np.diag(scale)

        # Shear matrix (upper triangular)
        H = np.array([
            [1, shear[0], shear[1]],
            [0, 1, shear[2]],
            [0, 0, 1],
        ])

        # Combined linear transformation
        linear = R @ S @ H

        # Build 4x4 matrix
        matrix = np.eye(4)
        matrix[:3, :3] = linear
        matrix[:3, 3] = translation

        return matrix

    @jaxcheck
    def generate(
        self,
        shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> Float[Array, "D H W 3"]:
        """
        Generate affine displacement field.

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

        if self._matrix is not None:
            matrix = self._matrix
        else:
            # Sample random parameters
            rotation = tuple(
                rng.uniform(*self.rotation_range) for _ in range(3)
            )
            scale = tuple(rng.uniform(*self.scale_range) for _ in range(3))
            shear = tuple(rng.uniform(*self.shear_range) for _ in range(3))
            translation = tuple(
                rng.uniform(*self.translation_range) for _ in range(3)
            )

            matrix = self._build_affine_matrix(
                rotation, scale, shear, translation
            )

        # Get identity grid and center coordinates
        grid = identity_grid(shape)
        center = np.array([s / 2 for s in shape])

        # Center coordinates, apply transformation, uncenter
        centered = grid - center

        # Apply affine transformation
        # Reshape for matrix multiplication: (D*H*W, 3)
        flat_coords = centered.reshape(-1, 3)

        # Apply linear transformation (3x3 part)
        transformed = flat_coords @ matrix[:3, :3].T

        # Add translation
        transformed += matrix[:3, 3]

        # Uncenter
        transformed += center

        # Reshape back
        new_coords = transformed.reshape(*shape, 3)

        # Convert to displacement
        displacement = new_coords - grid

        return displacement

# ruff: noqa: F722
"""Coordinate grid utilities for deformation fields."""

from __future__ import annotations

import numpy as np
from jaxtyping import Float
from numpy import ndarray as Array

from ._types import jaxcheck


@jaxcheck
def identity_grid(shape: tuple[int, int, int]) -> Float[Array, "D H W 3"]:
    """
    Create identity coordinate grid.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).

    Returns
    -------
    Float[Array, "D H W 3"]
        Coordinate grid where each voxel contains its own [d, h, w] coordinates.
    """
    d, h, w = shape
    coords = np.meshgrid(
        np.arange(d, dtype=np.float64),
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )
    return np.stack(coords, axis=-1)


@jaxcheck
def displacement_to_absolute(
    displacement: Float[Array, "D H W 3"],
) -> Float[Array, "D H W 3"]:
    """
    Convert displacement field to absolute coordinates.

    Parameters
    ----------
    displacement : Float[Array, "D H W 3"]
        Displacement field.

    Returns
    -------
    Float[Array, "D H W 3"]
        Absolute coordinate field.
    """
    shape = displacement.shape[:3]
    grid = identity_grid(shape)
    return grid + displacement

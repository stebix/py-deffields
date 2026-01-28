# ruff: noqa: F722
"""Compose multiple deformation fields."""

from __future__ import annotations

import numpy as np
from jaxtyping import Float
from numpy import ndarray as Array
from scipy.ndimage import map_coordinates

from ._types import jaxcheck
from .grid import displacement_to_absolute


@jaxcheck
def compose(
    field1: Float[Array, "d h w 3"],
    field2: Float[Array, "d h w 3"],
    order: int = 1,
) -> Float[Array, "d h w 3"]:
    """
    Compose two displacement fields.

    Computes the composition field1 followed by field2:
        x -> x + field1(x) + field2(x + field1(x))

    Parameters
    ----------
    field1 : Float[Array, "d h w 3"]
        First displacement field.
    field2 : Float[Array, "d h w 3"]
        Second displacement field.
    order : int
        Interpolation order for resampling field2.

    Returns
    -------
    Float[Array, "d h w 3"]
        Composed displacement field.
    """
    if field1.shape != field2.shape:
        raise ValueError(
            f"Field shapes must match: {field1.shape} vs {field2.shape}"
        )

    # Compute the coordinates after applying field1
    coords_after_field1 = displacement_to_absolute(field1)

    # Sample field2 at these new coordinates
    # Need to interpolate each component of field2
    coords_transposed = np.moveaxis(coords_after_field1, -1, 0)

    field2_resampled = np.zeros_like(field2)
    for i in range(3):
        field2_resampled[..., i] = map_coordinates(
            field2[..., i],
            coords_transposed,
            order=order,
            mode="nearest",
        )

    # Compose: displacement = field1 + field2(x + field1(x))
    return field1 + field2_resampled

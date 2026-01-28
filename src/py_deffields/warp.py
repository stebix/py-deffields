# ruff: noqa: F722
"""Apply deformation fields to warp volumes."""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
from jaxtyping import Float, Inexact
from numpy import ndarray as Array
from scipy.ndimage import map_coordinates

from ._types import jaxcheck
from .grid import displacement_to_absolute


@overload
def warp(
    volume: Inexact[Array, "d h w"],
    displacement: Float[Array, "d h w 3"],
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> Inexact[Array, "d h w"]: ...


@overload
def warp(
    volume: Inexact[Array, "d h w C"],
    displacement: Float[Array, "d h w 3"],
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> Inexact[Array, "d h w C"]: ...


@jaxcheck
def warp(
    volume: Inexact[Array, "d h w"] | Inexact[Array, "d h w C"],
    displacement: Float[Array, "d h w 3"],
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> Inexact[Array, "d h w"] | Inexact[Array, "d h w C"]:
    """
    Apply displacement field to warp a volume.

    Parameters
    ----------
    volume : Inexact[Array, "d h w"] | Inexact[Array, "d h w C"]
        Input volume, either single-channel (D, H, W) or multi-channel (D, H, W, C).
    displacement : Float[Array, "d h w 3"]
        Displacement field.
    order : int
        Interpolation order: 0=nearest, 1=linear, 3=cubic.
    mode : str
        Boundary handling mode: 'constant', 'nearest', 'reflect', 'wrap'.
    cval : float
        Value used for 'constant' mode outside boundaries.

    Returns
    -------
    Inexact[Array, "d h w"] | Inexact[Array, "d h w C"]
        Warped volume with same shape as input.
    """
    if volume.ndim not in (3, 4):
        raise ValueError(
            f"Volume must be 3D or 4D, got shape {volume.shape}"
        )
    if volume.shape[:3] != displacement.shape[:3]:
        raise ValueError(
            f"Volume shape {volume.shape[:3]} doesn't match "
            f"displacement shape {displacement.shape[:3]}"
        )

    # Convert displacement to absolute coordinates
    coords = displacement_to_absolute(displacement)

    # Prepare coordinate arrays for map_coordinates
    # map_coordinates expects (3, D, H, W) coordinate array
    coords_transposed = np.moveaxis(coords, -1, 0)

    if volume.ndim == 3:
        # Single channel volume
        warped = map_coordinates(
            volume,
            coords_transposed,
            order=order,
            mode=mode,
            cval=cval,
        )
    else:
        # Multi-channel volume: warp each channel separately
        n_channels = volume.shape[-1]
        warped = np.zeros_like(volume)
        for c in range(n_channels):
            warped[..., c] = map_coordinates(
                volume[..., c],
                coords_transposed,
                order=order,
                mode=mode,
                cval=cval,
            )

    return warped

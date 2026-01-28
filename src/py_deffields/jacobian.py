# ruff: noqa: F722
"""Jacobian determinant computation for deformation fields."""

from __future__ import annotations

import numpy as np
from jaxtyping import Float
from numpy import ndarray as Array

from ._types import jaxcheck


@jaxcheck
def jacobian_determinant(
    displacement: Float[Array, "d h w 3"],
) -> Float[Array, "d h w"]:
    """
    Compute Jacobian determinant of deformation field.

    The Jacobian determinant measures local volume change under the
    deformation. Values < 1 indicate compression, > 1 indicate expansion,
    and < 0 indicate folding (topology violation).

    Parameters
    ----------
    displacement : Float[Array, "d h w 3"]
        Displacement field.

    Returns
    -------
    Float[Array, "d h w"]
        Jacobian determinant.
        Values < 0 indicate folding (topology violation).
        Values = 1 indicate no volume change.
    """
    # Extract displacement components
    u = displacement[..., 0]  # d-direction displacement
    v = displacement[..., 1]  # h-direction displacement
    w = displacement[..., 2]  # w-direction displacement

    # Compute spatial gradients using central differences
    # np.gradient returns gradients along each axis
    du = np.gradient(u, axis=(0, 1, 2))  # du/dd, du/dh, du/dw
    dv = np.gradient(v, axis=(0, 1, 2))  # dv/dd, dv/dh, dv/dw
    dw = np.gradient(w, axis=(0, 1, 2))  # dw/dd, dw/dh, dw/dw

    # Build Jacobian matrix of the transformation T(x) = x + displacement(x)
    # J = I + gradient(displacement)
    # J[i,j] = delta_ij + d(displacement_i)/d(x_j)

    # Jacobian matrix elements (add 1 to diagonal for identity part)
    j00 = 1.0 + du[0]  # 1 + du/dd
    j01 = du[1]      # du/dh
    j02 = du[2]      # du/dw

    j10 = dv[0]      # dv/dd
    j11 = 1.0 + dv[1]  # 1 + dv/dh
    j12 = dv[2]      # dv/dw

    j20 = dw[0]      # dw/dd
    j21 = dw[1]      # dw/dh
    j22 = 1.0 + dw[2]  # 1 + dw/dw

    # Compute 3x3 determinant using expansion by minors
    det = (
          j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )

    return det

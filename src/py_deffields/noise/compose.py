# ruff: noqa: F722
"""Noise composition and application utilities.

This module provides functions for combining multiple noise fields
and applying noise to volumetric data.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Inexact
from numpy import ndarray as Array

from .._types import jaxcheck


@jaxcheck
def add_noise(
    volume: Inexact[Array, "d h w"],
    noise: Float[Array, "d h w"],
    weight: float = 1.0,
) -> Float[Array, "d h w"]:
    """
    Add noise to a volume with optional scaling.

    Parameters
    ----------
    volume : Inexact[Array, "d h w"]
        Input volume to augment.
    noise : Float[Array, "d h w"]
        Noise field to add.
    weight : float
        Scaling factor for the noise.

    Returns
    -------
    Float[Array, "d h w"]
        Volume with added noise.

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import generate_gaussian, add_noise
    >>> volume = np.ones((32, 64, 64))
    >>> noise = generate_gaussian((32, 64, 64), std=0.1)
    >>> noisy = add_noise(volume, noise, weight=0.5)
    """
    return (volume + weight * noise).astype(np.float64)


@jaxcheck
def multiply_noise(
    volume: Inexact[Array, "d h w"],
    noise: Float[Array, "d h w"],
    weight: float = 1.0,
) -> Float[Array, "d h w"]:
    """
    Apply multiplicative noise to a volume (speckle-like).

    The noise is applied as: volume * (1 + weight * noise)

    Parameters
    ----------
    volume : Inexact[Array, "d h w"]
        Input volume to augment.
    noise : Float[Array, "d h w"]
        Noise field (centered around 0 for best results).
    weight : float
        Scaling factor for the noise effect.

    Returns
    -------
    Float[Array, "d h w"]
        Volume with multiplicative noise applied.

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import generate_gaussian, multiply_noise
    >>> volume = np.ones((32, 64, 64)) * 100
    >>> noise = generate_gaussian((32, 64, 64), mean=0, std=1)
    >>> speckled = multiply_noise(volume, noise, weight=0.1)
    """
    return (volume * (1.0 + weight * noise)).astype(np.float64)


@jaxcheck
def blend_noise(
    *noises: Float[Array, "d h w"],
    weights: list[float] | None = None,
    normalize: bool = True,
) -> Float[Array, "d h w"]:
    """
    Blend multiple noise fields together.

    Parameters
    ----------
    *noises : Float[Array, "d h w"]
        Variable number of noise fields to blend.
    weights : list[float], optional
        Weights for each noise field. If None, equal weights are used.
    normalize : bool
        If True, normalize weights to sum to 1.

    Returns
    -------
    Float[Array, "d h w"]
        Blended noise field.

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import generate_gaussian, generate_perlin, blend_noise
    >>> rng = np.random.default_rng(42)
    >>> noise1 = generate_gaussian((32, 64, 64), rng=rng)
    >>> noise2 = generate_perlin((32, 64, 64), rng=rng)
    >>> blended = blend_noise(noise1, noise2, weights=[0.3, 0.7])
    """
    if len(noises) == 0:
        raise ValueError("At least one noise field is required")

    if weights is None:
        weights = [1.0] * len(noises)

    if len(weights) != len(noises):
        raise ValueError(f"Number of weights ({len(weights)}) must match number of noises ({len(noises)})")

    weights_arr = np.array(weights, dtype=np.float64)

    if normalize:
        weight_sum = weights_arr.sum()
        if weight_sum > 0:
            weights_arr = weights_arr / weight_sum

    result = np.zeros_like(noises[0], dtype=np.float64)
    for noise, weight in zip(noises, weights_arr):
        result += weight * noise

    return result


@jaxcheck
def normalize_noise(
    noise: Float[Array, "d h w"],
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> Float[Array, "d h w"]:
    """
    Normalize noise to a target range.

    Parameters
    ----------
    noise : Float[Array, "d h w"]
        Input noise field.
    target_min : float
        Minimum value of output range.
    target_max : float
        Maximum value of output range.

    Returns
    -------
    Float[Array, "d h w"]
        Normalized noise field.

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import generate_perlin, normalize_noise
    >>> noise = generate_perlin((32, 64, 64))  # Range approx [-1, 1]
    >>> normalized = normalize_noise(noise, 0, 255)  # Range [0, 255]
    """
    noise_min = noise.min()
    noise_max = noise.max()

    if noise_max - noise_min < 1e-10:
        # Constant noise, return middle of target range
        return np.full_like(noise, (target_min + target_max) / 2)

    # Linear rescaling
    normalized = (noise - noise_min) / (noise_max - noise_min)
    return (normalized * (target_max - target_min) + target_min).astype(np.float64)


@jaxcheck
def clip_noise(
    noise: Float[Array, "d h w"],
    min_val: float | None = None,
    max_val: float | None = None,
) -> Float[Array, "d h w"]:
    """
    Clip noise values to a specified range.

    Parameters
    ----------
    noise : Float[Array, "d h w"]
        Input noise field.
    min_val : float, optional
        Minimum value (no lower clipping if None).
    max_val : float, optional
        Maximum value (no upper clipping if None).

    Returns
    -------
    Float[Array, "d h w"]
        Clipped noise field.
    """
    return np.clip(noise, min_val, max_val).astype(np.float64)

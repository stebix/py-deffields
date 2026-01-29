# ruff: noqa: F722
"""Analytic deformation field generators for DVC benchmarking.

This module provides functional generators for creating synthetic displacement
fields commonly used in digital volume correlation (DVC) benchmarking:

- Star: Sinusoidal wave patterns with spatially varying periods
- Curve: Power-law gradients along each axis with optional rotation
- Sphere: Spherical coordinate-based tangential flow patterns
- Overall: Composite field combining sphere, star, and polynomial modulation
- Static: Uniform displacement field (baseline)

These fields are analytically defined (closed-form expressions) and fully
deterministic given a random seed, making them suitable for reproducible
benchmarking of DVC algorithms.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Num
from numpy import ndarray as Array

from .._types import jaxcheck


# -----------------------------------------------------------------------------
# Internal utilities
# -----------------------------------------------------------------------------


def _normalized_grid(
    shape: tuple[int, int, int],
) -> tuple[Float[Array, "d 1 1"], Float[Array, "1 h 1"], Float[Array, "1 1 w"]]:
    """
    Create normalized coordinate grids in [0, 1) range.

    Uses sparse meshgrid for memory efficiency.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).

    Returns
    -------
    tuple of arrays
        Sparse meshgrid arrays (gd, gh, gw) each normalized to [0, 1).
    """
    d, h, w = shape
    gd = np.arange(d, dtype=np.float64) / d
    gh = np.arange(h, dtype=np.float64) / h
    gw = np.arange(w, dtype=np.float64) / w
    return np.meshgrid(gd, gh, gw, sparse=True, indexing="ij")


def _build_rotation_matrix(
    angles: Float[Array, "3"],
) -> Float[Array, "3 3"]:
    """
    Build 3x3 rotation matrix from Euler angles (XYZ order).

    Parameters
    ----------
    angles : Float[Array, "3"]
        Rotation angles in radians for each axis [rx, ry, rz].

    Returns
    -------
    Float[Array, "3 3"]
        Combined rotation matrix R = Rz @ Ry @ Rx.
    """
    rx, ry, rz = angles
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def _rotate_displacement_field(
    field: Float[Array, "d h w 3"],
    angles: Float[Array, "3"],
) -> Float[Array, "d h w 3"]:
    """
    Rotate a displacement field by applying rotation to each displacement vector.

    Parameters
    ----------
    field : Float[Array, "d h w 3"]
        Input displacement field.
    angles : Float[Array, "3"]
        Rotation angles in radians [rx, ry, rz].

    Returns
    -------
    Float[Array, "d h w 3"]
        Rotated displacement field.
    """
    R = _build_rotation_matrix(angles)
    shape = field.shape
    flat = field.reshape(-1, 3)
    rotated = flat @ R.T
    return rotated.reshape(shape)


def _add_awgn(
    signal: Float[Array, "..."],
    snr_db: float,
    rng: np.random.Generator,
) -> Float[Array, "..."]:
    """
    Add Additive White Gaussian Noise at specified SNR.

    Parameters
    ----------
    signal : array
        Input signal.
    snr_db : float
        Signal-to-noise ratio in decibels.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    array
        Noisy signal with same shape as input.
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


# -----------------------------------------------------------------------------
# Star field
# -----------------------------------------------------------------------------


@jaxcheck
def generate_star(
    shape: tuple[int, int, int],
    max_displacement: float = 16.0,
    center: Float[Array, "3"] | None = None,
    period_range: tuple[float, float] = (0.0625, 0.375),
    rng: np.random.Generator | None = None,
) -> Float[Array, "d h w 3"]:
    """
    Generate star-pattern displacement field with sinusoidal waves.

    Creates a displacement field where each component is a sinusoidal function
    of a different spatial coordinate, with spatially varying periods that
    create star-like interference patterns.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).
    max_displacement : float
        Maximum displacement magnitude in voxels.
    center : Float[Array, "3"], optional
        Center point for the sinusoidal pattern in normalized [0, 1] coords.
        If None, randomly sampled near (0.5, 0.5, 0.5).
    period_range : tuple[float, float]
        Range of periods (T_min, T_max) for the sinusoidal variation.
        Periods vary spatially within this range.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    Float[Array, "d h w 3"]
        Star-pattern displacement field.
    """
    if rng is None:
        rng = np.random.default_rng()

    gd, gh, gw = _normalized_grid(shape)

    # Amplitude per axis (with small random variation)
    a0 = max_displacement - rng.random() * 0.1 * max_displacement
    a1 = max_displacement - rng.random() * 0.1 * max_displacement
    a2 = max_displacement - rng.random() * 0.1 * max_displacement

    # Center point (midpoint with small random jitter)
    if center is None:
        center = np.array([
            0.5 + (rng.random() * 2.0 - 1.0) * 0.1,
            0.5 + (rng.random() * 2.0 - 1.0) * 0.1,
            0.5 + (rng.random() * 2.0 - 1.0) * 0.1,
        ])

    # Spatially varying periods
    T_min, T_max = period_range
    T_min += rng.random() * 0.01
    T_max += rng.random() * 0.1

    # Period varies along perpendicular axis
    Td = T_min + (T_max - T_min) * gh  # varies with h
    Th = T_min + (T_max - T_min) * gw  # varies with w
    Tw = T_min + (T_max - T_min) * gd  # varies with d

    # Displacement components: each is sinusoidal in a different axis
    # d-component depends on w, h-component depends on d, w-component depends on h
    dd = a0 * np.sin(2 * np.pi * (gw - center[2]) / Td)
    dh = a1 * np.sin(2 * np.pi * (gd - center[0]) / Th)
    dw = a2 * np.cos(2 * np.pi * (gh - center[1]) / Tw)

    # Broadcast and stack
    displacement = np.stack(
        np.broadcast_arrays(dd, dh, dw),
        axis=-1,
    )

    return displacement


# -----------------------------------------------------------------------------
# Curve field
# -----------------------------------------------------------------------------


@jaxcheck
def generate_curve(
    shape: tuple[int, int, int],
    max_displacement: float = 16.0,
    gamma: Num[Array, "3"] | None = None,
    slope: Num[Array, "3"] | None = None,
    shift: Num[Array, "3"] | None = None,
    slope_ratio: float = 0.8,
    rotation: Num[Array, "3"] | None = None,
    rng: np.random.Generator | None = None,
) -> Float[Array, "d h w 3"]:
    """
    Generate curve/polynomial displacement field with power-law gradients.

    Creates a displacement field where each component follows a power-law
    curve along its corresponding axis: d_i = slope_i * x_i^gamma_i + shift_i.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).
    max_displacement : float
        Maximum displacement magnitude in voxels.
    gamma : Float[Array, "3"], optional
        Exponents for power-law in each axis. If None, random in [1.0, 2.0].
    slope : Float[Array, "3"], optional
        Slope coefficients. If None, random based on max_displacement.
    shift : Float[Array, "3"], optional
        Constant shift per axis. If None, random based on max_displacement.
    slope_ratio : float
        Fraction of max_displacement allocated to slope vs shift.
    rotation : Float[Array, "3"], optional
        Rotation angles in degrees [rx, ry, rz]. If None, random in [0, 360].
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    Float[Array, "d h w 3"]
        Curve displacement field.
    """
    if rng is None:
        rng = np.random.default_rng()

    gd, gh, gw = _normalized_grid(shape)

    # Power-law exponents
    if gamma is None:
        gamma = 1.0 + rng.random(3)

    # Slopes and shifts
    if slope is None:
        max_slope = max_displacement * min(1.0, slope_ratio)
        slope = (rng.random(3) * 2.0 - 1.0) * max_slope

    if shift is None:
        max_shift = max_displacement * (1.0 - min(1.0, slope_ratio))
        shift = (rng.random(3) * 2.0 - 1.0) * max_shift

    # Compute displacement components
    dd = slope[0] * (gd ** gamma[0]) + shift[0]
    dh = slope[1] * (gh ** gamma[1]) + shift[1]
    dw = slope[2] * (gw ** gamma[2]) + shift[2]

    # Broadcast and stack
    displacement = np.stack(
        np.broadcast_arrays(dd, dh, dw),
        axis=-1,
    )

    # Apply rotation
    if rotation is None:
        rotation = rng.random(3) * 360.0

    displacement = _rotate_displacement_field(
        displacement, np.radians(rotation)
    )

    return displacement


# -----------------------------------------------------------------------------
# Sphere field
# -----------------------------------------------------------------------------


@jaxcheck
def generate_sphere(
    shape: tuple[int, int, int],
    max_displacement: float = 16.0,
    center: Num[Array, "3"] | None = None,
    scale: Num[Array, "3"] | None = None,
    velocity: Num[Array, "3"] | None = None,
    phase: float | None = None,
    rng: np.random.Generator | None = None,
) -> Float[Array, "d h w 3"]:
    """
    Generate spherical tangential flow displacement field.

    Creates a displacement field with circular/vortex patterns emanating
    from a center point, using spherical coordinate transformations.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).
    max_displacement : float
        Maximum displacement magnitude in voxels.
    center : Num[Array, "3"], optional
        Center point in normalized [0, 1] coordinates.
        If None, randomly sampled.
    scale : Num[Array, "3"], optional
        Scaling factors for each axis. If None, random in [1.0, 2.0].
    velocity : Num[Array, "3"], optional
        Radial velocity components. If None, random in [-1, 1].
    phase : float, optional
        Phase offset in radians. If None, random in [0, 2π].
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    Float[Array, "d h w 3"]
        Spherical displacement field.
    """
    if rng is None:
        rng = np.random.default_rng()

    gd, gh, gw = _normalized_grid(shape)

    # Parameters
    if center is None:
        center = rng.random(3)
    if scale is None:
        scale = 1.0 + rng.random(3)
    if velocity is None:
        velocity = rng.random(3) * 2.0 - 1.0
    if phase is None:
        phase = rng.random() * 2.0 * np.pi

    # Shift and scale coordinates relative to center
    d_scaled = (gd - center[0]) / scale[0]
    h_scaled = (gh - center[1]) / scale[1]
    w_scaled = (gw - center[2]) / scale[2]

    # Convert to spherical coordinates
    r = np.sqrt(d_scaled**2 + h_scaled**2 + w_scaled**2)
    r = np.maximum(r, np.finfo(float).eps)  # avoid division by zero

    theta = np.arccos(w_scaled / r)  # polar angle
    phi = np.arctan2(h_scaled, d_scaled)  # azimuthal angle

    # Radial velocity components
    vr_d = np.ones_like(r) * velocity[0]
    vr_h = np.ones_like(r) * velocity[1]
    vr_w = np.ones_like(r) * velocity[2]

    # Tangential components (circular patterns)
    v_theta = -np.sin(phi + phase)  # polar direction
    v_phi = np.cos(theta + phase)  # azimuthal direction

    # Convert back to Cartesian
    # d-component
    u_d = (
        vr_d * np.sin(theta) * np.cos(phi)
        + v_theta * np.cos(theta) * np.cos(phi)
        - v_phi * np.sin(phi)
    ) * scale[0]

    # h-component
    u_h = (
        vr_h * np.sin(theta) * np.sin(phi)
        + v_theta * np.cos(theta) * np.sin(phi)
        + v_phi * np.cos(phi)
    ) * scale[1]

    # w-component
    u_w = (vr_w * np.cos(theta) - v_theta * np.sin(theta)) * scale[2]

    # Normalize to max_displacement
    uvw_max = max(np.abs(u_d).max(), np.abs(u_h).max(), np.abs(u_w).max())
    if uvw_max > 0:
        u_d = u_d / uvw_max * max_displacement
        u_h = u_h / uvw_max * max_displacement
        u_w = u_w / uvw_max * max_displacement

    # Stack into displacement field
    displacement = np.stack(
        np.broadcast_arrays(u_d, u_h, u_w),
        axis=-1,
    )

    return displacement


# -----------------------------------------------------------------------------
# Overall/composite field
# -----------------------------------------------------------------------------


@jaxcheck
def generate_overall(
    shape: tuple[int, int, int],
    max_displacement: float = 16.0,
    center: Num[Array, "3"] | None = None,
    scale: Num[Array, "3"] | None = None,
    velocity: Num[Array, "3"] | None = None,
    phase: float | None = None,
    gamma: Num[Array, "3"] | None = None,
    rotation: Num[Array, "3"] | None = None,
    star_weight: float | None = None,
    star_rotation: Num[Array, "3"] | None = None,
    snr_db: float | None = None,
    rng: np.random.Generator | None = None,
) -> Num[Array, "d h w 3"]:
    """
    Generate composite displacement field combining multiple deformation modes.

    Creates a complex, realistic displacement field by combining:
    - Spherical tangential flow (base)
    - Polynomial modulation (gamma exponents)
    - Star pattern overlay (weighted)
    - Optional rotation and noise

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).
    max_displacement : float
        Maximum displacement magnitude in voxels.
    center : Num[Array, "3"], optional
        Center point in normalized [0, 1] coordinates.
    scale : Num[Array, "3"], optional
        Scaling factors for spherical component.
    velocity : Num[Array, "3"], optional
        Radial velocity components.
    phase : float, optional
        Phase offset in radians.
    gamma : Num[Array, "3"], optional
        Exponents for polynomial modulation.
    rotation : Num[Array, "3"], optional
        Final rotation angles in degrees.
    star_weight : float, optional
        Weight for star pattern overlay [0, 1].
    star_rotation : Num[Array, "3"], optional
        Rotation for star pattern in degrees.
    snr_db : float, optional
        If provided, adds Gaussian noise at this SNR.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    Num[Array, "d h w 3"]
        Composite displacement field.
    """
    if rng is None:
        rng = np.random.default_rng()

    gd, gh, gw = _normalized_grid(shape)

    # Parameters with defaults
    if center is None:
        center = rng.random(3)
    if scale is None:
        scale = 1.0 + rng.random(3)
    if velocity is None:
        velocity = rng.random(3) * 2.0 - 1.0
    if phase is None:
        phase = rng.random() * 2.0 * np.pi
    if gamma is None:
        gamma = 1.0 + rng.random(3)
    if rotation is None:
        rotation = rng.random(3) * 360.0
    if star_weight is None:
        # Gaussian-like sampling centered at 0.1
        star_weight = np.clip(rng.normal(0.1, 0.033), 0.0, 0.2)
    if star_rotation is None:
        star_rotation = rng.random(3) * 360.0

    # Spherical coordinate computation (similar to generate_sphere)
    d_scaled = (gd - center[0]) / scale[0]
    h_scaled = (gh - center[1]) / scale[1]
    w_scaled = (gw - center[2]) / scale[2]

    r = np.sqrt(d_scaled**2 + h_scaled**2 + w_scaled**2)
    r = np.maximum(r, np.finfo(float).eps)

    theta = np.arccos(w_scaled / r)
    phi = np.arctan2(h_scaled, d_scaled)

    # Radial and tangential velocity components
    vr_d = np.ones_like(r) * velocity[0]
    vr_h = np.ones_like(r) * velocity[1]
    vr_w = np.ones_like(r) * velocity[2]

    v_theta = -np.sin(phi + phase)
    v_phi = np.cos(theta + phase)

    # Polynomial modulation factor for each axis
    mod_d = (1.0 + gd) ** gamma[0] - gd
    mod_h = (1.0 + gh) ** gamma[1] - gh
    mod_w = (1.0 + gw) ** gamma[2] - gw

    # Convert to Cartesian with modulation
    u_d = (
        vr_d * np.sin(theta) * np.cos(phi)
        + v_theta * np.cos(theta) * np.cos(phi)
        - v_phi * np.sin(phi)
    ) * scale[0] * mod_d

    u_h = (
        vr_h * np.sin(theta) * np.sin(phi)
        + v_theta * np.cos(theta) * np.sin(phi)
        + v_phi * np.cos(phi)
    ) * scale[1] * mod_h

    u_w = (vr_w * np.cos(theta) - v_theta * np.sin(theta)) * scale[2] * mod_w

    # Generate and add star field
    star_field = generate_star(
        shape,
        max_displacement=max_displacement,
        rng=rng,
    )
    star_field = _rotate_displacement_field(star_field, np.radians(star_rotation))

    # Blend star field (normalized contribution)
    u_d = u_d + star_weight * ((star_field[..., 0] / max_displacement) * 2.0 - 1.0)
    u_h = u_h + star_weight * ((star_field[..., 1] / max_displacement) * 2.0 - 1.0)
    u_w = u_w + star_weight * ((star_field[..., 2] / max_displacement) * 2.0 - 1.0)

    # Add noise if requested
    if snr_db is not None:
        u_d = _add_awgn(u_d, snr_db, rng)
        u_h = _add_awgn(u_h, snr_db, rng)
        u_w = _add_awgn(u_w, snr_db, rng)

    # Normalize to max_displacement
    uvw_max = max(np.abs(u_d).max(), np.abs(u_h).max(), np.abs(u_w).max())
    if uvw_max > 0:
        u_d = u_d / uvw_max * max_displacement
        u_h = u_h / uvw_max * max_displacement
        u_w = u_w / uvw_max * max_displacement

    # Stack and apply final rotation
    displacement = np.stack(
        np.broadcast_arrays(u_d, u_h, u_w),
        axis=-1,
    )

    displacement = _rotate_displacement_field(displacement, np.radians(rotation))

    return displacement


# -----------------------------------------------------------------------------
# Static/uniform field (simple baseline)
# -----------------------------------------------------------------------------


@jaxcheck
def generate_static(
    shape: tuple[int, int, int],
    displacement: Float[Array, "3"] | None = None,
    max_displacement: float = 16.0,
    snr_db: float | None = None,
    rng: np.random.Generator | None = None,
) -> Float[Array, "d h w 3"]:
    """
    Generate uniform/static displacement field.

    Creates a constant displacement across the entire volume,
    optionally with additive Gaussian noise.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).
    displacement : Float[Array, "3"], optional
        Constant displacement vector [dd, dh, dw].
        If None, random within [-max, +max].
    max_displacement : float
        Maximum displacement magnitude if displacement is None.
    snr_db : float, optional
        If provided, adds Gaussian noise at this SNR.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    Float[Array, "d h w 3"]
        Uniform displacement field.
    """
    if rng is None:
        rng = np.random.default_rng()

    if displacement is None:
        displacement = (rng.random(3) * 2.0 - 1.0) * max_displacement

    # Create constant field
    field = np.zeros((*shape, 3), dtype=np.float64)
    field[..., 0] = displacement[0]
    field[..., 1] = displacement[1]
    field[..., 2] = displacement[2]

    # Add noise if requested
    if snr_db is not None:
        for i in range(3):
            field[..., i] = _add_awgn(field[..., i], snr_db, rng)

    return field

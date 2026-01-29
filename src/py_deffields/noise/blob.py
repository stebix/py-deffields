# ruff: noqa: F722
"""Blob/sphere noise generator for particle-like texture patterns.

Blob noise creates randomly placed spherical features that simulate
particles, inclusions, or fiducial markers in volumetric data. These
discrete features provide excellent trackable patterns for DVC.

This implementation supports:
- Random sphere placement with controllable density
- Variable sphere sizes (uniform or Gaussian distribution)
- Soft (Gaussian falloff) or hard (binary) sphere edges
- Intensity variation between spheres
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float
from numpy import ndarray as Array

from .base import NoiseGenerator
from .._types import jaxcheck


# =============================================================================
# Internal utilities
# =============================================================================


def _place_sphere(
    volume: Float[Array, "d h w"],
    center: tuple[float, float, float],
    radius: float,
    intensity: float,
    soft: bool,
    falloff_sigma: float,
) -> None:
    """
    Place a single sphere into the volume (in-place modification).

    Parameters
    ----------
    volume : Float[Array, "d h w"]
        Volume to modify in-place.
    center : tuple[float, float, float]
        Sphere center coordinates (d, h, w).
    radius : float
        Sphere radius in voxels.
    intensity : float
        Peak intensity value at sphere center.
    soft : bool
        If True, use Gaussian falloff; if False, use hard binary edges.
    falloff_sigma : float
        For soft spheres, sigma of the Gaussian falloff relative to radius.
    """
    d, h, w = volume.shape
    cd, ch, cw = center

    # Compute bounding box (with margin for soft edges)
    margin = int(np.ceil(radius * (1 + 2 * falloff_sigma))) if soft else int(np.ceil(radius))

    d_min = max(0, int(cd - margin))
    d_max = min(d, int(cd + margin) + 1)
    h_min = max(0, int(ch - margin))
    h_max = min(h, int(ch + margin) + 1)
    w_min = max(0, int(cw - margin))
    w_max = min(w, int(cw + margin) + 1)

    if d_min >= d_max or h_min >= h_max or w_min >= w_max:
        return  # Sphere is outside volume

    # Create local coordinate grid
    dd = np.arange(d_min, d_max, dtype=np.float64) - cd
    dh = np.arange(h_min, h_max, dtype=np.float64) - ch
    dw = np.arange(w_min, w_max, dtype=np.float64) - cw

    # Distance from center
    gd, gh, gw = np.meshgrid(dd, dh, dw, indexing="ij")
    dist = np.sqrt(gd**2 + gh**2 + gw**2)

    if soft:
        # Gaussian falloff
        sigma = radius * falloff_sigma
        values = intensity * np.exp(-0.5 * (dist / sigma) ** 2)
        # Use max blending to handle overlapping spheres
        volume[d_min:d_max, h_min:h_max, w_min:w_max] = np.maximum(
            volume[d_min:d_max, h_min:h_max, w_min:w_max],
            values,
        )
    else:
        # Hard binary sphere
        mask = dist <= radius
        volume[d_min:d_max, h_min:h_max, w_min:w_max] = np.where(
            mask,
            np.maximum(volume[d_min:d_max, h_min:h_max, w_min:w_max], intensity),
            volume[d_min:d_max, h_min:h_max, w_min:w_max],
        )


# =============================================================================
# Functional interface
# =============================================================================


@jaxcheck
def generate_blobs(
    shape: tuple[int, int, int],
    num_blobs: int | None = None,
    density: float = 0.001,
    radius_mean: float = 5.0,
    radius_std: float = 1.0,
    intensity_mean: float = 1.0,
    intensity_std: float = 0.1,
    soft: bool = True,
    falloff_sigma: float = 0.5,
    min_spacing: float | None = None,
    rng: np.random.Generator | None = None,
) -> Float[Array, "d h w"]:
    """
    Generate blob/sphere noise field with randomly placed spherical features.

    Creates a volume with randomly positioned spherical blobs that simulate
    particles, inclusions, or fiducial markers. Excellent for DVC as the
    discrete features provide strong trackable patterns.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).
    num_blobs : int, optional
        Exact number of blobs to place. If None, computed from density.
    density : float
        Blob density as fraction of total voxels (used if num_blobs is None).
        A density of 0.001 means approximately 1 blob per 1000 voxels.
    radius_mean : float
        Mean blob radius in voxels.
    radius_std : float
        Standard deviation of blob radius.
    intensity_mean : float
        Mean intensity value at blob centers.
    intensity_std : float
        Standard deviation of blob intensities.
    soft : bool
        If True, blobs have Gaussian falloff edges.
        If False, blobs have hard binary edges.
    falloff_sigma : float
        For soft blobs, Gaussian sigma relative to radius.
        A value of 0.5 means sigma = 0.5 * radius.
    min_spacing : float, optional
        Minimum distance between blob centers. If None, no spacing constraint.
        Setting this helps avoid excessive blob overlap.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    Float[Array, "d h w"]
        Blob noise field with values in [0, max_intensity].

    Notes
    -----
    For DVC applications:
    - Use radius_mean ≈ 3-10 voxels for features visible in subvolumes
    - Set soft=True for smoother interpolation during correlation
    - density ≈ 0.0005-0.002 gives good texture without overcrowding

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import generate_blobs
    >>> rng = np.random.default_rng(42)
    >>> noise = generate_blobs((32, 64, 64), num_blobs=50, rng=rng)
    >>> noise.shape
    (32, 64, 64)
    """
    if rng is None:
        rng = np.random.default_rng()

    d, h, w = shape
    total_voxels = d * h * w

    # Determine number of blobs
    if num_blobs is None:
        # Estimate based on density and average blob volume
        avg_blob_volume = (4 / 3) * np.pi * radius_mean**3
        num_blobs = max(1, int(density * total_voxels / avg_blob_volume))

    # Initialize output volume
    volume = np.zeros(shape, dtype=np.float64)

    # Generate blob parameters
    centers = []
    max_attempts = num_blobs * 10  # Limit attempts for spacing constraint

    attempts = 0
    while len(centers) < num_blobs and attempts < max_attempts:
        # Random center position
        cd = rng.uniform(0, d)
        ch = rng.uniform(0, h)
        cw = rng.uniform(0, w)

        # Check minimum spacing constraint
        if min_spacing is not None and len(centers) > 0:
            centers_arr = np.array(centers)
            distances = np.sqrt(
                (centers_arr[:, 0] - cd) ** 2
                + (centers_arr[:, 1] - ch) ** 2
                + (centers_arr[:, 2] - cw) ** 2
            )
            if np.any(distances < min_spacing):
                attempts += 1
                continue

        centers.append((cd, ch, cw))
        attempts += 1

    # Generate radii and intensities
    radii = np.maximum(1.0, rng.normal(radius_mean, radius_std, len(centers)))
    intensities = np.maximum(0.1, rng.normal(intensity_mean, intensity_std, len(centers)))

    # Place blobs
    for center, radius, intensity in zip(centers, radii, intensities):
        _place_sphere(volume, center, radius, intensity, soft, falloff_sigma)

    return volume


# =============================================================================
# Class-based interface
# =============================================================================


class BlobNoise(NoiseGenerator):
    """
    Blob/sphere noise generator for particle-like texture patterns.

    Creates randomly positioned spherical features that simulate particles,
    inclusions, or fiducial markers. The discrete nature of blobs provides
    excellent trackable features for digital volume correlation.

    Parameters
    ----------
    num_blobs : int, optional
        Exact number of blobs to place. If None, computed from density.
    density : float
        Blob density as fraction of voxels (used if num_blobs is None).
    radius_mean : float
        Mean blob radius in voxels.
    radius_std : float
        Standard deviation of blob radius.
    intensity_mean : float
        Mean intensity at blob centers.
    intensity_std : float
        Standard deviation of intensities.
    soft : bool
        If True, blobs have Gaussian falloff edges.
    falloff_sigma : float
        Gaussian sigma relative to radius (for soft blobs).
    min_spacing : float, optional
        Minimum distance between blob centers.

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import BlobNoise
    >>> generator = BlobNoise(num_blobs=100, radius_mean=5.0, soft=True)
    >>> rng = np.random.default_rng(42)
    >>> noise = generator((32, 64, 64), rng=rng)
    >>> noise.shape
    (32, 64, 64)
    """

    def __init__(
        self,
        num_blobs: int | None = None,
        density: float = 0.001,
        radius_mean: float = 5.0,
        radius_std: float = 1.0,
        intensity_mean: float = 1.0,
        intensity_std: float = 0.1,
        soft: bool = True,
        falloff_sigma: float = 0.5,
        min_spacing: float | None = None,
    ) -> None:
        self.num_blobs = num_blobs
        self.density = density
        self.radius_mean = radius_mean
        self.radius_std = radius_std
        self.intensity_mean = intensity_mean
        self.intensity_std = intensity_std
        self.soft = soft
        self.falloff_sigma = falloff_sigma
        self.min_spacing = min_spacing

    @jaxcheck
    def generate(
        self,
        shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> Float[Array, "d h w"]:
        """
        Generate blob noise field.

        Parameters
        ----------
        shape : tuple[int, int, int]
            Volume dimensions (D, H, W).
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        Float[Array, "d h w"]
            Blob noise field.
        """
        return generate_blobs(
            shape,
            self.num_blobs,
            self.density,
            self.radius_mean,
            self.radius_std,
            self.intensity_mean,
            self.intensity_std,
            self.soft,
            self.falloff_sigma,
            self.min_spacing,
            rng,
        )

    def __repr__(self) -> str:
        blob_spec = f"num_blobs={self.num_blobs}" if self.num_blobs else f"density={self.density}"
        return (
            f"BlobNoise({blob_spec}, radius_mean={self.radius_mean}, "
            f"soft={self.soft})"
        )

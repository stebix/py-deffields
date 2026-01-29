# ruff: noqa: F722
"""Perlin noise generator for coherent texture patterns.

Perlin noise is a gradient-based coherent noise function that produces
smooth, natural-looking patterns. It's particularly useful for DVC
because it creates trackable texture features with controllable scale.

This implementation supports:
- 3D Perlin noise with configurable scale
- Octave layering (fractal Brownian motion / fBm) for multi-scale patterns
- Tileable/periodic boundaries (optional)
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float
from numpy import ndarray as Array
from scipy.ndimage import map_coordinates

from .base import NoiseGenerator
from .._types import jaxcheck


# =============================================================================
# Internal utilities
# =============================================================================


def _fade(t: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Quintic smoothstep function for Perlin noise interpolation.

    This is the improved fade function: 6t^5 - 15t^4 + 10t^3
    which has zero first and second derivatives at t=0 and t=1.
    """
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(
    a: Float[Array, "..."],
    b: Float[Array, "..."],
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Linear interpolation between a and b."""
    return a + t * (b - a)


def _generate_gradients(
    grid_shape: tuple[int, int, int],
    rng: np.random.Generator,
) -> Float[Array, "gd gh gw 3"]:
    """
    Generate random unit gradient vectors at each grid point.

    Parameters
    ----------
    grid_shape : tuple[int, int, int]
        Number of grid points in each dimension.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    Float[Array, "gd gh gw 3"]
        Unit gradient vectors at each grid point.
    """
    # Generate random vectors
    gradients = rng.standard_normal((*grid_shape, 3))

    # Normalize to unit length
    norms = np.linalg.norm(gradients, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # Avoid division by zero
    gradients = gradients / norms

    return gradients.astype(np.float64)


def _perlin_3d_single_octave(
    shape: tuple[int, int, int],
    scale: float,
    rng: np.random.Generator,
) -> Float[Array, "d h w"]:
    """
    Generate a single octave of 3D Perlin noise.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Output volume dimensions (D, H, W).
    scale : float
        Scale of the noise pattern (larger = more stretched features).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    Float[Array, "d h w"]
        Single octave of Perlin noise in range approximately [-1, 1].
    """
    d, h, w = shape

    # Determine grid size based on scale
    # Grid spacing in voxels
    grid_d = max(2, int(np.ceil(d / scale)) + 2)
    grid_h = max(2, int(np.ceil(h / scale)) + 2)
    grid_w = max(2, int(np.ceil(w / scale)) + 2)

    # Generate gradient vectors at grid points
    gradients = _generate_gradients((grid_d, grid_h, grid_w), rng)

    # Create coordinate arrays for the output volume
    # Map output coordinates to grid coordinates
    coords_d = np.linspace(0, (d - 1) / scale, d, dtype=np.float64)
    coords_h = np.linspace(0, (h - 1) / scale, h, dtype=np.float64)
    coords_w = np.linspace(0, (w - 1) / scale, w, dtype=np.float64)

    # Create meshgrid (full, not sparse, for indexing)
    gd, gh, gw = np.meshgrid(coords_d, coords_h, coords_w, indexing="ij")

    # Integer grid coordinates (corners of the cell)
    d0 = np.floor(gd).astype(int)
    h0 = np.floor(gh).astype(int)
    w0 = np.floor(gw).astype(int)

    d1 = d0 + 1
    h1 = h0 + 1
    w1 = w0 + 1

    # Fractional coordinates within cell
    fd = gd - d0
    fh = gh - h0
    fw = gw - w0

    # Apply fade function
    ud = _fade(fd)
    uh = _fade(fh)
    uw = _fade(fw)

    # Compute dot products with gradients at 8 corners
    def dot_grad(di: np.ndarray, hi: np.ndarray, wi: np.ndarray) -> np.ndarray:
        """Compute dot product of distance vector with gradient."""
        # Distance from corner to sample point
        dd = fd - (di - d0)
        dh = fh - (hi - h0)
        dw = fw - (wi - w0)

        # Get gradients at corner (clamp indices to valid range)
        di_clamped = np.clip(di, 0, grid_d - 1)
        hi_clamped = np.clip(hi, 0, grid_h - 1)
        wi_clamped = np.clip(wi, 0, grid_w - 1)

        grad = gradients[di_clamped, hi_clamped, wi_clamped]

        # Dot product
        return dd * grad[..., 0] + dh * grad[..., 1] + dw * grad[..., 2]

    # Dot products at 8 corners
    n000 = dot_grad(d0, h0, w0)
    n001 = dot_grad(d0, h0, w1)
    n010 = dot_grad(d0, h1, w0)
    n011 = dot_grad(d0, h1, w1)
    n100 = dot_grad(d1, h0, w0)
    n101 = dot_grad(d1, h0, w1)
    n110 = dot_grad(d1, h1, w0)
    n111 = dot_grad(d1, h1, w1)

    # Trilinear interpolation
    # Interpolate along w
    nx00 = _lerp(n000, n001, uw)
    nx01 = _lerp(n010, n011, uw)
    nx10 = _lerp(n100, n101, uw)
    nx11 = _lerp(n110, n111, uw)

    # Interpolate along h
    nxy0 = _lerp(nx00, nx01, uh)
    nxy1 = _lerp(nx10, nx11, uh)

    # Interpolate along d
    noise = _lerp(nxy0, nxy1, ud)

    return noise.astype(np.float64)


# =============================================================================
# Functional interface
# =============================================================================


@jaxcheck
def generate_perlin(
    shape: tuple[int, int, int],
    scale: float = 32.0,
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    normalize: bool = True,
    rng: np.random.Generator | None = None,
) -> Float[Array, "d h w"]:
    """
    Generate 3D Perlin noise with optional octave layering (fBm).

    Creates coherent, gradient-based noise that produces smooth, natural
    patterns. Multiple octaves can be combined to create fractal Brownian
    motion (fBm) with multi-scale detail.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).
    scale : float
        Base scale of the noise pattern in voxels. Larger values produce
        more stretched, smoother features.
    octaves : int
        Number of noise layers to combine. More octaves add finer detail
        but increase computation time.
    persistence : float
        Amplitude multiplier for each successive octave. Values < 1
        reduce the influence of higher-frequency octaves.
    lacunarity : float
        Frequency multiplier for each successive octave. Typically 2.0
        means each octave has twice the frequency of the previous.
    normalize : bool
        If True, normalize output to [-1, 1] range.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    Float[Array, "d h w"]
        Perlin noise field. If normalized, values are in [-1, 1].

    Notes
    -----
    The fractal Brownian motion (fBm) formula is:
        noise = sum_{i=0}^{octaves-1} persistence^i * perlin(freq * lacunarity^i)

    For DVC applications:
    - Use scale ≈ 16-64 for features visible in typical subvolume sizes
    - Use 3-5 octaves for natural-looking textures
    - persistence=0.5 gives a good balance of detail

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import generate_perlin
    >>> rng = np.random.default_rng(42)
    >>> noise = generate_perlin((32, 64, 64), scale=16, octaves=4, rng=rng)
    >>> noise.shape
    (32, 64, 64)
    >>> -1 <= noise.min() <= noise.max() <= 1  # Normalized
    True
    """
    if rng is None:
        rng = np.random.default_rng()

    if octaves < 1:
        raise ValueError("octaves must be >= 1")

    noise = np.zeros(shape, dtype=np.float64)
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        # Generate single octave at current frequency
        current_scale = scale / frequency
        octave_noise = _perlin_3d_single_octave(shape, current_scale, rng)

        # Accumulate
        noise += amplitude * octave_noise
        max_amplitude += amplitude

        # Update for next octave
        amplitude *= persistence
        frequency *= lacunarity

    # Normalize to [-1, 1] if requested
    if normalize and max_amplitude > 0:
        noise = noise / max_amplitude

    return noise


# =============================================================================
# Class-based interface
# =============================================================================


class PerlinNoise(NoiseGenerator):
    """
    Perlin noise generator with octave layering (fBm).

    Generates coherent, gradient-based noise that produces smooth,
    natural-looking texture patterns. Supports fractal Brownian motion
    (fBm) through octave layering for multi-scale detail.

    Parameters
    ----------
    scale : float
        Base scale of the noise pattern in voxels.
    octaves : int
        Number of noise layers to combine.
    persistence : float
        Amplitude decay per octave (typically 0.5).
    lacunarity : float
        Frequency growth per octave (typically 2.0).
    normalize : bool
        If True, normalize output to [-1, 1] range.

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import PerlinNoise
    >>> generator = PerlinNoise(scale=16, octaves=4)
    >>> rng = np.random.default_rng(42)
    >>> noise = generator((32, 64, 64), rng=rng)
    >>> noise.shape
    (32, 64, 64)
    """

    def __init__(
        self,
        scale: float = 32.0,
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        normalize: bool = True,
    ) -> None:
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.normalize = normalize

    @jaxcheck
    def generate(
        self,
        shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> Float[Array, "d h w"]:
        """
        Generate Perlin noise field.

        Parameters
        ----------
        shape : tuple[int, int, int]
            Volume dimensions (D, H, W).
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        Float[Array, "d h w"]
            Perlin noise field.
        """
        return generate_perlin(
            shape,
            self.scale,
            self.octaves,
            self.persistence,
            self.lacunarity,
            self.normalize,
            rng,
        )

    def __repr__(self) -> str:
        return (
            f"PerlinNoise(scale={self.scale}, octaves={self.octaves}, "
            f"persistence={self.persistence}, lacunarity={self.lacunarity})"
        )

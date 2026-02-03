"""Noise generators for volumetric data augmentation.

This module provides noise generators for creating texture patterns
in volumetric data, particularly useful for digital volume correlation (DVC).
DVC algorithms require unique texture patterns to accurately compute
correlation metrics between reference and deformed subvolumes.

Each noise type is available in two interfaces:
- **Functional**: Pure functions for one-off generation (e.g., `generate_gaussian`)
- **Class-based**: Configurable generators for repeated use (e.g., `GaussianNoise`)

Available Noise Types
--------------------
- **Gaussian**: Independent random noise at each voxel (baseline texture)
- **Perlin**: Coherent gradient noise with smooth, natural patterns
- **Blob**: Randomly placed spherical features (particle-like texture)
- **Poisson**: Signal-dependent shot noise (photon counting statistics)
- **Cellular**: Cell-like Worley/Voronoi tessellation patterns
- **Spectral**: Frequency-domain noise with prescribed power spectrum
- **Speckle**: Multiplicative coherent imaging noise (ultrasound, SAR, OCT)

Examples
--------
Functional interface (one-off generation):

>>> import numpy as np
>>> from py_deffields.noise import generate_perlin, generate_blobs, blend_noise
>>> rng = np.random.default_rng(42)
>>> perlin = generate_perlin((64, 128, 128), scale=16, octaves=4, rng=rng)
>>> blobs = generate_blobs((64, 128, 128), num_blobs=100, rng=rng)
>>> texture = blend_noise(perlin, blobs, weights=[0.7, 0.3])

Class-based interface (reusable configuration):

>>> from py_deffields.noise import PerlinNoise, BlobNoise
>>> perlin_gen = PerlinNoise(scale=16, octaves=4)
>>> blob_gen = BlobNoise(num_blobs=100, radius_mean=5.0)
>>> noise1 = perlin_gen((64, 128, 128), rng=rng)
>>> noise2 = perlin_gen((64, 128, 128), rng=rng)  # Same config, different result
"""

from .base import NoiseGenerator

# Gaussian noise
from .gaussian import GaussianNoise, generate_gaussian

# Perlin noise
from .perlin import PerlinNoise, generate_perlin

# Blob noise
from .blob import BlobNoise, generate_blobs

# Poisson noise
from .poisson import PoissonNoise, generate_poisson

# Cellular noise
from .cellular import CellularNoise, generate_cellular

# Spectral noise
from .spectral import SpectralNoise, generate_spectral

# Speckle noise
from .speckle import SpeckleNoise, generate_speckle

# Composition utilities
from .compose import (
    add_noise,
    multiply_noise,
    blend_noise,
    normalize_noise,
    clip_noise,
)

__all__ = [
    # Base class
    "NoiseGenerator",
    # Class-based generators
    "GaussianNoise",
    "PerlinNoise",
    "BlobNoise",
    "PoissonNoise",
    "CellularNoise",
    "SpectralNoise",
    "SpeckleNoise",
    # Functional generators
    "generate_gaussian",
    "generate_perlin",
    "generate_blobs",
    "generate_poisson",
    "generate_cellular",
    "generate_spectral",
    "generate_speckle",
    # Composition utilities
    "add_noise",
    "multiply_noise",
    "blend_noise",
    "normalize_noise",
    "clip_noise",
]

# ruff: noqa: F722
"""Cellular (Worley) noise generator for cell-like texture patterns.

Cellular noise produces tessellation patterns based on distances to
randomly scattered feature points. The resulting patterns resemble
biological cells, foam, grain boundaries, or Voronoi diagrams.

This implementation supports:
- Multiple distance metrics (Euclidean, Manhattan, Chebyshev)
- Configurable nth-closest point selection
- Combination modes (F2 - F1 produces vein/crack patterns)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from jaxtyping import Float
from numpy import ndarray as Array

from .base import NoiseGenerator
from .._types import jaxcheck


# =============================================================================
# Internal utilities
# =============================================================================


def _compute_distances(
    coords: Float[Array, "n 3"],
    points: Float[Array, "m 3"],
    metric: Literal["euclidean", "manhattan", "chebyshev"],
) -> Float[Array, "n m"]:
    """
    Compute pairwise distances between coordinates and feature points.

    Parameters
    ----------
    coords : Float[Array, "n 3"]
        Query coordinates, one per row.
    points : Float[Array, "m 3"]
        Feature point locations.
    metric : str
        Distance metric to use.

    Returns
    -------
    Float[Array, "n m"]
        Distance from each coordinate to each feature point.
    """
    # coords: (n, 3), points: (m, 3)
    # diff: (n, m, 3)
    diff = coords[:, np.newaxis, :] - points[np.newaxis, :, :]

    if metric == "euclidean":
        return np.sqrt(np.sum(diff**2, axis=-1))
    elif metric == "manhattan":
        return np.sum(np.abs(diff), axis=-1)
    elif metric == "chebyshev":
        return np.max(np.abs(diff), axis=-1)
    else:
        raise ValueError(f"Unknown distance metric: {metric!r}")


def _worley_3d(
    shape: tuple[int, int, int],
    points: Float[Array, "m 3"],
    metric: Literal["euclidean", "manhattan", "chebyshev"],
    nth_closest: int,
    chunk_size: int = 65536,
) -> Float[Array, "d h w"]:
    """
    Compute Worley noise values for a 3D volume.

    Processes voxels in chunks to limit memory usage for large volumes.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).
    points : Float[Array, "m 3"]
        Feature point locations in voxel coordinates.
    metric : str
        Distance metric.
    nth_closest : int
        Which closest point to use (1-indexed).
    chunk_size : int
        Number of voxels to process per chunk.

    Returns
    -------
    Float[Array, "d h w"]
        Raw distance values at each voxel.
    """
    d, h, w = shape
    total = d * h * w

    # Create coordinate grid
    gd, gh, gw = np.meshgrid(
        np.arange(d, dtype=np.float64),
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )
    coords = np.stack([gd.ravel(), gh.ravel(), gw.ravel()], axis=-1)

    result = np.empty(total, dtype=np.float64)

    # Process in chunks to limit memory
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk_coords = coords[start:end]

        # Compute distances to all feature points
        dists = _compute_distances(chunk_coords, points, metric)

        # Sort and select nth closest (1-indexed to 0-indexed)
        if nth_closest >= dists.shape[1]:
            # Not enough points; use the farthest available
            sorted_dists = np.sort(dists, axis=1)
            result[start:end] = sorted_dists[:, -1]
        else:
            # Partial sort is more efficient than full sort
            partitioned = np.partition(dists, nth_closest - 1, axis=1)
            # partition doesn't sort within the first k, so we need
            # to take the max of the first nth_closest elements
            result[start:end] = np.sort(
                partitioned[:, :nth_closest], axis=1
            )[:, nth_closest - 1]

    return result.reshape(shape)


# =============================================================================
# Functional interface
# =============================================================================


@jaxcheck
def generate_cellular(
    shape: tuple[int, int, int],
    num_points: int = 64,
    metric: Literal["euclidean", "manhattan", "chebyshev"] = "euclidean",
    nth_closest: int = 1,
    combination: Literal["F1", "F2", "F2-F1"] = "F1",
    normalize: bool = True,
    rng: np.random.Generator | None = None,
) -> Float[Array, "d h w"]:
    """
    Generate cellular (Worley) noise field.

    Creates cell-like, tessellation patterns based on distances to
    randomly scattered feature points. The resulting patterns resemble
    biological cells, foam structures, or grain boundaries.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).
    num_points : int
        Number of feature points to scatter. Controls cell density;
        more points produce smaller cells.
    metric : {"euclidean", "manhattan", "chebyshev"}
        Distance metric for computing proximity to feature points.
        - "euclidean": Round cells (standard Worley noise)
        - "manhattan": Diamond-shaped cells
        - "chebyshev": Square/cubic cells
    nth_closest : int
        Which closest feature point to use (1-indexed).
        - 1: Standard Worley noise (F1), produces convex cells
        - 2: F2 noise, produces rounder, more uniform patterns
    combination : {"F1", "F2", "F2-F1"}
        How to combine distance values.
        - "F1": Distance to 1st closest point (cell interiors)
        - "F2": Distance to 2nd closest point (rounder patterns)
        - "F2-F1": Difference between F2 and F1 (vein/crack patterns
          with values near zero at cell boundaries)
    normalize : bool
        If True, normalize output to [0, 1] range.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    Float[Array, "d h w"]
        Cellular noise field. If normalized, values are in [0, 1].

    Notes
    -----
    For DVC applications:
    - Use num_points ≈ 32-128 for features at typical subvolume scales
    - "F2-F1" combination produces edge-like features useful for
      tracking deformations near grain boundaries
    - "euclidean" metric produces the most natural-looking patterns

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import generate_cellular
    >>> rng = np.random.default_rng(42)
    >>> noise = generate_cellular((32, 64, 64), num_points=50, rng=rng)
    >>> noise.shape
    (32, 64, 64)

    Vein/crack pattern using F2-F1:

    >>> veins = generate_cellular(
    ...     (32, 64, 64), num_points=50, combination="F2-F1", rng=rng,
    ... )
    """
    if rng is None:
        rng = np.random.default_rng()

    if num_points < 1:
        raise ValueError("num_points must be >= 1")

    d, h, w = shape

    # Scatter feature points uniformly within the volume
    points = np.column_stack([
        rng.uniform(0.0, d, size=num_points),
        rng.uniform(0.0, h, size=num_points),
        rng.uniform(0.0, w, size=num_points),
    ])

    if combination == "F1":
        noise = _worley_3d(shape, points, metric, nth_closest=1)
    elif combination == "F2":
        noise = _worley_3d(shape, points, metric, nth_closest=2)
    elif combination == "F2-F1":
        f1 = _worley_3d(shape, points, metric, nth_closest=1)
        # Need at least 2 points for F2
        if num_points < 2:
            noise = np.zeros(shape, dtype=np.float64)
        else:
            f2 = _worley_3d(shape, points, metric, nth_closest=2)
            noise = f2 - f1
    else:
        raise ValueError(f"Unknown combination mode: {combination!r}")

    # Normalize to [0, 1] if requested
    if normalize:
        noise_min = noise.min()
        noise_max = noise.max()
        if noise_max - noise_min > 1e-10:
            noise = (noise - noise_min) / (noise_max - noise_min)
        else:
            noise = np.zeros_like(noise)

    return noise.astype(np.float64)


# =============================================================================
# Class-based interface
# =============================================================================


class CellularNoise(NoiseGenerator):
    """
    Cellular (Worley) noise generator for cell-like texture patterns.

    Creates tessellation patterns resembling biological cells, foam,
    or grain boundaries. Based on distances to randomly scattered
    feature points in 3D space.

    Parameters
    ----------
    num_points : int
        Number of feature points (controls cell density).
    metric : {"euclidean", "manhattan", "chebyshev"}
        Distance metric for proximity computation.
    nth_closest : int
        Which closest point to use (1-indexed).
    combination : {"F1", "F2", "F2-F1"}
        Distance combination mode.
    normalize : bool
        If True, normalize output to [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import CellularNoise
    >>> generator = CellularNoise(num_points=50, combination="F2-F1")
    >>> rng = np.random.default_rng(42)
    >>> noise = generator((32, 64, 64), rng=rng)
    >>> noise.shape
    (32, 64, 64)
    """

    def __init__(
        self,
        num_points: int = 64,
        metric: Literal["euclidean", "manhattan", "chebyshev"] = "euclidean",
        nth_closest: int = 1,
        combination: Literal["F1", "F2", "F2-F1"] = "F1",
        normalize: bool = True,
    ) -> None:
        self.num_points = num_points
        self.metric = metric
        self.nth_closest = nth_closest
        self.combination = combination
        self.normalize = normalize

    @jaxcheck
    def generate(
        self,
        shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> Float[Array, "d h w"]:
        """
        Generate cellular noise field.

        Parameters
        ----------
        shape : tuple[int, int, int]
            Volume dimensions (D, H, W).
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        Float[Array, "d h w"]
            Cellular noise field.
        """
        return generate_cellular(
            shape,
            self.num_points,
            self.metric,
            self.nth_closest,
            self.combination,
            self.normalize,
            rng,
        )

    def __repr__(self) -> str:
        return (
            f"CellularNoise(num_points={self.num_points}, "
            f"metric={self.metric!r}, combination={self.combination!r})"
        )

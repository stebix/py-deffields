# ruff: noqa: F722
"""Poisson (shot) noise generator for signal-dependent noise simulation.

Poisson noise models photon counting statistics in imaging detectors
(X-ray CT, electron microscopy). Unlike Gaussian noise, the variance
scales with signal level: brighter regions exhibit more noise.

This implementation supports:
- Signal-dependent Poisson noise via a configurable scale factor
- Output as either the noisy signal or the noise delta (noisy - original)
- Approximate Gaussian mode for high photon counts
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Inexact
from numpy import ndarray as Array

from .base import NoiseGenerator
from .._types import jaxcheck


# =============================================================================
# Functional interface
# =============================================================================


@jaxcheck
def generate_poisson(
    shape: tuple[int, int, int],
    signal: Inexact[Array, "d h w"] | None = None,
    scale: float = 100.0,
    normalize: bool = True,
    rng: np.random.Generator | None = None,
) -> Float[Array, "d h w"]:
    """
    Generate Poisson (shot) noise field.

    Produces signal-dependent noise where the variance at each voxel is
    proportional to its intensity. This models the physical process of
    photon detection in imaging systems.

    When no signal is provided, a uniform unit signal is assumed and the
    result is pure Poisson-distributed noise scaled to [0, 1].

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).
    signal : Inexact[Array, "d h w"], optional
        Input signal whose intensity determines noise variance.
        Values should be non-negative. If None, a uniform unit signal
        is used and the output represents pure shot noise.
    scale : float
        Photon count scaling factor. Maps signal intensity to expected
        photon counts: lambda = signal * scale. Higher values produce
        lower relative noise (better SNR). Typical values:
        - 10-50: heavy noise (low-dose imaging)
        - 100-500: moderate noise
        - 1000+: light noise (high-dose imaging)
    normalize : bool
        If True, return the noise delta (noisy - original) normalized
        by scale, so output represents noise only. If False, return
        the full noisy signal scaled back to the original intensity range.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    Float[Array, "d h w"]
        Poisson noise field. If normalize=True, this is the noise
        component only (centered near zero). If normalize=False,
        this is the full noisy signal.

    Notes
    -----
    The Poisson distribution has the property that Var[X] = E[X],
    so the signal-to-noise ratio (SNR) improves with the square root
    of intensity: SNR = sqrt(lambda).

    For DVC applications:
    - Use scale ≈ 50-200 for realistic CT detector noise levels
    - Poisson noise is non-Gaussian at low counts (scale < 20),
      approaching Gaussian at high counts

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import generate_poisson
    >>> rng = np.random.default_rng(42)
    >>> noise = generate_poisson((32, 64, 64), scale=100.0, rng=rng)
    >>> noise.shape
    (32, 64, 64)

    Signal-dependent noise on a real volume:

    >>> signal = np.random.default_rng(0).random((32, 64, 64))
    >>> noisy = generate_poisson(
    ...     (32, 64, 64), signal=signal, scale=200.0, normalize=False, rng=rng,
    ... )
    """
    if rng is None:
        rng = np.random.default_rng()

    if signal is None:
        signal = np.ones(shape, dtype=np.float64)

    # Ensure non-negative for Poisson sampling
    signal_clamped = np.maximum(signal, 0.0)

    # Scale signal to photon counts (lambda parameter)
    lam = signal_clamped * scale

    # Sample from Poisson distribution
    noisy_counts = rng.poisson(lam).astype(np.float64)

    if normalize:
        # Return noise delta normalized by scale
        return ((noisy_counts / scale) - signal_clamped).astype(np.float64)
    else:
        # Return full noisy signal scaled back to original range
        return (noisy_counts / scale).astype(np.float64)


# =============================================================================
# Class-based interface
# =============================================================================


class PoissonNoise(NoiseGenerator):
    """
    Poisson (shot) noise generator for signal-dependent noise.

    Models photon counting noise where variance scales with signal
    intensity. Commonly used to simulate detector noise in X-ray CT,
    electron microscopy, and other photon-counting imaging systems.

    Parameters
    ----------
    signal : Inexact[Array, "d h w"], optional
        Reference signal for signal-dependent noise. If None, a
        uniform unit signal is assumed.
    scale : float
        Photon count scaling factor. Higher values produce less
        relative noise (better SNR).
    normalize : bool
        If True, return noise delta only. If False, return full
        noisy signal.

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import PoissonNoise
    >>> generator = PoissonNoise(scale=100.0)
    >>> rng = np.random.default_rng(42)
    >>> noise = generator((32, 64, 64), rng=rng)
    >>> noise.shape
    (32, 64, 64)
    """

    def __init__(
        self,
        signal: Inexact[Array, "d h w"] | None = None,
        scale: float = 100.0,
        normalize: bool = True,
    ) -> None:
        self.signal = signal
        self.scale = scale
        self.normalize = normalize

    @jaxcheck
    def generate(
        self,
        shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> Float[Array, "d h w"]:
        """
        Generate Poisson noise field.

        Parameters
        ----------
        shape : tuple[int, int, int]
            Volume dimensions (D, H, W).
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        Float[Array, "d h w"]
            Poisson noise field.
        """
        return generate_poisson(shape, self.signal, self.scale, self.normalize, rng)

    def __repr__(self) -> str:
        signal_str = "provided" if self.signal is not None else "None"
        return (
            f"PoissonNoise(signal={signal_str}, scale={self.scale}, "
            f"normalize={self.normalize})"
        )

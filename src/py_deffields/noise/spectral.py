# ruff: noqa: F722
"""Spectral (frequency-domain) noise generator with prescribed power spectrum.

Spectral noise provides direct control over the frequency content of the
generated pattern by operating in Fourier space. Random phases are combined
with a shaped amplitude spectrum, then inverse-transformed to produce
spatial noise with a prescribed power spectral density (PSD).

This implementation supports:
- Power-law spectra: 1/f^beta (white, pink, brown/red, blue, violet)
- Optional bandpass filtering (freq_min, freq_max)
- Normalization to [-1, 1] range
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float
from numpy import ndarray as Array

from .base import NoiseGenerator
from .._types import jaxcheck


# =============================================================================
# Functional interface
# =============================================================================


@jaxcheck
def generate_spectral(
    shape: tuple[int, int, int],
    spectral_exponent: float = 0.0,
    freq_min: float | None = None,
    freq_max: float | None = None,
    normalize: bool = True,
    rng: np.random.Generator | None = None,
) -> Float[Array, "d h w"]:
    """
    Generate noise with a prescribed power spectral density.

    Creates noise by generating random phases in Fourier space and
    shaping the amplitude spectrum according to a power law 1/f^beta.
    The inverse FFT produces spatial noise with the desired frequency
    characteristics.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).
    spectral_exponent : float
        Power law exponent beta for the 1/f^beta amplitude spectrum.
        Controls the color/character of the noise:
        - 0.0: White noise (flat spectrum, all frequencies equal)
        - 0.5: Pink-ish noise (gentle high-frequency rolloff)
        - 1.0: Pink noise (1/f, equal energy per octave)
        - 2.0: Brown/red noise (1/f², smooth, cloud-like)
        - -1.0: Blue noise (high frequencies emphasized)
        - -2.0: Violet noise (even more high-frequency emphasis)
    freq_min : float, optional
        Minimum frequency in cycles/voxel (bandpass lower bound).
        Frequencies below this are zeroed out. Range: [0, 0.5].
        If None, no lower cutoff is applied.
    freq_max : float, optional
        Maximum frequency in cycles/voxel (bandpass upper bound).
        Frequencies above this are zeroed out. Range: [0, 0.5].
        If None, no upper cutoff is applied.
    normalize : bool
        If True, normalize output to [-1, 1] range.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    Float[Array, "d h w"]
        Spectral noise field. If normalized, values are in [-1, 1].

    Notes
    -----
    The power spectral density follows S(f) = 1/|f|^beta, where f is
    the radial spatial frequency and beta is the spectral exponent.

    Common noise colors and their properties:

    =========  ====  =========================================
    Color      Beta  Character
    =========  ====  =========================================
    Violet     -2    Very rough, emphasizes fine detail
    Blue       -1    Rough, anti-correlated
    White       0    Uncorrelated, equal power at all scales
    Pink        1    Natural-looking, scale-invariant
    Brown/Red   2    Very smooth, cloud-like, large features
    =========  ====  =========================================

    For DVC applications:
    - Pink noise (beta=1) produces natural-looking textures
    - Brown noise (beta=2) creates large-scale intensity variations
    - Bandpass filtering (freq_min/freq_max) lets you match the
      frequency content to the DVC subvolume size for optimal tracking

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import generate_spectral
    >>> rng = np.random.default_rng(42)

    Pink noise (1/f):

    >>> pink = generate_spectral((32, 64, 64), spectral_exponent=1.0, rng=rng)
    >>> pink.shape
    (32, 64, 64)

    Bandpass-filtered brown noise:

    >>> brown_bp = generate_spectral(
    ...     (32, 64, 64),
    ...     spectral_exponent=2.0,
    ...     freq_min=0.02,
    ...     freq_max=0.2,
    ...     rng=rng,
    ... )
    """
    if rng is None:
        rng = np.random.default_rng()

    d, h, w = shape

    # Generate random white noise in spatial domain, then FFT
    white = rng.standard_normal(shape)
    spectrum = np.fft.rfftn(white)

    # Build 3D frequency magnitude grid
    # np.fft.fftfreq gives frequencies in cycles/sample
    freq_d = np.fft.fftfreq(d)
    freq_h = np.fft.fftfreq(h)
    freq_w = np.fft.rfftfreq(w)  # rfft for last axis

    # Sparse meshgrid for memory efficiency
    fd, fh, fw = np.meshgrid(freq_d, freq_h, freq_w, sparse=True, indexing="ij")
    freq_magnitude = np.sqrt(fd**2 + fh**2 + fw**2)

    # Avoid division by zero at DC component
    freq_magnitude_safe = np.maximum(freq_magnitude, np.finfo(np.float64).eps)

    # Build amplitude filter: 1/f^(beta/2)
    # We use beta/2 because PSD ~ |amplitude|^2, so amplitude ~ 1/f^(beta/2)
    amplitude_filter = 1.0 / (freq_magnitude_safe ** (spectral_exponent / 2.0))

    # Zero out DC component to ensure zero-mean output
    amplitude_filter[0, 0, 0] = 0.0

    # Apply optional bandpass filter
    if freq_min is not None:
        amplitude_filter[freq_magnitude < freq_min] = 0.0

    if freq_max is not None:
        amplitude_filter[freq_magnitude > freq_max] = 0.0

    # Apply filter to spectrum
    shaped_spectrum = spectrum * amplitude_filter

    # Inverse FFT to get spatial noise
    noise = np.fft.irfftn(shaped_spectrum, s=shape)

    # Normalize to [-1, 1] if requested
    if normalize:
        noise_max = np.max(np.abs(noise))
        if noise_max > 1e-10:
            noise = noise / noise_max

    return noise.astype(np.float64)


# =============================================================================
# Class-based interface
# =============================================================================


class SpectralNoise(NoiseGenerator):
    """
    Spectral noise generator with prescribed power spectral density.

    Generates noise by shaping random phases in Fourier space according
    to a power-law amplitude spectrum 1/f^beta. Provides direct control
    over the frequency content of the generated pattern.

    Parameters
    ----------
    spectral_exponent : float
        Power law exponent beta (0=white, 1=pink, 2=brown).
    freq_min : float, optional
        Bandpass lower cutoff in cycles/voxel.
    freq_max : float, optional
        Bandpass upper cutoff in cycles/voxel.
    normalize : bool
        If True, normalize output to [-1, 1].

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import SpectralNoise
    >>> generator = SpectralNoise(spectral_exponent=1.0)
    >>> rng = np.random.default_rng(42)
    >>> noise = generator((32, 64, 64), rng=rng)
    >>> noise.shape
    (32, 64, 64)
    """

    def __init__(
        self,
        spectral_exponent: float = 0.0,
        freq_min: float | None = None,
        freq_max: float | None = None,
        normalize: bool = True,
    ) -> None:
        self.spectral_exponent = spectral_exponent
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.normalize = normalize

    @jaxcheck
    def generate(
        self,
        shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> Float[Array, "d h w"]:
        """
        Generate spectral noise field.

        Parameters
        ----------
        shape : tuple[int, int, int]
            Volume dimensions (D, H, W).
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        Float[Array, "d h w"]
            Spectral noise field.
        """
        return generate_spectral(
            shape,
            self.spectral_exponent,
            self.freq_min,
            self.freq_max,
            self.normalize,
            rng,
        )

    def __repr__(self) -> str:
        parts = [f"spectral_exponent={self.spectral_exponent}"]
        if self.freq_min is not None:
            parts.append(f"freq_min={self.freq_min}")
        if self.freq_max is not None:
            parts.append(f"freq_max={self.freq_max}")
        return f"SpectralNoise({', '.join(parts)})"

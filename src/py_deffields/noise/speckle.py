# ruff: noqa: F722
"""Speckle noise generator for coherent imaging simulation.

Speckle is a granular interference pattern inherent to coherent imaging
systems (ultrasound, SAR, OCT, laser imaging). It arises when a coherent
wave scatters from many sub-resolution scatterers and their random phases
cause constructive/destructive interference.

Unlike additive sensor noise, speckle is fundamentally multiplicative:
the observed intensity is I_observed = I_clean * S, where S is the
speckle field with E[S] = 1.

This implementation supports:
- Multiple statistical models (Gamma, Rayleigh)
- Configurable number of looks (L) controlling speckle contrast
- Spatially correlated speckle via anisotropic Gaussian filtering
- Three generation methods (direct sampling, phasor sum, filtered)
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


def _apply_spatial_correlation(
    field: np.ndarray,
    shape: tuple[int, int, int],
    correlation_length: tuple[float, float, float],
) -> np.ndarray:
    """
    Apply spatial correlation to a field via anisotropic Gaussian filtering
    in the frequency domain.

    Parameters
    ----------
    field : np.ndarray
        Input field to filter.
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).
    correlation_length : tuple[float, float, float]
        Correlation lengths (sigma_d, sigma_h, sigma_w) in voxels.

    Returns
    -------
    np.ndarray
        Spatially correlated field.
    """
    spectrum = np.fft.rfftn(field)

    # Build anisotropic Gaussian filter in frequency domain
    freq_d = np.fft.fftfreq(shape[0])
    freq_h = np.fft.fftfreq(shape[1])
    freq_w = np.fft.rfftfreq(shape[2])

    fd, fh, fw = np.meshgrid(freq_d, freq_h, freq_w, sparse=True, indexing="ij")

    sigma_d, sigma_h, sigma_w = correlation_length
    exponent = (
        (2.0 * np.pi * fd * sigma_d) ** 2
        + (2.0 * np.pi * fh * sigma_h) ** 2
        + (2.0 * np.pi * fw * sigma_w) ** 2
    ) / 2.0
    gaussian_filter = np.exp(-exponent)

    filtered = np.fft.irfftn(spectrum * gaussian_filter, s=shape)
    return filtered


def _generate_phasor_field(
    shape: tuple[int, int, int],
    correlation_length: tuple[float, float, float] | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a single-look speckle intensity field via the random phasor
    sum model (Goodman model).

    The complex field at each voxel is modeled as a sum of independent
    phasors. By the central limit theorem, the real and imaginary parts
    become independent Gaussians, yielding Rayleigh-distributed amplitude
    and exponentially-distributed intensity.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).
    correlation_length : tuple[float, float, float] or None
        If provided, applies anisotropic spatial correlation.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Single-look intensity field (exponentially distributed),
        normalized to mean 1.
    """
    real_part = rng.standard_normal(shape)
    imag_part = rng.standard_normal(shape)

    if correlation_length is not None:
        real_part = _apply_spatial_correlation(real_part, shape, correlation_length)
        imag_part = _apply_spatial_correlation(imag_part, shape, correlation_length)

    # Intensity = |complex field|^2
    intensity = real_part**2 + imag_part**2

    # Normalize to mean 1
    mean_intensity = intensity.mean()
    if mean_intensity > 1e-10:
        intensity = intensity / mean_intensity

    return intensity


# =============================================================================
# Functional interface
# =============================================================================


@jaxcheck
def generate_speckle(
    shape: tuple[int, int, int],
    num_looks: float = 1.0,
    correlation_length: float | tuple[float, float, float] | None = None,
    model: Literal["gamma", "rayleigh"] = "gamma",
    method: Literal["direct", "phasor", "filtered"] = "direct",
    normalize: bool = True,
    rng: np.random.Generator | None = None,
) -> Float[Array, "d h w"]:
    """
    Generate a multiplicative speckle noise field.

    Produces a speckle field S with E[S] = 1, intended for multiplicative
    application: I_observed = I_clean * S. The speckle contrast (ratio of
    standard deviation to mean) is controlled by the number of looks L:
    contrast = 1 / sqrt(L).

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume dimensions (D, H, W).
    num_looks : float
        Number of looks (L). Controls speckle contrast C = 1/sqrt(L).
        Must be >= 1.

        - 1.0: Fully developed speckle (maximum contrast, C = 1)
        - 4.0: Multi-look speckle (C = 0.5)
        - 16.0: Heavily averaged speckle (C = 0.25)

        Fractional values are valid (the Gamma distribution accepts
        non-integer shape parameters).
    correlation_length : float or tuple[float, float, float], optional
        Speckle grain size in voxels. Controls spatial correlation of
        the speckle pattern.

        - If a scalar, isotropic correlation is applied.
        - If a 3-tuple ``(sigma_d, sigma_h, sigma_w)``, anisotropic
          correlation is applied with different grain sizes per axis.
          This is typical of real imaging systems where axial and
          lateral resolutions differ (e.g., OCT, ultrasound).
        - If None, speckle is spatially uncorrelated (each voxel
          independent).
    model : {"gamma", "rayleigh"}
        Statistical model for the speckle field.

        - "gamma": Intensity speckle. The field follows a Gamma
          distribution with shape=L, scale=1/L, yielding mean=1
          and variance=1/L. This is the standard model for detected
          intensity in coherent imaging.
        - "rayleigh": Amplitude speckle. The field follows a Rayleigh-
          derived distribution (square root of Gamma-distributed
          intensity), normalized to mean 1. Models the amplitude
          (envelope) signal before squaring.
    method : {"direct", "phasor", "filtered"}
        Generation algorithm.

        - "direct": Sample directly from the Gamma distribution.
          Fast and exact for the marginal distribution, but produces
          spatially uncorrelated speckle. The ``correlation_length``
          parameter is ignored.
        - "phasor": Complex Gaussian field model (Goodman model).
          Generates independent real and imaginary Gaussian fields,
          optionally applies spatial correlation via FFT filtering,
          then computes intensity as |complex field|^2. Physically
          motivated. Always produces single-look (L=1) speckle;
          for L > 1, averages L independent phasor fields.
        - "filtered": Generates spatially correlated Gaussian noise
          via FFT filtering, then maps to the target Gamma marginal
          distribution via the probability integral transform (CDF
          inversion). Supports both spatial correlation and arbitrary
          num_looks in a single pass.
    normalize : bool
        If True, normalize the output field to have mean = 1.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    Float[Array, "d h w"]
        Multiplicative speckle noise field with E[S] ≈ 1.
        Apply as ``I_observed = I_clean * S``.

    Notes
    -----
    Speckle is fundamentally different from additive noise. The
    multiplicative model ``I = I_clean * S`` means that absolute noise
    amplitude scales with signal intensity: brighter regions have larger
    fluctuations but the same relative noise level.

    The number of looks L relates to the speckle contrast C by:

    .. math:: C = \\frac{\\sigma}{\\mu} = \\frac{1}{\\sqrt{L}}

    For DVC applications:

    - Use ``correlation_length`` of 3-5 voxels to match typical DVC
      subset sizes for optimal tracking performance
    - Anisotropic correlation (e.g., ``(2, 5, 5)``) simulates realistic
      ultrasound or OCT texture where axial and lateral resolutions differ
    - The ``"phasor"`` method produces the most physically realistic
      speckle patterns for coherent imaging simulation
    - ``num_looks=1`` gives maximum contrast; increase for smoother
      texture with less speckle variation

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import generate_speckle
    >>> rng = np.random.default_rng(42)

    Fully developed uncorrelated speckle:

    >>> speckle = generate_speckle((32, 64, 64), rng=rng)
    >>> speckle.shape
    (32, 64, 64)

    Multi-look speckle with anisotropic correlation:

    >>> speckle = generate_speckle(
    ...     (32, 64, 64),
    ...     num_looks=4.0,
    ...     correlation_length=(2.0, 5.0, 5.0),
    ...     method="filtered",
    ...     rng=rng,
    ... )

    Physically motivated speckle via phasor model:

    >>> speckle = generate_speckle(
    ...     (32, 64, 64),
    ...     correlation_length=3.0,
    ...     method="phasor",
    ...     rng=rng,
    ... )

    Apply speckle to a volume (multiplicative):

    >>> volume = np.ones((32, 64, 64))
    >>> noisy_volume = volume * speckle
    """
    if rng is None:
        rng = np.random.default_rng()

    if num_looks < 1.0:
        raise ValueError(f"num_looks must be >= 1, got {num_looks}")

    # Normalize correlation_length to a 3-tuple or None
    corr: tuple[float, float, float] | None = None
    if correlation_length is not None:
        if isinstance(correlation_length, (int, float)):
            corr = (float(correlation_length), float(correlation_length),
                     float(correlation_length))
        else:
            corr = (float(correlation_length[0]), float(correlation_length[1]),
                     float(correlation_length[2]))

    if method == "direct":
        speckle = _generate_direct(shape, num_looks, model, rng)
    elif method == "phasor":
        speckle = _generate_phasor(shape, num_looks, corr, model, rng)
    elif method == "filtered":
        speckle = _generate_filtered(shape, num_looks, corr, model, rng)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    # Normalize to mean 1 if requested
    if normalize:
        mean_val = speckle.mean()
        if mean_val > 1e-10:
            speckle = speckle / mean_val

    return speckle.astype(np.float64)


# =============================================================================
# Generation methods
# =============================================================================


def _generate_direct(
    shape: tuple[int, int, int],
    num_looks: float,
    model: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate speckle via direct distribution sampling.

    Samples from Gamma(shape=L, scale=1/L) for intensity speckle,
    giving E[S] = 1 and Var[S] = 1/L.
    """
    # Gamma(shape=L, scale=1/L) -> mean=1, var=1/L
    intensity = rng.gamma(shape=num_looks, scale=1.0 / num_looks, size=shape)

    if model == "rayleigh":
        return np.sqrt(intensity)

    return intensity


def _generate_phasor(
    shape: tuple[int, int, int],
    num_looks: float,
    correlation_length: tuple[float, float, float] | None,
    model: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate speckle via the random phasor sum model.

    For L=1, generates a single complex Gaussian field and takes |·|^2.
    For L>1, averages L independent single-look fields. Integer L uses
    exact summation; fractional L rounds up and weights the final look
    to achieve the target contrast.
    """
    int_looks = int(np.ceil(num_looks))

    if int_looks == 1:
        intensity = _generate_phasor_field(shape, correlation_length, rng)
    else:
        total = np.zeros(shape, dtype=np.float64)
        for i in range(int_looks):
            single = _generate_phasor_field(shape, correlation_length, rng)
            if i == int_looks - 1 and num_looks != int_looks:
                # Fractional last look: weight to achieve target contrast
                frac_weight = num_looks - (int_looks - 1)
                total += frac_weight * single
            else:
                total += single
        intensity = total / num_looks

    if model == "rayleigh":
        return np.sqrt(intensity)

    return intensity


def _generate_filtered(
    shape: tuple[int, int, int],
    num_looks: float,
    correlation_length: tuple[float, float, float] | None,
    model: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate spatially correlated multi-look speckle via filtered
    Gaussian noise with probability integral transform.

    1. Generate white Gaussian noise
    2. Apply spatial correlation via FFT filtering
    3. Map to uniform via Gaussian CDF
    4. Map to Gamma via inverse CDF (ppf)
    """
    from scipy import stats

    # Generate white Gaussian noise
    white = rng.standard_normal(shape)

    # Apply spatial correlation if requested
    if correlation_length is not None:
        correlated = _apply_spatial_correlation(white, shape, correlation_length)
    else:
        correlated = white

    # Standardize to zero mean, unit variance for CDF transform
    mean = correlated.mean()
    std = correlated.std()
    if std > 1e-10:
        correlated = (correlated - mean) / std

    # Gaussian CDF -> uniform [0, 1]
    uniform = stats.norm.cdf(correlated)

    # Clip to avoid numerical issues at distribution tails
    uniform = np.clip(uniform, 1e-10, 1.0 - 1e-10)

    # Inverse CDF: uniform -> Gamma(shape=L, scale=1/L)
    # This gives mean=1, var=1/L
    intensity = stats.gamma.ppf(uniform, a=num_looks, scale=1.0 / num_looks)

    if model == "rayleigh":
        return np.sqrt(intensity)

    return intensity


# =============================================================================
# Class-based interface
# =============================================================================


class SpeckleNoise(NoiseGenerator):
    """
    Speckle noise generator for coherent imaging simulation.

    Generates multiplicative speckle noise fields S with E[S] = 1,
    modeling the granular interference pattern in coherent imaging
    systems (ultrasound, SAR, OCT, laser imaging).

    Parameters
    ----------
    num_looks : float
        Number of looks (L). Controls speckle contrast C = 1/sqrt(L).
    correlation_length : float or tuple[float, float, float], optional
        Speckle grain size in voxels. Scalar for isotropic, 3-tuple
        for anisotropic (sigma_d, sigma_h, sigma_w).
    model : {"gamma", "rayleigh"}
        Statistical model ("gamma" for intensity, "rayleigh" for
        amplitude speckle).
    method : {"direct", "phasor", "filtered"}
        Generation algorithm ("direct" for fast uncorrelated,
        "phasor" for physically motivated, "filtered" for correlated
        multi-look).
    normalize : bool
        If True, normalize output to mean = 1.

    Examples
    --------
    >>> import numpy as np
    >>> from py_deffields.noise import SpeckleNoise
    >>> generator = SpeckleNoise(num_looks=4.0, correlation_length=3.0)
    >>> rng = np.random.default_rng(42)
    >>> speckle = generator((32, 64, 64), rng=rng)
    >>> speckle.shape
    (32, 64, 64)
    """

    def __init__(
        self,
        num_looks: float = 1.0,
        correlation_length: float | tuple[float, float, float] | None = None,
        model: Literal["gamma", "rayleigh"] = "gamma",
        method: Literal["direct", "phasor", "filtered"] = "direct",
        normalize: bool = True,
    ) -> None:
        self.num_looks = num_looks
        self.correlation_length = correlation_length
        self.model = model
        self.method = method
        self.normalize = normalize

    @jaxcheck
    def generate(
        self,
        shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> Float[Array, "d h w"]:
        """
        Generate speckle noise field.

        Parameters
        ----------
        shape : tuple[int, int, int]
            Volume dimensions (D, H, W).
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        Float[Array, "d h w"]
            Multiplicative speckle noise field.
        """
        return generate_speckle(
            shape,
            self.num_looks,
            self.correlation_length,
            self.model,
            self.method,
            self.normalize,
            rng,
        )

    def __repr__(self) -> str:
        parts = [f"num_looks={self.num_looks}"]
        if self.correlation_length is not None:
            parts.append(f"correlation_length={self.correlation_length}")
        parts.append(f"model={self.model!r}")
        parts.append(f"method={self.method!r}")
        return f"SpeckleNoise({', '.join(parts)})"

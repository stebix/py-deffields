"""Deformation field generators."""

from .base import DeformationField
from .elastic import ElasticDeformation
from .bspline import BSplineDeformation
from .affine import AffineDeformation
from .analytic import (
    generate_star,
    generate_curve,
    generate_sphere,
    generate_overall,
    generate_static,
)

__all__ = [
    # Base class
    "DeformationField",
    # Class-based generators
    "ElasticDeformation",
    "BSplineDeformation",
    "AffineDeformation",
    # Functional analytic generators
    "generate_star",
    "generate_curve",
    "generate_sphere",
    "generate_overall",
    "generate_static",
]

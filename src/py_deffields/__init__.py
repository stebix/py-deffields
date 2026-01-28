"""
py-deffields: Volumetric Deformation Field Generation Package.

A NumPy/SciPy library for generating and applying volumetric deformation
fields for 3D image augmentation and registration tasks.

Example
-------
>>> import numpy as np
>>> from py_deffields import ElasticDeformation, warp
>>>
>>> volume = np.random.rand(64, 128, 128)
>>> elastic = ElasticDeformation(alpha=15.0, sigma=4.0)
>>> field = elastic.generate(volume.shape)
>>> warped = warp(volume, field)
"""

from .fields import (
    DeformationField,
    ElasticDeformation,
    BSplineDeformation,
    AffineDeformation,
)
from .warp import warp
from .compose import compose
from .jacobian import jacobian_determinant
from .grid import identity_grid, displacement_to_absolute

__all__ = [
    # Field generators
    "DeformationField",
    "ElasticDeformation",
    "BSplineDeformation",
    "AffineDeformation",
    # Operations
    "warp",
    "compose",
    "jacobian_determinant",
    # Grid utilities
    "identity_grid",
    "displacement_to_absolute",
]

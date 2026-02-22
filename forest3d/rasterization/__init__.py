"""Rasterization routines (hot path).

This package is reserved for creating synthetic rasters (e.g., CHMs) from validated
inputs and geospatial metadata.
"""

from forest3d.rasterization.analytic_dsm import make_analytic_dsm
from forest3d.rasterization.dsm import make_dsm

__all__ = [
    "make_analytic_dsm",
    "make_dsm",
]

"""Distance-field utilities and regular-grid sampling.

This package contains:
- `VoxelGridInterpolator`: a JAX-native, trilinear-only, SciPy-shaped interpolator
  specialized for uniform voxel grids.
- Distance-field construction helpers used to precompute distance fields for
  coregistration workflows.
"""

from forest3d.distance.field import DistanceField
from forest3d.distance.interpolator import VoxelGridInterpolator

__all__ = [
    "DistanceField",
    "VoxelGridInterpolator",
]

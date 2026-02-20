"""Distance-field generators for CHM coregistration workflows.

Cold path vs hot path
---------------------
- Cold path (NumPy/SciPy): build a voxel distance field from a CHM-like surface.
- Hot path (JAX): query the precomputed field using `VoxelGridInterpolator`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from forest3d.geospatial.coordinates import CoordinateSystem


@dataclass(frozen=True, slots=True)
class DistanceField:
    """A regular grid distance field with SciPy-style axis order.

    Attributes:
        x, y, z: 1D coordinate vectors (strictly increasing, uniform spacing).
        values: 3D array with shape (len(x), len(y), len(z)).
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    values: np.ndarray


def voxel_centers_from_coordinate_system(
    cs: CoordinateSystem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 1D coordinate vectors for voxel *cell centers* from a `CoordinateSystem`.

    `CoordinateSystem.ijk_to_xyz` uses raster-style indexing where `i` increases as
    `y` decreases. For SciPy-style grids we require `y` to be strictly increasing,
    so this function returns y-centers in ascending order (south->north).
    """
    # Compute axis center vectors directly using the same formulas as
    # `CoordinateSystem.ijk_to_xyz` for grid="voxel".
    j = np.arange(cs.nx, dtype=np.float32)
    i = np.arange(cs.ny, dtype=np.float32)
    k = np.arange(cs.nz, dtype=np.float32)

    x = np.float32(cs.xmin) + (j + np.float32(0.5)) * np.float32(cs.dx)
    y_desc = np.float32(cs.ymax) - (i + np.float32(0.5)) * np.float32(cs.dy)
    z = np.float32(cs.zmin) + (k + np.float32(0.5)) * np.float32(cs.dz)

    # Convert from raster-style y (descending) to SciPy-style axis order (ascending).
    y = y_desc[::-1].copy()
    return x, y, z


def distance_field_from_surface(
    surface: np.ndarray,
    *,
    cs: CoordinateSystem,
) -> DistanceField:
    """Build a 3D distance field from a surface raster via a distance transform.

    Args:
        surface: 2D array (ny, nx) aligned with `cs` in raster indexing (row-major).
            Values are interpreted as **absolute z** at the surface.
        cs: Coordinate system defining voxel geometry (nx, ny, nz and dx,dy,dz).

    Returns:
        DistanceField with SciPy-style axis order: `values` has shape (nx, ny, nz)
        and axis vectors (x, y, z) are strictly increasing.
    """
    surface = np.asarray(surface)
    if surface.shape != (cs.ny, cs.nx):
        raise ValueError(
            "surface must have shape (cs.ny, cs.nx)="
            f"{(cs.ny, cs.nx)}; got {surface.shape}."
        )
    if not np.isfinite(surface).all():
        raise ValueError("surface contains non-finite values (nan/inf).")

    # Cold-path validation: do not tolerate a surface outside the voxel grid.
    #
    # - x/y bounds are enforced by shape alignment with `cs.nx/cs.ny`.
    # - z bounds are enforced by value range: the surface must satisfy
    #   z ∈ [cs.zmin, cs.zmax) everywhere (half-open on the max edge).
    smin = float(surface.min())
    smax = float(surface.max())
    if smin < float(cs.zmin) or smax >= float(cs.zmax):
        raise ValueError(
            "surface z-values exceed voxel grid vertical bounds: "
            f"surface∈[{smin},{smax}], cs.z∈[{cs.zmin},{cs.zmax})."
        )

    x, y, z = voxel_centers_from_coordinate_system(cs)

    # Convert surface from raster orientation (ny, nx) with y descending to (nx, ny)
    # with y ascending.
    surface_y_asc = surface[::-1, :]
    surface_z = surface_y_asc.T  # (nx, ny)

    # Map surface z to the voxel bin index k (0..nz-1):
    # bins are [cs.zmin + k*dz, cs.zmin + (k+1)*dz).
    k_float = (surface_z - np.float32(cs.zmin)) / np.float32(cs.dz)
    k_idx = np.floor(k_float).astype(np.int32)
    if k_idx.min() < 0 or k_idx.max() >= cs.nz:
        # Should be impossible given the z-range check above, but keep a hard fail
        # to avoid silently clipping.
        raise ValueError("surface produces out-of-range voxel indices in z.")

    # Build a binary volume where surface voxels are zeros; others are ones.
    vol = np.ones((cs.nx, cs.ny, cs.nz), dtype=np.uint8)
    ix = np.arange(cs.nx)[:, None]
    iy = np.arange(cs.ny)[None, :]
    vol[ix, iy, k_idx] = 0

    # EDT distance in coordinate units using voxel spacings.
    dist = distance_transform_edt(vol, sampling=(cs.dx, cs.dy, cs.dz)).astype(
        np.float32
    )
    return DistanceField(x=x, y=y, z=z, values=dist)

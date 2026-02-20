"""JAX-native trilinear interpolation on a uniform voxel grid.

This module provides `VoxelGridInterpolator`, intended as a drop-in replacement for
SciPy's `RegularGridInterpolator` for the **3D, linear** case on **uniformly spaced**
grids.

Important differences vs SciPy
------------------------------
- Only supports 3D trilinear interpolation (`method="linear"`).
- Requires `points=(x, y, z)` to be strictly increasing and uniformly spaced.
- Out-of-bounds behavior is specialized for distance-field evaluation:

  For any query point `p`, let `p_clamped = clip(p, mins, maxs)`.
  The returned value is:

      v(p) = v_trilinear(p_clamped) + ||p - p_clamped||_2

  This makes extrapolation increase smoothly outside the grid by Euclidean distance
  in coordinate units.

The class is a JAX pytree, so it can be used inside `jax.jit`-compiled functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def _as_1d_points(name: str, p: ArrayLike) -> Array:
    arr = jnp.asarray(p)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got shape {arr.shape}.")
    if arr.size < 2:
        raise ValueError(f"{name} must have length >= 2; got {arr.size}.")
    return arr


def _assert_uniform_increasing(name: str, p: Array) -> tuple[Array, Array]:
    """Return (p0, dp) and validate strictly increasing + uniform spacing."""
    diffs = jnp.diff(p)
    if not bool(jnp.all(diffs > 0)):
        raise ValueError(f"{name} must be strictly increasing.")
    dp = diffs[0]
    # Uniform spacing check (Python-side). This is intentionally eager since it is
    # validation on static grid metadata.
    if not bool(jnp.allclose(diffs, dp)):
        raise ValueError(f"{name} must be uniformly spaced (constant d{name}).")
    return p[0], dp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class VoxelGridInterpolator:
    """Trilinear interpolator for uniform 3D grids.

    This interpolator is designed for **uniform** voxel grids, storing a compact
    representation of the grid (origin + spacing + sizes) rather than the full
    `x/y/z` vectors.

    Args:
        points: Tuple/list `(x, y, z)` of 1D coordinate vectors.
            - Each vector must be **1D**, **strictly increasing**, and **uniformly
              spaced** (constant `dx`, `dy`, `dz`).
        values: Grid values with shape `(len(x), len(y), len(z))`, i.e. axis order is
            `(x_index, y_index, z_index)`.

    Attributes:
        x0, y0, z0: Coordinate values at index 0 for each axis (i.e. `x[0]`, `y[0]`,
            `z[0]`).
        dx, dy, dz: Constant spacing between adjacent points along each axis:
            `dx = x[1] - x[0]` (and similarly for y/z).
        nx, ny, nz: Number of grid points along each axis: `nx=len(x)`, `ny=len(y)`,
            `nz=len(z)`.
        xmin, xmax: Minimum/maximum coordinate values along x, equal to `x[0]` and
            `x[-1]`.
        ymin, ymax: Minimum/maximum coordinate values along y, equal to `y[0]` and
            `y[-1]`.
        zmin, zmax: Minimum/maximum coordinate values along z, equal to `z[0]` and
            `z[-1]`.
        values: The underlying 3D grid values with shape `(nx, ny, nz)`.
            - `values[i, j, k]` corresponds to coordinate
              `(x0 + i*dx, y0 + j*dy, z0 + k*dz)`.
            - Query points are **clamped** to `[xmin,xmax]×[ymin,ymax]×[zmin,zmax]`
              for the trilinear part, and then the **Euclidean outside distance**
              `||p - clamp(p)||_2` is added to the interpolated value.
    """

    # Grid metadata
    x0: Array
    y0: Array
    z0: Array
    dx: Array
    dy: Array
    dz: Array
    nx: int
    ny: int
    nz: int

    # Domain bounds in coordinate units
    xmin: Array
    xmax: Array
    ymin: Array
    ymax: Array
    zmin: Array
    zmax: Array

    # Values
    values: Array

    def __init__(
        self,
        points: Sequence[ArrayLike],
        values: ArrayLike,
    ):
        if len(points) != 3:
            raise ValueError("points must be a 3-tuple (x, y, z).")
        x = _as_1d_points("x", points[0])
        y = _as_1d_points("y", points[1])
        z = _as_1d_points("z", points[2])

        x0, dx = _assert_uniform_increasing("x", x)
        y0, dy = _assert_uniform_increasing("y", y)
        z0, dz = _assert_uniform_increasing("z", z)

        vals = jnp.asarray(values)
        nx, ny, nz = int(x.size), int(y.size), int(z.size)
        if vals.shape != (nx, ny, nz):
            raise ValueError(
                "values must have shape (len(x), len(y), len(z)); "
                f"expected {(nx, ny, nz)}, got {vals.shape}."
            )

        object.__setattr__(self, "x0", jnp.asarray(x0))
        object.__setattr__(self, "y0", jnp.asarray(y0))
        object.__setattr__(self, "z0", jnp.asarray(z0))
        object.__setattr__(self, "dx", jnp.asarray(dx))
        object.__setattr__(self, "dy", jnp.asarray(dy))
        object.__setattr__(self, "dz", jnp.asarray(dz))
        object.__setattr__(self, "nx", nx)
        object.__setattr__(self, "ny", ny)
        object.__setattr__(self, "nz", nz)

        object.__setattr__(self, "xmin", x[0])
        object.__setattr__(self, "xmax", x[-1])
        object.__setattr__(self, "ymin", y[0])
        object.__setattr__(self, "ymax", y[-1])
        object.__setattr__(self, "zmin", z[0])
        object.__setattr__(self, "zmax", z[-1])

        object.__setattr__(self, "values", vals)

    def tree_flatten(self):
        children = (
            self.x0,
            self.y0,
            self.z0,
            self.dx,
            self.dy,
            self.dz,
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.zmin,
            self.zmax,
            self.values,
        )
        aux = dict(nx=self.nx, ny=self.ny, nz=self.nz)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            x0,
            y0,
            z0,
            dx,
            dy,
            dz,
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            values,
        ) = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "x0", x0)
        object.__setattr__(obj, "y0", y0)
        object.__setattr__(obj, "z0", z0)
        object.__setattr__(obj, "dx", dx)
        object.__setattr__(obj, "dy", dy)
        object.__setattr__(obj, "dz", dz)
        object.__setattr__(obj, "nx", int(aux_data["nx"]))
        object.__setattr__(obj, "ny", int(aux_data["ny"]))
        object.__setattr__(obj, "nz", int(aux_data["nz"]))
        object.__setattr__(obj, "xmin", xmin)
        object.__setattr__(obj, "xmax", xmax)
        object.__setattr__(obj, "ymin", ymin)
        object.__setattr__(obj, "ymax", ymax)
        object.__setattr__(obj, "zmin", zmin)
        object.__setattr__(obj, "zmax", zmax)
        object.__setattr__(obj, "values", values)
        return obj

    def __call__(self, xi: ArrayLike) -> Array:
        """Evaluate interpolated values at query points `xi`.

        Args:
            xi: Array-like of shape (..., 3) containing (x, y, z) query locations.

        Returns:
            Array of shape (...) containing interpolated values with distance-style
            out-of-bounds handling.
        """
        pts = jnp.asarray(xi)
        if pts.ndim < 1 or pts.shape[-1] != 3:
            raise ValueError(f"xi must have shape (..., 3); got {pts.shape}.")

        orig_shape = pts.shape[:-1]
        p = pts.reshape((-1, 3))
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]

        # Clamp to grid domain for interpolation part.
        x_clamped = jnp.clip(x, self.xmin, self.xmax)
        y_clamped = jnp.clip(y, self.ymin, self.ymax)
        z_clamped = jnp.clip(z, self.zmin, self.zmax)

        # Euclidean distance outside bounds (0 for in-bounds).
        oob_dx = x - x_clamped
        oob_dy = y - y_clamped
        oob_dz = z - z_clamped
        oob_dist = jnp.sqrt(oob_dx * oob_dx + oob_dy * oob_dy + oob_dz * oob_dz)

        clamped_value = _trilinear_uniform(
            self.values,
            x_clamped,
            y_clamped,
            z_clamped,
            x0=self.x0,
            y0=self.y0,
            z0=self.z0,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
        )
        out = (clamped_value + oob_dist).reshape(orig_shape)
        return out


def _trilinear_uniform(
    values: Array,
    x: Array,
    y: Array,
    z: Array,
    *,
    x0: Array,
    y0: Array,
    z0: Array,
    dx: Array,
    dy: Array,
    dz: Array,
    nx: int,
    ny: int,
    nz: int,
) -> Array:
    """Trilinear interpolation for uniform grids.

    Args:
        values: Array (nx, ny, nz).
        x, y, z: Arrays of query coordinates (already clamped to bounds).
    """
    # Continuous (fractional) grid indices in coordinate units.
    x_index = (x - x0) / dx
    y_index = (y - y0) / dy
    z_index = (z - z0) / dz

    # Bracketing neighbor indices for interpolation: (idx_lo, idx_hi=idx_lo+1).
    # Clip `idx_lo` so that `idx_hi` stays in-bounds; this also correctly handles
    # queries on the upper boundary where the weight becomes 1.0.
    x_idx_lo = jnp.clip(jnp.floor(x_index).astype(jnp.int32), 0, nx - 2)
    y_idx_lo = jnp.clip(jnp.floor(y_index).astype(jnp.int32), 0, ny - 2)
    z_idx_lo = jnp.clip(jnp.floor(z_index).astype(jnp.int32), 0, nz - 2)
    x_idx_hi = x_idx_lo + 1
    y_idx_hi = y_idx_lo + 1
    z_idx_hi = z_idx_lo + 1

    # Interpolation weights in [0, 1] along each axis.
    x_weight = x_index - x_idx_lo.astype(x_index.dtype)
    y_weight = y_index - y_idx_lo.astype(y_index.dtype)
    z_weight = z_index - z_idx_lo.astype(z_index.dtype)

    # 8 corner values for the surrounding voxel cell.
    #
    # Naming convention:
    # - `cXYZ` uses X/Y/Z ∈ {0,1} to indicate whether we take the low (`idx_lo`) or
    #   high (`idx_hi`) neighbor index along each axis.
    # - Example: `c101` means x uses high, y uses low, z uses high:
    #     values[x_idx_hi, y_idx_lo, z_idx_hi]
    c000 = values[x_idx_lo, y_idx_lo, z_idx_lo]
    c100 = values[x_idx_hi, y_idx_lo, z_idx_lo]
    c010 = values[x_idx_lo, y_idx_hi, z_idx_lo]
    c110 = values[x_idx_hi, y_idx_hi, z_idx_lo]
    c001 = values[x_idx_lo, y_idx_lo, z_idx_hi]
    c101 = values[x_idx_hi, y_idx_lo, z_idx_hi]
    c011 = values[x_idx_lo, y_idx_hi, z_idx_hi]
    c111 = values[x_idx_hi, y_idx_hi, z_idx_hi]

    # Interpolate in three stages: x -> y -> z.
    #
    # After interpolating along x, we have four edge values:
    # - `c00`: (y=lo, z=lo) edge, interpolated between c000 and c100
    # - `c10`: (y=hi, z=lo) edge, interpolated between c010 and c110
    # - `c01`: (y=lo, z=hi) edge, interpolated between c001 and c101
    # - `c11`: (y=hi, z=hi) edge, interpolated between c011 and c111
    #
    # Then interpolate those along y to get two face values:
    # - `c0`: z=lo face value (between c00 and c10)
    # - `c1`: z=hi face value (between c01 and c11)
    #
    # Finally interpolate along z between `c0` and `c1` to get the trilinear value.
    c00 = c000 * (1 - x_weight) + c100 * x_weight
    c10 = c010 * (1 - x_weight) + c110 * x_weight
    c01 = c001 * (1 - x_weight) + c101 * x_weight
    c11 = c011 * (1 - x_weight) + c111 * x_weight

    c0 = c00 * (1 - y_weight) + c10 * y_weight
    c1 = c01 * (1 - y_weight) + c11 * y_weight

    return c0 * (1 - z_weight) + c1 * z_weight

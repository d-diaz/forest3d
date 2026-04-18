"""Digital surface model (DSM) rasterization and queries (hot path).

This module provides JAX-friendly utilities for building a DSM raster from
simulated point clouds and for querying the resulting raster (e.g., codominance
classification).

Current definition: DSM is computed as per-pixel maximum absolute z of all
points that fall into the pixel (no ground subtraction).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from forest3d.geospatial.coordinates import CoordinateSystem
from forest3d.geospatial.enums import GridKind, IntegerMode


def make_synthetic_dsm(
    points: ArrayLike,
    *,
    cs: CoordinateSystem,
    fill_value: ArrayLike = jnp.nan,
) -> Array:
    """Rasterize points into a DSM raster (max-z per pixel).

    DSM is computed as per-pixel maximum z of all points whose (x, y) fall into the
    pixel. Bounds are treated as half-open on max edges: x∈[xmin,xmax), y∈[ymin,ymax).

    Args:
        points (ArrayLike): Point cloud coordinates `(x, y, z)` with shape (N, 3)
            or (B, N, 3).
        cs (CoordinateSystem): Coordinate system defining the raster grid geometry.
        fill_value (ArrayLike): Value used for pixels with no points (defaults to NaN).

    Returns:
        (Array): A `jax.Array` of shape (n_rows, n_cols) containing max-z per pixel.

    Raises:
        ValueError: If `points` does not have shape (N,3) or (B,N,3).
    """
    pts = jnp.asarray(points)

    if pts.ndim < 2 or pts.shape[-1] != 3:
        raise ValueError("points must have shape (N,3) or (B,N,3).")

    pts_flat = pts.reshape((-1, 3))
    x = pts_flat[:, 0]
    y = pts_flat[:, 1]
    z = pts_flat[:, 2]

    i, j, _k = cs.xyz_to_ijk(x, y, grid=GridKind.RASTER, integers=IntegerMode.FLOOR)
    row = i
    col = j

    finite = jnp.isfinite(x) & jnp.isfinite(y) & jnp.isfinite(z)
    in_bounds = finite & (col >= 0) & (col < cs.nx) & (row >= 0) & (row < cs.ny)

    flat = row * jnp.int32(cs.nx) + col
    flat_safe = jnp.where(in_bounds, flat, jnp.int32(0))
    z_safe = jnp.where(in_bounds, z, -jnp.inf)

    flat_raster = jnp.full(
        (cs.ny * cs.nx,),
        -jnp.inf,
        dtype=z.dtype,
    )
    flat_raster = flat_raster.at[flat_safe].max(z_safe)

    raster = flat_raster.reshape((cs.ny, cs.nx))
    fill_value = jnp.asarray(fill_value, dtype=raster.dtype)
    raster = jnp.where(jnp.isneginf(raster), fill_value, raster)
    return raster


def is_codominant_from_dsm(
    *,
    dsm: ArrayLike,
    i: ArrayLike,
    j: ArrayLike,
    z_apex: ArrayLike,
    epsilon: float = 1e-5,
) -> Array:
    """Test whether each tree's apex is codominant on a merged DSM.

    For each tree k, reads `dsm[i_k, j_k]` and compares to `z_apex_k`.
    A tree is codominant when its apex is at or above the height of the DSM
    at its apex cell.  The test is one-sided:

        dsm[i_k, j_k] - z_apex_k <= epsilon

    Trees overtopped by a neighbor crown (DSM exceeds apex by more than
    epsilon) are not codominant.

    Args:
        dsm: Merged analytic DSM array, shape (ny, nx).
        i: Per-tree row indices into the DSM, shape (B,). Typically from
            `CoordinateSystem.xyz_to_ijk(..., grid=RASTER, integers=FLOOR)`.
        j: Per-tree column indices into the DSM, shape (B,).
        z_apex: Model apex elevation per tree, shape (B,).
        epsilon: Absolute tolerance for the one-sided test (scalar, >= 0).
            Defaults to 1e-5 to absorb floating-point rounding.

    Returns:
        Boolean array of shape (B,). True where
        `dsm[i_k, j_k] - z_apex_k <= epsilon` and the index is in bounds.
        Trees whose (i, j) falls outside the DSM shape are marked False.
    """
    dsm = jnp.asarray(dsm)
    i = jnp.asarray(i)
    j = jnp.asarray(j)
    z_apex = jnp.asarray(z_apex)
    ny, nx = dsm.shape

    in_bounds = (i >= 0) & (i < ny) & (j >= 0) & (j < nx)

    i_safe = jnp.where(in_bounds, i, 0)
    j_safe = jnp.where(in_bounds, j, 0)
    dsm_at_apex = dsm[i_safe, j_safe]

    match = (dsm_at_apex - z_apex) <= epsilon
    return in_bounds & match

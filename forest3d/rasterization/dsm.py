"""Digital surface model (DSM) rasterization (hot path).

This module provides JAX-friendly rasterization utilities for turning simulated
point clouds (e.g., crown hull surface points) into a **DSM** raster.

Current definition
------------------
For this first pass, DSM is computed as: per-pixel maximum absolute `z` of all
points that fall into the pixel (no ground subtraction).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from forest3d.geospatial.coordinates import CoordinateSystem
from forest3d.geospatial.enums import GridKind, IntegerMode


def make_dsm(
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

"""Analytic DSM construction from crown parameters.

This module provides a JAX-friendly alternative to point-cloud rasterization:
instead of sampling crown surface points then taking max-z per pixel, we evaluate
the analytic upper-crown surface height at a representative location per pixel.

Separation of concerns:
- This module owns raster/window/pixel policy and max-scatter reduction.
- Crown surface math is delegated to `forest3d.geometry.*` (kernels + `CrownModel`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import jax.numpy as jnp
from jax import Array, lax

from forest3d.geometry.crown import CrownModel
from forest3d.geometry.evaluators.surface import (
    upper_surface_z_local,
)
from forest3d.geometry.params import CrownHullParams, CrownSurfaceParams
from forest3d.geometry.primitives import AzimuthalProfile
from forest3d.geospatial.coordinates import CoordinateSystem


class DsmPixelLocation(StrEnum):
    """Where to calculate the DSM value within each pixel."""

    CENTER = "center"
    RAY_ENTRY = "ray_entry"


def make_analytic_dsm(
    params: CrownHullParams | CrownSurfaceParams,
    *,
    cs: CoordinateSystem,
    fill_value: Array | float = jnp.nan,
    max_crown_radius: float,
    window_margin: float = 0.5,
    dsm_pixel_location: DsmPixelLocation = DsmPixelLocation.RAY_ENTRY,
    enforce_apex: bool = True,
) -> Array:
    """Analytically rasterize a batch of trees into a Digital Surface Model."""
    window_radius = float(max_crown_radius) + float(window_margin)
    if not (window_radius > 0.0) or not math.isfinite(window_radius):
        raise ValueError("max_crown_radius + window_margin must be finite and > 0.")

    half_window_i = int(math.ceil(window_radius / float(cs.dy)))
    half_window_j = int(math.ceil(window_radius / float(cs.dx)))
    window_ny = int(2 * half_window_i + 1)
    window_nx = int(2 * half_window_j + 1)

    dtype = jnp.asarray(params.top_height).dtype
    neg_inf = jnp.asarray(-jnp.inf, dtype=dtype)

    ny = int(cs.ny)
    nx = int(cs.nx)
    ny_i32 = jnp.int32(ny)
    nx_i32 = jnp.int32(nx)

    dsm_flat0 = jnp.full((ny * nx,), neg_inf, dtype=dtype)

    raster_geom = _RasterGeom(
        xmin=jnp.asarray(cs.xmin, dtype=dtype),
        ymax=jnp.asarray(cs.ymax, dtype=dtype),
        dx=jnp.asarray(cs.dx, dtype=dtype),
        dy=jnp.asarray(cs.dy, dtype=dtype),
        ny_i32=ny_i32,
        nx_i32=nx_i32,
    )

    window_di = (jnp.arange(window_ny, dtype=jnp.int32) - jnp.int32(half_window_i))[
        :, None
    ]
    window_dj = (jnp.arange(window_nx, dtype=jnp.int32) - jnp.int32(half_window_j))[
        None, :
    ]

    period = jnp.asarray(2.0 * math.pi, dtype=dtype)

    if dsm_pixel_location == DsmPixelLocation.CENTER:

        def query_xy_fn(
            *, apex_x: Array, apex_y: Array, win: _WindowGeometry
        ) -> tuple[Array, Array]:
            _ = apex_x, apex_y
            return _query_xy_center(win)

    else:

        def query_xy_fn(
            *, apex_x: Array, apex_y: Array, win: _WindowGeometry
        ) -> tuple[Array, Array]:
            return _query_xy_ray_entry(
                apex_x=apex_x,
                apex_y=apex_y,
                win=win,
                raster=raster_geom,
                dtype=dtype,
            )

    def scan_tree(dsm_flat: Array, tree: object) -> tuple[Array, Any]:
        tree_params = cast(CrownHullParams | CrownSurfaceParams, tree)
        model = CrownModel.from_params(tree_params, dtype=dtype)
        apex_x = (model.pose.tx + model.apex.x).astype(dtype)
        apex_y = (model.pose.ty + model.apex.y).astype(dtype)
        apex_z = (model.pose.tz + model.apex.z).astype(dtype)
        win = _WindowGeometry.from_apex_xy(
            apex_x=apex_x,
            apex_y=apex_y,
            raster=raster_geom,
            window_di=window_di,
            window_dj=window_dj,
            dtype=dtype,
        )

        query_x, query_y = query_xy_fn(apex_x=apex_x, apex_y=apex_y, win=win)
        r, theta = model.local_polar(query_x=query_x, query_y=query_y, dtype=dtype)
        az = AzimuthalProfile.from_model(
            model=model, theta=theta, period=period, dtype=dtype
        )
        z_local, inside = upper_surface_z_local(
            r=r,
            crown_edge_radius=az.crown_edge_radius,
            periph_z_local=az.periph_z_local,
            apex_z_local=model.apex.z,
            top_shape=az.top_shape,
        )
        z_global = z_local + model.pose.tz

        dsm_flat = _scatter_max_window(
            dsm_flat=dsm_flat,
            win=win,
            z_global=z_global.astype(dtype),
            inside=inside,
            nx_i32=nx_i32,
            neg_inf=neg_inf,
        )

        if enforce_apex:
            dsm_flat = _enforce_apex_height(
                dsm_flat=dsm_flat,
                apex_z=apex_z,
                win=win,
                ny_i32=ny_i32,
                nx_i32=nx_i32,
                neg_inf=neg_inf,
            )

        return dsm_flat, jnp.asarray(0, dtype=jnp.int32)

    scan_fn: Callable[[Array, object], tuple[Array, Any]] = scan_tree
    dsm_flat, _ = lax.scan(scan_fn, dsm_flat0, params)
    dsm = dsm_flat.reshape((ny, nx))
    fill_value = jnp.asarray(fill_value, dtype=dtype)
    return jnp.where(jnp.isneginf(dsm), fill_value, dsm)


@dataclass(frozen=True)
class _WindowGeometry:
    apex_i: Array
    apex_j: Array
    win_i: Array
    win_j: Array
    in_raster: Array
    pixel_center_x: Array
    pixel_center_y: Array

    @staticmethod
    def from_apex_xy(
        *,
        apex_x: Array,
        apex_y: Array,
        raster: _RasterGeom,
        window_di: Array,
        window_dj: Array,
        dtype: jnp.dtype,
    ) -> _WindowGeometry:
        apex_i = jnp.floor((raster.ymax - apex_y) / raster.dy).astype(jnp.int32)
        apex_j = jnp.floor((apex_x - raster.xmin) / raster.dx).astype(jnp.int32)

        win_i = apex_i + window_di
        win_j = apex_j + window_dj
        in_raster = (
            (win_i >= 0)
            & (win_i < raster.ny_i32)
            & (win_j >= 0)
            & (win_j < raster.nx_i32)
        )

        pixel_center_x = (
            raster.xmin
            + (win_j.astype(dtype) + jnp.asarray(0.5, dtype=dtype)) * raster.dx
        )
        pixel_center_y = (
            raster.ymax
            - (win_i.astype(dtype) + jnp.asarray(0.5, dtype=dtype)) * raster.dy
        )

        return _WindowGeometry(
            apex_i=apex_i,
            apex_j=apex_j,
            win_i=win_i,
            win_j=win_j,
            in_raster=in_raster,
            pixel_center_x=pixel_center_x,
            pixel_center_y=pixel_center_y,
        )


@dataclass(frozen=True)
class _RasterGeom:
    xmin: Array
    ymax: Array
    dx: Array
    dy: Array
    ny_i32: Array
    nx_i32: Array


def _query_xy_center(win: _WindowGeometry) -> tuple[Array, Array]:
    """Return pixel-center query coordinates for a window.

    Args:
        win: Per-tree window geometry with pixel-center coordinates.

    Returns:
        Tuple `(query_x, query_y)` for each pixel in the local window.
    """
    return win.pixel_center_x, win.pixel_center_y


def _query_xy_ray_entry(
    *,
    apex_x: Array,
    apex_y: Array,
    win: _WindowGeometry,
    raster: _RasterGeom,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Return query coordinates at ray-entry into each pixel cell.

    A ray is traced from the tree apex to each pixel center. The query point is
    clamped to the first intersection with the pixel bounds.

    Args:
        inv: Tree invariants for the current tree.
        win: Window geometry for local raster indices and pixel centers.
        raster: Global raster geometry and resolution.
        dtype: Floating dtype for intermediate calculations.

    Returns:
        Tuple `(query_x, query_y)` with the same shape as the window arrays.
    """
    pixel_x_min = raster.xmin + win.win_j.astype(dtype) * raster.dx
    pixel_x_max = pixel_x_min + raster.dx
    pixel_y_max = raster.ymax - win.win_i.astype(dtype) * raster.dy
    pixel_y_min = pixel_y_max - raster.dy

    ray_dx = win.pixel_center_x - apex_x
    ray_dy = win.pixel_center_y - apex_y

    t_x0 = (pixel_x_min - apex_x) / ray_dx
    t_x1 = (pixel_x_max - apex_x) / ray_dx
    t_y0 = (pixel_y_min - apex_y) / ray_dy
    t_y1 = (pixel_y_max - apex_y) / ray_dy

    tmin_x = jnp.minimum(t_x0, t_x1)
    tmax_x = jnp.maximum(t_x0, t_x1)
    tmin_y = jnp.minimum(t_y0, t_y1)
    tmax_y = jnp.maximum(t_y0, t_y1)

    t_neg_inf = jnp.asarray(-jnp.inf, dtype=dtype)
    t_pos_inf = jnp.asarray(jnp.inf, dtype=dtype)
    tmin_x = jnp.where(ray_dx == 0, t_neg_inf, tmin_x)
    tmax_x = jnp.where(ray_dx == 0, t_pos_inf, tmax_x)
    tmin_y = jnp.where(ray_dy == 0, t_neg_inf, tmin_y)
    tmax_y = jnp.where(ray_dy == 0, t_pos_inf, tmax_y)

    t_entry = jnp.maximum(tmin_x, tmin_y)
    t_entry = jnp.clip(t_entry, 0.0, 1.0)
    t_entry = jnp.where((ray_dx == 0) & (ray_dy == 0), 0.0, t_entry)

    query_x = apex_x + t_entry * ray_dx
    query_y = apex_y + t_entry * ray_dy
    return query_x, query_y


def _scatter_max_window(
    *,
    dsm_flat: Array,
    win: _WindowGeometry,
    z_global: Array,
    inside: Array,
    nx_i32: Array,
    neg_inf: Array,
) -> Array:
    """Scatter local window values into flattened DSM with max reduction.

    Args:
        dsm_flat: Flattened DSM accumulator.
        win: Window geometry with flattened index components.
        z_global: Candidate z values in global space for the window.
        inside: Mask indicating valid crown-support points.
        nx_i32: Raster width used for flattening `(i, j)` indices.
        neg_inf: Sentinel value used for masked-out cells.

    Returns:
        Updated flattened DSM where window cells are max-reduced.
    """
    z_safe = jnp.where(win.in_raster & inside, z_global, neg_inf)
    flat_idx = win.win_i * nx_i32 + win.win_j
    flat_idx_safe = jnp.where(win.in_raster, flat_idx, jnp.int32(0)).reshape((-1,))
    z_flat = z_safe.reshape((-1,))
    return dsm_flat.at[flat_idx_safe].max(z_flat)


def _enforce_apex_height(
    *,
    dsm_flat: Array,
    apex_z: Array,
    win: _WindowGeometry,
    ny_i32: Array,
    nx_i32: Array,
    neg_inf: Array,
) -> Array:
    """Ensure the apex pixel reaches at least the tree apex elevation.

    Args:
        dsm_flat: Flattened DSM accumulator.
        inv: Tree invariants containing apex global z.
        win: Window geometry containing apex raster indices.
        ny_i32: Raster height for bounds checking.
        nx_i32: Raster width for bounds checking and flattening.
        neg_inf: Sentinel value when apex is out of bounds.

    Returns:
        Updated flattened DSM after max-updating the apex pixel.
    """
    apex_in_bounds = (
        (win.apex_i >= 0)
        & (win.apex_i < ny_i32)
        & (win.apex_j >= 0)
        & (win.apex_j < nx_i32)
    )
    flat_apex = win.apex_i * nx_i32 + win.apex_j
    flat_apex_safe = jnp.where(apex_in_bounds, flat_apex, jnp.int32(0))
    z_apex_safe = jnp.where(apex_in_bounds, apex_z, neg_inf)
    return dsm_flat.at[flat_apex_safe].max(z_apex_safe)

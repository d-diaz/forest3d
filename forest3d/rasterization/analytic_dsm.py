"""Analytic DSM construction from crown parameters.

This module provides a JAX-friendly alternative to point-cloud rasterization:
instead of sampling crown surface points then taking max-z per pixel, we evaluate
the analytic upper-crown surface height at a representative location per pixel.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast

import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike

from forest3d.geometry.crown import (
    AzimuthalProfile,
    CrownAnchors,
    CrownApex,
    PeripheralPoints,
    TreePose,
)
from forest3d.geometry.params import CrownHullParams, CrownSurfaceParams
from forest3d.geospatial.coordinates import CoordinateSystem

_SurfaceLikeParams = CrownHullParams | CrownSurfaceParams


class _TreeLike(Protocol):
    stem_base: ArrayLike
    top_height: ArrayLike
    crown_ratio: ArrayLike
    lean_direction: ArrayLike
    lean_severity: ArrayLike
    crown_radii: ArrayLike
    crown_edge_heights: ArrayLike


class DsmPixelLocation(StrEnum):
    """Where to calculate the DSM value within each pixel."""

    CENTER = "center"
    RAY_ENTRY = "ray_entry"


def make_analytic_dsm(
    params: _SurfaceLikeParams,
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
            *,
            inv: _TreeInvariants,
            win: _WindowGeometry,
        ) -> tuple[Array, Array]:
            return _query_xy_center(win)

    else:

        def query_xy_fn(
            *,
            inv: _TreeInvariants,
            win: _WindowGeometry,
        ) -> tuple[Array, Array]:
            return _query_xy_ray_entry(
                inv=inv,
                win=win,
                raster=raster_geom,
                dtype=dtype,
            )

    def scan_tree(dsm_flat: Array, tree: object) -> tuple[Array, Any]:
        tree_params = cast(_SurfaceLikeParams, tree)
        inv = _TreeInvariants.from_tree(tree_params, dtype=dtype)
        win = _WindowGeometry.from_invariants(
            inv=inv,
            raster=raster_geom,
            window_di=window_di,
            window_dj=window_dj,
            dtype=dtype,
        )

        query_x, query_y = query_xy_fn(inv=inv, win=win)
        r, theta = _local_polar(
            query_x=query_x,
            query_y=query_y,
            inv=inv,
            dtype=dtype,
        )

        az = _azimuthal_profile(inv=inv, theta=theta, period=period, dtype=dtype)
        z_local, inside = _upper_crown_surface_z(r=r, inv=inv, az=az)
        z_global = z_local + inv.top_tz

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
                inv=inv,
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
class _TreeInvariants:
    top_tx: Array
    top_ty: Array
    top_tz: Array
    apex_x_local: Array
    apex_y_local: Array
    apex_z_local: Array
    apex_x: Array
    apex_y: Array
    apex_z: Array
    periph_x: Array
    periph_y: Array
    periph_z: Array
    top_shapes: Array

    @staticmethod
    def from_tree(tree: _SurfaceLikeParams, *, dtype: jnp.dtype) -> _TreeInvariants:
        pose = TreePose.from_tree(cast(_TreeLike, tree))
        if isinstance(tree, CrownHullParams):
            anchors = CrownAnchors.from_hull(tree)
        else:
            anchors = CrownAnchors.from_crown_surface_params(tree)

        top_tx, top_ty, top_tz = pose.tx, pose.ty, pose.tz

        apex = CrownApex.from_params(
            crown_radii=anchors.crown_radii,
            top_height=jnp.asarray(tree.top_height, dtype=dtype),
            crown_ratio=jnp.asarray(tree.crown_ratio, dtype=dtype),
        ).local
        apex_x_local, apex_y_local, apex_z_local = apex.x, apex.y, apex.z
        apex_x = (top_tx + apex_x_local).astype(dtype)
        apex_y = (top_ty + apex_y_local).astype(dtype)
        apex_z = (top_tz + apex_z_local).astype(dtype)

        periph = PeripheralPoints.from_params(
            crown_radii=anchors.crown_radii,
            crown_edge_heights=anchors.crown_edge_heights,
            top_height=jnp.asarray(tree.top_height, dtype=dtype),
            crown_ratio=jnp.asarray(tree.crown_ratio, dtype=dtype),
        )
        periph_x, periph_y, periph_z = periph.x, periph.y, periph.z

        return _TreeInvariants(
            top_tx=top_tx,
            top_ty=top_ty,
            top_tz=top_tz,
            apex_x_local=apex_x_local,
            apex_y_local=apex_y_local,
            apex_z_local=apex_z_local,
            apex_x=apex_x,
            apex_y=apex_y,
            apex_z=apex_z,
            periph_x=periph_x,
            periph_y=periph_y,
            periph_z=periph_z,
            top_shapes=anchors.top_shapes,
        )


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
    def from_invariants(
        *,
        inv: _TreeInvariants,
        raster: _RasterGeom,
        window_di: Array,
        window_dj: Array,
        dtype: jnp.dtype,
    ) -> _WindowGeometry:
        apex_i = jnp.floor((raster.ymax - inv.apex_y) / raster.dy).astype(jnp.int32)
        apex_j = jnp.floor((inv.apex_x - raster.xmin) / raster.dx).astype(jnp.int32)

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
    return win.pixel_center_x, win.pixel_center_y


def _query_xy_ray_entry(
    *,
    inv: _TreeInvariants,
    win: _WindowGeometry,
    raster: _RasterGeom,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    pixel_x_min = raster.xmin + win.win_j.astype(dtype) * raster.dx
    pixel_x_max = pixel_x_min + raster.dx
    pixel_y_max = raster.ymax - win.win_i.astype(dtype) * raster.dy
    pixel_y_min = pixel_y_max - raster.dy

    ray_dx = win.pixel_center_x - inv.apex_x
    ray_dy = win.pixel_center_y - inv.apex_y

    t_x0 = (pixel_x_min - inv.apex_x) / ray_dx
    t_x1 = (pixel_x_max - inv.apex_x) / ray_dx
    t_y0 = (pixel_y_min - inv.apex_y) / ray_dy
    t_y1 = (pixel_y_max - inv.apex_y) / ray_dy

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

    query_x = inv.apex_x + t_entry * ray_dx
    query_y = inv.apex_y + t_entry * ray_dy
    return query_x, query_y


def _local_polar(
    *,
    query_x: Array,
    query_y: Array,
    inv: _TreeInvariants,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    query_x_local = query_x - inv.top_tx
    query_y_local = query_y - inv.top_ty
    dx_local = query_x_local - inv.apex_x_local
    dy_local = query_y_local - inv.apex_y_local

    r = jnp.hypot(dy_local, dx_local)
    theta = jnp.arctan2(dy_local, dx_local).astype(dtype)
    return r, theta


def _upper_crown_surface_z(
    *,
    r: Array,
    inv: _TreeInvariants,
    az: AzimuthalProfile,
) -> tuple[Array, Array]:
    crown_edge_radius = az.crown_edge_radius
    periph_z_local = az.periph_z_local
    apex_z_local = inv.apex_z_local
    top_shape = az.top_shape

    crown_edge_radius_safe = jnp.where(crown_edge_radius == 0, 1.0, crown_edge_radius)
    top_shape_safe = jnp.where(top_shape == 0, 1.0, top_shape)

    r_frac = (r / crown_edge_radius_safe) ** top_shape_safe
    inner = jnp.maximum(1.0 - r_frac, 0.0)
    u = inner ** (1.0 / top_shape_safe)

    z_local = periph_z_local + (apex_z_local - periph_z_local) * u
    inside = (crown_edge_radius > 0) & (r <= crown_edge_radius)
    return z_local, inside


def _azimuthal_profile(
    *,
    inv: _TreeInvariants,
    theta: Array,
    period: Array,
    dtype: jnp.dtype,
) -> AzimuthalProfile:
    """Compute theta-dependent surface profile terms for analytic DSM."""
    periph_drop_from_apex = inv.apex_z_local - inv.periph_z
    periph_radius_from_apex = jnp.hypot(
        inv.periph_y - inv.apex_y_local, inv.periph_x - inv.apex_x_local
    )
    periph_theta = jnp.arctan2(
        inv.periph_y - inv.apex_y_local, inv.periph_x - inv.apex_x_local
    ).astype(dtype)

    crown_edge_radius = jnp.interp(
        theta, periph_theta, periph_radius_from_apex.astype(dtype), period=period
    )
    periph_drop = jnp.interp(
        theta, periph_theta, periph_drop_from_apex.astype(dtype), period=period
    )
    periph_z_local = inv.apex_z_local - periph_drop
    top_shape = jnp.interp(
        theta, periph_theta, inv.top_shapes.astype(dtype), period=period
    )
    return AzimuthalProfile(
        crown_edge_radius=crown_edge_radius,
        periph_z_local=periph_z_local,
        top_shape=top_shape,
    )


def _scatter_max_window(
    *,
    dsm_flat: Array,
    win: _WindowGeometry,
    z_global: Array,
    inside: Array,
    nx_i32: Array,
    neg_inf: Array,
) -> Array:
    z_safe = jnp.where(win.in_raster & inside, z_global, neg_inf)
    flat_idx = win.win_i * nx_i32 + win.win_j
    flat_idx_safe = jnp.where(win.in_raster, flat_idx, jnp.int32(0)).reshape((-1,))
    z_flat = z_safe.reshape((-1,))
    return dsm_flat.at[flat_idx_safe].max(z_flat)


def _enforce_apex_height(
    *,
    dsm_flat: Array,
    inv: _TreeInvariants,
    win: _WindowGeometry,
    ny_i32: Array,
    nx_i32: Array,
    neg_inf: Array,
) -> Array:
    apex_in_bounds = (
        (win.apex_i >= 0)
        & (win.apex_i < ny_i32)
        & (win.apex_j >= 0)
        & (win.apex_j < nx_i32)
    )
    flat_apex = win.apex_i * nx_i32 + win.apex_j
    flat_apex_safe = jnp.where(apex_in_bounds, flat_apex, jnp.int32(0))
    z_apex_safe = jnp.where(apex_in_bounds, inv.apex_z, neg_inf)
    return dsm_flat.at[flat_apex_safe].max(z_apex_safe)

"""Analytic DSM construction from crown parameters.

This module provides a JAX-friendly alternative to point-cloud rasterization:
instead of sampling crown surface points then taking max-z per pixel, we evaluate
the analytic upper-crown surface height at a representative location per pixel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import jax.numpy as jnp
from jax import Array, lax

from forest3d.geometry.crown_hull import (
    _get_hull_apex_and_base,
    _get_peripheral_points,
    _get_treetop_location,
)
from forest3d.geometry.params import CrownSurfaceParams, TreeHullParams
from forest3d.geospatial.coordinates import CoordinateSystem

_SurfaceLikeParams = TreeHullParams | CrownSurfaceParams


class DsmPixelLocation(StrEnum):
    """Where to calculate the DSM value within each pixel."""

    CENTER = "center"
    RAY_ENTRY = "ray_entry"


@dataclass(frozen=True)
class _TreeInvariants:
    """Per-tree arrays and derived invariants used by rasterization.

    These values are computed once per tree (inside the scan body) and then reused
    across the window evaluation. This keeps the hot loop readable and avoids
    recomputing crown/hull geometry terms at every pixel.

    Attributes:
        top_height (jax.Array): Tree top height above the stem base (scalar).
        crown_ratio (jax.Array): Crown ratio parameter used by hull geometry (scalar).
        crown_radii (jax.Array): Crown radii at the azimuthal anchors with shape (4,)
            (E,N,W,S) for a single tree.
        crown_edge_heights (jax.Array): Crown edge heights (fraction of top height) at
            the azimuthal anchors with shape (4,) (E,N,W,S).
        top_shape_anchors (jax.Array): Upper-crown shape coefficients at the azimuthal
            anchors with shape (4,) (E,N,W,S).
        top_tx (jax.Array): Global x translation applied by the treetop/lean model.
        top_ty (jax.Array): Global y translation applied by the treetop/lean model.
        top_tz (jax.Array): Global z translation applied by the treetop model.
        apex_x_local (jax.Array): Apex x coordinate in the *local* crown frame.
        apex_y_local (jax.Array): Apex y coordinate in the *local* crown frame.
        apex_z_local (jax.Array): Apex z coordinate in the *local* crown frame.
        apex_x (jax.Array): Apex x coordinate in the *global* frame.
        apex_y (jax.Array): Apex y coordinate in the *global* frame.
        apex_z (jax.Array): Apex z coordinate in the *global* frame (dtype-cast).
    """

    top_height: Array
    crown_ratio: Array
    crown_radii: Array
    crown_edge_heights: Array
    top_shape_anchors: Array
    top_tx: Array
    top_ty: Array
    top_tz: Array
    apex_x_local: Array
    apex_y_local: Array
    apex_z_local: Array
    apex_x: Array
    apex_y: Array
    apex_z: Array

    @staticmethod
    def from_tree(tree: _SurfaceLikeParams, *, dtype: jnp.dtype) -> _TreeInvariants:
        """Build invariants from a single-tree params container."""
        stem_base = jnp.asarray(tree.stem_base)
        top_height = jnp.asarray(tree.top_height)
        crown_ratio = jnp.asarray(tree.crown_ratio)
        lean_direction = jnp.asarray(tree.lean_direction)
        lean_severity = jnp.asarray(tree.lean_severity)
        crown_radii = jnp.asarray(tree.crown_radii)
        crown_edge_heights = jnp.asarray(tree.crown_edge_heights)
        top_shape_anchors = jnp.asarray(tree.crown_top_shapes)

        # Global translation used by the existing crown hull generator.
        top_tx, top_ty, top_tz = _get_treetop_location(
            stem_base, top_height, lean_direction, lean_severity
        )

        # Local-frame apex location from the crown model.
        hull_apex_local, _hull_base_local = _get_hull_apex_and_base(
            crown_radii, top_height, crown_ratio
        )
        apex_x_local, apex_y_local, apex_z_local = hull_apex_local

        # Global apex location.
        apex_x = apex_x_local + top_tx
        apex_y = apex_y_local + top_ty
        apex_z = (apex_z_local + top_tz).astype(dtype)

        return _TreeInvariants(
            top_height=top_height,
            crown_ratio=crown_ratio,
            crown_radii=crown_radii,
            crown_edge_heights=crown_edge_heights,
            top_shape_anchors=top_shape_anchors,
            top_tx=top_tx,
            top_ty=top_ty,
            top_tz=top_tz,
            apex_x_local=apex_x_local,
            apex_y_local=apex_y_local,
            apex_z_local=apex_z_local,
            apex_x=apex_x,
            apex_y=apex_y,
            apex_z=apex_z,
        )


@dataclass(frozen=True)
class _WindowGeometry:
    """Static apex-centered window geometry and pixel centers.

    This describes the fixed-size (static-shape) raster window evaluated per tree.
    The window is centered on the pixel containing the tree apex.

    Attributes:
        apex_i (jax.Array): Raster row index of the apex pixel (int32 scalar).
        apex_j (jax.Array): Raster column index of the apex pixel (int32 scalar).
        win_i (jax.Array): Raster row indices for the full window with shape
            (window_ny, window_nx) (int32).
        win_j (jax.Array): Raster column indices for the full window with shape
            (window_ny, window_nx) (int32).
        in_raster (jax.Array): Boolean mask with shape (window_ny, window_nx) marking
            which window pixels fall inside the raster bounds.
        pixel_center_x (jax.Array): Global x coordinate of each window pixel center
            with shape (window_ny, window_nx).
        pixel_center_y (jax.Array): Global y coordinate of each window pixel center
            with shape (window_ny, window_nx).
    """

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
        """Compute static window indices and pixel centers around the apex pixel."""
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
class _AzimuthalProfiles:
    """Theta-dependent crown edge radius/height and top-shape.

    These are evaluated on the per-pixel azimuth angle `theta` and feed directly
    into the analytic upper-crown surface evaluation.

    Attributes:
        crown_edge_radius (jax.Array): Radius from the apex axis to the crown edge
            along azimuth `theta`, shape (window_ny, window_nx).
        periph_z_local (jax.Array): Local-frame z coordinate of the peripheral crown
            edge along azimuth `theta`, shape (window_ny, window_nx).
        top_shape (jax.Array): Upper-crown shape coefficient along azimuth `theta`,
            shape (window_ny, window_nx).
    """

    crown_edge_radius: Array
    periph_z_local: Array
    top_shape: Array

    @staticmethod
    def from_invariants(
        *,
        inv: _TreeInvariants,
        theta: Array,
        period: Array,
        dtype: jnp.dtype,
    ) -> _AzimuthalProfiles:
        """Interpolate peripheral-line radius/height and top-shape as functions of
        theta.
        """
        peripheral_points = _get_peripheral_points(
            crown_radii=inv.crown_radii,
            crown_edge_heights=inv.crown_edge_heights,
            top_height=inv.top_height,
            crown_ratio=inv.crown_ratio,
        )
        periph_x = peripheral_points[:, 0]
        periph_y = peripheral_points[:, 1]
        periph_z = peripheral_points[:, 2]

        periph_drop_from_apex = inv.apex_z_local - periph_z
        periph_radius_from_apex = jnp.hypot(
            periph_y - inv.apex_y_local, periph_x - inv.apex_x_local
        )
        periph_theta = jnp.arctan2(
            periph_y - inv.apex_y_local, periph_x - inv.apex_x_local
        ).astype(dtype)

        crown_edge_radius = jnp.interp(
            theta, periph_theta, periph_radius_from_apex.astype(dtype), period=period
        )
        periph_drop = jnp.interp(
            theta, periph_theta, periph_drop_from_apex.astype(dtype), period=period
        )
        periph_z_local = inv.apex_z_local - periph_drop
        top_shape = jnp.interp(
            theta, periph_theta, inv.top_shape_anchors.astype(dtype), period=period
        )
        return _AzimuthalProfiles(
            crown_edge_radius=crown_edge_radius,
            periph_z_local=periph_z_local,
            top_shape=top_shape,
        )


@dataclass(frozen=True)
class _RasterGeom:
    """Raster geometry constants (as JAX arrays).

    This bundles `CoordinateSystem` fields into dtype-cast JAX arrays so the scan
    body can remain “arrays-only” and avoid repeatedly calling `jnp.asarray(...)`.

    Attributes:
        xmin (jax.Array): Raster x minimum (origin) in global coordinates.
        ymax (jax.Array): Raster y maximum (origin) in global coordinates.
        dx (jax.Array): Raster pixel width in x.
        dy (jax.Array): Raster pixel height in y.
        ny_i32 (jax.Array): Number of raster rows (int32 scalar).
        nx_i32 (jax.Array): Number of raster columns (int32 scalar).
    """

    xmin: Array
    ymax: Array
    dx: Array
    dy: Array
    ny_i32: Array
    nx_i32: Array


def _query_xy_center(
    win: _WindowGeometry,
) -> tuple[Array, Array]:
    """Sample at pixel centers."""
    return win.pixel_center_x, win.pixel_center_y


def _query_xy_ray_entry(
    *,
    inv: _TreeInvariants,
    win: _WindowGeometry,
    raster: _RasterGeom,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Sample at the entry point of the apex→center segment into the pixel box."""
    pixel_x_min = raster.xmin + win.win_j.astype(dtype) * raster.dx
    pixel_x_max = pixel_x_min + raster.dx
    pixel_y_max = raster.ymax - win.win_i.astype(dtype) * raster.dy
    pixel_y_min = pixel_y_max - raster.dy

    # Segment direction from apex→pixel-center.
    ray_dx = win.pixel_center_x - inv.apex_x
    ray_dy = win.pixel_center_y - inv.apex_y

    # Slab intersection in parameter t, where P(t)=apex+t*(center-apex), t∈[0,1].
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
    """Convert global query XY to (r, theta) about the local-frame apex axis."""
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
    az: _AzimuthalProfiles,
) -> tuple[Array, Array]:
    """Evaluate analytic upper-crown surface and inside-mask at radius r."""
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


def _scatter_max_window(
    *,
    dsm_flat: Array,
    win: _WindowGeometry,
    z_global: Array,
    inside: Array,
    nx_i32: Array,
    neg_inf: Array,
) -> Array:
    """Max-update DSM accumulator at window pixels."""
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
    """Ensure the pixel containing the apex is at least the apex height."""
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
    """Analytically rasterize a batch of trees into a Digital Surface Model.

    Args:
        params (TreeHullParams | CrownSurfaceParams): Batched tree parameters with
            leading dimension `B`.
        cs (CoordinateSystem): Coordinate system defining raster geometry
            (typically closed-over under `jit`).
        fill_value (jax.Array | float): Value for pixels with no canopy
            contribution.
        max_crown_radius (float): Maximum crown radius (in the same units as
            `cs`) expected in `params`. Determines the static evaluation window size.
        window_margin (float): Extra margin (in the same units as `cs`) added to
            the window radius.
        dsm_pixel_location (DsmPixelLocation): Where to calculate surface height within
            each pixel. Valid options:
            - `CENTER`: at the pixel center.
            - `RAY_ENTRY`: at the first intersection of the apex→center ray
              with the pixel footprint.
        enforce_apex (bool): If True, ensure the pixel containing the apex is at
            least the apex height (max-update).

    Returns:
        dsm (jax.Array): DSM raster of shape `(cs.ny, cs.nx)`.
    """
    # Compute static window shape (in cells). These must be treated as static under jit.
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

    # Flattened DSM for efficient scatter-max.
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

    def scan_tree(dsm_flat: Array, tree: _SurfaceLikeParams) -> tuple[Array, Array]:
        """Accumulate a single tree's canopy surface into the DSM buffer.

        This is the per-tree loop body used by `jax.lax.scan`. It evaluates the
        analytic upper-crown surface within a static apex-centered window and
        updates `dsm_flat` via scatter-max.

        Args:
            dsm_flat (jax.Array): Flattened DSM accumulator of shape `(cs.ny * cs.nx,)`.
                Entries start at `-inf` and are max-updated in-place (functionally)
                for each tree.
            tree (TreeHullParams | CrownSurfaceParams): Single-tree parameters (a
                single slice of the batched `params` PyTree).

        Returns:
            dsm_flat (jax.Array): Updated flattened DSM accumulator.
            aux (jax.Array): Placeholder scan output (unused).
        """
        inv = _TreeInvariants.from_tree(tree, dtype=dtype)
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

        az = _AzimuthalProfiles.from_invariants(
            inv=inv,
            theta=theta,
            period=period,
            dtype=dtype,
        )

        z_local, inside = _upper_crown_surface_z(
            r=r,
            inv=inv,
            az=az,
        )
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

    dsm_flat, _ = lax.scan(scan_tree, dsm_flat0, params)
    dsm = dsm_flat.reshape((ny, nx))
    fill_value = jnp.asarray(fill_value, dtype=dtype)

    return jnp.where(jnp.isneginf(dsm), fill_value, dsm)

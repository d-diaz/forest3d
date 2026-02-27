"""Pure kernels for crown-surface evaluation.

This module contains crown-domain evaluation kernels that operate on arrays and
encode the crown surface mathematics (local frames, azimuthal profiles, and
surface height evaluation).

Non-goals:
- No raster/window/pixel policy logic (belongs to `forest3d.rasterization.*`).
- No point-cloud sampling loops (belongs to `forest3d.geometry.evaluators.points`).

Frame conventions:
- Inputs ending with `_local` are in the crown-local frame.
- `pose_tx`/`pose_ty` translate from local → global in XY.
- Azimuth angles are in radians.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from forest3d.geometry.kernels import interp_periodic

if TYPE_CHECKING:
    from forest3d.geometry.primitives import PeripheralRelative


def local_polar_from_pose_and_apex(
    *,
    query_x: Array,
    query_y: Array,
    pose_tx: Array,
    pose_ty: Array,
    apex_x_local: Array,
    apex_y_local: Array,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Convert global query coordinates to local polar coordinates.

    Args:
        query_x: Query x coordinates in global space.
        query_y: Query y coordinates in global space.
        pose_tx: Local-to-global translation in x.
        pose_ty: Local-to-global translation in y.
        apex_x_local: Apex x coordinate in local space.
        apex_y_local: Apex y coordinate in local space.
        dtype: Output dtype for azimuth values.

    Returns:
        Tuple `(r, theta)` in the local crown frame about the apex axis.
    """
    query_x_local = query_x - pose_tx
    query_y_local = query_y - pose_ty
    dx_local = query_x_local - apex_x_local
    dy_local = query_y_local - apex_y_local
    r = jnp.hypot(dy_local, dx_local)
    theta = jnp.arctan2(dy_local, dx_local).astype(dtype)
    return r, theta


def azimuthal_profile_from_relative(
    *,
    theta: Array,
    relative: PeripheralRelative,
    apex_z_local: Array,
    top_shapes: Array,
    period: Array,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array]:
    """Compute upper-surface azimuthal profile components from relative periphery terms.

    Args:
        theta: Query azimuth angles (local frame, radians).
        relative: Peripheral points expressed relative to the crown apex.
        apex_z_local: Apex z in local frame.
        top_shapes: Top-shape exponent values at peripheral anchors.
        period: Interpolation period (typically `2*pi`).
        dtype: Floating dtype for interpolation outputs.

    Returns:
        Tuple `(crown_edge_radius, periph_z_local, top_shape)` sampled at `theta`.
    """
    crown_edge_radius = interp_periodic(
        theta=theta,
        key_theta=relative.thetas,
        values=relative.radii_xy.astype(dtype),
        period=period,
    )
    periph_drop = interp_periodic(
        theta=theta,
        key_theta=relative.thetas,
        values=relative.drop_from_ref_z.astype(dtype),
        period=period,
    )
    periph_z_local = apex_z_local - periph_drop
    top_shape = interp_periodic(
        theta=theta,
        key_theta=relative.thetas,
        values=top_shapes.astype(dtype),
        period=period,
    )
    return crown_edge_radius, periph_z_local, top_shape


def _azimuthal_profile_from_peripheral(
    *,
    theta: Array,
    periph_x: Array,
    periph_y: Array,
    periph_z: Array,
    apex_x_local: Array,
    apex_y_local: Array,
    apex_z_local: Array,
    top_shapes: Array,
    period: Array,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array]:
    """Compute upper-surface azimuthal profile components.

    Args:
        theta: Query azimuth angles.
        periph_x: Peripheral x coordinates.
        periph_y: Peripheral y coordinates.
        periph_z: Peripheral z coordinates.
        apex_x_local: Apex x in local frame.
        apex_y_local: Apex y in local frame.
        apex_z_local: Apex z in local frame.
        top_shapes: Top-shape values at peripheral anchors.
        period: Interpolation period (typically `2*pi`).
        dtype: Floating dtype for interpolation outputs.

    Returns:
        Tuple `(crown_edge_radius, periph_z_local, top_shape)` sampled at `theta`.
    """
    periph_drop_from_apex = apex_z_local - periph_z
    periph_radius_from_apex = jnp.hypot(
        periph_y - apex_y_local, periph_x - apex_x_local
    )
    periph_theta = jnp.arctan2(periph_y - apex_y_local, periph_x - apex_x_local).astype(
        dtype
    )

    crown_edge_radius = interp_periodic(
        theta=theta,
        key_theta=periph_theta,
        values=periph_radius_from_apex.astype(dtype),
        period=period,
    )
    periph_drop = interp_periodic(
        theta=theta,
        key_theta=periph_theta,
        values=periph_drop_from_apex.astype(dtype),
        period=period,
    )
    periph_z_local = apex_z_local - periph_drop
    top_shape = interp_periodic(
        theta=theta,
        key_theta=periph_theta,
        values=top_shapes.astype(dtype),
        period=period,
    )
    return crown_edge_radius, periph_z_local, top_shape


def upper_surface_z_local(
    *,
    r: Array,
    crown_edge_radius: Array,
    periph_z_local: Array,
    apex_z_local: Array,
    top_shape: Array,
) -> tuple[Array, Array]:
    """Evaluate local upper-crown z and inside-mask.

    Args:
        r: Local radial distances from the apex axis.
        crown_edge_radius: Interpolated crown-edge radii.
        periph_z_local: Interpolated peripheral z values.
        apex_z_local: Apex z value in local frame.
        top_shape: Interpolated top-shape exponent.

    Returns:
        Tuple `(z_local, inside)` where `inside` marks supported points.
    """
    crown_edge_radius_safe = jnp.where(crown_edge_radius == 0, 1.0, crown_edge_radius)
    top_shape_safe = jnp.where(top_shape == 0, 1.0, top_shape)

    r_frac = (r / crown_edge_radius_safe) ** top_shape_safe
    inner = jnp.maximum(1.0 - r_frac, 0.0)
    u = inner ** (1.0 / top_shape_safe)

    z_local = periph_z_local + (apex_z_local - periph_z_local) * u
    inside = (crown_edge_radius > 0) & (r <= crown_edge_radius)
    return z_local, inside

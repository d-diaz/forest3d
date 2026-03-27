"""Generic math kernels used by geometry and evaluators.

This module hosts small, JAX-friendly helper kernels that are:
- pure (array-in/array-out),
- domain-agnostic (no crown-specific semantics), and
- reusable across multiple geometry evaluators.

Non-goals:
- No crown model or primitive assembly.
- No raster/window/pixel policy logic.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def interp_periodic(
    *,
    theta: Array,
    key_theta: Array,
    values: Array,
    period: Array,
) -> Array:
    """Periodic 1-D interpolation over an angular domain.

    Args:
        theta (Array): Query coordinates (often azimuth angles in radians).
        key_theta (Array): Source coordinates for known samples.
        values (Array): Sample values aligned with `key_theta`.
        period (Array): Interpolation period (typically `2*pi`).

    Returns:
        Interpolated values at `theta` (Array).
    """
    return jnp.interp(theta, key_theta, values, period=period)


def theta_z_grid(
    *, num_theta: int, num_z: int, base_z: Array, apex_z: Array
) -> tuple[Array, Array]:
    """Build meshgrid arrays for (theta, z) sampling.

    Args:
        num_theta (int): Number of azimuth samples.
        num_z (int): Number of vertical samples.
        base_z (Array): Lower z bound.
        apex_z (Array): Upper z bound.

    Returns:
        grid_theta, grid_z (Arrays) with shape (num_z, num_theta)
    """
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(num_theta))
    z = jnp.linspace(base_z, apex_z, int(num_z))
    grid_theta, grid_z = jnp.meshgrid(theta, z)
    return grid_theta, grid_z


def rotate_xy(*, x: ArrayLike, y: ArrayLike, theta: ArrayLike) -> tuple[Array, Array]:
    """Rotate 2D coordinates counterclockwise about the origin (+z out).

    Standard right-handed rotation in map/plan view:
        x' = cos(θ) x - sin(θ) y
        y' = sin(θ) x + cos(θ) y

    Args:
        x: X coordinates; any shape broadcastable with *y* and *theta*.
        y: Y coordinates; same broadcast contract as *x*.
        theta: Rotation angle in radians (scalar or broadcastable).

    Returns:
        x', y' (Arrays) of rotated coordinates.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    theta = jnp.asarray(theta)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    return cos_theta * x - sin_theta * y, sin_theta * x + cos_theta * y


def stem_xy_world(
    *,
    center_x: ArrayLike,
    center_y: ArrayLike,
    local_x: ArrayLike,
    local_y: ArrayLike,
    offset_x: ArrayLike,
    offset_y: ArrayLike,
    theta: ArrayLike,
) -> tuple[Array, Array]:
    """Compute world-space stem XY from plot center, local offsets, and rotation.

    Builds per-tree layout vectors `u = local + offset`, rotates them by
    `theta` about the origin, then translates by the plot center:

        u_x = local_x + offset_x
        u_y = local_y + offset_y
        stem_x = center_x + rotate_xy(u_x, u_y, theta).x
        stem_y = center_y + rotate_xy(u_x, u_y, theta).y

    When `theta = 0` this reduces to `center + local + offset`, matching
    the existing formula in `distance_field_energy`.

    Args:
        center_x: Plot-center x (scalar or broadcastable).
        center_y: Plot-center y (scalar or broadcastable).
        local_x: Per-tree local x offset from plot center, shape (B,).
        local_y: Per-tree local y offset from plot center, shape (B,).
        offset_x: Per-tree perturbation x (latent), shape (B,).
        offset_y: Per-tree perturbation y (latent), shape (B,).
        theta: Global rotation angle in radians (scalar or broadcastable).

    Returns:
        stem_x, stem_y (Arrays) of world-space stem coordinates.

    Note:
        Internal angle convention: 0 rad = +x (east), pi/2 = +y (north), CCW
        positive.  This is the same frame used by `rotate_xy` and consistent
        with `BearingPrior.from_degrees` in `forest3d.simulation.priors`
        (which maps FIA degrees-from-north via `deg2rad(90 - bearing_deg)`).
        For grid search, theta represents a global plot rotation about the
        plot center in this frame.
    """
    u_x = jnp.asarray(local_x) + jnp.asarray(offset_x)
    u_y = jnp.asarray(local_y) + jnp.asarray(offset_y)
    rot_x, rot_y = rotate_xy(x=u_x, y=u_y, theta=theta)
    return jnp.asarray(center_x) + rot_x, jnp.asarray(center_y) + rot_y


def polar_to_xy(
    *,
    radii: ArrayLike,
    theta: ArrayLike,
    center_x: ArrayLike,
    center_y: ArrayLike,
) -> tuple[Array, Array]:
    """Convert polar coordinates to cartesian about a center.

    Args:
        radii: Radial distances.
        theta: Azimuth angles in radians.
        center_x: Cartesian center x.
        center_y: Cartesian center y.

    Returns:
        x, y (Arrays) of cartesian coordinates.
    """
    theta = jnp.asarray(theta)
    radii = jnp.asarray(radii)
    center_x = jnp.asarray(center_x)
    center_y = jnp.asarray(center_y)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    return radii * cos_theta + center_x, radii * sin_theta + center_y

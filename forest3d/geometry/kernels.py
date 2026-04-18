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

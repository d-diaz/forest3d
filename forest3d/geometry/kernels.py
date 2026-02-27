"""Generic math kernels used by geometry and evaluators.

This module hosts small, JAX-friendly helper kernels that are:
- pure (array-in/array-out),
- domain-agnostic (no crown-specific semantics), and
- reusable across multiple geometry evaluators.

Non-goals:
- No crown model or primitive assembly.
- No raster/window/pixel policy logic.

API note:
- Only `interp_periodic` is intended as a stable public helper.
- Underscored helpers are internal implementation details for evaluator modules.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def interp_periodic(
    *,
    theta: Array,
    key_theta: Array,
    values: Array,
    period: Array,
) -> Array:
    """Periodic 1-D interpolation over an angular domain.

    Args:
        theta: Query coordinates (often azimuth angles in radians).
        key_theta: Source coordinates for known samples.
        values: Sample values aligned with `key_theta`.
        period: Interpolation period (typically `2*pi`).

    Returns:
        Interpolated values at `theta`.
    """
    return jnp.interp(theta, key_theta, values, period=period)


def _theta_z_grid(
    *, num_theta: int, num_z: int, base_z: Array, apex_z: Array
) -> tuple[Array, Array]:
    """Build meshgrid arrays for `(theta, z)` sampling.

    Args:
        num_theta: Number of azimuth samples.
        num_z: Number of vertical samples.
        base_z: Lower z bound.
        apex_z: Upper z bound.

    Returns:
        Tuple `(grid_thetas, grid_zs)` with shape `(num_z, num_theta)`.
    """
    thetas = jnp.linspace(0.0, 2.0 * jnp.pi, int(num_theta))
    zs = jnp.linspace(base_z, apex_z, int(num_z))
    grid_thetas, grid_zs = jnp.meshgrid(thetas, zs)
    return grid_thetas, grid_zs


def _polar_to_xy(
    *,
    radii: Array,
    cos_theta: Array,
    sin_theta: Array,
    center_x: Array,
    center_y: Array,
) -> tuple[Array, Array]:
    """Convert polar components to cartesian coordinates about a center.

    Args:
        radii: Radial distances.
        cos_theta: Cosine of azimuth angles.
        sin_theta: Sine of azimuth angles.
        center_x: Cartesian center x.
        center_y: Cartesian center y.

    Returns:
        Tuple `(x, y)` coordinates.
    """
    return radii * cos_theta + center_x, radii * sin_theta + center_y

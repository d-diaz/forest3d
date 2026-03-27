"""Crown hull point-sampling evaluators.

This module generates point clouds from `CrownModel` geometry.

Non-goals:
- No raster/window/pixel policy logic (belongs to `forest3d.rasterization.*`).
- No analytic raster evaluation (belongs to `forest3d.rasterization.analytic_dsm`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from forest3d.geometry.crown import CrownModel
from forest3d.geometry.kernels import interp_periodic, polar_to_xy, theta_z_grid
from forest3d.geometry.params import CrownHullParams
from forest3d.geometry.primitives import peripheral_relative


def sample_hull_points(
    *,
    model: CrownModel,
    num_theta: int = 32,
    num_z: int = 50,
) -> Array:
    """Sample surface points on the full crown hull.

    Args:
        model: Crown model with local geometry and global pose.
        num_theta: Number of azimuth samples around the crown.
        num_z: Number of vertical samples between base and apex.

    Returns:
        Global-coordinate point cloud with shape `(num_z * num_theta, 3)` and
        columns `(x, y, z)`.

    The `(theta, z)` sampling grid is built with `meshgrid(thetas, zs)`, so when
    flattened in row-major (C) order the output is grouped by z-level: theta
    varies fastest (all thetas for a fixed z), then z.
    """
    anchors = model.anchors
    periph = model.peripheral

    grid_thetas, grid_zs = theta_z_grid(
        num_theta=num_theta,
        num_z=num_z,
        base_z=model.base.z,
        apex_z=model.apex.z,
    )

    top_periph = peripheral_relative(
        periph=periph, ref=model.apex, dtype=grid_thetas.dtype
    )
    period = jnp.asarray(2.0 * jnp.pi, dtype=grid_thetas.dtype)
    apex_periph_line_radii = interp_periodic(
        theta=grid_thetas,
        key_theta=top_periph.thetas,
        values=top_periph.radii_xy,
        period=period,
    )
    periph_height_interp = interp_periodic(
        theta=grid_thetas,
        key_theta=top_periph.thetas,
        values=top_periph.drop_from_ref_z,
        period=period,
    )
    top_shapes_interp = interp_periodic(
        theta=grid_thetas,
        key_theta=top_periph.thetas,
        values=anchors.top_shapes,
        period=period,
    )

    cos_grid_thetas = jnp.cos(grid_thetas)
    sin_grid_thetas = jnp.sin(grid_thetas)
    periph_line_xs, periph_line_ys = polar_to_xy(
        radii=apex_periph_line_radii,
        cos_theta=cos_grid_thetas,
        sin_theta=sin_grid_thetas,
        center_x=model.apex.x,
        center_y=model.apex.y,
    )
    periph_line_zs = model.apex.z - periph_height_interp

    top_hull_radii = _hull_radii_from_profile(
        delta_z=jnp.maximum(grid_zs - periph_line_zs, 0.0),
        denom_z=model.apex.z - periph_line_zs,
        edge_radii=apex_periph_line_radii,
        shape=top_shapes_interp,
    )

    base_periph = peripheral_relative(
        periph=periph, ref=model.base, dtype=grid_thetas.dtype
    )
    bottom_periph_line_thetas = jnp.arctan2(
        periph_line_ys - model.base.y,
        periph_line_xs - model.base.x,
    )
    base_periph_line_radii = jnp.hypot(
        periph_line_xs - model.base.x,
        periph_line_ys - model.base.y,
    )
    bottom_shapes_interp = interp_periodic(
        theta=bottom_periph_line_thetas,
        key_theta=base_periph.thetas,
        values=anchors.bottom_shapes,
        period=period,
    )

    bottom_hull_radii = _hull_radii_from_profile(
        delta_z=jnp.maximum(periph_line_zs - grid_zs, 0.0),
        denom_z=periph_line_zs - model.base.z,
        edge_radii=base_periph_line_radii,
        shape=bottom_shapes_interp,
    )

    grid_top = grid_zs >= periph_line_zs
    hull_radii = jnp.where(grid_top, top_hull_radii, bottom_hull_radii)
    cos_bottom_thetas = jnp.cos(bottom_periph_line_thetas)
    sin_bottom_thetas = jnp.sin(bottom_periph_line_thetas)
    top_grid_xs, top_grid_ys = polar_to_xy(
        radii=hull_radii,
        cos_theta=cos_grid_thetas,
        sin_theta=sin_grid_thetas,
        center_x=model.apex.x,
        center_y=model.apex.y,
    )
    bottom_grid_xs, bottom_grid_ys = polar_to_xy(
        radii=hull_radii,
        cos_theta=cos_bottom_thetas,
        sin_theta=sin_bottom_thetas,
        center_x=model.base.x,
        center_y=model.base.y,
    )
    grid_xs = jnp.where(grid_top, top_grid_xs, bottom_grid_xs)
    grid_ys = jnp.where(grid_top, top_grid_ys, bottom_grid_ys)

    crown_xs = grid_xs + model.pose.tx
    crown_ys = grid_ys + model.pose.ty
    crown_zs = grid_zs + model.pose.tz
    return jnp.column_stack((crown_xs.ravel(), crown_ys.ravel(), crown_zs.ravel()))


def make_crown_hull(
    params: CrownHullParams, *, num_theta: int = 32, num_z: int = 50
) -> Array:
    """Generate crown hull points for a single tree.

    Args:
        params: Single-tree crown hull parameters.
        num_theta: Number of azimuth samples around the crown.
        num_z: Number of vertical samples between base and apex.

    Returns:
        Hull point cloud with shape `(num_z * num_theta, 3)`.
    """
    dtype = jnp.asarray(params.top_height).dtype
    model = CrownModel.from_params(params, dtype=dtype)
    return sample_hull_points(model=model, num_theta=num_theta, num_z=num_z)


def make_crown_hull_batched(
    params: CrownHullParams, *, num_theta: int = 32, num_z: int = 50
) -> Array:
    """Vectorize crown hull generation over a tree batch.

    Args:
        params: Batched crown hull parameters.
        num_theta: Number of azimuth samples around each crown.
        num_z: Number of vertical samples between base and apex.

    Returns:
        Batched hull point clouds with leading batch dimension.
    """

    def _single(p: CrownHullParams) -> Array:
        return make_crown_hull(p, num_theta=num_theta, num_z=num_z)

    return jax.vmap(_single)(params)


def _hull_radii_from_profile(
    *, delta_z: Array, denom_z: Array, edge_radii: Array, shape: Array
) -> Array:
    """Evaluate crown profile radii from the power-law form.

    Args:
        delta_z: Vertical distance from profile origin to query z.
        denom_z: Full vertical span used for normalization.
        edge_radii: Profile radius at peripheral boundary.
        shape: Power-law exponent.

    Returns:
        Profile radii at query locations.
    """
    denom_z_safe = jnp.where(denom_z == 0, 1.0, denom_z)
    inner = 1.0 - delta_z**shape / denom_z_safe**shape
    inner = jnp.maximum(inner, 0.0)
    return (inner * edge_radii**shape) ** (1.0 / shape)

"""Crown model definition (derived invariants).

This module defines `CrownModel`, a compact derived representation of crown
geometry assembled from parameter containers (`CrownHullParams` /
`CrownSurfaceParams`).

The model is intended to be *cheap to build* and to serve as a reusable input to
multiple downstream evaluators:
- point-cloud sampling (`forest3d.geometry.evaluators.points`)
- analytic raster evaluation (`forest3d.rasterization.analytic_dsm`)

Non-goals:
- No raster/window/pixel policy logic.
- No point-cloud/raster generation loops.

Construction conventions (developer contract):
- `CrownModel.from_params(...)` is the canonical entrypoint from sampled parameter
  containers (`geometry.params`) into derived invariants used by evaluators.
- For stable named concepts, prefer primitives from `geometry.primitives` over
  ad-hoc tuple/array plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, tree_util
from jax.typing import ArrayLike

from forest3d.geometry.evaluators.surface import local_polar_from_pose_and_apex
from forest3d.geometry.params import CrownHullParams, CrownSurfaceParams
from forest3d.geometry.primitives import (
    CrownAnchors,
    PeripheralPoints,
    Point3D,
    TreePose,
)


@tree_util.register_dataclass
@dataclass(frozen=True)
class CrownModel:
    """Derived crown geometry that can drive both hull sampling and surface queries."""

    top_height: Array
    crown_ratio: Array
    pose: TreePose
    anchors: CrownAnchors
    apex: Point3D
    base: Point3D
    peripheral: PeripheralPoints

    @staticmethod
    def from_params(
        params: CrownHullParams | CrownSurfaceParams,
        *,
        dtype: jnp.dtype,
    ) -> CrownModel:
        """Build a crown model from crown parameter containers.

        Args:
            params: Crown geometry parameters for either hull-only or surface mode.
            dtype: Dtype kept explicit at call sites for numeric consistency.

        Returns:
            Crown model with eagerly computed anchors, apex/base, and peripheral points.
        """
        top_height = jnp.asarray(params.top_height)
        crown_ratio = jnp.asarray(params.crown_ratio)
        pose = TreePose.from_tree(params)
        if isinstance(params, CrownHullParams):
            anchors = CrownAnchors.from_hull(params)
        else:
            anchors = CrownAnchors.from_crown_surface_params(params)

        apex_arr, base_arr = _hull_apex_and_base_local(
            crown_radii=anchors.crown_radii,
            top_height=top_height,
            crown_ratio=crown_ratio,
        )
        apex = Point3D.from_array(apex_arr, axis=0)
        base = Point3D.from_array(base_arr, axis=0)
        peripheral = PeripheralPoints.from_params(
            crown_radii=anchors.crown_radii,
            crown_edge_heights=anchors.crown_edge_heights,
            top_height=top_height,
            crown_ratio=crown_ratio,
        )
        # dtype is currently used at call sites; keep in signature to make it explicit.
        _ = dtype
        return CrownModel(
            top_height=top_height,
            crown_ratio=crown_ratio,
            pose=pose,
            anchors=anchors,
            apex=apex,
            base=base,
            peripheral=peripheral,
        )

    def apex_global(self, *, dtype: jnp.dtype) -> Array:
        """Return apex `(x, y, z)` in global coordinates.

        Args:
            dtype: Output dtype for the returned coordinate vector.

        Returns:
            Apex point as a length-3 array in global frame.
        """
        a = Point3D(
            x=self.apex.x + self.pose.tx,
            y=self.apex.y + self.pose.ty,
            z=self.apex.z + self.pose.tz,
        )
        return a.as_array(axis=0).astype(dtype)

    def base_global(self, *, dtype: jnp.dtype) -> Array:
        """Return base `(x, y, z)` in global coordinates.

        Args:
            dtype: Output dtype for the returned coordinate vector.

        Returns:
            Base point as a length-3 array in global frame.
        """
        b = Point3D(
            x=self.base.x + self.pose.tx,
            y=self.base.y + self.pose.ty,
            z=self.base.z + self.pose.tz,
        )
        return b.as_array(axis=0).astype(dtype)

    def local_polar(
        self,
        *,
        query_x: Array,
        query_y: Array,
        dtype: jnp.dtype,
    ) -> tuple[Array, Array]:
        """Convert global query XY to local polar coordinates around the apex axis.

        Args:
            query_x: Query x coordinates in global frame.
            query_y: Query y coordinates in global frame.
            dtype: Output dtype used for azimuth values.

        Returns:
            Tuple `(r, theta)` in the local crown frame.
        """
        return local_polar_from_pose_and_apex(
            query_x=query_x,
            query_y=query_y,
            pose_tx=self.pose.tx,
            pose_ty=self.pose.ty,
            apex_x_local=self.apex.x,
            apex_y_local=self.apex.y,
            dtype=dtype,
        )


def _hull_center_xy(crown_radii: ArrayLike) -> Array:
    """Compute local-frame center of the crown XY projection.

    Args:
        crown_radii: Four crown radii ordered `(east, north, west, south)`.

    Returns:
        Length-2 array `(center_x, center_y)`.
    """
    crown_radii = jnp.asarray(crown_radii)
    r_e, r_n, r_w, r_s = crown_radii
    center_x = (r_w - r_e) / 2.0
    center_y = (r_s - r_n) / 2.0
    return jnp.array((center_x, center_y))


def _hull_eccentricity_factors(crown_radii: ArrayLike, crown_ratio: ArrayLike) -> Array:
    """Compute eccentricity factors for apex/base XY displacement.

    Args:
        crown_radii: Four crown radii ordered `(east, north, west, south)`.
        crown_ratio: Crown ratio controlling top/bottom eccentricity magnitude.

    Returns:
        Array of shape `(2, 2)`:
        - row 0: top factors `(east-west, north-south)`
        - row 1: bottom factors `(east-west, north-south)`.
    """
    crown_radii_array = jnp.asarray(crown_radii)
    crown_ratio_array = jnp.asarray(crown_ratio)
    center_xy = _hull_center_xy(crown_radii_array)
    center_x, center_y = center_xy
    r_e, r_n, r_w, r_s = crown_radii_array
    mean_eastwest = (r_e + r_w) / 2.0
    mean_northsouth = (r_n + r_s) / 2.0
    eccen = jnp.array((center_x / mean_eastwest, center_y / mean_northsouth))
    idx = jnp.array(
        (
            -2 / jnp.pi * jnp.arctan(eccen) * crown_ratio_array,
            2 / jnp.pi * jnp.arctan(eccen) * crown_ratio_array,
        )
    )
    return idx


def _hull_apex_and_base_local(
    *, crown_radii: ArrayLike, top_height: ArrayLike, crown_ratio: ArrayLike
) -> tuple[Array, Array]:
    """Compute local-frame crown apex and base coordinates.

    Args:
        crown_radii: Four crown radii ordered `(east, north, west, south)`.
        top_height: Top-of-crown local z value.
        crown_ratio: Fraction controlling crown base height and eccentricity.

    Returns:
        Tuple `(apex, base)` where each entry is a length-3 array `(x, y, z)`.
    """
    crown_radii_array = jnp.asarray(crown_radii)
    top_height_array = jnp.asarray(top_height)
    crown_ratio_array = jnp.asarray(crown_ratio)

    center_xy = _hull_center_xy(crown_radii_array)
    eccen_idx = _hull_eccentricity_factors(crown_radii_array, crown_ratio_array)
    center_x, center_y = center_xy
    r_e, r_n, r_w, r_s = crown_radii_array
    top_eccen_eastwest, top_eccen_northsouth = eccen_idx[0]
    bottom_eccen_eastwest, bottom_eccen_northsouth = eccen_idx[1]

    apex = jnp.array(
        (
            center_x + (r_w - r_e) * top_eccen_eastwest,
            center_y + (r_s - r_n) * top_eccen_northsouth,
            top_height_array,
        ),
        dtype=float,
    )
    base = jnp.array(
        (
            center_x + (r_w - r_e) * bottom_eccen_eastwest,
            center_y + (r_s - r_n) * bottom_eccen_northsouth,
            top_height_array * (1 - crown_ratio_array),
        ),
        dtype=float,
    )
    return apex, base

"""Reusable crown/geometry primitives and low-level helpers.

This module provides small, PyTree-friendly dataclasses used to express crown
geometry with named fields (e.g., `Point3D.x` rather than integer indexing).

Frame conventions:
- \"local\" refers to the crown-local frame used by `CrownModel` (before applying
  the global `TreePose` translation).
- Coordinates are stored as JAX arrays and may be scalar or batched.

Non-goals:
- No raster/window/pixel policy logic (belongs to `forest3d.rasterization.*`).
- No long evaluation loops; those live in evaluator modules.

Construction conventions (developer contract):
- `from_array(...)`: convert raw array layouts into a named primitive.
- `from_params(...)`: build a primitive from parameter containers / parameter fields.
- `from_model(...)`: build a primitive from `CrownModel` derived invariants.
- If a kernel exists for the underlying math, constructors should delegate to it so
  there is exactly one math implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array, tree_util
from jax.typing import ArrayLike

from forest3d.geometry.evaluators.surface import azimuthal_profile_from_relative
from forest3d.geometry.params import CrownHullParams, CrownSurfaceParams

if TYPE_CHECKING:
    from forest3d.geometry.crown import CrownModel


@tree_util.register_dataclass
@dataclass(frozen=True)
class Point3D:
    """A small 3D coordinate primitive (arrays-only, PyTree-friendly)."""

    x: Array
    y: Array
    z: Array

    @staticmethod
    def from_array(a: ArrayLike, *, axis: int | None = None) -> Point3D:
        """Smart constructor from arrays shaped like (3, ...) or (..., 3)."""
        arr = jnp.asarray(a)
        if axis is None:
            if arr.ndim >= 1 and arr.shape[0] == 3:
                axis = 0
            elif arr.ndim >= 1 and arr.shape[-1] == 3:
                axis = -1
            else:
                raise ValueError("Expected shape (3, ...) or (..., 3) for Point3D.")

        if axis == 0:
            x, y, z = arr[0], arr[1], arr[2]
        elif axis == -1:
            x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]
        else:
            raise ValueError("axis must be 0, -1, or None.")
        return Point3D(x=jnp.asarray(x), y=jnp.asarray(y), z=jnp.asarray(z))

    def as_array(self, *, axis: int = 0) -> Array:
        """Return stacked xyz as an array."""
        if axis == 0:
            return jnp.stack((self.x, self.y, self.z), axis=0)
        if axis == -1:
            return jnp.stack((self.x, self.y, self.z), axis=-1)
        raise ValueError("axis must be 0 or -1.")


@tree_util.register_dataclass
@dataclass(frozen=True)
class TreePose:
    """Placement of the crown local frame in global coordinates."""

    stem_base: Point3D
    top_height: Array
    lean_direction: Array
    lean_severity: Array
    t_global: Point3D

    @property
    def tx(self) -> Array:
        return self.t_global.x

    @property
    def ty(self) -> Array:
        return self.t_global.y

    @property
    def tz(self) -> Array:
        return self.t_global.z

    @staticmethod
    def from_tree(tree: CrownHullParams | CrownSurfaceParams) -> TreePose:
        """Build `TreePose` from crown parameter containers.

        Note: The argument name `tree` reflects legacy naming; the input is a crown
        *parameter container* (`CrownHullParams` or `CrownSurfaceParams`).
        """
        stem_base = Point3D.from_array(tree.stem_base)
        top_height = jnp.asarray(tree.top_height)
        lean_direction = jnp.asarray(tree.lean_direction)
        lean_severity = jnp.asarray(tree.lean_severity)
        t_arr = TreePose._translation_global_from_stem_base(
            stem_base.as_array(axis=0), top_height, lean_direction, lean_severity
        )
        t_global = Point3D.from_array(t_arr, axis=0)
        return TreePose(
            stem_base=stem_base,
            top_height=top_height,
            lean_direction=lean_direction,
            lean_severity=lean_severity,
            t_global=t_global,
        )

    @staticmethod
    def _translation_global_from_stem_base(
        stem_base: ArrayLike,
        top_height: ArrayLike,
        lean_direction: ArrayLike | None = None,
        lean_severity: ArrayLike | None = None,
    ) -> Array:
        """Compute the global translation that places a crown in space."""
        stem_base = jnp.asarray(stem_base)
        top_height = jnp.asarray(top_height)
        stem = Point3D.from_array(stem_base)
        stem_x, stem_y, stem_z = stem.x, stem.y, stem.z

        if lean_direction is None:
            lean_direction = jnp.zeros_like(stem_x)
        lean_direction = jnp.asarray(lean_direction)

        if lean_severity is None:
            lean_severity = jnp.zeros_like(stem_x)
        lean_severity = jnp.asarray(lean_severity)

        theta_lean = jnp.deg2rad(lean_direction)
        phi_lean = jnp.deg2rad(lean_severity)

        tx = stem_x + top_height * jnp.tan(phi_lean) * jnp.cos(theta_lean)
        ty = stem_y + top_height * jnp.tan(phi_lean) * jnp.sin(theta_lean)
        tz = stem_z
        return jnp.array((tx, ty, tz))


@tree_util.register_dataclass
@dataclass(frozen=True)
class CrownAnchors:
    """E/N/W/S anchor parameters for crown geometry."""

    crown_radii: Array
    crown_edge_heights: Array
    top_shapes: Array
    bottom_shapes: Array

    @staticmethod
    def from_hull(params: CrownHullParams) -> CrownAnchors:
        return CrownAnchors(
            crown_radii=jnp.asarray(params.crown_radii),
            crown_edge_heights=jnp.asarray(params.crown_edge_heights),
            top_shapes=jnp.asarray(params.crown_top_shapes),
            bottom_shapes=jnp.asarray(params.crown_bottom_shapes),
        )

    @staticmethod
    def from_crown_surface_params(params: CrownSurfaceParams) -> CrownAnchors:
        top_shapes = jnp.asarray(params.crown_top_shapes)
        return CrownAnchors(
            crown_radii=jnp.asarray(params.crown_radii),
            crown_edge_heights=jnp.asarray(params.crown_edge_heights),
            top_shapes=top_shapes,
            bottom_shapes=jnp.ones_like(top_shapes),
        )


@tree_util.register_dataclass
@dataclass(frozen=True)
class PeripheralPoints:
    """Local-frame peripheral points (E/N/W/S)."""

    points_local: Array

    @property
    def x(self) -> Array:
        return self.points_local[:, 0]

    @property
    def y(self) -> Array:
        return self.points_local[:, 1]

    @property
    def z(self) -> Array:
        return self.points_local[:, 2]

    @staticmethod
    def from_params(
        *,
        crown_radii: ArrayLike,
        crown_edge_heights: ArrayLike,
        top_height: ArrayLike,
        crown_ratio: ArrayLike,
    ) -> PeripheralPoints:
        pts = PeripheralPoints._points_local(
            crown_radii=crown_radii,
            crown_edge_heights=crown_edge_heights,
            top_height=top_height,
            crown_ratio=crown_ratio,
        )
        return PeripheralPoints(points_local=pts)

    @staticmethod
    def _points_local(
        *,
        crown_radii: ArrayLike,
        crown_edge_heights: ArrayLike,
        top_height: ArrayLike,
        crown_ratio: ArrayLike,
    ) -> Array:
        """Compute peripheral points in the local crown frame."""
        crown_base_height = jnp.asarray(top_height) * (1 - jnp.asarray(crown_ratio))
        crown_length = jnp.asarray(crown_ratio) * jnp.asarray(top_height)
        crown_radii = jnp.asarray(crown_radii)
        crown_edge_heights = jnp.asarray(crown_edge_heights)
        r_e, r_n, r_w, r_s = crown_radii
        eh_e, eh_n, eh_w, eh_s = crown_edge_heights

        east = jnp.array(
            (r_e, 0.0, crown_base_height + eh_e * crown_length), dtype=float
        )
        north = jnp.array(
            (0.0, r_n, crown_base_height + eh_n * crown_length), dtype=float
        )
        west = jnp.array(
            (-r_w, 0.0, crown_base_height + eh_w * crown_length), dtype=float
        )
        south = jnp.array(
            (0.0, -r_s, crown_base_height + eh_s * crown_length), dtype=float
        )
        return jnp.stack((east, north, west, south))


@tree_util.register_dataclass
@dataclass(frozen=True)
class PeripheralRelative:
    """Peripheral points expressed relative to a reference point.

    This dataclass exists to centralize a common geometric decomposition used by
    both point-cloud sampling and analytic surface evaluation.

    Contract:
    - `radii_xy` and `thetas` are computed from `periph.(x,y)` relative to `ref.(x,y)`.
    - `drop_from_ref_z` is computed as `ref.z - periph.z` (positive means periphery is
      below the reference).
    """

    radii_xy: Array
    thetas: Array
    drop_from_ref_z: Array


def peripheral_relative(
    *,
    periph: PeripheralPoints,
    ref: Point3D,
    dtype: jnp.dtype,
) -> PeripheralRelative:
    """Compute periphery polar coordinates and vertical drop about a reference point.

    Args:
        periph: Peripheral anchor points in the same frame as `ref`.
        ref: Reference point in the same frame as `periph`.
        dtype: Floating dtype used for the returned azimuth angles (radians).

    Returns:
        Relative periphery terms for reuse across evaluators. The returned `thetas`
        are in radians.
    """
    dx = periph.x - ref.x
    dy = periph.y - ref.y
    return PeripheralRelative(
        radii_xy=jnp.hypot(dy, dx),
        thetas=jnp.arctan2(dy, dx).astype(dtype),
        drop_from_ref_z=(ref.z - periph.z),
    )


@tree_util.register_dataclass
@dataclass(frozen=True)
class AzimuthalProfile:
    """Theta-dependent interpolants for the upper crown surface."""

    crown_edge_radius: Array
    periph_z_local: Array
    top_shape: Array

    @staticmethod
    def from_model(
        *,
        model: CrownModel,
        theta: Array,
        period: Array,
        dtype: jnp.dtype,
    ) -> AzimuthalProfile:
        rel = peripheral_relative(periph=model.peripheral, ref=model.apex, dtype=dtype)
        crown_edge_radius, periph_z_local, top_shape = azimuthal_profile_from_relative(
            theta=theta,
            relative=rel,
            apex_z_local=model.apex.z,
            top_shapes=model.anchors.top_shapes,
            period=period,
            dtype=dtype,
        )
        return AzimuthalProfile(
            crown_edge_radius=crown_edge_radius,
            periph_z_local=periph_z_local,
            top_shape=top_shape,
        )

"""Crown geometry primitives and hull/surface evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array, tree_util
from jax.typing import ArrayLike

from forest3d.geometry.params import CrownHullParams, CrownSurfaceParams


@tree_util.register_dataclass
@dataclass(frozen=True)
class Point3D:
    """A small 3D coordinate primitive (arrays-only, PyTree-friendly)."""

    x: Array
    y: Array
    z: Array

    @staticmethod
    def from_array(a: ArrayLike, *, axis: int | None = None) -> Point3D:
        """Smart constructor from arrays shaped like (3, ...) or (..., 3).

        Args:
            a: Array-like holding xyz in either leading or trailing axis.
            axis: If provided, must be 0 (leading) or -1 (trailing). If None, infer
                from shape.
        """
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
    """Placement of the crown local frame in global coordinates.

    This object holds the inputs that define crown placement (stem base + lean) and
    a derived translation `t_global` that is applied to *all* local-frame crown
    coordinates to obtain global coordinates.

    In particular, the global crown apex is:

    - `apex_global = t_global + apex_local`

    where `apex_local` is computed from crown radii/asymmetry (see `CrownApex`).
    """

    stem_base: Point3D
    top_height: Array
    lean_direction: Array
    lean_severity: Array
    t_global: Point3D  # add to local crown coords to get global

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
    def from_tree(tree: _TreeLike) -> TreePose:
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
        """Compute the global translation that places a crown in space.

        Returns `(tx, ty, tz)` as a stacked array with leading xyz axis. This is
        designed to be added to local-frame crown coordinates (where z is measured
        from the stem base) to obtain global coordinates.

        `lean_direction` and `lean_severity` are in degrees. Lean contributes a
        horizontal offset of magnitude `top_height * tan(lean_severity)` in the
        azimuth `lean_direction`. The returned `tz` is the stem-base z.
        """
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

    crown_radii: Array  # (4,) E,N,W,S
    crown_edge_heights: Array  # (4,) E,N,W,S
    top_shapes: Array  # (4,) E,N,W,S
    bottom_shapes: Array  # (4,) E,N,W,S (required for full hull sampling)

    # ---- Directional accessors (E,N,W,S) ----
    @property
    def r_e(self) -> Array:
        return self.crown_radii[0]

    @property
    def r_n(self) -> Array:
        return self.crown_radii[1]

    @property
    def r_w(self) -> Array:
        return self.crown_radii[2]

    @property
    def r_s(self) -> Array:
        return self.crown_radii[3]

    @property
    def edge_height_e(self) -> Array:
        return self.crown_edge_heights[0]

    @property
    def edge_height_n(self) -> Array:
        return self.crown_edge_heights[1]

    @property
    def edge_height_w(self) -> Array:
        return self.crown_edge_heights[2]

    @property
    def edge_height_s(self) -> Array:
        return self.crown_edge_heights[3]

    @property
    def top_shape_e(self) -> Array:
        return self.top_shapes[0]

    @property
    def top_shape_n(self) -> Array:
        return self.top_shapes[1]

    @property
    def top_shape_w(self) -> Array:
        return self.top_shapes[2]

    @property
    def top_shape_s(self) -> Array:
        return self.top_shapes[3]

    @property
    def bottom_shape_e(self) -> Array:
        return self.bottom_shapes[0]

    @property
    def bottom_shape_n(self) -> Array:
        return self.bottom_shapes[1]

    @property
    def bottom_shape_w(self) -> Array:
        return self.bottom_shapes[2]

    @property
    def bottom_shape_s(self) -> Array:
        return self.bottom_shapes[3]

    # ---- Grouped axis helpers ----
    @property
    def radii_eastwest(self) -> Array:
        """(E,W) radii as shape (2,)."""
        return self.crown_radii[0::2]

    @property
    def radii_northsouth(self) -> Array:
        """(N,S) radii as shape (2,)."""
        return self.crown_radii[1::2]

    @property
    def edge_heights_eastwest(self) -> Array:
        return self.crown_edge_heights[0::2]

    @property
    def edge_heights_northsouth(self) -> Array:
        return self.crown_edge_heights[1::2]

    @property
    def top_shapes_eastwest(self) -> Array:
        return self.top_shapes[0::2]

    @property
    def top_shapes_northsouth(self) -> Array:
        return self.top_shapes[1::2]

    @property
    def bottom_shapes_eastwest(self) -> Array:
        return self.bottom_shapes[0::2]

    @property
    def bottom_shapes_northsouth(self) -> Array:
        return self.bottom_shapes[1::2]

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
        # Surface-only container lacks bottom shapes; fill with a safe positive
        # default to keep the object arrays-only. (Do not use for full hull sampling.)
        top_shapes = jnp.asarray(params.crown_top_shapes)
        return CrownAnchors(
            crown_radii=jnp.asarray(params.crown_radii),
            crown_edge_heights=jnp.asarray(params.crown_edge_heights),
            top_shapes=top_shapes,
            bottom_shapes=jnp.ones_like(top_shapes),
        )


@tree_util.register_dataclass
@dataclass(frozen=True)
class CrownApex:
    """Local-frame crown apex point."""

    local: Point3D

    @staticmethod
    def from_params(
        *, crown_radii: ArrayLike, top_height: ArrayLike, crown_ratio: ArrayLike
    ) -> CrownApex:
        apex = _hull_apex_local(
            crown_radii=crown_radii, top_height=top_height, crown_ratio=crown_ratio
        )
        return CrownApex(local=Point3D.from_array(apex, axis=0))


@tree_util.register_dataclass
@dataclass(frozen=True)
class CrownBase:
    """Local-frame crown base point."""

    local: Point3D

    @staticmethod
    def from_params(
        *, crown_radii: ArrayLike, top_height: ArrayLike, crown_ratio: ArrayLike
    ) -> CrownBase:
        base = _hull_base_local(
            crown_radii=crown_radii, top_height=top_height, crown_ratio=crown_ratio
        )
        return CrownBase(local=Point3D.from_array(base, axis=0))


@tree_util.register_dataclass
@dataclass(frozen=True)
class PeripheralPoints:
    """Local-frame peripheral points (E/N/W/S)."""

    points_local: Array  # (4,3)

    @property
    def x(self) -> Array:
        return self.points_local[:, 0]

    @property
    def y(self) -> Array:
        return self.points_local[:, 1]

    @property
    def z(self) -> Array:
        return self.points_local[:, 2]

    # ---- Directional point accessors (E,N,W,S) ----
    @property
    def east(self) -> Array:
        """Peripheral point at East anchor, shape (3,)."""
        return self.points_local[0]

    @property
    def north(self) -> Array:
        """Peripheral point at North anchor, shape (3,)."""
        return self.points_local[1]

    @property
    def west(self) -> Array:
        """Peripheral point at West anchor, shape (3,)."""
        return self.points_local[2]

    @property
    def south(self) -> Array:
        """Peripheral point at South anchor, shape (3,)."""
        return self.points_local[3]

    @property
    def x_e(self) -> Array:
        return self.points_local[0, 0]

    @property
    def y_e(self) -> Array:
        return self.points_local[0, 1]

    @property
    def z_e(self) -> Array:
        return self.points_local[0, 2]

    @property
    def x_n(self) -> Array:
        return self.points_local[1, 0]

    @property
    def y_n(self) -> Array:
        return self.points_local[1, 1]

    @property
    def z_n(self) -> Array:
        return self.points_local[1, 2]

    @property
    def x_w(self) -> Array:
        return self.points_local[2, 0]

    @property
    def y_w(self) -> Array:
        return self.points_local[2, 1]

    @property
    def z_w(self) -> Array:
        return self.points_local[2, 2]

    @property
    def x_s(self) -> Array:
        return self.points_local[3, 0]

    @property
    def y_s(self) -> Array:
        return self.points_local[3, 1]

    @property
    def z_s(self) -> Array:
        return self.points_local[3, 2]

    # ---- Grouped helpers ----
    @property
    def points_eastwest(self) -> Array:
        """(E,W) points, shape (2,3)."""
        return self.points_local[0::2]

    @property
    def points_northsouth(self) -> Array:
        """(N,S) points, shape (2,3)."""
        return self.points_local[1::2]

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
        """Compute peripheral points in the local crown frame.

        Returns an array of shape `(4, 3)` ordered as `(E, N, W, S)`, with columns
        `(x, y, z)`. West and south x/y are negative by convention.
        """
        crown_base_height = jnp.asarray(top_height) * (1 - jnp.asarray(crown_ratio))
        crown_length = jnp.asarray(crown_ratio) * jnp.asarray(top_height)

        crown_radii = jnp.asarray(crown_radii)
        crown_edge_heights = jnp.asarray(crown_edge_heights)
        r_e, r_n, r_w, r_s = crown_radii
        eh_e, eh_n, eh_w, eh_s = crown_edge_heights

        east = jnp.array(
            (
                r_e,
                0.0,
                crown_base_height + eh_e * crown_length,
            ),
            dtype=float,
        )
        north = jnp.array(
            (
                0.0,
                r_n,
                crown_base_height + eh_n * crown_length,
            ),
            dtype=float,
        )
        west = jnp.array(
            (
                -r_w,
                0.0,
                crown_base_height + eh_w * crown_length,
            ),
            dtype=float,
        )
        south = jnp.array(
            (
                0.0,
                -r_s,
                crown_base_height + eh_s * crown_length,
            ),
            dtype=float,
        )
        return jnp.stack((east, north, west, south))


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
        periph = model.peripheral
        apex = model.apex.local
        apex_x, apex_y, apex_z = apex.x, apex.y, apex.z

        periph_x = periph.x
        periph_y = periph.y
        periph_z = periph.z

        periph_drop_from_apex = apex_z - periph_z
        periph_radius_from_apex = jnp.hypot(periph_y - apex_y, periph_x - apex_x)
        periph_theta = jnp.arctan2(periph_y - apex_y, periph_x - apex_x).astype(dtype)

        crown_edge_radius = jnp.interp(
            theta, periph_theta, periph_radius_from_apex.astype(dtype), period=period
        )
        periph_drop = jnp.interp(
            theta, periph_theta, periph_drop_from_apex.astype(dtype), period=period
        )
        periph_z_local = apex_z - periph_drop
        top_shape = jnp.interp(
            theta, periph_theta, model.anchors.top_shapes.astype(dtype), period=period
        )
        return AzimuthalProfile(
            crown_edge_radius=crown_edge_radius,
            periph_z_local=periph_z_local,
            top_shape=top_shape,
        )


@tree_util.register_dataclass
@dataclass(frozen=True)
class CrownModel:
    """Derived crown geometry that can drive both hull sampling and surface queries."""

    top_height: Array
    crown_ratio: Array
    pose: TreePose
    anchors: CrownAnchors
    apex: CrownApex
    base: CrownBase | None
    peripheral: PeripheralPoints

    @property
    def apex_local(self) -> Point3D:
        return self.apex.local

    @property
    def base_local(self) -> Point3D:
        if self.base is None:
            raise ValueError("CrownModel.base is not available (include_base=False).")
        return self.base.local

    @staticmethod
    def from_tree(
        tree: CrownHullParams | CrownSurfaceParams,
        *,
        dtype: jnp.dtype,
        include_base: bool = False,
    ) -> CrownModel:
        top_height = jnp.asarray(tree.top_height)
        crown_ratio = jnp.asarray(tree.crown_ratio)
        pose = TreePose.from_tree(cast(_TreeLike, tree))
        if isinstance(tree, CrownHullParams):
            anchors = CrownAnchors.from_hull(tree)
        else:
            anchors = CrownAnchors.from_crown_surface_params(tree)

        if include_base:
            apex_arr, base_arr = _hull_apex_and_base_local(
                crown_radii=anchors.crown_radii,
                top_height=top_height,
                crown_ratio=crown_ratio,
            )
            apex = CrownApex(local=Point3D.from_array(apex_arr, axis=0))
            base: CrownBase | None = CrownBase(
                local=Point3D.from_array(base_arr, axis=0)
            )
        else:
            apex = CrownApex.from_params(
                crown_radii=anchors.crown_radii,
                top_height=top_height,
                crown_ratio=crown_ratio,
            )
            base = None
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
        a = Point3D(
            x=self.apex_local.x + self.pose.tx,
            y=self.apex_local.y + self.pose.ty,
            z=self.apex_local.z + self.pose.tz,
        )
        return a.as_array(axis=0).astype(dtype)

    def base_global(self, *, dtype: jnp.dtype) -> Array:
        b = Point3D(
            x=self.base_local.x + self.pose.tx,
            y=self.base_local.y + self.pose.ty,
            z=self.base_local.z + self.pose.tz,
        )
        return b.as_array(axis=0).astype(dtype)

    def local_polar(
        self,
        *,
        query_x: Array,
        query_y: Array,
        dtype: jnp.dtype,
    ) -> tuple[Array, Array]:
        """Convert global query XY to (r, theta) about the apex axis in local frame."""
        query_x_local = query_x - self.pose.tx
        query_y_local = query_y - self.pose.ty
        dx_local = query_x_local - self.apex_local.x
        dy_local = query_y_local - self.apex_local.y
        r = jnp.hypot(dy_local, dx_local)
        theta = jnp.arctan2(dy_local, dx_local).astype(dtype)
        return r, theta

    def upper_surface_z_local(
        self,
        *,
        r: Array,
        az: AzimuthalProfile,
    ) -> tuple[Array, Array]:
        """Analytic upper-crown surface z (local frame) and inside-mask."""
        crown_edge_radius = az.crown_edge_radius
        periph_z_local = az.periph_z_local
        apex_z_local = self.apex_local.z
        top_shape = az.top_shape

        crown_edge_radius_safe = jnp.where(
            crown_edge_radius == 0, 1.0, crown_edge_radius
        )
        top_shape_safe = jnp.where(top_shape == 0, 1.0, top_shape)

        r_frac = (r / crown_edge_radius_safe) ** top_shape_safe
        inner = jnp.maximum(1.0 - r_frac, 0.0)
        u = inner ** (1.0 / top_shape_safe)

        z_local = periph_z_local + (apex_z_local - periph_z_local) * u
        inside = (crown_edge_radius > 0) & (r <= crown_edge_radius)
        return z_local, inside

    def sample_hull_points(self, *, num_theta: int = 32, num_z: int = 50) -> Array:
        """Sample surface points on the full crown hull.

        Returns a global-coordinate point cloud with shape `(num_z * num_theta, 3)`
        and columns `(x, y, z)`.

        The `(theta, z)` sampling grid is built with `meshgrid(thetas, zs)`, so when
        flattened in row-major (C) order the output is grouped by z-level: theta
        varies fastest (all thetas for a fixed z), then z.
        """
        anchors = self.anchors
        periph = self.peripheral
        apex_x, apex_y, apex_z = self.apex_local.x, self.apex_local.y, self.apex_local.z
        base_x, base_y, base_z = self.base_local.x, self.base_local.y, self.base_local.z

        thetas = jnp.linspace(0.0, 2.0 * jnp.pi, int(num_theta))
        zs = jnp.linspace(base_z, apex_z, int(num_z))
        grid_thetas, grid_zs = jnp.meshgrid(thetas, zs)

        periph_points_height_from_apex = apex_z - periph.z
        top_periph_points_radii = jnp.hypot(periph.y - apex_y, periph.x - apex_x)
        apex_vs_periph_points_thetas = jnp.arctan2(periph.y - apex_y, periph.x - apex_x)

        apex_periph_line_radii = jnp.interp(
            grid_thetas,
            apex_vs_periph_points_thetas,
            top_periph_points_radii,
            period=2.0 * jnp.pi,
        )
        periph_line_xs = apex_periph_line_radii * jnp.cos(grid_thetas) + apex_x
        periph_line_ys = apex_periph_line_radii * jnp.sin(grid_thetas) + apex_y
        periph_line_zs = apex_z - jnp.interp(
            grid_thetas,
            apex_vs_periph_points_thetas,
            periph_points_height_from_apex,
            period=2.0 * jnp.pi,
        )

        grid_top = grid_zs >= periph_line_zs

        top_shapes_interp = jnp.interp(
            grid_thetas,
            apex_vs_periph_points_thetas,
            anchors.top_shapes,
            period=2.0 * jnp.pi,
        )

        top_delta_z = jnp.maximum(grid_zs - periph_line_zs, 0.0)
        top_inner = (
            1.0
            - top_delta_z**top_shapes_interp
            / (apex_z - periph_line_zs) ** top_shapes_interp
        )
        top_inner = jnp.maximum(top_inner, 0.0)
        top_hull_radii = ((top_inner) * apex_periph_line_radii**top_shapes_interp) ** (
            1.0 / top_shapes_interp
        )

        base_vs_periph_points_thetas = jnp.arctan2(periph.y - base_y, periph.x - base_x)
        grid_bottom = grid_zs < periph_line_zs
        bottom_periph_line_thetas = jnp.arctan2(
            periph_line_ys - base_y, periph_line_xs - base_x
        )
        base_periph_line_radii = jnp.hypot(
            periph_line_xs - base_x, periph_line_ys - base_y
        )

        bottom_shapes_interp = jnp.interp(
            bottom_periph_line_thetas,
            base_vs_periph_points_thetas,
            anchors.bottom_shapes,
            period=2.0 * jnp.pi,
        )

        bottom_delta_z = jnp.maximum(periph_line_zs - grid_zs, 0.0)
        bottom_denom_z = periph_line_zs - base_z
        bottom_denom_z_safe = jnp.where(bottom_denom_z == 0, 1.0, bottom_denom_z)
        bottom_inner = (
            1.0
            - bottom_delta_z**bottom_shapes_interp
            / bottom_denom_z_safe**bottom_shapes_interp
        )
        bottom_inner = jnp.maximum(bottom_inner, 0.0)
        bottom_hull_radii = (
            (bottom_inner) * base_periph_line_radii**bottom_shapes_interp
        ) ** (1.0 / bottom_shapes_interp)

        hull_radii = jnp.where(grid_bottom, bottom_hull_radii, top_hull_radii)

        grid_xs = jnp.where(
            grid_top,
            hull_radii * jnp.cos(grid_thetas) + apex_x,
            hull_radii * jnp.cos(bottom_periph_line_thetas) + base_x,
        )
        grid_ys = jnp.where(
            grid_top,
            hull_radii * jnp.sin(grid_thetas) + apex_y,
            hull_radii * jnp.sin(bottom_periph_line_thetas) + base_y,
        )

        crown_xs = grid_xs + self.pose.tx
        crown_ys = grid_ys + self.pose.ty
        crown_zs = grid_zs + self.pose.tz
        return jnp.column_stack((crown_xs.ravel(), crown_ys.ravel(), crown_zs.ravel()))


def make_crown_hull(
    params: CrownHullParams, *, num_theta: int = 32, num_z: int = 50
) -> Array:
    """Generate crown hull points for a single tree params container."""
    dtype = jnp.asarray(params.top_height).dtype
    model = CrownModel.from_tree(params, dtype=dtype, include_base=True)
    return model.sample_hull_points(num_theta=num_theta, num_z=num_z)


def make_crown_hull_batched(
    params: CrownHullParams, *, num_theta: int = 32, num_z: int = 50
) -> Array:
    """Vectorized crown hull points over a batch of trees."""

    def _single(p: CrownHullParams) -> Array:
        return make_crown_hull(p, num_theta=num_theta, num_z=num_z)

    return jax.vmap(_single)(params)


@runtime_checkable
class _TreeLike(Protocol):
    """Structural protocol for tree-like crown parameter containers."""

    stem_base: ArrayLike
    top_height: ArrayLike
    crown_ratio: ArrayLike
    lean_direction: ArrayLike
    lean_severity: ArrayLike
    crown_radii: ArrayLike
    crown_edge_heights: ArrayLike


def _hull_center_xy(crown_radii: ArrayLike) -> Array:
    """Local-frame center of crown projection.

    The input radii are ordered `(E, N, W, S)` as positive distances. The returned
    `(center_x, center_y)` are offsets (in local coordinates) for the center of
    the crown footprint, computed as:

    - `center_x = (W - E) / 2`
    - `center_y = (S - N) / 2`
    """
    crown_radii = jnp.asarray(crown_radii)
    r_e, r_n, r_w, r_s = crown_radii
    center_x = (r_w - r_e) / 2.0
    center_y = (r_s - r_n) / 2.0
    return jnp.array((center_x, center_y))


def _hull_eccentricity_idx(crown_radii: ArrayLike, crown_ratio: ArrayLike) -> Array:
    """Eccentricity-index terms used for apex/base xy offsets.

    Returns a `(2, 2)` array where each row is an `(x, y)` scaling factor derived
    from the eccentricity of the crown footprint (via `arctan`) and scaled by
    `crown_ratio`:

    - row 0 ("top"): negative index terms used for the apex offset
    - row 1 ("bottom"): positive index terms used for the base offset
    """
    crown_radii_array = jnp.asarray(crown_radii)
    crown_ratio_array = jnp.asarray(crown_ratio)
    center_xy = _hull_center_xy(crown_radii_array)
    center_x, center_y = center_xy
    r_e, r_n, r_w, r_s = crown_radii_array
    mean_eastwest = (r_e + r_w) / 2.0
    mean_northsouth = (r_n + r_s) / 2.0

    eccen = jnp.array(
        (
            center_x / mean_eastwest,
            center_y / mean_northsouth,
        )
    )
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
    """Local-frame crown apex and base.

    Returns `(apex, base)` where each is an `(x, y, z)` array in the local crown
    frame. The xy components include a center offset (from `_hull_center_xy`) and
    an eccentricity-driven offset (from `_hull_eccentricity_idx`) so asymmetric
    radii shift the apex/base away from the stem axis.
    """
    crown_radii_array = jnp.asarray(crown_radii)
    top_height_array = jnp.asarray(top_height)
    crown_ratio_array = jnp.asarray(crown_ratio)

    center_xy = _hull_center_xy(crown_radii_array)
    eccen_idx = _hull_eccentricity_idx(crown_radii_array, crown_ratio_array)

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


def _hull_apex_local(
    *, crown_radii: ArrayLike, top_height: ArrayLike, crown_ratio: ArrayLike
) -> Array:
    """Local-frame crown apex point `(x, y, z)`."""
    apex, _base = _hull_apex_and_base_local(
        crown_radii=crown_radii, top_height=top_height, crown_ratio=crown_ratio
    )
    return apex


def _hull_base_local(
    *, crown_radii: ArrayLike, top_height: ArrayLike, crown_ratio: ArrayLike
) -> Array:
    """Local-frame crown base point `(x, y, z)`."""
    _apex, base = _hull_apex_and_base_local(
        crown_radii=crown_radii, top_height=top_height, crown_ratio=crown_ratio
    )
    return base

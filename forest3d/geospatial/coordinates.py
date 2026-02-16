"""Coordinate system definition shared by rasters and voxel grids.

This module defines a single source of truth for mapping between:
- geographic coordinates: (x, y, z) in a projected coordinate system
- grid coordinates: (i, j, k) indices for raster (2D) and voxel (3D) grids

Conventions
-----------
- Half-open bounds on max edges:
    x ∈ [xmin, xmax), y ∈ [ymin, ymax), z ∈ [zmin, zmax)
- Indexing:
    i = row index (increases downward as y decreases)
    j = col index (increases as x increases)
    k = vertical index (increases as z increases)
- Cell centers:
    `ijk_to_xyz` maps integer indices to *cell centers*.
- Float-first:
    `xyz_to_ijk` returns float indices by default; integer indices are returned only
    when `integers` is set.

Rotation/shear
--------------
For now we assume north-up grids (no rotation/shear). `from_raster` will reject
rotated transforms (b!=0 or d!=0).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from forest3d.geospatial.enums import GridKind, IntegerMode


def _snap_min_down(v: float, dv: float) -> float:
    """Snap a value down to the nearest gridline.

    Args:
        v (float): Value to snap.
        dv (float): Grid spacing.

    Returns:
        (float): Snapped value such that it is an integer multiple of `dv` and <= `v`.
    """
    return math.floor(v / dv) * dv


def _snap_max_up(v_min_adj: float, v_max: float, dv: float) -> float:
    """Snap a max edge up so an interval contains the requested max.

    This is designed for half-open bounds: the returned `v_max_adj` satisfies
    `v_max <= v_max_adj` and makes `(v_max_adj - v_min_adj)` evenly divisible by `dv`.

    Args:
        v_min_adj (float): Adjusted min edge (already snapped).
        v_max (float): Requested max edge.
        dv (float): Grid spacing.

    Returns:
        (float): Adjusted max edge that is an integer multiple of `dv` away from
            `v_min_adj`.
    """
    return v_min_adj + math.ceil((v_max - v_min_adj) / dv) * dv


def _require_positive(name: str, v: float) -> None:
    """Validate that a numeric value is finite and strictly positive.

    Args:
        name (str): Parameter name to use in error messages.
        v (float): Value to validate.

    Raises:
        ValueError: If `v` is not finite or is <= 0.
    """
    if not (v > 0) or not math.isfinite(v):
        raise ValueError(f"{name} must be finite and > 0. Got {v!r}.")


@dataclass(frozen=True, slots=True)
class CoordinateSystem:
    """Shared coordinate system for raster (2D) and voxel (3D) grids."""

    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float
    dx: float
    dy: float
    dz: float
    # Stored Rasterio-like corner affine coefficients:
    # x = a*j + b*i + c
    # y = d*j + e*i + f
    # For north-up rasters: b=d=0, a=dx, e=-dy, c=xmin, f=ymax
    a: float | None = None
    b: float | None = None
    c: float | None = None
    d: float | None = None
    e: float | None = None
    f: float | None = None

    @staticmethod
    def from_bounds(
        *,
        xmin: float,
        ymin: float,
        zmin: float,
        xmax: float,
        ymax: float,
        zmax: float,
        dx: float,
        dy: float,
        dz: float,
    ) -> CoordinateSystem:
        """Create a grid-aligned coordinate system by snapping bounds outward.

        The requested bounds are treated as flexible. We:
        - snap mins down to the nearest gridline
        - snap max edges up so the adjusted box fully contains the requested bounds

        This produces extents that are evenly divisible by dx/dy/dz under half-open
        max-edge semantics: x∈[xmin,xmax), y∈[ymin,ymax), z∈[zmin,zmax).

        Args:
            xmin (float): Requested x min (geographic coordinates).
            ymin (float): Requested y min (geographic coordinates).
            zmin (float): Requested z min (geographic coordinates).
            xmax (float): Requested x max (geographic coordinates).
            ymax (float): Requested y max (geographic coordinates).
            zmax (float): Requested z max (geographic coordinates).
            dx (float): Grid spacing along x (> 0).
            dy (float): Grid spacing along y (> 0).
            dz (float): Grid spacing along z (> 0).

        Returns:
            (CoordinateSystem): A grid-aligned `CoordinateSystem` with snapped bounds
                and a north-up raster-style affine (i increases downward, j increases
                to the right).

        Raises:
            ValueError: If spacings are non-positive/non-finite, bounds are not finite,
                or bounds are degenerate (max <= min).
        """
        _require_positive("dx", float(dx))
        _require_positive("dy", float(dy))
        _require_positive("dz", float(dz))

        xmin = float(xmin)
        ymin = float(ymin)
        zmin = float(zmin)
        xmax = float(xmax)
        ymax = float(ymax)
        zmax = float(zmax)

        if not (math.isfinite(xmin) and math.isfinite(ymin) and math.isfinite(zmin)):
            raise ValueError("xmin/ymin/zmin must be finite.")
        if not (math.isfinite(xmax) and math.isfinite(ymax) and math.isfinite(zmax)):
            raise ValueError("xmax/ymax/zmax must be finite.")
        if not (xmax > xmin and ymax > ymin and zmax > zmin):
            raise ValueError("Expected xmax>xmin, ymax>ymin, zmax>zmin.")

        xmin_adj = _snap_min_down(xmin, dx)
        ymin_adj = _snap_min_down(ymin, dy)
        zmin_adj = _snap_min_down(zmin, dz)

        xmax_adj = _snap_max_up(xmin_adj, xmax, dx)
        ymax_adj = _snap_max_up(ymin_adj, ymax, dy)
        zmax_adj = _snap_max_up(zmin_adj, zmax, dz)

        # Corner affine for north-up raster-style indexing.
        a, b, c = dx, 0.0, xmin_adj
        d, e, f = 0.0, -dy, ymax_adj
        return CoordinateSystem(
            xmin=xmin_adj,
            ymin=ymin_adj,
            zmin=zmin_adj,
            xmax=xmax_adj,
            ymax=ymax_adj,
            zmax=zmax_adj,
            dx=dx,
            dy=dy,
            dz=dz,
            a=a,
            b=b,
            c=c,
            d=d,
            e=e,
            f=f,
        )

    @staticmethod
    def from_raster(
        path_to_raster: str | os.PathLike,
        *,
        zmin: float,
        zmax: float,
        dz: float,
    ) -> CoordinateSystem:
        """Create a coordinate system from a raster.

        Args:
            path_to_raster (str | os.PathLike): Path to a raster readable by rasterio.
            zmin (float): Minimum z for the associated voxel space (geographic
                coordinates).
            zmax (float): Maximum z for the associated voxel space (geographic
                coordinates).
            dz (float): Vertical spacing for voxels.

        Returns:
            (CoordinateSystem): A `CoordinateSystem` whose x/y bounds and affine are
                inferred from the raster metadata, and whose z bounds are provided by
                the caller.

        Raises:
            ValueError: If dz <= 0, z bounds are invalid, or the raster transform is
                rotated/sheared.
        """
        _require_positive("dz", float(dz))
        zmin = float(zmin)
        zmax = float(zmax)
        if not (zmax > zmin):
            raise ValueError("Expected zmax > zmin.")

        # Lazy import to keep rasterio off the hot path.
        import rasterio

        with rasterio.open(Path(path_to_raster)) as ds:
            transform = ds.transform
            bounds = ds.bounds
            width = ds.width
            height = ds.height

        # Reject rotation/shear.
        if not (transform.b == 0 and transform.d == 0):
            raise ValueError(
                "Rotated/sheared rasters are not supported yet (transform.b/d must "
                "be 0)."
            )

        dx = float(transform.a)
        dy = float(-transform.e)  # e is typically negative for north-up rasters
        _require_positive("dx", dx)
        _require_positive("dy", dy)

        xmin = float(bounds.left)
        xmax = float(bounds.right)
        ymin = float(bounds.bottom)
        ymax = float(bounds.top)

        # Ensure extents are exactly width/height * dx/dy (avoid tiny drift).
        xmax = xmin + float(width) * dx
        ymax = ymin + float(height) * dy

        return CoordinateSystem(
            xmin=xmin,
            ymin=ymin,
            zmin=zmin,
            xmax=xmax,
            ymax=ymax,
            zmax=zmax,
            dx=dx,
            dy=dy,
            dz=float(dz),
            a=float(transform.a),
            b=float(transform.b),
            c=float(transform.c),
            d=float(transform.d),
            e=float(transform.e),
            f=float(transform.f),
        )

    @staticmethod
    def from_las(
        path_to_las: str | os.PathLike,
        *,
        dx: float,
        dy: float,
        dz: float,
        zmin: float | None = None,
        zmax: float | None = None,
    ) -> CoordinateSystem:
        """Create a coordinate system from a LAS/LAZ header or file.

        Args:
            path_to_las (str | os.PathLike): Path to a LAS/LAZ readable by laspy.
            dx (float): Grid spacing along x (> 0).
            dy (float): Grid spacing along y (> 0).
            dz (float): Vertical spacing (> 0).
            zmin (float | None): Optional z min. If None, inferred from the LAS header
                min z.
            zmax (float | None): Optional z max. If None, inferred from the LAS header
                max z.

        Returns:
            (CoordinateSystem): A grid-aligned `CoordinateSystem` snapped outward from
                LAS header bounds, using a north-up raster-style indexing convention.

        Raises:
            ValueError: If spacings are invalid.
        """
        _require_positive("dx", float(dx))
        _require_positive("dy", float(dy))
        _require_positive("dz", float(dz))

        # Lazy import to keep laspy off the hot path.
        import laspy

        las = laspy.read(Path(path_to_las))
        header = las.header

        xmin, ymin, zmin_h = map(float, header.mins)
        xmax, ymax, zmax_h = map(float, header.maxs)
        if zmin is None:
            zmin = zmin_h
        if zmax is None:
            zmax = zmax_h

        return CoordinateSystem.from_bounds(
            xmin=xmin,
            ymin=ymin,
            zmin=float(zmin),
            xmax=xmax,
            ymax=ymax,
            zmax=float(zmax),
            dx=float(dx),
            dy=float(dy),
            dz=float(dz),
        )

    @property
    def nx(self) -> int:
        return int(round((self.xmax - self.xmin) / self.dx))

    @property
    def ny(self) -> int:
        return int(round((self.ymax - self.ymin) / self.dy))

    @property
    def nz(self) -> int:
        return int(round((self.zmax - self.zmin) / self.dz))

    def xyz_to_ijk(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike | None = None,
        *,
        grid: GridKind,
        integers: IntegerMode | None = None,
    ) -> tuple[Array, Array, Array]:
        """Convert geographic coordinates to grid indices (float-first).

        Args:
            x (ArrayLike): geographic x coordinate(s).
            y (ArrayLike): geographic y coordinate(s).
            z (ArrayLike | None): geographic z coordinate(s). Required when `grid` is
                `GridKind.VOXEL`.
            grid (GridKind): Which grid to compute indices for.
            integers (IntegerMode | None): If None, return floating indices. Otherwise
                return integer indices using the selected mode:
                - `IntegerMode.FLOOR`: floor
                - `IntegerMode.CEIL`: ceil
                - `IntegerMode.INT`: truncate toward zero (safe when indices >= 0)

        Returns:
            (tuple[Array, Array, Array]): A tuple `(i, j, k)` of indices. For
                `GridKind.RASTER`, `k` is returned as zeros (treat raster as nz=1).

        Raises:
            ValueError: If `z` is missing when indices are computed for voxels, or if
                `integers` is an invalid value.
        """
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        j = (x - self.xmin) / self.dx
        i = (self.ymax - y) / self.dy

        if grid == GridKind.RASTER:
            k = jnp.zeros_like(i)
        else:
            if z is None:
                raise ValueError("z is required when grid='voxel'.")
            z = jnp.asarray(z)
            k = (z - self.zmin) / self.dz

        if integers == IntegerMode.FLOOR:
            return (
                jnp.floor(i).astype(jnp.int32),
                jnp.floor(j).astype(jnp.int32),
                jnp.floor(k).astype(jnp.int32),
            )
        if integers == IntegerMode.CEIL:
            return (
                jnp.ceil(i).astype(jnp.int32),
                jnp.ceil(j).astype(jnp.int32),
                jnp.ceil(k).astype(jnp.int32),
            )
        if integers == IntegerMode.INT:
            return i.astype(jnp.int32), j.astype(jnp.int32), k.astype(jnp.int32)

        return i, j, k

    def ijk_to_xyz(
        self,
        i: ArrayLike,
        j: ArrayLike,
        k: ArrayLike = 0,
        *,
        grid: GridKind,
    ) -> tuple[Array, Array, Array]:
        """Convert grid indices to geographic coordinates at cell centers.

        Args:
            i (ArrayLike): Row index/indices (0-based).
            j (ArrayLike): Column index/indices (0-based).
            k (ArrayLike): Vertical index/indices (0-based). For rasters, this is
                typically 0.
            grid (GridKind): Which grid to interpret indices for.

        Returns:
            (tuple[Array, Array, Array]): A tuple `(x, y, z)` of geographic
                coordinates at cell centers.

        """
        i = jnp.asarray(i)
        j = jnp.asarray(j)
        k = jnp.asarray(k)

        x = self.xmin + (j + 0.5) * self.dx
        y = self.ymax - (i + 0.5) * self.dy
        z = self.zmin + (k + 0.5) * self.dz
        return x, y, z

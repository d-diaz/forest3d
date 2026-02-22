"""DEM / raster helper utilities (cold path).

These functions depend on raster I/O (`rasterio`) and geometry libraries
(`shapely`). Keep them out of `forest3d.geometry` so the hot path stays focused
on JAX-friendly numeric routines.
"""

from __future__ import annotations

import os

import numpy as np
import rasterio
from shapely.geometry import Point, Polygon


def get_raster_bbox_as_polygon(path_to_raster: str | os.PathLike) -> Polygon:
    """Return a Shapely polygon for the bounding box of a raster."""
    with rasterio.open(path_to_raster) as raster_src:
        west_edge, south_edge, east_edge, north_edge = raster_src.bounds

    points = [
        Point(west_edge, south_edge),
        Point(west_edge, north_edge),
        Point(east_edge, north_edge),
        Point(east_edge, south_edge),
    ]
    return Polygon([(p.x, p.y) for p in points])


def get_elevation(
    dem: str | os.PathLike, x: float | np.ndarray, y: float | np.ndarray
) -> float | np.ndarray:
    """Sample elevation(s) from a DEM at (x, y) coordinate(s)."""
    x, y = np.asanyarray(x), np.asanyarray(y)
    _arrays_equal_shape(x, y)

    with rasterio.open(dem) as src:
        terrain = src.read(1)

        if x.shape == ():  # scalar case
            row, col = src.index(float(x), float(y))
            return float(terrain[row, col])

        rows: list[int] = []
        cols: list[int] = []
        for x_val, y_val in zip(x.ravel(), y.ravel()):
            row, col = src.index(float(x_val), float(y_val))
            rows.append(row)
            cols.append(col)
        rows_arr = np.asarray(rows)
        cols_arr = np.asarray(cols)

    try:
        elev = terrain[rows_arr, cols_arr].reshape(x.shape)
    except IndexError:
        with rasterio.open(dem) as src2:
            bounds = src2.bounds
        raise IndexError(
            f"(x,y) location outside bounds of elevation raster:\n{bounds}"
        )

    return elev


def get_circular_plot_boundary(
    *,
    x: np.ndarray,
    y: np.ndarray,
    radius: np.ndarray,
    dem: str | os.PathLike | None = None,
    n: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boundary (x,y,z) samples along a circular plot circumference."""
    thetas = np.linspace(0, 2 * np.pi, int(n))
    xs = radius * np.cos(thetas) + x
    ys = radius * np.sin(thetas) + y
    zs = np.asanyarray(get_elevation(dem, xs, ys) if dem else np.zeros(int(n)))
    return xs, ys, zs


def _arrays_equal_shape(*args: np.ndarray, raise_exc: bool = True) -> bool:
    """Return True if all inputs have equal shape (after `np.asanyarray`)."""
    arrs = [np.asanyarray(arg) for arg in args]
    shapes = np.array([arr.shape for arr in arrs])
    equal_shapes = bool(np.all(shapes == shapes[0]))

    if not equal_shapes and raise_exc:
        raise ValueError(f"Input shapes mismatch: {shapes}")

    return equal_shapes

import numpy as np
import pytest

from forest3d.distance.field import distance_field_from_surface
from forest3d.geospatial.coordinates import CoordinateSystem


def test_distance_field_surface_shapes_and_monotonic_axes():
    cs = CoordinateSystem.from_bounds(
        xmin=0.0,
        ymin=0.0,
        zmin=0.0,
        xmax=2.0,
        ymax=2.0,
        zmax=2.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )
    dsm = np.zeros((cs.ny, cs.nx), dtype=np.float32)

    df = distance_field_from_surface(dsm, cs=cs)

    assert df.values.shape == (cs.nx, cs.ny, cs.nz)
    assert df.x.shape == (cs.nx,)
    assert df.y.shape == (cs.ny,)
    assert df.z.shape == (cs.nz,)

    assert np.all(np.diff(df.x) > 0)
    assert np.all(np.diff(df.y) > 0)
    assert np.all(np.diff(df.z) > 0)


def test_distance_field_surface_z_clamps_to_bounds():
    cs = CoordinateSystem.from_bounds(
        xmin=0.0,
        ymin=0.0,
        zmin=0.0,
        xmax=2.0,
        ymax=2.0,
        zmax=2.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )
    # z-bins cover [0,2) with dz=1. Surfaces outside cs.z-bounds should error.
    msg = r"surface z-values exceed voxel grid vertical bounds"

    surface_hi = np.full((cs.ny, cs.nx), 999.0, dtype=np.float32)
    with pytest.raises(ValueError, match=msg):
        distance_field_from_surface(surface_hi, cs=cs)

    surface_lo = np.full((cs.ny, cs.nx), -999.0, dtype=np.float32)
    with pytest.raises(ValueError, match=msg):
        distance_field_from_surface(surface_lo, cs=cs)

    # Exactly at the max edge is also out-of-bounds (half-open).
    surface_eq_max = np.full((cs.ny, cs.nx), float(cs.zmax), dtype=np.float32)
    with pytest.raises(ValueError, match=msg):
        distance_field_from_surface(surface_eq_max, cs=cs)


def test_distance_field_surface_y_orientation_matches_scipy_axes():
    # ny=2 so we can distinguish north (top row) vs south (bottom row).
    cs = CoordinateSystem.from_bounds(
        xmin=0.0,
        ymin=0.0,
        zmin=0.0,
        xmax=1.0,
        ymax=2.0,
        zmax=2.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )
    # Top row (north, higher y) set to top layer; bottom row (south) set to bottom.
    # Raster indexing: row 0 is north/high y, row 1 is south/low y.
    surface = np.array([[1.5], [0.5]], dtype=np.float32)  # shape (ny=2, nx=1)
    df = distance_field_from_surface(surface, cs=cs)

    # SciPy-style y-axis is ascending (south->north), so y index 0 corresponds to
    # raster row 1 (south), and y index 1 corresponds to raster row 0 (north).
    assert df.values.shape == (1, 2, 2)
    assert df.values[0, 0, 0] == 0.0  # south row at k=0 is on the surface
    assert df.values[0, 1, 1] == 0.0  # north row at k=1 is on the surface

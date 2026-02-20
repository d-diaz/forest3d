import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin

from forest3d.geospatial.coordinates import CoordinateSystem
from forest3d.geospatial.enums import GridKind, IntegerMode


def test_coordinates_from_bounds_resolution_and_shape():
    cs = CoordinateSystem.from_bounds(
        xmin=0.0,
        ymin=0.0,
        zmin=0.0,
        xmax=10.0,
        ymax=10.0,
        zmax=10.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )
    assert cs.nx == 10
    assert cs.ny == 10
    assert cs.nz == 10


def test_coordinates_from_bounds_snaps_outward_to_even_divisions():
    # Requested bounds are flexible; from_bounds snaps mins down and maxes up.
    cs = CoordinateSystem.from_bounds(
        xmin=0.0,
        ymin=0.0,
        zmin=0.0,
        xmax=10.0,
        ymax=10.0,
        zmax=10.0,
        dx=1.1,
        dy=1.1,
        dz=1.1,
    )
    # 10 / 1.1 = 9.09..., so we expand outward to 11.0 (10 steps).
    assert cs.nx == 10
    assert cs.ny == 10
    assert cs.nz == 10
    assert np.isclose(cs.xmax - cs.xmin, 11.0)
    assert np.isclose(cs.ymax - cs.ymin, 11.0)
    assert np.isclose(cs.zmax - cs.zmin, 11.0)


def test_xyz_to_ijk_float_first_and_integer_modes():
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

    i_f, j_f, k_f = cs.xyz_to_ijk(0.25, 1.75, grid=GridKind.RASTER)
    assert float(i_f) == 0.25
    assert float(j_f) == 0.25
    assert float(k_f) == 0.0

    i, j, k = cs.xyz_to_ijk(
        0.25, 1.75, grid=GridKind.RASTER, integers=IntegerMode.FLOOR
    )
    assert int(i) == 0
    assert int(j) == 0
    assert int(k) == 0


def test_coordinates_from_raster_reads_bounds_and_resolution(tmp_path):
    # 3 cols (x), 2 rows (y). Pixel size: dx=1.0, dy=2.0
    xmin = 100.0
    ymax = 200.0
    dx = 1.0
    dy = 2.0
    width = 3
    height = 2
    transform = from_origin(xmin, ymax, dx, dy)

    path = tmp_path / "test.tif"
    data = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    cs = CoordinateSystem.from_raster(path, zmin=0.0, zmax=10.0, dz=1.0)
    assert cs.dx == dx
    assert cs.dy == dy
    assert cs.xmin == xmin
    assert cs.ymax == ymax
    assert cs.nx == width
    assert cs.ny == height
    assert cs.zmin == 0.0
    assert cs.zmax == 10.0
    assert cs.dz == 1.0


def test_coordinates_from_las_snaps_bounds_and_infers_z(tmp_path):
    # Points define mins/maxs that are not aligned to dx/dy/dz.
    xs = np.array([0.2, 2.3], dtype=float)
    ys = np.array([10.2, 12.3], dtype=float)
    zs = np.array([5.2, 6.9], dtype=float)

    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = xs
    las.y = ys
    las.z = zs

    path = tmp_path / "test.las"
    las.write(path)

    cs = CoordinateSystem.from_las(path, dx=1.0, dy=1.0, dz=0.5)
    # mins snap down, maxs snap up (outward)
    assert np.isclose(cs.xmin, 0.0)
    assert np.isclose(cs.xmax, 3.0)
    assert np.isclose(cs.ymin, 10.0)
    assert np.isclose(cs.ymax, 13.0)
    assert np.isclose(cs.zmin, 5.0)
    assert np.isclose(cs.zmax, 7.0)
    assert cs.nx == 3
    assert cs.ny == 3
    assert cs.nz == 4

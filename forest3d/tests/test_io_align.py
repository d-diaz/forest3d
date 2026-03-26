import pytest

from forest3d.io.align import RasterGrid, assert_grids_equal


def test_assert_grids_equal_accepts_identical_grids():
    g1 = RasterGrid(
        crs_wkt="EPSG:32610",
        transform_gdal=(0.0, 1.0, 0.0, 10.0, 0.0, -1.0),
        width=10,
        height=20,
    )
    g2 = RasterGrid(
        crs_wkt="EPSG:32610",
        transform_gdal=(0.0, 1.0, 0.0, 10.0, 0.0, -1.0),
        width=10,
        height=20,
    )
    assert_grids_equal(g1, g2)


def test_assert_grids_equal_raises_with_informative_message():
    dem = RasterGrid(
        crs_wkt="EPSG:32610",
        transform_gdal=(0.0, 1.0, 0.0, 10.0, 0.0, -1.0),
        width=10,
        height=20,
    )
    dsm = RasterGrid(
        crs_wkt="EPSG:32611",
        transform_gdal=(0.0, 1.0, 0.0, 10.0, 0.0, -1.0),
        width=11,
        height=20,
    )
    with pytest.raises(ValueError, match=r"provided grids are not equal"):
        assert_grids_equal(dem, dsm)

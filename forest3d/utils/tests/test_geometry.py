import os

import pytest
from jax import Array

from forest3d.utils.geometry import (
    _get_treetop_location,  # _get_raster_bbox_as_polygon,; get_elevation,
)

THIS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(THIS_DIR, "sample_data_for_testing")


TEST_DEM = os.path.join(DATA_DIR, "elevation_raster.tif")


# def test_raster_bbox_getter_returns_polygon():
#     """Checks whether _get_raster_bbox_... return a Polygon object."""
#
#     result = _get_raster_bbox_as_polygon(TEST_DEM)
#     assert isinstance(result, Polygon)


# def test_elevation_getter_single_point_out_of_bounds():
#     """Checks whether get_elevation returns IndexError with invalid when
#     point outside bounding box is queried."""
#
#     poly = _get_raster_bbox_as_polygon(TEST_DEM)
#     west_edge, south_edge, east_edge, north_edge = poly.bounds
#     x = west_edge - 1.0  # outside left bound
#     y = south_edge - 1.0  # outside bottom bound
#
#     with pytest.raises(IndexError):
#         get_elevation(TEST_DEM, x, y)


# def test_elevation_getter_single_point_dtype():
#     """Checks whether get_elevation returns a float or int when a valid
#     query location is used."""
#
#     poly = _get_raster_bbox_as_polygon(TEST_DEM)
#     west_edge, south_edge, east_edge, north_edge = poly.bounds
#     x = np.mean((west_edge, east_edge))  # middle of xs
#     y = np.mean((south_edge, north_edge))  # middle of ys
#
#     result = get_elevation(TEST_DEM, x, y)
#     result_type = result.dtype.kind
#
#     assert result_type in ["i", "u", "f"]
#
#
# def test_elevation_getter_array_dtype():
#     """Checks whether get_elevation returns an array with dtype float or
#     int when a valid query location is used."""
#
#     poly = _get_raster_bbox_as_polygon(TEST_DEM)
#     west_edge, south_edge, east_edge, north_edge = poly.bounds
#     mean_x = np.mean((west_edge, east_edge))  # middle of xs
#     mean_y = np.mean((south_edge, north_edge))  # middle of ys
#     x = np.array((mean_x, mean_x))
#     y = np.array((mean_y, mean_y))
#
#     result = get_elevation(TEST_DEM, x, y)
#     result_type = result.dtype.kind
#     assert isinstance(result, np.ndarray)
#     assert result_type in ["i", "u", "f"]


# def test_treetop_getter_invalid_lean():
# TODO: Add test for invalid lean severity once type checking with jax implemented
#     """get_treetop_location raises ValueError when lean >= 90 degrees specified."""
#
#     args = [(0, 0, 0), 100]  # x, y, z, and height
#     kwargs = {"lean_severity": 90}
#     with pytest.raises(ValueError):
#         _get_treetop_location(*args, **kwargs)


# def test_treetop_getter_invalid_height():
# TODO: Add test for invalid height once type checking with jax implemented
#     """get_treetop_location raises ValueError when height < 0 specified."""
#
#     args = [(0, 0, 0), -1]  # x, y, z, and height
#
#     with pytest.raises(ValueError):
#         _get_treetop_location(*args)


def test_treetop_getter_single_point_result_format():
    """_get_treetop_location returns array with shape (3,) when single point provided."""

    args = [(0, 0, 0), 100]  # x, y, z, and height
    result = _get_treetop_location(*args)
    assert isinstance(result, Array)
    assert result.shape[0] == 3


def test_treetop_getter_array_result_format():
    """Checks whether _get_treetop_location returns a numpy array with shape
    (3,N) when lists of points are provided."""

    x = [0, 1.0]
    y = [0, 2.0]
    z = [0, 3.0]
    height = [100, 75]
    args = [(x, y, z), height]  # x, y, z, and height
    result = _get_treetop_location(*args)
    assert result.shape == (3, len(x))


def test_treetop_getter_input_arrays_diff_shapes():
    """get_treetop_location returns ValueError for arrays of different shapes."""

    x = [0, 0]
    y = [0, 5]
    z = [3.0, 2.0]
    height = [100, 75, 85]
    args = [x, y, z, height]

    with pytest.raises(ValueError):
        _get_treetop_location(*args)

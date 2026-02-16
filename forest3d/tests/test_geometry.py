import numpy as np
import pytest
from jax import Array

from forest3d.geometry.crown_hull import _get_treetop_location


def test_treetop_getter_single_point_result_format():
    """_get_treetop_location returns shape (3,) when single point provided."""
    args = [(0, 0, 0), 100]  # x, y, z, and height
    result = _get_treetop_location(*args)
    assert isinstance(result, Array)
    assert result.shape[0] == 3


def test_treetop_getter_array_result_format():
    """_get_treetop_location returns array with shape (3,N) for list inputs."""
    x = [0, 1.0]
    y = [0, 2.0]
    z = [0, 3.0]
    height = [100, 75]
    args = [(x, y, z), height]  # x, y, z, and height
    result = _get_treetop_location(*args)
    assert result.shape == (3, len(x))


def test_treetop_getter_input_arrays_diff_shapes():
    """_get_treetop_location raises ValueError for arrays of different shapes."""
    x = [0, 0]
    y = [0, 5]
    z = [3.0, 2.0]
    height = [100, 75, 85]
    args = [x, y, z, height]

    with pytest.raises(ValueError):
        _get_treetop_location(*args)


def test_treetop_getter_lean_changes_xy():
    """Non-zero lean should affect x/y translation, not z translation."""
    stem = (0.0, 0.0, 7.0)
    height = 10.0
    no_lean = np.asarray(_get_treetop_location(stem, height, lean_severity=0.0))
    with_lean = np.asarray(
        _get_treetop_location(stem, height, lean_direction=45.0, lean_severity=10.0)
    )

    assert np.allclose(no_lean[2], with_lean[2])
    assert not np.allclose(no_lean[0], with_lean[0])
    assert not np.allclose(no_lean[1], with_lean[1])

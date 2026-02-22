import numpy as np
import pytest
from jax import Array

from forest3d.geometry.crown import TreePose
from forest3d.geometry.params import CrownHullParams


def test_treetop_getter_single_point_result_format():
    """`TreePose` translation returns shape (3,) for a single tree."""
    params = CrownHullParams(
        stem_base=(0.0, 0.0, 0.0),
        top_height=100.0,
        crown_ratio=0.65,
        lean_direction=0.0,
        lean_severity=0.0,
        crown_radii=(1.0, 1.0, 1.0, 1.0),
        crown_edge_heights=(0.3, 0.3, 0.3, 0.3),
        crown_shapes=np.full((2, 4), 2.0),
    )
    result = TreePose.from_tree(params).t_global.as_array(axis=0)
    assert isinstance(result, Array)
    assert result.shape[0] == 3


def test_treetop_getter_array_result_format():
    """`TreePose` translation returns shape (3,N) for vectorized local inputs."""
    x = [0, 1.0]
    y = [0, 2.0]
    z = [0, 3.0]
    height = [100, 75]
    params = CrownHullParams(
        stem_base=(x, y, z),  # -> jnp.asarray => (3,N)
        top_height=height,
        crown_ratio=np.full((len(x),), 0.65),
        lean_direction=np.zeros((len(x),)),
        lean_severity=np.zeros((len(x),)),
        crown_radii=np.ones((len(x), 4)),
        crown_edge_heights=np.full((len(x), 4), 0.3),
        crown_shapes=np.full((len(x), 2, 4), 2.0),
    )
    result = TreePose.from_tree(params).t_global.as_array(axis=0)
    assert result.shape == (3, len(x))


def test_treetop_getter_input_arrays_diff_shapes():
    """`TreePose` requires a 3D stem_base (x,y,z)."""
    params = CrownHullParams(
        stem_base=(0.0, 0.0),  # invalid: only (x,y)
        top_height=100.0,
        crown_ratio=0.65,
        lean_direction=0.0,
        lean_severity=0.0,
        crown_radii=(1.0, 1.0, 1.0, 1.0),
        crown_edge_heights=(0.3, 0.3, 0.3, 0.3),
        crown_shapes=np.full((2, 4), 2.0),
    )
    with pytest.raises(ValueError):
        _ = TreePose.from_tree(params).t_global.as_array(axis=0)


def test_treetop_getter_lean_changes_xy():
    """Non-zero lean should affect x/y translation, not z translation."""
    stem = (0.0, 0.0, 7.0)
    height = 10.0
    p0 = CrownHullParams(
        stem_base=stem,
        top_height=height,
        crown_ratio=0.65,
        lean_direction=0.0,
        lean_severity=0.0,
        crown_radii=(1.0, 1.0, 1.0, 1.0),
        crown_edge_heights=(0.3, 0.3, 0.3, 0.3),
        crown_shapes=np.full((2, 4), 2.0),
    )
    p1 = CrownHullParams(
        stem_base=stem,
        top_height=height,
        crown_ratio=0.65,
        lean_direction=45.0,
        lean_severity=10.0,
        crown_radii=(1.0, 1.0, 1.0, 1.0),
        crown_edge_heights=(0.3, 0.3, 0.3, 0.3),
        crown_shapes=np.full((2, 4), 2.0),
    )
    no_lean = np.asarray(TreePose.from_tree(p0).t_global.as_array(axis=0))
    with_lean = np.asarray(TreePose.from_tree(p1).t_global.as_array(axis=0))

    assert np.allclose(no_lean[2], with_lean[2])
    assert not np.allclose(no_lean[0], with_lean[0])
    assert not np.allclose(no_lean[1], with_lean[1])

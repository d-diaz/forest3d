import numpy as np
from jax import jit

from forest3d.geometry.crown import _hull_apex_and_base_local
from forest3d.geometry.evaluators.points import make_crown_hull
from forest3d.geometry.params import CrownHullParams
from forest3d.geometry.primitives import Point3D, TreePose
from forest3d.schemas.tree import Tree


def test_stem_x_hull_isolation():
    """Changes in stem_x coordinate alter expected coordinates describing crown."""
    p1 = Tree(
        species="Douglas-fir", dbh=7.5, top_height=85, stem_x=0, stem_y=0, stem_z=0
    ).crown()
    p2 = Tree(
        species="Douglas-fir", dbh=7.5, top_height=85, stem_x=10, stem_y=0, stem_z=0
    ).crown()

    assert not np.allclose(p1[:, 0], p2[:, 0])
    assert np.allclose(p1[:, 1], p2[:, 1])
    assert np.allclose(p1[:, 2], p2[:, 2])


def test_stem_y_hull_isolation():
    """Changes in stem_y coordinate alter expected coordinates describing crown."""
    p1 = Tree(
        species="Douglas-fir", dbh=7.5, top_height=85, stem_x=0, stem_y=0, stem_z=0
    ).crown()
    p2 = Tree(
        species="Douglas-fir", dbh=7.5, top_height=85, stem_x=0, stem_y=10, stem_z=0
    ).crown()

    assert np.allclose(p1[:, 0], p2[:, 0])
    assert not np.allclose(p1[:, 1], p2[:, 1])
    assert np.allclose(p1[:, 2], p2[:, 2])


def test_stem_z_hull_isolation():
    """Changes in stem_z coordinate alter expected coordinates describing crown."""
    p1 = Tree(
        species="Douglas-fir", dbh=7.5, top_height=85, stem_x=0, stem_y=0, stem_z=0
    ).crown()
    p2 = Tree(
        species="Douglas-fir", dbh=7.5, top_height=85, stem_x=0, stem_y=0, stem_z=10
    ).crown()

    assert np.allclose(p1[:, 0], p2[:, 0])
    assert np.allclose(p1[:, 1], p2[:, 1])
    assert not np.allclose(p1[:, 2], p2[:, 2])


def test_treetop_stem_x_isolation():
    """Changes in treetop stem_x alter expected coordinates describing crown."""
    stem1 = (0, 0, 0)
    stem2 = (10, 0, 0)

    # Use a minimal params container to drive the pose model.
    p1 = CrownHullParams(
        stem_base=np.asarray(stem1, dtype=float),
        top_height=np.asarray(75.0),
        crown_ratio=np.asarray(0.65),
        lean_direction=np.asarray(0.0),
        lean_severity=np.asarray(0.0),
        crown_radii=np.asarray((1.0, 1.0, 1.0, 1.0)),
        crown_edge_heights=np.asarray((0.3, 0.3, 0.3, 0.3)),
        crown_shapes=np.full((2, 4), 2.0),
    )
    p2 = CrownHullParams(
        stem_base=np.asarray(stem2, dtype=float),
        top_height=np.asarray(75.0),
        crown_ratio=np.asarray(0.65),
        lean_direction=np.asarray(0.0),
        lean_severity=np.asarray(0.0),
        crown_radii=np.asarray((1.0, 1.0, 1.0, 1.0)),
        crown_edge_heights=np.asarray((0.3, 0.3, 0.3, 0.3)),
        crown_shapes=np.full((2, 4), 2.0),
    )

    trans1 = TreePose.from_tree(p1).t_global.as_array(axis=0)
    trans2 = TreePose.from_tree(p2).t_global.as_array(axis=0)

    assert not np.allclose(trans1[0], trans2[0])
    assert np.allclose(trans1[1], trans2[1])
    assert np.allclose(trans1[2], trans2[2])


def test_treetop_stem_y_isolation():
    """Changes in treetop stem_y alter expected coordinates describing crown."""
    stem1 = (0, 0, 0)
    stem2 = (0, 10, 0)

    p1 = CrownHullParams(
        stem_base=np.asarray(stem1, dtype=float),
        top_height=np.asarray(75.0),
        crown_ratio=np.asarray(0.65),
        lean_direction=np.asarray(0.0),
        lean_severity=np.asarray(0.0),
        crown_radii=np.asarray((1.0, 1.0, 1.0, 1.0)),
        crown_edge_heights=np.asarray((0.3, 0.3, 0.3, 0.3)),
        crown_shapes=np.full((2, 4), 2.0),
    )
    p2 = CrownHullParams(
        stem_base=np.asarray(stem2, dtype=float),
        top_height=np.asarray(75.0),
        crown_ratio=np.asarray(0.65),
        lean_direction=np.asarray(0.0),
        lean_severity=np.asarray(0.0),
        crown_radii=np.asarray((1.0, 1.0, 1.0, 1.0)),
        crown_edge_heights=np.asarray((0.3, 0.3, 0.3, 0.3)),
        crown_shapes=np.full((2, 4), 2.0),
    )

    trans1 = TreePose.from_tree(p1).t_global.as_array(axis=0)
    trans2 = TreePose.from_tree(p2).t_global.as_array(axis=0)

    assert np.allclose(trans1[0], trans2[0])
    assert not np.allclose(trans1[1], trans2[1])
    assert np.allclose(trans1[2], trans2[2])


def test_treetop_stem_z_isolation():
    """Changes in treetop stem_z alter expected coordinates describing crown."""
    stem1 = (0, 0, 0)
    stem2 = (0, 0, 1)

    p1 = CrownHullParams(
        stem_base=np.asarray(stem1, dtype=float),
        top_height=np.asarray(75.0),
        crown_ratio=np.asarray(0.65),
        lean_direction=np.asarray(0.0),
        lean_severity=np.asarray(0.0),
        crown_radii=np.asarray((1.0, 1.0, 1.0, 1.0)),
        crown_edge_heights=np.asarray((0.3, 0.3, 0.3, 0.3)),
        crown_shapes=np.full((2, 4), 2.0),
    )
    p2 = CrownHullParams(
        stem_base=np.asarray(stem2, dtype=float),
        top_height=np.asarray(75.0),
        crown_ratio=np.asarray(0.65),
        lean_direction=np.asarray(0.0),
        lean_severity=np.asarray(0.0),
        crown_radii=np.asarray((1.0, 1.0, 1.0, 1.0)),
        crown_edge_heights=np.asarray((0.3, 0.3, 0.3, 0.3)),
        crown_shapes=np.full((2, 4), 2.0),
    )

    trans1 = TreePose.from_tree(p1).t_global.as_array(axis=0)
    trans2 = TreePose.from_tree(p2).t_global.as_array(axis=0)

    assert np.allclose(trans1[0], trans2[0])
    assert np.allclose(trans1[1], trans2[1])
    assert not np.allclose(trans1[2], trans2[2])


def test_hull_apex_and_base_consistent():
    """Tree.crown() has the same apex and base as crown local geometry helpers."""
    tree = Tree(
        species="Douglas-fir",
        dbh=8.5,
        top_height=80,
        stem_x=0,
        stem_y=0,
        stem_z=0,
        crown_radii=(10, 10, 10, 10),
        crown_ratio=0.5,
    )

    params = CrownHullParams(
        stem_base=tree.stem_base,
        top_height=tree.top_height,
        crown_ratio=tree.crown_ratio,
        lean_direction=tree.lean_direction,
        lean_severity=tree.lean_severity,
        crown_radii=tree.crown_radii,
        crown_edge_heights=tree.crown_edge_heights,
        crown_shapes=tree.crown_shapes,
    )

    pose = TreePose.from_tree(params)
    apex_arr, base_arr = _hull_apex_and_base_local(
        crown_radii=params.crown_radii,
        top_height=params.top_height,
        crown_ratio=params.crown_ratio,
    )
    apex_local = Point3D.from_array(apex_arr, axis=0)
    base_local = Point3D.from_array(base_arr, axis=0)
    apex1 = np.asarray(
        (apex_local.as_array(axis=0) + pose.t_global.as_array(axis=0)).reshape((3,))
    )
    base1 = np.asarray(
        (base_local.as_array(axis=0) + pose.t_global.as_array(axis=0)).reshape((3,))
    )
    points = tree.crown()

    apex2 = (tree.stem_x, tree.stem_y, points[:, 2].max())
    base2 = (tree.stem_x, tree.stem_y, points[:, 2].min())

    assert np.allclose(apex1, apex2)
    assert np.allclose(base1, base2)


def test_hull_apex_y_offset_depends_on_northsouth_center_not_eastwest():
    """Apex y-offset should not change when only (E,W) radii change.

    This is a regression test for an implementation error where `center_x` was used
    in the apex y-offset term, coupling E/W asymmetry into a N/S shift.
    """
    common = dict(
        stem_base=np.asarray((0.0, 0.0, 0.0)),
        top_height=np.asarray(30.0),
        crown_ratio=np.asarray(0.6),
        lean_direction=np.asarray(0.0),
        lean_severity=np.asarray(0.0),
        crown_edge_heights=np.asarray((0.3, 0.3, 0.3, 0.3)),
        crown_shapes=np.full((2, 4), 2.0),
    )

    # Keep N/S symmetric so center_y == 0 and (N - S) == 0.
    p1 = CrownHullParams(crown_radii=np.asarray((1.0, 2.0, 3.0, 2.0)), **common)
    p2 = CrownHullParams(crown_radii=np.asarray((5.0, 2.0, 1.0, 2.0)), **common)

    a1, _ = _hull_apex_and_base_local(
        crown_radii=p1.crown_radii, top_height=p1.top_height, crown_ratio=p1.crown_ratio
    )
    a2, _ = _hull_apex_and_base_local(
        crown_radii=p2.crown_radii, top_height=p2.top_height, crown_ratio=p2.crown_ratio
    )
    a1 = Point3D.from_array(a1, axis=0)
    a2 = Point3D.from_array(a2, axis=0)

    assert np.allclose(np.asarray(a1.y), 0.0)
    assert np.allclose(np.asarray(a2.y), 0.0)
    assert np.allclose(np.asarray(a1.y), np.asarray(a2.y))


def test_hull_num_theta_and_num_z():
    """Providing different num_theta and num_z values alters shape of hull."""
    num_theta_1 = 32
    num_theta_2 = 64
    num_z_1 = 50
    num_z_2 = 100

    tree = Tree(
        species="Douglas-fir",
        dbh=8.5,
        top_height=80,
        stem_x=0,
        stem_y=0,
        stem_z=0,
        crown_radii=(10, 10, 10, 10),
        crown_ratio=0.5,
    )
    p1 = tree.crown(num_theta=num_theta_1, num_z=num_z_1)
    p2 = tree.crown(num_theta=num_theta_2, num_z=num_z_2)

    assert p1.shape == (num_z_1 * num_theta_1, 3)
    assert p2.shape == (num_z_2 * num_theta_2, 3)


def test_hull_jit_returns_finite_vertices_for_valid_inputs():
    params = CrownHullParams(
        stem_base=np.array([0.0, 0.0, 0.0]),
        top_height=30.0,
        crown_ratio=0.6,
        lean_direction=0.0,
        lean_severity=0.0,
        crown_radii=np.array([5.0, 4.0, 5.0, 4.0]),
        crown_edge_heights=np.array([0.2, 0.2, 0.2, 0.2]),
        crown_shapes=np.full((2, 4), 2.0),
    )
    pts = jit(lambda: make_crown_hull(params, num_theta=16, num_z=10))()
    assert np.isfinite(np.asarray(pts)).all()


def test_hull_jit_allows_edge_heights_at_zero_and_returns_finite_vertices():
    params = CrownHullParams(
        stem_base=np.array([0.0, 0.0, 0.0]),
        top_height=30.0,
        crown_ratio=0.6,
        lean_direction=0.0,
        lean_severity=0.0,
        crown_radii=np.array([5.0, 4.0, 5.0, 4.0]),
        crown_edge_heights=np.array([0.0, 0.2, 0.0, 0.2]),
        crown_shapes=np.full((2, 4), 2.0),
    )
    pts = jit(lambda: make_crown_hull(params, num_theta=16, num_z=10))()
    assert np.isfinite(np.asarray(pts)).all()

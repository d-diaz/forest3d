import numpy as np
from jax import jit

from forest3d.geometry.crown_hull import (
    _get_hull_apex_and_base,
    _get_treetop_location,
    _make_crown_hull,
)
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

    trans1 = _get_treetop_location(stem1, 75)
    trans2 = _get_treetop_location(stem2, 75)

    assert not np.allclose(trans1[0], trans2[0])
    assert np.allclose(trans1[1], trans2[1])
    assert np.allclose(trans1[2], trans2[2])


def test_treetop_stem_y_isolation():
    """Changes in treetop stem_y alter expected coordinates describing crown."""
    stem1 = (0, 0, 0)
    stem2 = (0, 10, 0)

    trans1 = _get_treetop_location(stem1, 75)
    trans2 = _get_treetop_location(stem2, 75)

    assert np.allclose(trans1[0], trans2[0])
    assert not np.allclose(trans1[1], trans2[1])
    assert np.allclose(trans1[2], trans2[2])


def test_treetop_stem_z_isolation():
    """Changes in treetop stem_z alter expected coordinates describing crown."""
    stem1 = (0, 0, 0)
    stem2 = (0, 0, 1)

    trans1 = _get_treetop_location(stem1, 75)
    trans2 = _get_treetop_location(stem2, 75)

    assert np.allclose(trans1[0], trans2[0])
    assert np.allclose(trans1[1], trans2[1])
    assert not np.allclose(trans1[2], trans2[2])


def test_hull_apex_and_base_consistent():
    """Tree.crown() has same apex and base as _get_hull_apex_and_base()."""
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

    apex1, base1 = _get_hull_apex_and_base(
        tree.crown_radii, tree.top_height, tree.crown_ratio
    )
    points = tree.crown()

    apex2 = (tree.stem_x, tree.stem_y, points[:, 2].max())
    base2 = (tree.stem_x, tree.stem_y, points[:, 2].min())

    assert np.allclose(apex1, apex2)
    assert np.allclose(base1, base2)


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
    params = dict(
        stem_base=np.array([0.0, 0.0, 0.0]),
        top_height=30.0,
        crown_ratio=0.6,
        lean_direction=0.0,
        lean_severity=0.0,
        crown_radii=np.array([5.0, 4.0, 5.0, 4.0]),
        crown_edge_heights=np.array([0.2, 0.2, 0.2, 0.2]),
        crown_shapes=np.full((2, 4), 2.0),
        num_theta=16,
        num_z=10,
    )
    pts = jit(lambda: _make_crown_hull(**params))()
    assert np.isfinite(np.asarray(pts)).all()


def test_hull_jit_allows_edge_heights_at_zero_and_returns_finite_vertices():
    params = dict(
        stem_base=np.array([0.0, 0.0, 0.0]),
        top_height=30.0,
        crown_ratio=0.6,
        lean_direction=0.0,
        lean_severity=0.0,
        crown_radii=np.array([5.0, 4.0, 5.0, 4.0]),
        crown_edge_heights=np.array([0.0, 0.2, 0.0, 0.2]),
        crown_shapes=np.full((2, 4), 2.0),
        num_theta=16,
        num_z=10,
    )
    pts = jit(lambda: _make_crown_hull(**params))()
    assert np.isfinite(np.asarray(pts)).all()

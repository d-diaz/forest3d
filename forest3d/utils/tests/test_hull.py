import numpy as np
from jax import jit

from forest3d.models.dataclass import Tree
from forest3d.utils.geometry import (
    _get_hull_apex_and_base,
    _get_treetop_location,
    _make_crown_hull,
)
from forest3d.utils.hull_checks import make_crown_hull_checked


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
    """Changes in treetop_stem_x alter expected coordinates describing crown."""

    stem1 = (0, 0, 0)
    stem2 = (10, 0, 0)

    trans1 = _get_treetop_location(stem1, 75)
    trans2 = _get_treetop_location(stem2, 75)

    assert not np.allclose(trans1[0], trans2[0])
    assert np.allclose(trans1[1], trans2[1])
    assert np.allclose(trans1[2], trans2[2])


def test_treetop_stem_y_isolation():
    """ "Changes in treetop_stem_y alter expected coordinates describing crown."""

    stem1 = (0, 0, 0)
    stem2 = (0, 10, 0)

    trans1 = _get_treetop_location(stem1, 75)
    trans2 = _get_treetop_location(stem2, 75)

    assert np.allclose(trans1[0], trans2[0])
    assert not np.allclose(trans1[1], trans2[1])
    assert np.allclose(trans1[2], trans2[2])


def test_treetop_stem_z_isolation():
    """Changes in treetop_stem_z alter expected coordinates describing crown."""

    stem1 = (0, 0, 0)
    stem2 = (0, 0, 1)

    trans1 = _get_treetop_location(stem1, 75)
    trans2 = _get_treetop_location(stem2, 75)

    assert np.allclose(trans1[0], trans2[0])
    assert np.allclose(trans1[1], trans2[1])
    assert not np.allclose(trans1[2], trans2[2])


def test_hull_apex_and_base_consistent():
    """make_hull() has same apex and base as _get_hull_apex_and_base()."""

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
    """providing different num_theta and num_z values alters shape of hull."""
    NUM_THETA_1 = 32
    NUM_THETA_2 = 64
    NUM_Z_1 = 50
    NUM_Z_2 = 100

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
    p1 = tree.crown(num_theta=NUM_THETA_1, num_z=NUM_Z_1)
    p2 = tree.crown(num_theta=NUM_THETA_2, num_z=NUM_Z_2)

    assert p1.shape == (NUM_Z_1 * NUM_THETA_1, 3)
    assert p2.shape == (NUM_Z_2 * NUM_THETA_2, 3)


def _checked_hull_err_message(**overrides) -> str | None:
    """Runs the checked hull under `jit` and returns the first error message (or None)."""
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
    params.update(overrides)

    err, _pts = jit(lambda: make_crown_hull_checked(**params))()
    return err.get()


def test_checked_hull_jit_errors_on_lean_90():
    msg = _checked_hull_err_message(lean_severity=90.0)
    assert msg is not None
    assert "lean_severity" in msg


def test_checked_hull_jit_errors_on_non_positive_shapes():
    msg = _checked_hull_err_message(crown_shapes=np.zeros((2, 4), dtype=float))
    assert msg is not None
    assert "crown_shapes" in msg


def test_checked_hull_jit_errors_on_edge_heights_at_one():
    msg = _checked_hull_err_message(
        crown_edge_heights=np.array([1.0, 0.2, 0.2, 0.2], dtype=float)
    )
    assert msg is not None
    assert "crown_edge_heights" in msg


def _checked_hull_run(**overrides):
    """Runs the checked hull under `jit` and returns (err, points)."""
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
    params.update(overrides)
    return jit(lambda: make_crown_hull_checked(**params))()


def test_checked_hull_jit_returns_finite_vertices_for_valid_inputs():
    err, pts = _checked_hull_run()
    assert err.get() is None
    assert np.isfinite(np.asarray(pts)).all()


def test_checked_hull_jit_allows_edge_heights_at_zero_and_returns_finite_vertices():
    # Edge heights of 0 are valid (max width at crown base). This case used to be a
    # source of NaNs due to evaluation of unused regions under tracing.
    err, pts = _checked_hull_run(crown_edge_heights=np.array([0.0, 0.2, 0.0, 0.2]))
    assert err.get() is None
    assert np.isfinite(np.asarray(pts)).all()


def test_checked_vs_unchecked_hull_outputs_match_for_valid_inputs():
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

    err, pts_checked = jit(lambda: make_crown_hull_checked(**params))()
    assert err.get() is None

    pts_unchecked = np.asarray(_make_crown_hull(**params))
    pts_checked_np = np.asarray(pts_checked)

    assert pts_checked_np.shape == pts_unchecked.shape
    # JIT + checkify instrumentation can slightly change operation ordering and
    # rounding behavior (especially near the apex where radii should be ~0). We
    # only require the checked path to be numerically consistent with the unchecked
    # path within a small tolerance.
    assert np.allclose(pts_checked_np, pts_unchecked, rtol=1e-5, atol=3e-3)

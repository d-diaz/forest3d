import jax.numpy as jnp
import pytest

from forest3d.geometry.kernels import polar_to_xy, rotate_xy, stem_xy_world

# ---------------------------------------------------------------------------
# rotate_xy
# ---------------------------------------------------------------------------


def test_rotate_xy_identity():
    x = jnp.array([1.0, 0.0, 3.0])
    y = jnp.array([0.0, 1.0, -2.0])
    xr, yr = rotate_xy(x=x, y=y, theta=0.0)
    assert jnp.allclose(xr, x)
    assert jnp.allclose(yr, y)


@pytest.mark.parametrize(
    "x,y,expected_x,expected_y",
    [
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, -1.0, 0.0),
        (1.0, 1.0, -1.0, 1.0),
    ],
)
def test_rotate_xy_pi_half(x, y, expected_x, expected_y):
    xr, yr = rotate_xy(x=x, y=y, theta=jnp.pi / 2)
    assert jnp.allclose(xr, expected_x, atol=1e-6)
    assert jnp.allclose(yr, expected_y, atol=1e-6)


def test_rotate_xy_pi():
    xr, yr = rotate_xy(x=2.0, y=0.0, theta=jnp.pi)
    assert jnp.allclose(xr, -2.0, atol=1e-6)
    assert jnp.allclose(yr, 0.0, atol=1e-6)


def test_rotate_xy_batched_theta_broadcasts():
    """Per-element rotation angles with shape (B,)."""
    x = jnp.array([1.0, 1.0, 1.0])
    y = jnp.array([0.0, 0.0, 0.0])
    theta = jnp.array([0.0, jnp.pi / 2, jnp.pi])
    xr, yr = rotate_xy(x=x, y=y, theta=theta)
    expected_xr, expected_yr = jnp.array([1.0, 0.0, -1.0]), jnp.array([0.0, 1.0, 0.0])
    assert jnp.allclose(xr, expected_xr, atol=1e-6)
    assert jnp.allclose(yr, expected_yr, atol=1e-6)


# ---------------------------------------------------------------------------
# stem_xy_world
# ---------------------------------------------------------------------------


def test_stem_xy_world_theta_zero_matches_add():
    """theta=0 reduces to center + local + offset (no rotation)."""
    cx, cy = 100.0, 200.0
    lx = jnp.array([1.0, -2.0, 0.5])
    ly = jnp.array([3.0, 0.0, -1.0])
    ox = jnp.array([0.1, 0.2, 0.3])
    oy = jnp.array([-0.1, 0.4, 0.0])
    sx, sy = stem_xy_world(
        center_x=cx,
        center_y=cy,
        local_x=lx,
        local_y=ly,
        offset_x=ox,
        offset_y=oy,
        theta=0.0,
    )
    expected_sx, expected_sy = cx + lx + ox, cy + ly + oy
    assert jnp.allclose(sx, expected_sx, atol=1e-6)
    assert jnp.allclose(sy, expected_sy, atol=1e-6)


def test_stem_xy_world_theta_pi_half():
    """Single tree at (1, 0) local+offset rotated 90 deg CCW -> (0, 1)."""
    sx, sy = stem_xy_world(
        center_x=10.0,
        center_y=20.0,
        local_x=1.0,
        local_y=0.0,
        offset_x=0.0,
        offset_y=0.0,
        theta=jnp.pi / 2,
    )
    expected_sx, expected_sy = 10.0, 21.0
    assert jnp.allclose(sx, expected_sx, atol=1e-6)
    assert jnp.allclose(sy, expected_sy, atol=1e-6)


def test_stem_xy_world_batched():
    """Multiple trees with scalar theta."""
    lx = jnp.array([1.0, 0.0])
    ly = jnp.array([0.0, 1.0])
    sx, sy = stem_xy_world(
        center_x=0.0,
        center_y=0.0,
        local_x=lx,
        local_y=ly,
        offset_x=0.0,
        offset_y=0.0,
        theta=jnp.pi,
    )
    expected_sx, expected_sy = jnp.array([-1.0, 0.0]), jnp.array([0.0, -1.0])
    assert jnp.allclose(sx, expected_sx, atol=1e-6)
    assert jnp.allclose(sy, expected_sy, atol=1e-6)


# ---------------------------------------------------------------------------
# polar_to_xy
# ---------------------------------------------------------------------------


def test_polar_to_xy_basic():
    x, y = polar_to_xy(radii=1.0, theta=0.0, center_x=0.0, center_y=0.0)
    assert jnp.allclose(x, 1.0)
    assert jnp.allclose(y, 0.0)


def test_polar_to_xy_with_center():
    x, y = polar_to_xy(radii=1.0, theta=jnp.pi / 2, center_x=5.0, center_y=3.0)
    expected_x, expected_y = 5.0, 4.0
    assert jnp.allclose(x, expected_x, atol=1e-6)
    assert jnp.allclose(y, expected_y, atol=1e-6)


def test_polar_to_xy_batched():
    radii = jnp.array([1.0, 2.0])
    theta = jnp.array([0.0, jnp.pi])
    x, y = polar_to_xy(radii=radii, theta=theta, center_x=0.0, center_y=0.0)
    expected_x, expected_y = jnp.array([1.0, -2.0]), jnp.array([0.0, 0.0])
    assert jnp.allclose(x, expected_x, atol=1e-6)
    assert jnp.allclose(y, expected_y, atol=1e-6)

import jax.numpy as jnp
import pytest

from forest3d.geometry.kernels import polar_to_xy, rotate_xy


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

import jax.numpy as jnp

from forest3d.geometry.kernels import polar_to_xy


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

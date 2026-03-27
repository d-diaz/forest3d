import jax
import jax.numpy as jnp
import numpy as np

from forest3d.simulation.distributions import UniformDisk2D


def test_uniform_disk2d_shape_semantics():
    B = 4
    S = 3
    d = UniformDisk2D(
        center_x=jnp.asarray(0.0, dtype=jnp.float32),
        center_y=jnp.asarray(0.0, dtype=jnp.float32),
        radius=jnp.asarray(2.0, dtype=jnp.float32),
        batch_shape=(B,),
    )

    assert d.batch_shape == (B,)
    assert d.event_shape == (2,)

    s0 = d.sample(jax.random.PRNGKey(0))
    assert np.asarray(s0).shape == (B, 2)

    s = d.sample(jax.random.PRNGKey(1), sample_shape=(S,))
    assert np.asarray(s).shape == (S, B, 2)

    lp = d.log_prob(s)
    assert np.asarray(lp).shape == (S, B)


def test_uniform_disk2d_samples_within_support():
    cx = jnp.asarray(1.25, dtype=jnp.float32)
    cy = jnp.asarray(-0.5, dtype=jnp.float32)
    r = jnp.asarray(3.0, dtype=jnp.float32)
    d = UniformDisk2D(center_x=cx, center_y=cy, radius=r, batch_shape=())

    s = d.sample(jax.random.PRNGKey(0), sample_shape=(10_000,))
    s_np = np.asarray(s)
    dx = s_np[:, 0] - float(cx)
    dy = s_np[:, 1] - float(cy)
    rho2 = dx * dx + dy * dy
    assert np.all(rho2 <= float(r * r) + 1e-5)


def test_uniform_disk2d_log_prob_constant_inside_and_neg_inf_outside():
    cx = jnp.asarray(0.0, dtype=jnp.float32)
    cy = jnp.asarray(0.0, dtype=jnp.float32)
    r = jnp.asarray(2.0, dtype=jnp.float32)
    d = UniformDisk2D(center_x=cx, center_y=cy, radius=r, batch_shape=())

    inside = jnp.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=jnp.float32)
    outside = jnp.asarray([[3.0, 0.0], [0.0, -3.0]], dtype=jnp.float32)

    lp_in = np.asarray(d.log_prob(inside))
    lp_out = np.asarray(d.log_prob(outside))

    assert np.isfinite(lp_in).all()
    assert np.allclose(lp_in[0], lp_in[1])
    assert np.isneginf(lp_out).all()


def test_uniform_disk2d_broadcasting_centers():
    B = 5
    S = 7
    cx = jnp.linspace(-1.0, 1.0, B, dtype=jnp.float32)
    cy = jnp.linspace(2.0, 3.0, B, dtype=jnp.float32)
    r = jnp.asarray(1.5, dtype=jnp.float32)

    d = UniformDisk2D(center_x=cx, center_y=cy, radius=r, batch_shape=(B,))
    s = d.sample(jax.random.PRNGKey(0), sample_shape=(S,))
    assert np.asarray(s).shape == (S, B, 2)
    lp = d.log_prob(s)
    assert np.asarray(lp).shape == (S, B)


def test_uniform_disk2d_mc_sanity_mean_and_r2_moment():
    # For a uniform disk of radius R centered at (cx, cy):
    #   E[x] = cx, E[y] = cy, and E[rho^2] = R^2 / 2.
    N = 50_000
    cx = jnp.asarray(0.7, dtype=jnp.float32)
    cy = jnp.asarray(-1.3, dtype=jnp.float32)
    r = jnp.asarray(2.0, dtype=jnp.float32)
    d = UniformDisk2D(center_x=cx, center_y=cy, radius=r, batch_shape=())

    s = d.sample(jax.random.PRNGKey(0), sample_shape=(N,))
    x = s[:, 0]
    y = s[:, 1]
    mx = float(jnp.mean(x))
    my = float(jnp.mean(y))
    rho2 = (x - cx) ** 2 + (y - cy) ** 2
    m_rho2 = float(jnp.mean(rho2))

    # Loose tolerances for test stability.
    assert abs(mx - float(cx)) < 0.03
    assert abs(my - float(cy)) < 0.03
    assert abs(m_rho2 - float(r * r / 2.0)) < 0.05

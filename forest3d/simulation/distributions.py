"""Custom NumPyro distribution helpers for forest3d simulation models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.distributions import constraints


class UniformDisk2D(dist.Distribution):
    """Uniform distribution over a 2D disk (event shape (2,))."""

    arg_constraints = {
        "center_x": constraints.real,
        "center_y": constraints.real,
        "radius": constraints.positive,
    }
    support = constraints.real_vector

    def __init__(
        self,
        center_x: ArrayLike,
        center_y: ArrayLike,
        radius: ArrayLike,
        *,
        batch_shape: tuple[int, ...] = (),
        validate_args: bool | None = None,
    ):
        self.center_x = jnp.asarray(center_x)
        self.center_y = jnp.asarray(center_y)
        self.radius = jnp.asarray(radius)
        super().__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        key_u, key_t = jax.random.split(key)
        shape = sample_shape + self.batch_shape
        u = jax.random.uniform(key_u, shape=shape, minval=0.0, maxval=1.0)
        theta = jax.random.uniform(key_t, shape=shape, minval=-jnp.pi, maxval=jnp.pi)
        r = jnp.sqrt(u) * self.radius
        x = self.center_x + r * jnp.cos(theta)
        y = self.center_y + r * jnp.sin(theta)
        return jnp.stack((x, y), axis=-1)

    def log_prob(self, value):
        v = jnp.asarray(value)
        dx = v[..., 0] - self.center_x
        dy = v[..., 1] - self.center_y
        inside = (dx * dx + dy * dy) <= (self.radius * self.radius)
        log_area = jnp.log(jnp.pi * self.radius * self.radius)
        neg_inf = jnp.asarray(-jnp.inf, dtype=v.dtype)
        return jnp.where(inside, -log_area, neg_inf)

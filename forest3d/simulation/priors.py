"""Prior specifications for NumPyro simulation/optimization models.

These dataclasses are *prior specs* (hyperparameters), not NumPyro distributions.
Model code should convert them into `numpyro.distributions.*` at sampling time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import numpyro
import numpyro.distributions as dist
from jax import numpy as jnp
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.infer.reparam import CircularReparam, Reparam

from forest3d.simulation.distributions import UniformDisk2D


@dataclass(frozen=True)
class BasePrior(ABC):
    """Base class for all prior specifications.

    This class is intended to be used for creating Prior classes that allow for easily
    generating NumPyro distributions and sample sites from distribution parameters which
    can be provided as scalars or arrays. These parameters are expected to be static
    configuration settings for the instances of BasePrior subclasses.

    Each subclass should define class attributes for each of the parameters used to
    instantiate the `_base_dist`, a NumPyro distribution class that must be defined as
    a property of the subclass. The class attribute parameters should be returned as a
    dict from a `params` property of the subclass (which must also be defined).

    An optional `_reparam` class attribute can be employed to specify a default
    reparameterization for the prior. If provided, it will be used to reparameterize
    the sample site unless overridden by a `reparam` argument to `sample`.
    """

    _reparam: Reparam | None = field(default=None, kw_only=True)

    @property
    @abstractmethod
    def _base_dist(self) -> Callable[..., dist.Distribution]:
        """The base distribution for the prior class."""
        ...

    @property
    @abstractmethod
    def params(self) -> dict[str, ArrayLike]:
        """Parameters for instantiating the base distribution."""
        ...

    def dist(self, batch_shape: tuple[int, ...] = ()) -> dist.Distribution:
        """Generate an instance of the base distribution.

        The instance may also be useful for extracting attributes from the instantiated
        distribution, such as the mean, variance, etc. (e.g., `self.dist().mean`)

        Args:
            batch_shape(tuple[int, ...]): Optional batch shape to invoke when creating
                the distribution. Defaults to an empty tuple, which means no batch
                dimension will be prepended.
        """
        base = self._base_dist(**self.params)
        if batch_shape:
            return base.expand(batch_shape)
        return base

    def sample(
        self,
        name: str,
        batch_shape: tuple[int, ...] = (),
        sample_shape: tuple[int, ...] = (),
        reparam: Reparam | None = None,
    ) -> ArrayLike:
        """Generate a NumPyro sample site for this prior.

        If you want to generate samples from the base distribution (e.g., for testing
        or development), you should use the `self.dist().sample(rng_key)` method, which
        does not require a name, and which does require a random key.

        Args:
            name (str): Name of the sample site.
            batch_shape (tuple[int, ...]): Optional batch shape. Defaults to an empty
                tuple, which means a no batch dimension will be prepended.
            sample_shape (tuple[int, ...]): Optional shape of the sample. Defaults to
                an empty tuple, which means a single sample is generated.
            reparam (Reparam | None): Optional reparameterization to use for this
                sample site. Call-site `reparam` takes precedence over `_reparam`.

        Returns:
            A NumPyro sample site.
        """

        config: dict[str, Reparam] = {}
        if self._reparam is not None:
            config[name] = self._reparam
        if reparam is not None:
            config[name] = reparam  # call-site overrides defaults

        if config:
            with handlers.reparam(config=config):
                return numpyro.sample(
                    name=name,
                    fn=self.dist(batch_shape=batch_shape),
                    sample_shape=sample_shape,
                )

        return numpyro.sample(
            name=name,
            fn=self.dist(batch_shape=batch_shape),
            sample_shape=sample_shape,
        )

    def __post_init__(self):
        """Validate the args that will be used to instantiate the base distribution.

        This should catch errors in parameter support, like passing negative values for
        a scale parameter.
        """
        self.dist().validate_args()


@dataclass(frozen=True, slots=True)
class NormalPrior(BasePrior):
    """Normal prior.

    Used for Gaussian priors, can be used for scalar or vector-valued `loc` parameters
    to represent single or multi-dimensional Gaussian distributions. For example,
    providing a scalar `loc` and `scale` will create a univariate Normal distribution,
    while providing a vector `loc` and scalar `scale` will create a multivariate Normal
    distribution.

    Args:
        loc: Location parameter(s) of the Normal distribution.
        scale: Scale parameter(s) of the Normal distribution.
    """

    loc: ArrayLike = 0
    scale: ArrayLike = 1

    @property
    def _base_dist(self) -> Callable[..., dist.Distribution]:
        return dist.Normal

    @property
    def params(self) -> dict[str, ArrayLike]:
        return {"loc": jnp.asarray(self.loc), "scale": jnp.asarray(self.scale)}


@dataclass(frozen=True, slots=True)
class UniformDisk2DPrior(BasePrior):
    """Uniform prior on a coordinate in 2D within a disk."""

    center_x: ArrayLike = 0
    center_y: ArrayLike = 0
    radius: ArrayLike = 1

    @property
    def _base_dist(self) -> Callable[..., dist.Distribution]:
        return UniformDisk2D

    @property
    def params(self) -> dict[str, ArrayLike]:
        return {
            "center_x": jnp.asarray(self.center_x),
            "center_y": jnp.asarray(self.center_y),
            "radius": jnp.asarray(self.radius),
        }


@dataclass(frozen=True, slots=True)
class BearingPrior(BasePrior):
    """VonMises prior for a circular bearing (radians).

    Args:
        loc: Location parameter(s) of the VonMises distribution in radians.
        scale: Scale parameter(s) of the VonMises distribution in radians.
        _reparam: Optional reparameterization to use for the sample site. Defaults to
            `CircularReparam`.
    """

    loc: ArrayLike = 0.0
    scale: ArrayLike = 0.35  # heuristic small-angle scale
    _reparam: Reparam | None = CircularReparam()

    @property
    def _base_dist(self) -> Callable[..., dist.Distribution]:
        return dist.VonMises

    @property
    def concentration(self) -> ArrayLike:
        return 1.0 / jnp.asarray(self.scale)

    @property
    def params(self) -> dict[str, ArrayLike]:
        loc = jnp.asarray(self.loc)
        return {"loc": loc, "concentration": self.concentration}

    @staticmethod
    def from_degrees(
        *,
        loc_deg: ArrayLike = 0.0,
        scale_deg: ArrayLike = 20.0,
    ) -> BearingPrior:
        """Construct a bearing prior from degrees from north, with 0/360 = north.

        This follows the common convention used in US Forest Inventory and Analysis
        (FIA) field methods where azimuth of a tree from plot center is measured with
        0/360 = north, 90 = east, 180 = south, 270 = west.

        The rest of the codebase uses the convention: 0 radians = +x (east),
        pi/2 = +y (north). Therefore we map: bearing_rad = deg2rad(90 - bearing_deg)

        Args:
            loc_deg: Location parameter(s) of the VonMises distribution in degrees from
                north.
            scale_deg: Scale parameter(s) of the VonMises distribution in degrees.
                Represents the standard deviation of the bearing in degrees (e.g., to
                reflect measurement error).

        Raises:
            ValueError: If `loc_deg` is not in [0, 360] or `scale_deg` is not > 0.
        """
        if jnp.any(jnp.asarray(loc_deg) < 0) or jnp.any(jnp.asarray(loc_deg) > 360):
            raise ValueError("loc_deg must be in [0, 360]")
        loc_rad = jnp.deg2rad(jnp.asarray(90.0) - jnp.asarray(loc_deg))
        scale_rad = jnp.deg2rad(jnp.asarray(scale_deg))
        return BearingPrior(loc=loc_rad, scale=scale_rad)

    def __post_init__(self):
        """Validate the args that will be used to instantiate the base distribution.

        Extends base class validation to include checking that `loc` is in [-pi, pi].
        """
        if jnp.any(jnp.asarray(self.loc) < -jnp.pi) or jnp.any(
            jnp.asarray(self.loc) > jnp.pi
        ):
            raise ValueError("loc must be in [-pi, pi]")
        self.dist().validate_args()


@dataclass(frozen=True, slots=True)
class LogNormalPrior(BasePrior):
    """LogNormal prior."""

    loc: ArrayLike = 1.0
    scale: ArrayLike = 0.1

    @property
    def _base_dist(self) -> Callable[..., dist.Distribution]:
        return dist.LogNormal

    @property
    def params(self) -> dict[str, ArrayLike]:
        return {"loc": jnp.asarray(self.loc), "scale": jnp.asarray(self.scale)}

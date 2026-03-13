import jax.numpy as jnp
import numpy as np
import pytest
from jax.random import PRNGKey
from numpyro import handlers
from numpyro.infer.reparam import LocScaleReparam

from forest3d.simulation.priors import (
    BearingPrior,
    LogNormalPrior,
    NormalPrior,
    UniformDisk2DPrior,
)

ALL_PRIORS = [NormalPrior, BearingPrior, LogNormalPrior, UniformDisk2DPrior]


def test_normal_prior_raises_on_invalid_support():
    with pytest.raises(ValueError):
        NormalPrior(loc=0, scale=-1)  # scale must be > 0


def test_lognormal_prior_raises_on_invalid_support():
    with pytest.raises(ValueError):
        LogNormalPrior(loc=0, scale=-1)  # scale must be > 0


def test_bearing_prior_raises_on_invalid_support():
    with pytest.raises(ValueError):
        BearingPrior(loc=0, scale=-1)  # scale must be > 0
    with pytest.raises(ValueError):
        BearingPrior(loc=-5, scale=1)  # loc must be in [-pi, pi]
    with pytest.raises(ValueError):
        BearingPrior.from_degrees(
            loc_deg=-5, scale_deg=1
        )  # loc_deg must be in [0, 360]
    with pytest.raises(ValueError):
        BearingPrior.from_degrees(
            loc_deg=365, scale_deg=1
        )  # loc_deg must be in [0, 360]
    with pytest.raises(ValueError):
        BearingPrior.from_degrees(loc_deg=0, scale_deg=-1)  # scale_deg must be > 0


def test_uniform_disk_prior_raises_on_invalid_support():
    with pytest.raises(ValueError):
        UniformDisk2DPrior(center_x=0, center_y=10, radius=-1)  # radius must be > 0


@pytest.mark.parametrize("prior", ALL_PRIORS)
def test_all_prior_params_exist_as_prior_attributes(prior):
    p = prior()
    params = p.params
    for param in params:
        assert hasattr(p, param)


def test_prior_builds_distribution_with_expected_moments():
    prior = NormalPrior(loc=0, scale=2)
    expected_mean = 0.0
    expected_variance = 4.0
    d = prior.dist()
    assert np.isclose(d.mean, expected_mean)
    assert np.isclose(d.variance, expected_variance)


def test_normalprior_to_dist_shapes():
    prior = NormalPrior(
        loc=np.array([0.0, 0.0], dtype=np.float32),
        scale=np.float32(1.0),
    )
    d = prior.dist()
    s = np.asarray(d.sample(PRNGKey(0)))
    assert s.shape == (2,)


def test_uniform_disk_prior_to_dist_support():
    prior = UniformDisk2DPrior(
        center_x=np.float32(0.0),
        center_y=np.float32(0.0),
        radius=np.float32(1.0),
    )
    d = prior.dist()
    s = np.asarray(d.sample(PRNGKey(0)))
    assert s.shape == (2,)
    assert (s[0] ** 2 + s[1] ** 2) <= 1.0 + 1e-6


def test_bearing_prior_batch_shape_and_support():
    prior = BearingPrior(loc=np.float32(0.0), scale=np.float32(0.5))
    s = np.asarray(prior.dist(batch_shape=(3,)).sample(PRNGKey(0)))
    assert s.shape == (3,)
    assert np.all(s >= -np.pi) and np.all(s <= np.pi)


def test_bearing_prior_from_degrees_north_is_pi_over_2():
    prior = BearingPrior.from_degrees(
        loc_deg=np.float32(0.0),
        scale_deg=np.float32(10.0),
    )
    assert np.isclose(np.asarray(prior.loc), np.pi / 2)
    assert np.isclose(np.asarray(prior.scale), np.deg2rad(10.0))


def test_lognormal_prior_batch_shape_and_support():
    prior = LogNormalPrior(loc=np.float32(1.0), scale=np.float32(0.25))
    s = np.asarray(prior.dist(batch_shape=(3,)).sample(PRNGKey(0)))
    assert s.shape == (3,)
    assert np.all(s > 0.0)


def test_baseprior_sample_accepts_reparam_object():
    def model_obj():
        NormalPrior(loc=np.float32(0.0), scale=np.float32(1.0)).sample(
            "theta", reparam=LocScaleReparam()
        )

    t1 = handlers.trace(handlers.seed(model_obj, PRNGKey(0))).get_trace()
    assert "theta" in t1


def test_bearing_prior_default_reparam_creates_aux_site_and_support():
    prior = BearingPrior(loc=np.float32(0.0), scale=np.float32(0.5))

    def model():
        prior.sample("tree_bearing", batch_shape=(3,))

    # CircularReparam uses an ImproperUniform aux site; substitute a value so
    # `seed+trace` doesn't attempt to sample it.
    seeded = handlers.seed(model, PRNGKey(0))
    subbed = handlers.substitute(
        seeded,
        data={"tree_bearing_unwrapped": jnp.zeros((3,), dtype=jnp.float32)},
    )
    tr = handlers.trace(subbed).get_trace()

    # Default circular reparam introduces an auxiliary site and makes the
    # original site deterministic.
    assert "tree_bearing_unwrapped" in tr
    assert tr["tree_bearing_unwrapped"]["type"] == "sample"
    assert tr["tree_bearing"]["type"] == "deterministic"

    bearing = np.asarray(tr["tree_bearing"]["value"])
    assert bearing.shape == (3,)
    assert np.all(bearing >= -np.pi) and np.all(bearing <= np.pi)


def test_normal_prior_reparam_creates_aux_site_and_samples_in_support():
    prior = NormalPrior(loc=np.float32(0.0), scale=np.float32(1.0))

    def model():
        prior.sample("theta", reparam=LocScaleReparam())

    tr = handlers.trace(handlers.seed(model, PRNGKey(0))).get_trace()

    # LocScaleReparam introduces an auxiliary Normal site and makes the
    # original site deterministic.
    assert "theta_decentered" in tr
    assert tr["theta_decentered"]["type"] == "sample"
    assert tr["theta"]["type"] == "deterministic"

    # The deterministic sample should remain in the expected support
    # (real line for Normal).
    theta = np.asarray(tr["theta"]["value"])
    assert np.isfinite(theta).all()

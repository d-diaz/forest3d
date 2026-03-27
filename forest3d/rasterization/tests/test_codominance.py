import jax.numpy as jnp
import pytest

from forest3d.rasterization.dsm import is_codominant_from_dsm

# A simple 3x3 DSM for most tests.
#   row 0: [10, 12, 10]
#   row 1: [10, 15, 10]
#   row 2: [10, 10,  8]
DSM_3x3 = jnp.array(
    [
        [10.0, 12.0, 10.0],
        [10.0, 15.0, 10.0],
        [10.0, 10.0, 8.0],
    ]
)


def test_codominant_match():
    """Tree apex z equals DSM value at its cell -> codominant."""
    result = is_codominant_from_dsm(
        dsm=DSM_3x3,
        i=jnp.array([1]),
        j=jnp.array([1]),
        z_apex=jnp.array([15.0]),
        epsilon=0.5,
    )
    assert result.shape == (1,)
    assert result[0]


def test_overtopped():
    """Neighbor crown raises DSM above tree's apex z -> not codominant."""
    result = is_codominant_from_dsm(
        dsm=DSM_3x3,
        i=jnp.array([1]),
        j=jnp.array([1]),
        z_apex=jnp.array([12.0]),
        epsilon=0.5,
    )
    assert not result[0]


def test_apex_above_dsm():
    """Tree apex above DSM (e.g., lidar underestimation) -> still codominant."""
    result = is_codominant_from_dsm(
        dsm=DSM_3x3,
        i=jnp.array([2]),
        j=jnp.array([2]),
        z_apex=jnp.array([20.0]),
        epsilon=0.5,
    )
    assert result[0]


@pytest.mark.parametrize(
    "z_apex,expected",
    [
        (14.5, True),  # DSM is 0.5 above apex — exactly at epsilon boundary
        (14.49, False),  # DSM is 0.51 above apex — just beyond epsilon
        (15.5, True),  # apex 0.5 above DSM — always codominant (one-sided)
        (100.0, True),  # apex far above DSM — still codominant
    ],
)
def test_epsilon_boundary(z_apex, expected):
    """Boundary behaviour around the epsilon tolerance (one-sided)."""
    result = is_codominant_from_dsm(
        dsm=DSM_3x3,
        i=jnp.array([1]),
        j=jnp.array([1]),
        z_apex=jnp.array([z_apex]),
        epsilon=0.5,
    )
    assert result[0] == expected


@pytest.mark.parametrize(
    "i,j",
    [
        (-1, 0),  # negative row
        (0, -1),  # negative column
        (3, 0),  # row >= ny
        (0, 3),  # column >= nx
        (3, 3),  # both out of bounds
    ],
)
def test_out_of_bounds(i, j):
    """Trees with OOB indices are marked not codominant."""
    result = is_codominant_from_dsm(
        dsm=DSM_3x3,
        i=jnp.array([i]),
        j=jnp.array([j]),
        z_apex=jnp.array([10.0]),
        epsilon=1.0,
    )
    assert not result[0]


def test_batched_mixed():
    """B=3 trees: codominant, overtopped, and OOB -> correct shape and values."""
    result = is_codominant_from_dsm(
        dsm=DSM_3x3,
        i=jnp.array([0, 1, 5]),
        j=jnp.array([0, 1, 0]),
        z_apex=jnp.array([10.0, 12.0, 10.0]),
        epsilon=0.5,
    )
    assert result.shape == (3,)
    expected = jnp.array([True, False, False])
    assert jnp.array_equal(result, expected)

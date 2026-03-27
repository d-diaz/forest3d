import numpy as np
from jax import Array, jit

from forest3d.geospatial.coordinates import CoordinateSystem
from forest3d.rasterization.dsm import make_dsm


def test_make_dsm_max_z_per_pixel():
    cs = CoordinateSystem.from_bounds(
        xmin=0.0,
        ymin=0.0,
        zmin=0.0,
        xmax=2.0,
        ymax=2.0,
        zmax=1.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )
    pts = np.array(
        [
            [0.25, 1.75, 5.0],  # row 0, col 0
            [0.25, 1.75, 7.0],  # same pixel, larger z
            [1.25, 0.25, 3.0],  # row 1, col 1
        ],
        dtype=float,
    )
    raster = make_dsm(pts, cs=cs, fill_value=-1.0)
    assert isinstance(raster, Array)
    assert raster.shape == (2, 2)
    np.testing.assert_allclose(
        np.asarray(raster),
        np.array(
            [
                [7.0, -1.0],
                [-1.0, 3.0],
            ],
            dtype=float,
        ),
    )


def test_make_dsm_ignores_out_of_bounds_points():
    cs = CoordinateSystem.from_bounds(
        xmin=0.0,
        ymin=0.0,
        zmin=0.0,
        xmax=2.0,
        ymax=2.0,
        zmax=1.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )
    pts = np.array(
        [
            [0.25, 1.75, 5.0],  # in
            [2.0, 1.0, 100.0],  # x==xmax => out (half-open)
            [1.0, 0.0, 50.0],  # y==ymin => out (half-open)
        ],
        dtype=float,
    )
    raster = make_dsm(pts, cs=cs, fill_value=-1.0)
    np.testing.assert_allclose(np.asarray(raster)[0, 0], 5.0)
    assert (np.asarray(raster) <= 5.0).all()


def test_make_dsm_batched_input_equivalent_to_concatenated():
    cs = CoordinateSystem.from_bounds(
        xmin=0.0,
        ymin=0.0,
        zmin=0.0,
        xmax=2.0,
        ymax=2.0,
        zmax=1.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )
    pts_batched = np.array(
        [
            [
                [0.25, 1.75, 5.0],
                [1.25, 1.75, 6.0],
            ],
            [
                [0.25, 0.25, 1.0],
                [1.25, 0.25, 9.0],
            ],
        ],
        dtype=float,
    )  # (B=2, N=2, 3)
    pts_concat = pts_batched.reshape((-1, 3))

    r1 = make_dsm(pts_batched, cs=cs, fill_value=-1.0)
    r2 = make_dsm(pts_concat, cs=cs, fill_value=-1.0)
    np.testing.assert_allclose(np.asarray(r1), np.asarray(r2))


def test_make_dsm_jittable():
    cs = CoordinateSystem.from_bounds(
        xmin=0.0,
        ymin=0.0,
        zmin=0.0,
        xmax=2.0,
        ymax=2.0,
        zmax=1.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )
    pts = np.array([[0.25, 1.75, 5.0], [1.25, 0.25, 3.0]], dtype=float)

    f = jit(lambda p: make_dsm(p, cs=cs, fill_value=-1.0))
    out = f(pts)
    assert out.shape == (2, 2)


def test_make_dsm_ignores_non_finite_points():
    cs = CoordinateSystem.from_bounds(
        xmin=0.0,
        ymin=0.0,
        zmin=0.0,
        xmax=2.0,
        ymax=2.0,
        zmax=1.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )
    pts = np.array(
        [
            [0.25, 1.75, 5.0],  # valid -> row 0 col 0
            [np.nan, 1.75, 9.0],  # invalid x
            [0.25, np.inf, 9.0],  # invalid y
            [0.25, 1.75, np.nan],  # invalid z
        ],
        dtype=float,
    )
    raster = make_dsm(pts, cs=cs, fill_value=-1.0)
    assert np.asarray(raster)[0, 0] == 5.0
    assert (np.asarray(raster) >= -1.0).all()


def test_make_dsm_preserves_float32_dtype_when_possible():
    # Guard against accidental dtype promotion:
    # `make_dsm` uses -inf as a sentinel during the max-reduction, then replaces
    # sentinel pixels with `fill_value`. If `fill_value` (or the accumulator) ends
    # up as float64, the output raster can silently upcast even when point z-values
    # are float32. This test ensures float32 in -> float32 out when possible.
    cs = CoordinateSystem.from_bounds(
        xmin=0.0,
        ymin=0.0,
        zmin=0.0,
        xmax=2.0,
        ymax=2.0,
        zmax=1.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )
    pts = np.array([[0.25, 1.75, 5.0]], dtype=np.float32)
    raster = make_dsm(pts, cs=cs, fill_value=np.float32(-1.0))
    assert np.asarray(raster).dtype == np.float32

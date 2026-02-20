import numpy as np
import pytest
from jax import jit

from forest3d.distance.interpolator import VoxelGridInterpolator


def test_voxel_grid_interpolator_trilinear_exact_for_affine_function():
    # For an affine function in x,y,z, trilinear interpolation is exact.
    x = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([10.0, 11.0, 12.0], dtype=float)
    z = np.array([100.0, 101.0, 102.0], dtype=float)

    # f(x,y,z) = ax + by + cz + d
    a, b, c, d = 1.25, -2.0, 0.5, 7.0
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    values = (a * xx + b * yy + c * zz + d).astype(np.float32)

    rgi = VoxelGridInterpolator((x, y, z), values)

    pts = np.array(
        [
            [0.25, 10.5, 100.0],
            [1.5, 11.25, 101.75],
            [2.0, 12.0, 102.0],
        ],
        dtype=float,
    )
    out = np.asarray(rgi(pts))
    expected = a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_voxel_grid_interpolator_handles_exact_domain_boundaries():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    z = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    # Make values equal to x+y+z on the grid; trilinear should be exact.
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    values = (xx + yy + zz).astype(np.float32)
    rgi = VoxelGridInterpolator((x, y, z), values)

    pts = np.array(
        [
            [0.0, 0.0, 0.0],  # xmin,ymin,zmin corner
            [2.0, 0.0, 0.0],  # xmax face
            [0.0, 2.0, 0.0],  # ymax face
            [0.0, 0.0, 2.0],  # zmax face
            [2.0, 2.0, 2.0],  # xmax,ymax,zmax corner
        ],
        dtype=np.float32,
    )
    out = np.asarray(rgi(pts))
    np.testing.assert_allclose(out, pts.sum(axis=1), rtol=0, atol=0)


def test_voxel_grid_interpolator_out_of_bounds_adds_euclidean_distance():
    x = np.array([0.0, 1.0], dtype=float)
    y = np.array([0.0, 1.0], dtype=float)
    z = np.array([0.0, 1.0], dtype=float)
    values = np.zeros((2, 2, 2), dtype=np.float32)

    rgi = VoxelGridInterpolator((x, y, z), values)

    pts = np.array(
        [
            [-1.0, 0.5, 0.5],  # 1m outside in x
            [0.5, 2.0, 0.5],  # 1m outside in y
            [0.5, 0.5, 4.0],  # 3m outside in z
            [-1.0, 2.0, 4.0],  # sqrt(1^2 + 1^2 + 3^2)
        ],
        dtype=float,
    )
    out = np.asarray(rgi(pts))
    expected = np.array([1.0, 1.0, 3.0, np.sqrt(1.0**2 + 1.0**2 + 3.0**2)], dtype=float)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_voxel_grid_interpolator_jittable():
    x = np.array([0.0, 1.0], dtype=float)
    y = np.array([0.0, 1.0], dtype=float)
    z = np.array([0.0, 1.0], dtype=float)
    values = np.ones((2, 2, 2), dtype=np.float32)
    rgi = VoxelGridInterpolator((x, y, z), values)

    f = jit(lambda p: rgi(p))
    pts = np.array([[0.25, 0.25, 0.25], [2.0, 0.5, 0.5]], dtype=float)
    out = np.asarray(f(pts))
    # First point in-bounds => interpolated 1.0, second out-of-bounds in x by 1.0.
    np.testing.assert_allclose(out, np.array([1.0, 2.0], dtype=float))


def test_voxel_grid_interpolator_rejects_non_uniform_or_non_monotone_axes():
    y = np.array([0.0, 1.0], dtype=float)
    z = np.array([0.0, 1.0], dtype=float)
    values = np.zeros((3, 2, 2), dtype=np.float32)

    x_non_uniform = np.array([0.0, 1.0, 3.0], dtype=float)
    with pytest.raises(ValueError, match="uniformly spaced"):
        VoxelGridInterpolator((x_non_uniform, y, z), values)

    x_decreasing = np.array([2.0, 1.0, 0.0], dtype=float)
    with pytest.raises(ValueError, match="strictly increasing"):
        VoxelGridInterpolator((x_decreasing, y, z), values)


def test_voxel_grid_interpolator_preserves_float32_when_inputs_float32():
    """Guard against accidental dtype promotion in the hot path (performance)."""
    x = np.array([0.0, 1.0], dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)
    z = np.array([0.0, 1.0], dtype=np.float32)
    values = np.ones((2, 2, 2), dtype=np.float32)
    rgi = VoxelGridInterpolator((x, y, z), values)

    pts = np.array([[0.25, 0.25, 0.25]], dtype=np.float32)
    out = rgi(pts)
    assert np.asarray(out).dtype == np.float32


def test_voxel_grid_interpolator_rejects_invalid_xi_shape():
    """Fail fast on malformed query inputs to avoid silent broadcasting bugs."""
    x = np.array([0.0, 1.0], dtype=float)
    y = np.array([0.0, 1.0], dtype=float)
    z = np.array([0.0, 1.0], dtype=float)
    values = np.zeros((2, 2, 2), dtype=np.float32)
    rgi = VoxelGridInterpolator((x, y, z), values)

    with pytest.raises(ValueError, match=r"xi must have shape"):
        rgi(np.array([0.1, 0.2], dtype=float))  # (...,2) not (...,3)

    with pytest.raises(ValueError, match=r"xi must have shape"):
        rgi(np.zeros((4, 4), dtype=float))  # (4,4) last dim != 3


def test_voxel_grid_interpolator_supports_higher_rank_xi_shapes():
    """Ensure __call__ preserves leading batch dimensions (...,3) -> (...)."""
    x = np.array([0.0, 1.0], dtype=float)
    y = np.array([0.0, 1.0], dtype=float)
    z = np.array([0.0, 1.0], dtype=float)
    values = np.ones((2, 2, 2), dtype=np.float32)
    rgi = VoxelGridInterpolator((x, y, z), values)

    pts = np.zeros((2, 3, 4, 3), dtype=float)  # (B1,B2,N,3)
    out = np.asarray(rgi(pts))
    assert out.shape == (2, 3, 4)


def test_voxel_grid_interpolator_nan_query_propagates():
    """Make NaN behavior explicit: NaNs in xi should produce NaNs in output."""
    x = np.array([0.0, 1.0], dtype=float)
    y = np.array([0.0, 1.0], dtype=float)
    z = np.array([0.0, 1.0], dtype=float)
    values = np.ones((2, 2, 2), dtype=np.float32)
    rgi = VoxelGridInterpolator((x, y, z), values)

    pts = np.array([[np.nan, 0.5, 0.5]], dtype=float)
    out = np.asarray(rgi(pts))
    assert np.isnan(out).all()

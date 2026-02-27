"""DSM benchmarking suite (pytest-benchmark).

Motivation
----------
These benchmarks are intended to approximate the *intended usage* for plot-scale
simulations and LiDAR-derived DSM rasters:

- Raster resolution: `dx=dy=0.5` (typical DSM pixel size) and a finer vertical
  resolution `dz=0.1` to better resolve height differences.
- Tree geometry: a grid of stems with realistic-ish crown sizes, so the evaluation
  window / raster extents scale plausibly with `B`.
- Fair point-cloud baseline: the point-cloud path is sampled densely enough
  relative to `dx` to reduce DSM pits. Using overly sparse sampling can make the
  point-cloud approach look artificially fast, but it is not representative of
  pit-resistant outputs.

Benchmarks included
-------------------
- `make_dsm` (rasterize pre-generated points): isolates rasterization cost.
- End-to-end point-cloud DSM: includes crown hull point generation + rasterization.
- Analytic DSM: benchmarks the intended parameter container (`CrownSurfaceParams`).
  This path avoids generating a full point cloud, and only calculates a single point
  per pixel using the upper crown surface.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from forest3d.geometry.evaluators.points import make_crown_hull_batched
from forest3d.geometry.params import CrownHullParams, CrownSurfaceParams
from forest3d.geospatial.coordinates import CoordinateSystem
from forest3d.rasterization.analytic_dsm import (
    make_analytic_dsm,
)
from forest3d.rasterization.dsm import make_dsm


def _pc_sampling_for_raster(*, dx: float, max_crown_radius: float) -> tuple[int, int]:
    """Point-cloud sampling tuned to raster resolution.

    The point-cloud DSM baseline should be sampled densely enough (relative to
    `dx`) that pits are reduced. Otherwise, the benchmark unfairly favors the
    point-cloud approach at the cost of accuracy.

    Args:
        dx (float): Raster pixel size along x (same units as crown radii).
        max_crown_radius (float): Expected upper bound on crown radius used to
            size the sampling density.

    Returns:
        num_theta (int): Number of azimuth samples.
        num_z (int): Number of vertical samples.
    """
    # Choose azimuthal spacing ~ dx/2 along the crown perimeter.
    num_theta = int(
        np.ceil((2.0 * np.pi * float(max_crown_radius)) / (0.5 * float(dx)))
    )
    # Make divisible by 4 for consistency with E/N/W/S anchors and clamp.
    num_theta = int(min(1024, max(256, 4 * int(np.ceil(num_theta / 4)))))

    # Vertical sampling mainly affects surface fidelity; keep moderately high.
    num_z = 256
    return num_theta, num_z


def _make_batched_hull_params(
    B: int,
    *,
    stem_spacing: float = 15.0,
    origin_xy: tuple[float, float] = (10.0, 10.0),
) -> CrownHullParams:
    """Deterministic grid of stems; constant crown geometry.

    `stem_spacing` keeps tree density roughly constant as `B` grows. It is in the
    same units as the `CoordinateSystem` used for rasterization.

    This benchmark uses a fixed crown geometry so that runtime is dominated by
    algorithmic scaling (trees × pixels / points), not by parameter variability.

    Args:
        B (int): Number of trees.
        stem_spacing (float): Spacing between stems in the same units as the
            `CoordinateSystem`.
        origin_xy (tuple[float, float]): (x, y) origin for the stem grid.

    Returns:
        params (TreeHullParams): Batched hull parameters with leading dimension `B`.
    """
    B = int(B)
    g = int(np.ceil(np.sqrt(B)))
    idx = jnp.arange(B, dtype=jnp.int32)
    x0, y0 = origin_xy
    x = (idx % g).astype(jnp.float32) * jnp.float32(stem_spacing) + jnp.float32(x0)
    y = (idx // g).astype(jnp.float32) * jnp.float32(stem_spacing) + jnp.float32(y0)
    stem_base = jnp.stack([x, y, jnp.zeros_like(x)], axis=1)  # (B,3)

    top_height = jnp.full((B,), 30.0, dtype=jnp.float32)
    crown_ratio = jnp.full((B,), 0.6, dtype=jnp.float32)
    lean_direction = jnp.zeros((B,), dtype=jnp.float32)
    lean_severity = jnp.zeros((B,), dtype=jnp.float32)

    crown_radii = jnp.full((B, 4), 6.0, dtype=jnp.float32)  # E,N,W,S
    crown_edge_heights = jnp.full((B, 4), 0.3, dtype=jnp.float32)
    crown_shapes = jnp.full((B, 2, 4), 2.0, dtype=jnp.float32)

    return CrownHullParams(
        stem_base=stem_base,
        top_height=top_height,
        crown_ratio=crown_ratio,
        lean_direction=lean_direction,
        lean_severity=lean_severity,
        crown_radii=crown_radii,
        crown_edge_heights=crown_edge_heights,
        crown_shapes=crown_shapes,
    )


def _make_cs_for_params(
    params: CrownHullParams,
    *,
    dx: float = 0.5,
    dy: float = 0.5,
    dz: float = 0.1,
    zmin: float = 0.0,
    zmax: float = 60.0,
    margin_xy: float = 10.0,
) -> CoordinateSystem:
    """Create a raster coordinate system that covers the tree batch extent.

    Motivation for defaults:
    - `dx=dy=0.5` approximates common LiDAR DSM resolution at plot scale.
    - `dz=0.1` provides finer vertical resolution for height comparisons.
    - A generous `margin_xy` avoids edge effects in benchmarks as `B` changes.

    Args:
        params (TreeHullParams): Batched hull parameters used to estimate xy extent.
        dx (float): Raster x spacing.
        dy (float): Raster y spacing.
        dz (float): Raster z spacing.
        zmin (float): Minimum z bound.
        zmax (float): Maximum z bound.
        margin_xy (float): Extra xy margin around the canopy extent.

    Returns:
        cs (CoordinateSystem): Raster coordinate system.
    """
    stems = np.asarray(params.stem_base, dtype=float)  # (B,3)
    radii = np.asarray(params.crown_radii, dtype=float)  # (B,4) E,N,W,S

    x_min = float(stems[:, 0].min() - radii[:, 2].max() - margin_xy)
    x_max = float(stems[:, 0].max() + radii[:, 0].max() + margin_xy)
    y_min = float(stems[:, 1].min() - radii[:, 3].max() - margin_xy)
    y_max = float(stems[:, 1].max() + radii[:, 1].max() + margin_xy)

    return CoordinateSystem.from_bounds(
        xmin=x_min,
        ymin=y_min,
        zmin=float(zmin),
        xmax=x_max,
        ymax=y_max,
        zmax=float(zmax),
        dx=float(dx),
        dy=float(dy),
        dz=float(dz),
    )


@pytest.mark.parametrize("B", [1, 10, 100, 1000])
def test_bench_make_dsm_from_points(benchmark, B: int):
    """Benchmark rasterization cost given a pre-generated (dense) point cloud.

    This isolates the cost of `make_dsm` itself and uses dense sampling (relative
    to `dx`) to reduce pits.
    """
    params = _make_batched_hull_params(B)
    cs = _make_cs_for_params(params)
    max_r = 8.0
    num_theta, num_z = _pc_sampling_for_raster(dx=float(cs.dx), max_crown_radius=max_r)
    pts = make_crown_hull_batched(params, num_theta=num_theta, num_z=num_z)  # (B,N,3)
    fill = jnp.asarray(cs.zmin, dtype=jnp.float32)

    g = jax.jit(lambda p: make_dsm(p, cs=cs, fill_value=fill))
    g(pts).block_until_ready()  # warmup compile

    def run():
        out = g(pts)
        out.block_until_ready()

    benchmark.group = "dsm"
    benchmark(run)


@pytest.mark.parametrize("B", [1, 10, 100, 1000])
def test_bench_make_dsm_end_to_end(benchmark, B: int):
    """Benchmark point-cloud DSM end-to-end (hull generation + rasterization).

    This measures the full cost of the point-cloud workflow at a sampling density
    that is intended to be pit-resistant for the chosen raster resolution.
    """
    params = _make_batched_hull_params(B)
    cs = _make_cs_for_params(params)
    fill = jnp.asarray(cs.zmin, dtype=jnp.float32)

    max_r = 8.0
    num_theta, num_z = _pc_sampling_for_raster(dx=float(cs.dx), max_crown_radius=max_r)
    h = jax.jit(lambda p: make_crown_hull_batched(p, num_theta=num_theta, num_z=num_z))
    g = jax.jit(lambda pts: make_dsm(pts, cs=cs, fill_value=fill))

    # warmup compile
    pts0 = h(params)
    g(pts0).block_until_ready()

    def run():
        pts = h(params)
        out = g(pts)
        out.block_until_ready()

    benchmark.group = "dsm"
    benchmark(run)


@pytest.mark.parametrize("B", [1, 10, 100, 1000])
def test_bench_make_analytic_dsm_surface_params(benchmark, B: int):
    """Benchmark analytic DSM using the intended lightweight parameter container.

    In simulation/optimization we typically only need the upper-crown surface for
    DSM evaluation, so `CrownSurfaceParams` is the relevant input to benchmark.
    """
    hull = _make_batched_hull_params(B)
    cs = _make_cs_for_params(hull)
    surface = CrownSurfaceParams.from_hull(hull)
    fill = jnp.asarray(cs.zmin, dtype=jnp.float32)
    max_r = 8.0

    f = jax.jit(
        lambda p: make_analytic_dsm(
            p,
            cs=cs,
            fill_value=fill,
            max_crown_radius=max_r,
        )
    )
    f(surface).block_until_ready()

    def run():
        out = f(surface)
        out.block_until_ready()

    benchmark.group = "dsm"
    benchmark(run)

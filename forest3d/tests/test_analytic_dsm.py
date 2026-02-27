import math

import numpy as np
import pytest
from jax import jit

from forest3d.geometry.evaluators.points import make_crown_hull_batched
from forest3d.geometry.params import CrownHullParams, CrownSurfaceParams
from forest3d.geospatial.coordinates import CoordinateSystem
from forest3d.rasterization.analytic_dsm import DsmPixelLocation, make_analytic_dsm
from forest3d.rasterization.dsm import make_dsm


@pytest.fixture
def single_tree_params():
    def _make(*, stem_x: float, stem_y: float) -> CrownHullParams:
        # Symmetric radii => local apex at (0,0), so global apex xy == stem xy (no lean)
        stem_base = np.array([[stem_x, stem_y, 0.0]], dtype=np.float32)  # (B=1,3)
        top_height = np.array([10.0], dtype=np.float32)  # (B,)
        crown_ratio = np.array([0.65], dtype=np.float32)
        lean_direction = np.array([0.0], dtype=np.float32)
        lean_severity = np.array([0.0], dtype=np.float32)
        crown_radii = np.array(
            [[0.5, 0.5, 0.5, 0.5]], dtype=np.float32
        )  # (B,4) E,N,W,S
        crown_edge_heights = np.array([[0.3, 0.3, 0.3, 0.3]], dtype=np.float32)
        crown_shapes = np.full((1, 2, 4), fill_value=2.0, dtype=np.float32)  # (B,2,4)
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

    return _make


def test_make_analytic_dsm_jittable_and_apex_height(single_tree_params):
    cs = CoordinateSystem.from_bounds(
        xmin=0.0,
        ymin=0.0,
        zmin=0.0,
        xmax=3.0,
        ymax=3.0,
        zmax=20.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )

    # Cell centers for dx=dy=1.0 are (0.5,1.5,2.5).
    # Pick apex at (1.5,1.5) => (i=1,j=1).
    params = single_tree_params(stem_x=1.5, stem_y=1.5)

    f = jit(
        lambda p: make_analytic_dsm(
            p,
            cs=cs,
            fill_value=np.float32(-1.0),
            max_crown_radius=1.0,
        )
    )
    dsm = np.asarray(f(params))

    assert dsm.shape == (cs.ny, cs.nx)
    assert dsm.dtype == np.float32
    assert np.isclose(dsm[1, 1], 10.0)  # stem_z=0, top_height=10


def test_make_analytic_dsm_outside_crown_is_fill_value(single_tree_params):
    cs = CoordinateSystem.from_bounds(
        xmin=0.0,
        ymin=0.0,
        zmin=0.0,
        xmax=3.0,
        ymax=3.0,
        zmax=20.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )
    params = single_tree_params(stem_x=1.5, stem_y=1.5)

    fill = np.float32(-1.0)
    dsm = np.asarray(
        make_analytic_dsm(params, cs=cs, fill_value=fill, max_crown_radius=1.0)
    )

    # (i=0,j=0) center is (0.5,2.5). Distance to apex (1.5,1.5) is sqrt(2) > 0.5 radius.
    assert np.isclose(dsm[0, 0], float(fill))


def test_make_analytic_dsm_enforce_apex_updates_containing_pixel(single_tree_params):
    cs = CoordinateSystem.from_bounds(
        xmin=0.0,
        ymin=0.0,
        zmin=0.0,
        xmax=3.0,
        ymax=3.0,
        zmax=20.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )

    # Apex at (1.1, 1.1) lies inside pixel (i=1,j=1), whose center is (1.5,1.5).
    # With a small crown radius (0.5), the center is outside the crown, so without
    # enforce_apex we'd see fill_value; with enforce_apex we should see the apex height.
    params = single_tree_params(stem_x=1.1, stem_y=1.1)
    fill = np.float32(-1.0)

    dsm_no = np.asarray(
        make_analytic_dsm(
            params,
            cs=cs,
            fill_value=fill,
            max_crown_radius=1.0,
            dsm_pixel_location=DsmPixelLocation.CENTER,
            enforce_apex=False,
        )
    )
    assert np.isclose(dsm_no[1, 1], float(fill))

    dsm_yes = np.asarray(
        make_analytic_dsm(
            params,
            cs=cs,
            fill_value=fill,
            max_crown_radius=1.0,
            dsm_pixel_location=DsmPixelLocation.CENTER,
            enforce_apex=True,
        )
    )
    assert np.isclose(dsm_yes[1, 1], 10.0)


def test_make_analytic_dsm_accepts_crown_surface_params(single_tree_params):
    cs = CoordinateSystem.from_bounds(
        xmin=0.0,
        ymin=0.0,
        zmin=0.0,
        xmax=3.0,
        ymax=3.0,
        zmax=20.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )
    hull = single_tree_params(stem_x=1.5, stem_y=1.5)
    surface = CrownSurfaceParams.from_hull(hull)

    d1 = np.asarray(
        make_analytic_dsm(
            hull,
            cs=cs,
            fill_value=np.float32(-1.0),
            max_crown_radius=1.0,
        )
    )
    d2 = np.asarray(
        make_analytic_dsm(
            surface, cs=cs, fill_value=np.float32(-1.0), max_crown_radius=1.0
        )
    )
    np.testing.assert_allclose(d1, d2)


def _unbatch_params(p: CrownHullParams) -> CrownHullParams:
    """Convert a (B=1,...) CrownHullParams to single-tree shapes."""
    return CrownHullParams(
        stem_base=np.asarray(p.stem_base)[0],
        top_height=np.asarray(p.top_height)[0],
        crown_ratio=np.asarray(p.crown_ratio)[0],
        lean_direction=np.asarray(p.lean_direction)[0],
        lean_severity=np.asarray(p.lean_severity)[0],
        crown_radii=np.asarray(p.crown_radii)[0],
        crown_edge_heights=np.asarray(p.crown_edge_heights)[0],
        crown_shapes=np.asarray(p.crown_shapes)[0],
    )


def _make_forest_params(*, B: int = 9, spacing: float = 6.0) -> CrownHullParams:
    """Deterministic small forest with varied crown parameters (batched)."""
    rng = np.random.default_rng(0)
    g = int(np.ceil(np.sqrt(B)))
    xs = (np.arange(B) % g).astype(np.float32) * np.float32(spacing) + np.float32(10.0)
    ys = (np.arange(B) // g).astype(np.float32) * np.float32(spacing) + np.float32(10.0)

    stem_base = np.stack([xs, ys, np.zeros(B, dtype=np.float32)], axis=1)
    top_height = rng.uniform(12.0, 28.0, size=B).astype(np.float32)
    crown_ratio = rng.uniform(0.45, 0.8, size=B).astype(np.float32)
    lean_direction = np.zeros(B, dtype=np.float32)
    lean_severity = np.zeros(B, dtype=np.float32)

    # Radii vary by tree and direction (E,N,W,S); use plot-scale crowns.
    crown_radii = rng.uniform(4.0, 10.0, size=(B, 4)).astype(np.float32)
    crown_edge_heights = rng.uniform(0.05, 0.6, size=(B, 4)).astype(np.float32)

    # Shapes: keep bottom fixed, vary top only.
    crown_shapes = np.full((B, 2, 4), 2.0, dtype=np.float32)
    crown_shapes[:, 0, :] = rng.uniform(1.2, 3.0, size=(B, 4)).astype(np.float32)

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


@pytest.fixture
def forest_params() -> CrownHullParams:
    # Small deterministic forest with varied tree geometry.
    return _make_forest_params(B=9, spacing=15.0)


def test_make_analytic_dsm_close_to_pointcloud_rasterized(forest_params):
    # Forest-scale regression: analytic DSM should be broadly consistent with
    # point-cloud rasterization for a small deterministic forest of varied trees.

    # Use a very dense hull point cloud as the reference DSM to reduce sampling
    # artifacts ("pits") in the point-cloud rasterization. Choose azimuthal
    # sampling density based on crown size vs raster resolution so that the
    # point cloud remains "dense" as crown radii change.
    dx = dy = dz = 0.1
    max_r = float(np.asarray(forest_params.crown_radii).max())
    num_theta = int(math.ceil(2.0 * math.pi * max_r / float(dx)))
    num_theta = int(min(1024, max(192, 4 * math.ceil(num_theta / 4))))
    num_z = 192

    crown_points_batched = make_crown_hull_batched(
        forest_params, num_theta=num_theta, num_z=num_z
    )
    crown_points = crown_points_batched.reshape((-1, 3))
    crown_np = np.asarray(crown_points, dtype=np.float32)
    # Compare on a reasonably fine raster so center-sampling is meaningful.

    margin_xy = 0.5
    margin_z = 0.5
    cs = CoordinateSystem.from_bounds(
        xmin=float(crown_np[:, 0].min() - margin_xy),
        ymin=float(crown_np[:, 1].min() - margin_xy),
        zmin=float(crown_np[:, 2].min() - margin_z),
        xmax=float(crown_np[:, 0].max() + margin_xy),
        ymax=float(crown_np[:, 1].max() + margin_xy),
        zmax=float(crown_np[:, 2].max() + margin_z),
        dx=dx,
        dy=dy,
        dz=dz,
    )

    fill = np.float32(-1.0)
    dsm_pc = np.asarray(
        make_dsm(crown_points, cs=cs, fill_value=fill), dtype=np.float32
    )
    dsm_an = np.asarray(
        make_analytic_dsm(
            forest_params,
            cs=cs,
            fill_value=fill,
            max_crown_radius=max_r,
            enforce_apex=True,
        ),
        dtype=np.float32,
    )

    mask_pc = dsm_pc != fill
    mask_an = dsm_an != fill
    assert mask_pc.any()
    assert mask_an.any()

    # The analytic DSM uses *upper crown only*, while the point-cloud DSM is built
    # from the full hull surface (upper + lower). As a result, the point-cloud DSM
    # can have additional low-height pixels near crown bases that the analytic DSM
    # intentionally leaves empty. We therefore compare overlap primarily on the
    # analytic support (where the upper surface exists).
    overlap = mask_pc & mask_an
    assert float(overlap.sum() / mask_an.sum()) > 0.90

    absdiff = np.abs(dsm_pc[overlap] - dsm_an[overlap])
    # Tie tolerances to vertical grid spacing for interpretability.
    # `make_dsm` is a max-reduction over sampled points, while `make_analytic_dsm`
    # samples the continuous surface at cell centers. Differences on the order of
    # a small multiple of `dz` are expected.
    dz = float(cs.dz)
    assert float(np.median(absdiff)) < 0.75 * dz
    assert float(np.quantile(absdiff, 0.95)) < 6.0 * dz

    # Peak canopy height should agree closely when apex enforcement is enabled.
    assert np.isclose(
        float(dsm_pc[mask_pc].max()),
        float(dsm_an[mask_an].max()),
        atol=1e-3,
    )

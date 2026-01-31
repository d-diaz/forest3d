"""JAX-friendly validation utilities for crown hull generation.

This module is intended to hold checkify-based 'fail fast' wrappers around the crown
hull generator without cluttering the core geometry implementation.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jax.typing import ArrayLike

from forest3d.models.hull_params import TreeHullParams
from forest3d.utils.geometry import _make_crown_hull


def make_crown_hull_checked(
    *,
    stem_base: ArrayLike,
    top_height: ArrayLike,
    crown_ratio: ArrayLike,
    lean_direction: ArrayLike,
    lean_severity: ArrayLike,
    crown_radii: ArrayLike,
    crown_edge_heights: ArrayLike,
    crown_shapes: ArrayLike,
    num_theta: int = 32,
    num_z: int = 50,
) -> tuple[checkify.Error, Array]:
    """Checks for valid inputs to the crown hull generator.

    `_make_crown_hull` (in `forest3d.utils.geometry`) is intentionally free of eager
    validation so it composes with JAX transforms. But some invalid/degenerate inputs
    can yield NaNs/infs (e.g., division-by-zero in the radius equations, `tan(pi/2)`
    at 90° lean, invalid exponents when shape coefficients are non-positive).

    This function provides the “plumbing” for Phase 1:
    - It wraps the hull math in `checkify.checkify(...)`, returning `(err, points)`.
    - It enables `checkify.float_checks` so division-by-zero and NaN-producing
      floating point ops are caught under `jit`.
    - Later Phase 1 steps add explicit `checkify.check(...)` assertions for clearer,
      domain-specific error messages.

    Returns
    -------
    err : checkify.Error
        A checkify error object (call `err.throw()` in Python to raise).
    points : jax.Array, shape (num_z * num_theta, 3)
        3D points on the crown hull surface.
    """

    # NOTE: checks MUST be inside the function passed to `checkify.checkify`.
    # If you call `checkify.check(...)` directly in a `jit`ted function without
    # being functionalized, JAX will error.
    def _checked_impl(
        *,
        stem_base: ArrayLike,
        top_height: ArrayLike,
        crown_ratio: ArrayLike,
        lean_direction: ArrayLike,
        lean_severity: ArrayLike,
        crown_radii: ArrayLike,
        crown_edge_heights: ArrayLike,
        crown_shapes: ArrayLike,
        num_theta: int,
        num_z: int,
    ) -> Array:
        stem_base_array = jnp.asarray(stem_base)
        top_height_array = jnp.asarray(top_height)
        crown_ratio_array = jnp.asarray(crown_ratio)
        lean_direction_array = jnp.asarray(lean_direction)
        lean_severity_array = jnp.asarray(lean_severity)
        crown_radii_array = jnp.asarray(crown_radii)
        crown_edge_heights_array = jnp.asarray(crown_edge_heights)
        crown_shapes_array = jnp.asarray(crown_shapes)

        # Value invariants. These produce user-friendly messages under `jit`.
        checkify.check(
            jnp.all(jnp.isfinite(stem_base_array)),
            "stem_base must be finite (no NaN/inf).",
        )
        checkify.check(
            jnp.isfinite(top_height_array) & (top_height_array > 0),
            "top_height must be finite and > 0.",
        )
        checkify.check(
            jnp.isfinite(crown_ratio_array)
            & (crown_ratio_array > 0)
            & (crown_ratio_array <= 1),
            "crown_ratio must be finite and in (0, 1].",
        )
        checkify.check(
            jnp.isfinite(lean_direction_array),
            "lean_direction must be finite (no NaN/inf).",
        )
        # Strict < 90: `_get_treetop_location` uses tan(phi_lean).
        checkify.check(
            jnp.isfinite(lean_severity_array)
            & (lean_severity_array >= 0)
            & (lean_severity_array < 90),
            "lean_severity must be finite and in [0, 90).",
        )

        checkify.check(
            jnp.all(jnp.isfinite(crown_radii_array)),
            "crown_radii must be finite (no NaN/inf).",
        )
        checkify.check(
            jnp.all(crown_radii_array >= 0),
            "crown_radii must be >= 0 (E,N,W,S).",
        )
        checkify.check(
            jnp.any(crown_radii_array > 0),
            "crown_radii cannot be all zeros.",
        )

        checkify.check(
            jnp.all(jnp.isfinite(crown_edge_heights_array)),
            "crown_edge_heights must be finite (no NaN/inf).",
        )
        checkify.check(
            jnp.all((crown_edge_heights_array >= 0) & (crown_edge_heights_array < 1)),
            "crown_edge_heights must be in [0, 1).",
        )

        checkify.check(
            jnp.all(jnp.isfinite(crown_shapes_array)),
            "crown_shapes must be finite (no NaN/inf).",
        )
        checkify.check(
            jnp.all(crown_shapes_array > 0),
            "crown_shapes must be > 0 (invalid exponents otherwise).",
        )

        # NOTE: shape mismatches are not asserted here because JAX shapes are static;
        # a mismatch will raise a (non-checkify) shape error at trace time. If we
        # want runtime shape errors in the future, we can add explicit reshapes here.
        return _make_crown_hull(
            stem_base=stem_base_array,
            top_height=top_height_array,
            crown_ratio=crown_ratio_array,
            lean_direction=lean_direction_array,
            lean_severity=lean_severity_array,
            crown_radii=crown_radii_array,
            crown_edge_heights=crown_edge_heights_array,
            crown_shapes=crown_shapes_array,
            num_theta=num_theta,
            num_z=num_z,
        )

    checked = checkify.checkify(
        _checked_impl,
        errors=checkify.user_checks | checkify.float_checks,
    )
    return checked(
        stem_base=stem_base,
        top_height=top_height,
        crown_ratio=crown_ratio,
        lean_direction=lean_direction,
        lean_severity=lean_severity,
        crown_radii=crown_radii,
        crown_edge_heights=crown_edge_heights,
        crown_shapes=crown_shapes,
        num_theta=num_theta,
        num_z=num_z,
    )


def make_crown_hull_checked_from_params(
    params: TreeHullParams, *, num_theta: int = 32, num_z: int = 50
) -> tuple[checkify.Error, Array]:
    """Checks for valid inputs to the crown hull generator from `TreeHullParams`.

    This is the params-container equivalent of `make_crown_hull_checked`, intended
    for `vmap`/`jit` workflows where you carry parameters as a single PyTree.
    """
    return make_crown_hull_checked(
        stem_base=params.stem_base,
        top_height=params.top_height,
        crown_ratio=params.crown_ratio,
        lean_direction=params.lean_direction,
        lean_severity=params.lean_severity,
        crown_radii=params.crown_radii,
        crown_edge_heights=params.crown_edge_heights,
        crown_shapes=params.crown_shapes,
        num_theta=num_theta,
        num_z=num_z,
    )

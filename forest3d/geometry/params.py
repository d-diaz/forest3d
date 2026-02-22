from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import tree_util
from jax.typing import ArrayLike


@tree_util.register_dataclass
@dataclass(frozen=True)
class CrownHullParams:
    """Parameter container for the crown hull (PyTree-friendly).

    This is intended for vmap/jit workflows, where parameters are passed around as a
    single object. Each field may be a scalar, a JAX array, or a NumPy array; they
    will be converted via `jnp.asarray()` inside the hull implementation.

    Batched usage
    ------------
    For batching with `vmap`, each field should be stacked with a leading batch
    dimension `B` (e.g., `stem_base` has shape `(B,3)`, `crown_radii` has shape
    `(B,4)`, etc.).
    """

    stem_base: ArrayLike
    top_height: ArrayLike
    crown_ratio: ArrayLike
    lean_direction: ArrayLike
    lean_severity: ArrayLike
    crown_radii: ArrayLike
    crown_edge_heights: ArrayLike
    crown_shapes: ArrayLike  # (...,2,4); [0]=top, [1]=bottom; E,N,W,S

    @property
    def crown_top_shapes(self) -> jnp.ndarray:
        """Upper-crown shape coefficients with shape (...,4)."""
        return jnp.asarray(self.crown_shapes)[..., 0, :]

    @property
    def crown_bottom_shapes(self) -> jnp.ndarray:
        """Lower-crown shape coefficients with shape (...,4)."""
        return jnp.asarray(self.crown_shapes)[..., 1, :]


@tree_util.register_dataclass
@dataclass(frozen=True)
class CrownSurfaceParams:
    """Parameter container for analytic crown *surface* evaluators (PyTree-friendly).

    This is a lightweight alternative to `CrownHullParams` intended for workflows
    (simulation/optimization) that only require the *upper-crown surface* and do not
    need the lower-crown shape parameters.

    Batched usage
    ------------
    For batching with `vmap`, each field should be stacked with a leading batch
    dimension `B` (e.g., `stem_base` has shape `(B,3)`, `crown_radii` has shape
    `(B,4)`, and `crown_top_shapes` has shape `(B,4)`).
    """

    stem_base: ArrayLike
    top_height: ArrayLike
    crown_ratio: ArrayLike
    lean_direction: ArrayLike
    lean_severity: ArrayLike
    crown_radii: ArrayLike
    crown_edge_heights: ArrayLike
    crown_top_shapes: ArrayLike  # (4,) or (B,4); E,N,W,S

    @staticmethod
    def from_hull(params: CrownHullParams) -> CrownSurfaceParams:
        """Create surface params from full hull params.

        This drops the lower-crown shape parameters, keeping only the top-of-crown
        shape coefficients.
        """
        return CrownSurfaceParams(
            stem_base=params.stem_base,
            top_height=params.top_height,
            crown_ratio=params.crown_ratio,
            lean_direction=params.lean_direction,
            lean_severity=params.lean_severity,
            crown_radii=params.crown_radii,
            crown_edge_heights=params.crown_edge_heights,
            crown_top_shapes=params.crown_top_shapes,
        )

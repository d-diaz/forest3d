from __future__ import annotations

from dataclasses import dataclass

from jax import tree_util
from jax.typing import ArrayLike


@tree_util.register_dataclass
@dataclass(frozen=True)
class TreeHullParams:
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
    crown_shapes: ArrayLike

"""Crown geometry evaluators (products) and kernels.

This package groups crown evaluation code by *output kind*:
- `surface`: crown-domain surface kernels (array-in/array-out; no raster policy)
- `points`: point-cloud generation from `CrownModel`

Pattern guidelines (developer contract):
- Use **dataclass smart constructors** (`from_array`, `from_params`, `from_model`) to
  build stable named concepts in `forest3d.geometry.primitives` and `CrownModel`.
- Use **pure functions** for reusable math kernels (`forest3d.geometry.kernels`) and
  for evaluators that produce products like point clouds.
- When both exist, constructors should delegate to kernels (single source of truth).

Non-goals:
- No end-to-end user-facing API; future wrappers belong in `forest3d/generators/`.
"""

"""Geospatial domain enums.

These are shared across coordinate transforms, rasterization, and voxel workflows.
Keep them in `forest3d.geospatial` to avoid coupling to schemas or hot-path modules.
"""

from __future__ import annotations

from enum import StrEnum


class GridKind(StrEnum):
    RASTER = "raster"
    VOXEL = "voxel"


class IntegerMode(StrEnum):
    FLOOR = "floor"
    INT = "int"
    CEIL = "ceil"

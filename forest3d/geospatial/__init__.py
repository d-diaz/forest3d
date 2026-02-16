"""Geospatial utilities and metadata types.

Coordinate transforms, bounds/windows, and CRS/affine helpers live here so they do not
end up in a generic `utils` bucket.
"""

from forest3d.geospatial.enums import GridKind, IntegerMode

__all__ = ["GridKind", "IntegerMode"]

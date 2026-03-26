"""Raster alignment validation."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import rasterio


@dataclass(frozen=True, slots=True)
class RasterGrid:
    """A raster grid.

    Properties:
        crs_wkt (str | None): The CRS of the raster grid.
        transform_gdal (tuple[float, float, float, float, float, float]): The GDAL
            affine transform of the raster grid.
        width (int): The number of columns in the raster grid.
        height (int): The number of rows in the raster grid.
    """

    crs_wkt: str | None
    transform_gdal: tuple[float, float, float, float, float, float]
    width: int
    height: int

    @staticmethod
    def _from_rasterio_dataset(ds: rasterio.DatasetReader) -> RasterGrid:
        """Create a `RasterGrid` from a rasterio dataset.

        Args:
            ds (rasterio.DatasetReader): A rasterio dataset.

        Returns:
            (RasterGrid): A `RasterGrid` object.
        """
        # Rasterio CRS may be None; stringify to a stable representation.
        crs_wkt = ds.crs.to_wkt() if ds.crs is not None else None
        t = ds.transform
        transform_gdal = (
            float(t.c),
            float(t.a),
            float(t.b),
            float(t.f),
            float(t.d),
            float(t.e),
        )
        return RasterGrid(
            crs_wkt=crs_wkt,
            transform_gdal=transform_gdal,
            width=int(ds.width),
            height=int(ds.height),
        )

    @staticmethod
    def from_raster(path_to_raster: str | os.PathLike) -> RasterGrid:
        """Create a `RasterGrid` from a path to a raster file.

        Args:
            path_to_raster (str | os.PathLike): Path to a raster file.

        Returns:
            (RasterGrid): A `RasterGrid` object.
        """

        with rasterio.open(path_to_raster) as ds:
            return RasterGrid._from_rasterio_dataset(ds)


def assert_grids_equal_from_paths(
    path1: str | os.PathLike,
    path2: str | os.PathLike,
    transform_rtol: float = 0.0,
    transform_atol: float = 1e-9,
) -> None:
    """Raise if two rasters are not on the exact same grid (within tolerances).

    Args:
        path1 (str | os.PathLike): Path to the first raster.
        path2 (str | os.PathLike): Path to the second raster.
        transform_rtol (float): The relative tolerance for the transform.
        transform_atol (float): The absolute tolerance for the transform.
    """
    rg1 = RasterGrid.from_raster(path1)
    rg2 = RasterGrid.from_raster(path2)
    assert_grids_equal(
        rg1,
        rg2,
        transform_rtol=transform_rtol,
        transform_atol=transform_atol,
    )


def assert_grids_equal(
    grid1: RasterGrid,
    grid2: RasterGrid,
    transform_rtol: float = 0.0,
    transform_atol: float = 1e-9,
) -> None:
    """Raise if two `RasterGrid` objects are not identical (within tolerances).

    Args:
        rg1 (RasterGrid): The first raster grid.
        rg2 (RasterGrid): The second raster grid.
        transform_rtol (float): Relative tolerance for the GDAL transform parameters.
        transform_atol (float): Absolute tolerance for the GDAL transform parameters.

    Raises:
        ValueError: If the rasters are not on the same grid (within tolerances).
    """
    problems: list[str] = []

    if grid1.crs_wkt != grid2.crs_wkt:
        problems.append("CRS differs between the rasters.")

    if grid1.width != grid2.width or grid1.height != grid2.height:
        problems.append(
            f"Shape differs: grid1 (height,width)=({grid1.height},{grid1.width}) "
            f"vs grid2 (height,width)=({grid2.height},{grid2.width})."
        )

    if not np.allclose(
        np.asarray(grid1.transform_gdal),
        np.asarray(grid2.transform_gdal),
        rtol=float(transform_rtol),
        atol=float(transform_atol),
    ):
        problems.append(
            "Affine transform differs (GDAL order: c,a,b,f,d,e):\n"
            f"  grid1: {grid1.transform_gdal}\n"
            f"  grid2: {grid2.transform_gdal}"
        )

    # Fail fast with a single informative error.
    if problems:
        raise ValueError(
            "The provided grids are not equal (within tolerances).\n"
            + "\n".join(f"- {p}" for p in problems)
        )

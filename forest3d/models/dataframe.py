from __future__ import annotations

import os
from typing import Any

import geopandas as gpd
import pandas as pd
import pandera.pandas as pa


class TreeListDataFrameModel(pa.DataFrameModel):
    """A DataFrame model for a Tree List."""

    species: str
    dbh: float = pa.Field(gt=0)
    stem_x: float
    stem_y: float
    top_height: float = pa.Field(gt=0)
    crown_ratio: float = pa.Field(default=0.65, ge=0, le=1.0)

    class Config:
        """Config for TreeListDataFrameModel."""

        name = "TreeListDataFrameModel"
        coerce = True
        ordered = False
        strict = False

    @pa.dataframe_check
    def not_empty(cls, df):
        """Raises if dataframe is empty."""
        return not df.empty

    @classmethod
    def from_csv(cls, path_to_csv: str | os.PathLike) -> TreeListDataFrameModel:
        """Reads and validates a treelist dataframe from a CSV."""
        return cls(pd.read_csv(path_to_csv))


class TreeListGeoDataFrameModel(TreeListDataFrameModel):
    """A GeoDataFrame model for a Tree List."""

    geometry: gpd.array.GeometryDtype

    class Config:
        """Config for TreeListGeoDataFrameModel."""

        name = "TreeListGeoDataFrameModel"
        coerce = True
        ordered = False
        strict = False

    @classmethod
    def from_file(cls, path_to_file: str | os.PathLike) -> TreeListGeoDataFrameModel:
        """Reads and validates treelist gdf from any file type recognized by GeoPandas."""
        return cls(gpd.read_file(path_to_file))

    @classmethod
    def from_dataframe(
        cls, dataframe: TreeListDataFrameModel, crs: Any
    ) -> TreeListGeoDataFrameModel:
        """Reads and validates treelist from a dataframe.

        Geometry field will be derived from stem_x and stem_y in dataframe.

        Args:
            dataframe (pd.DataFrame): the treelist as a dataframe.
            crs (CRS): the coordinate reference system to use for building a
                GeoDataFrame. Can accept any format used by pyproj.CRS.from_user_input.

        Raises:
            pandera.errors.SchemaError: if underlying dataframe is not valid according
                to TreeListDataFrameModel.
        """
        return cls(
            gpd.GeoDataFrame(
                TreeListDataFrameModel(dataframe),
                geometry=gpd.points_from_xy(dataframe.stem_x, dataframe.stem_y),
                crs=crs,
            )
        )

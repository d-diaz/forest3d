import geopandas as gpd
import pandas as pd
import pytest
from pandera.errors import SchemaError

from forest3d.schemas.treelist import TreeListDataFrameModel, TreeListGeoDataFrameModel

TEST_GEO_FILENAME = "test.geojson"
TEST_CSV_FILENAME = "test.csv"

df = pd.DataFrame(
    {
        "stem_x": [x**2 for x in range(1, 4)],
        "stem_y": [x for x in range(1, 4)],
        "species": ["A", "b", "C"],
        "crown_ratio": [x / 4.0 for x in range(1, 4)],
        "dbh": [x * 1.20 for x in range(1, 4)],
        "top_height": [x * 2 for x in range(1, 4)],
    }
)

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.stem_x, df.stem_y), crs=4326)


def test_empty_df_pandera(tmp_path):
    """Test whether providing an empty tree list fails tree_list_checker."""
    empty_df = df[0:0]
    assert len(empty_df) == 0
    with pytest.raises(SchemaError):
        TreeListDataFrameModel.validate(empty_df)


def test_csv_loader_pandera(tmp_path):
    outpath = tmp_path / TEST_CSV_FILENAME
    df.to_csv(outpath)
    validated = TreeListDataFrameModel.from_csv(outpath)
    assert isinstance(validated, pd.DataFrame)
    assert len(validated) == 3


def test_df_missing_required_column():
    with pytest.raises(SchemaError):
        TreeListDataFrameModel(gdf.drop("stem_x", axis=1))


def test_names_ok_but_not_datatype():
    """tree_list_checker returns data can't be coerced to correct dtype."""
    test_df = df.copy()
    test_df["stem_x"] = test_df["stem_x"].astype(str)

    TreeListDataFrameModel(test_df)  # doesn't raise Schema Error

    test_df["stem_x"] = ["a", "b", "c"]
    with pytest.raises(SchemaError):
        TreeListDataFrameModel(test_df)


def test_data_ok_with_extra_columns():
    """tree_list_checker returns True when supplementary columns exist."""
    test_df = df.copy()
    test_df["extra"] = [x for x in range(1, 4)]
    validated = TreeListDataFrameModel(test_df)
    assert isinstance(validated, pd.DataFrame)
    assert len(validated) == 3
    assert "extra" in validated.columns


def test_gdf_file_loader(tmp_path):
    """Valid gdf returned from file."""
    outpath = tmp_path / TEST_GEO_FILENAME
    gdf.to_file(outpath)
    validated = TreeListGeoDataFrameModel.from_file(outpath)
    assert isinstance(validated, gpd.GeoDataFrame)
    assert len(validated) == 3


def test_gdf_from_df():
    validated = TreeListGeoDataFrameModel.from_dataframe(df, crs=4326)
    assert isinstance(validated, gpd.GeoDataFrame)
    assert len(validated) == 3


def test_gdf_without_geometry():
    """SchemaError when geometry missing."""
    with pytest.raises(SchemaError):
        TreeListGeoDataFrameModel(gdf.drop("geometry", axis=1))

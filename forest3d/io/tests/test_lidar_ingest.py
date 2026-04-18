from __future__ import annotations

import json
from typing import Any

import pytest

import forest3d.io.lidar.ingest as ingest_mod
from forest3d.io.lidar.ingest import (
    make_lidar_dem,
    make_lidar_dsm,
    make_lidar_local_context,
)
from forest3d.io.lidar.pipeline import LidarPipeline


@pytest.fixture
def patch_run_capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_run(self: LidarPipeline, *, reader: object, writer: object) -> object:
        pipeline = LidarPipeline._compose(
            self._stages_with_io(reader=reader, writer=writer)
        )
        assert pipeline is not None, "Expected a non-empty pipeline"
        captured["stages"] = json.loads(pipeline.toJSON())
        return pipeline

    monkeypatch.setattr(ingest_mod.LidarPipeline, "run", fake_run)
    return captured


@pytest.fixture
def patch_exists_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest_mod.os.path, "exists", lambda _p: True)


@pytest.mark.usefixtures("patch_run_capture", "patch_exists_true")
def test_make_lidar_dem_reader_writer_and_ground_filter(
    patch_run_capture: dict[str, Any],
):
    make_lidar_dem("in.laz", "out.tif")

    stages = patch_run_capture["stages"]
    reader = stages[0]
    writer = stages[-1]

    assert reader["type"] == "readers.las"
    assert reader["filename"] == "in.laz"

    assert writer["type"] == "writers.gdal"
    assert writer["where"] == "Classification == 2 && Withheld == 0"
    assert writer["filename"] == "out.tif"


@pytest.mark.usefixtures("patch_run_capture", "patch_exists_true")
def test_make_lidar_dem_non_default_options(
    patch_run_capture: dict[str, Any],
):
    make_lidar_dem(
        "a.las",
        "b.tif",
        output_type="mean",
        resolution=1.0,
        radius=5.0,
        window_size=10.0,
    )

    writer = patch_run_capture["stages"][-1]
    assert writer["output_type"] == "mean"
    assert writer["resolution"] == 1.0
    assert writer["radius"] == 5.0
    assert writer["window_size"] == 10.0


@pytest.mark.usefixtures("patch_run_capture", "patch_exists_true")
def test_make_lidar_dsm_max_reduction_and_noise_filter(
    patch_run_capture: dict[str, Any],
):
    make_lidar_dsm("tile.laz", "dsm.tif", resolution=0.25, radius=2.0)

    stages = patch_run_capture["stages"]
    reader = stages[0]
    writer = stages[-1]

    assert reader["type"] == "readers.las"
    assert reader["filename"] == "tile.laz"

    assert writer["type"] == "writers.gdal"
    assert writer["output_type"] == "max"
    assert (
        writer["where"]
        == "Classification != 7 && Classification != 18 && Withheld == 0"
    )
    assert writer["filename"] == "dsm.tif"


def test_make_lidar_local_context_invalid_extension_raises():
    with pytest.raises(ValueError, match="Invalid source file"):
        make_lidar_local_context("bad.txt", "out", lon=-122.0, lat=47.0, radius=100.0)


def test_make_lidar_local_context_invalid_list_source_raises():
    with pytest.raises(ValueError, match="Invalid source file"):
        make_lidar_local_context(
            ["a.laz", "bad.xyz"],
            "out",
            lon=-122.0,
            lat=47.0,
            radius=50.0,
        )


@pytest.mark.usefixtures("patch_run_capture", "patch_exists_true")
def test_make_lidar_local_context_laz_path_reader_is_las_plus_crop(
    patch_run_capture: dict[str, Any],
):
    make_lidar_local_context(
        "/data/tile.laz",
        "local",
        lon=-122.3,
        lat=47.6,
        radius=75.0,
    )

    stages = patch_run_capture["stages"]
    assert stages[0]["type"] == "readers.las"
    assert stages[0]["filename"] == "/data/tile.laz"
    assert stages[1]["type"] == "filters.crop"
    assert "([-75.0, 75.0], [-75.0, 75.0])" in stages[1]["bounds"]


@pytest.mark.usefixtures("patch_run_capture", "patch_exists_true")
def test_make_lidar_local_context_ept_json_uses_ept_reader(
    patch_run_capture: dict[str, Any],
):
    make_lidar_local_context(
        "/ept/tile/ept.json",
        "ctx",
        lon=-100.0,
        lat=40.0,
        radius=25.0,
    )

    stages = patch_run_capture["stages"]
    assert stages[0]["type"] == "readers.ept"
    assert stages[0]["filename"] == "/ept/tile/ept.json"
    assert "([-25.0, 25.0], [-25.0, 25.0])" in stages[0]["bounds"]


@pytest.mark.usefixtures("patch_run_capture", "patch_exists_true")
def test_make_lidar_local_context_multi_las_merge_and_crop(
    patch_run_capture: dict[str, Any],
):
    make_lidar_local_context(
        "/c.laz",
        "not_merged",
        lon=0.0,
        lat=0.0,
        radius=10.0,
    )
    stages = patch_run_capture["stages"]
    types = [s["type"] for s in stages]
    assert "filters.merge" not in types

    make_lidar_local_context(
        ["/a.laz", "/b.laz"],
        "merged",
        lon=0.0,
        lat=0.0,
        radius=10.0,
    )

    stages = patch_run_capture["stages"]
    assert stages[0]["type"] == "readers.las"
    assert stages[0]["filename"] == "/a.laz"
    assert stages[1]["type"] == "readers.las"
    assert stages[1]["filename"] == "/b.laz"
    assert stages[2]["type"] == "filters.merge"
    assert stages[3]["type"] == "filters.crop"


@pytest.mark.usefixtures("patch_run_capture", "patch_exists_true")
def test_make_lidar_local_context_processing_stages_and_writer(
    patch_run_capture: dict[str, Any],
):
    make_lidar_local_context(
        "one.laz",
        "outbase",
        lon=-121.0,
        lat=45.0,
        radius=40.0,
    )

    stages = patch_run_capture["stages"]
    types = [s["type"] for s in stages]

    assert types[0] == "readers.las"
    assert "filters.reprojection" in types
    assert "filters.crop" in types
    assert "filters.smrf" in types
    assert "filters.outlier" in types
    assert "filters.hag_delaunay" in types
    assert types[-1] == "writers.copc"

    writer = stages[-1]
    assert writer["filename"] == "outbase.copc.laz"
    assert writer["extra_dims"] == "all"

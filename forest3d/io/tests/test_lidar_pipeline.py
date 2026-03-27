from __future__ import annotations

import json
import subprocess

import pdal
import pytest

from forest3d.io.lidar.pipeline import (
    LidarPipeline,
    validate_pipeline,
)


class FakeStage:
    def __init__(self, stage_type: str, /, **options) -> None:
        self.type = stage_type
        self.options = {"type": stage_type, **options}

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeStage) and self.options == other.options

    def pipeline(self) -> FakePipeline:
        return FakePipeline([self.options])

    def __or__(self, other: FakeStage) -> FakePipeline:
        return FakePipeline([self.options, other.options])


class FakePipeline:
    def __init__(self, stages: list[dict[str, object]]) -> None:
        self.stages = stages
        self.execute_calls = 0

    def __or__(self, other: FakeStage) -> FakePipeline:
        return FakePipeline([*self.stages, other.options])

    def toJSON(self) -> str:
        return json.dumps(self.stages)

    def execute(self) -> None:
        self.execute_calls += 1


@pytest.fixture
def patch_pdal_stage(monkeypatch) -> None:
    monkeypatch.setattr("forest3d.io.lidar.pipeline.pdal.Stage", FakeStage)


def _valid_result() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["pdal", "pipeline", "--stdin", "--validate"],
        returncode=0,
        stdout="valid",
        stderr="",
    )


def _stats() -> FakeStage:
    return FakeStage("filters.stats", count="Classification")


@pytest.mark.usefixtures("patch_pdal_stage")
def test_lidar_pipeline_add_is_immutable():
    pipeline = LidarPipeline()
    stage_a = FakeStage("filters.range", limits="Z[0:10]")
    stage_b = FakeStage("filters.crop", bounds="([0,1],[2,3])")

    updated = pipeline.add(stage_a)
    expanded = updated.add(stage_b)

    assert pipeline.stages == ()
    assert updated.stages == (stage_a,)
    assert expanded.stages == (stage_a, stage_b)


def test_lidar_pipeline_stores_processing_stages():
    stages = (FakeStage("filters.range", limits="Z[0:10]"),)
    pipeline = LidarPipeline(stages=stages)

    assert pipeline.stages == stages


@pytest.mark.usefixtures("patch_pdal_stage")
def test_lidar_pipeline_to_json_and_hash_are_stable_and_order_sensitive():
    stage_a = FakeStage("filters.range", limits="Z[0:10]")
    stage_b = FakeStage("filters.crop", bounds="([0,1],[2,3])")
    pipeline_one = LidarPipeline(stats=_stats()).add(stage_a).add(stage_b)
    pipeline_two = LidarPipeline(stats=_stats()).add(stage_a).add(stage_b)
    reordered = LidarPipeline(stats=_stats()).add(stage_b).add(stage_a)

    assert json.loads(pipeline_one.to_json()) == [
        {"type": "filters.range", "limits": "Z[0:10]"},
        {"type": "filters.crop", "bounds": "([0,1],[2,3])"},
        {"type": "filters.stats", "count": "Classification"},
    ]
    assert pipeline_one.pipeline_hash == pipeline_two.pipeline_hash
    assert pipeline_one.pipeline_hash != reordered.pipeline_hash


@pytest.mark.usefixtures("patch_pdal_stage")
def test_lidar_pipeline_to_json_and_hash_support_disabling_stats():
    pipeline = LidarPipeline(
        stages=(FakeStage("filters.range", limits="Z[0:10]"),),
        stats=None,
    )

    assert json.loads(pipeline.to_json()) == [
        {"type": "filters.range", "limits": "Z[0:10]"},
    ]
    assert (
        pipeline.pipeline_hash
        != LidarPipeline(
            stages=(FakeStage("filters.range", limits="Z[0:10]"),),
        ).pipeline_hash
    )


@pytest.mark.usefixtures("patch_pdal_stage")
def test_lidar_pipeline_validate_delegates_to_module_helper(monkeypatch):
    pipeline = LidarPipeline(stats=_stats()).add(
        FakeStage("filters.range", limits="Z[0:10]")
    )
    expected = _valid_result()

    monkeypatch.setattr(
        "forest3d.io.lidar.pipeline.validate_pipeline",
        lambda pipeline_json: expected,
    )

    assert pipeline.validate(reader=FakeStage("readers.ept")) is expected


@pytest.mark.usefixtures("patch_pdal_stage")
def test_lidar_pipeline_add_rejects_reader_and_writer_stages():
    with pytest.raises(
        ValueError, match="Processing stages cannot be PDAL readers or writers"
    ):
        LidarPipeline().add(FakeStage("readers.ept"))

    with pytest.raises(
        ValueError, match="Processing stages cannot be PDAL readers or writers"
    ):
        LidarPipeline().add(FakeStage("writers.las"))


@pytest.mark.usefixtures("patch_pdal_stage")
def test_lidar_pipeline_add_replaces_stats_stage_with_warning():
    custom_stats = FakeStage("filters.stats", count="ReturnNumber")

    with pytest.warns(
        UserWarning, match="The previously configured stats stage is being replaced"
    ):
        pipeline = LidarPipeline(stats=_stats()).add(custom_stats)

    assert pipeline.stages == ()
    assert pipeline.stats == custom_stats
    assert json.loads(pipeline.to_json()) == [
        {"type": "filters.stats", "count": "ReturnNumber"}
    ]


@pytest.mark.usefixtures("patch_pdal_stage")
def test_lidar_pipeline_run_executes_in_reader_stats_writer_order(monkeypatch):
    pipeline = LidarPipeline(stats=_stats()).add(
        FakeStage("filters.range", limits="Z[0:10]")
    )
    monkeypatch.setattr(
        "forest3d.io.lidar.pipeline.validate_pipeline",
        lambda pipeline_json: _valid_result(),
    )

    result = pipeline.run(
        reader=FakeStage("readers.ept", filename="https://example.com/ept.json"),
        writer=FakeStage("writers.las", filename="tile.laz"),
    )

    assert result.execute_calls == 1
    assert json.loads(result.toJSON()) == [
        {"type": "readers.ept", "filename": "https://example.com/ept.json"},
        {"type": "filters.range", "limits": "Z[0:10]"},
        {"type": "filters.stats", "count": "Classification"},
        {"type": "writers.las", "filename": "tile.laz"},
    ]


@pytest.mark.usefixtures("patch_pdal_stage")
def test_lidar_pipeline_validate_uses_stage_json_with_default_stats(monkeypatch):
    pipeline = LidarPipeline(stats=_stats()).add(
        FakeStage("filters.range", limits="Z[0:10]")
    )
    captured: list[str] = []
    expected = _valid_result()

    def fake_validate(pipeline_json: str) -> subprocess.CompletedProcess[str]:
        captured.append(pipeline_json)
        return expected

    monkeypatch.setattr(
        "forest3d.io.lidar.pipeline.validate_pipeline",
        fake_validate,
    )

    assert (
        pipeline.validate(
            reader=FakeStage("readers.ept", filename="https://example.com/ept.json"),
            writer=FakeStage("writers.las", filename="tile.laz"),
        )
        is expected
    )
    assert json.loads(captured[0]) == [
        {"type": "readers.ept", "filename": "https://example.com/ept.json"},
        {"type": "filters.range", "limits": "Z[0:10]"},
        {"type": "filters.stats", "count": "Classification"},
        {"type": "writers.las", "filename": "tile.laz"},
    ]


@pytest.mark.usefixtures("patch_pdal_stage")
def test_lidar_pipeline_validate_allows_disabling_default_stats(monkeypatch):
    pipeline = LidarPipeline(
        stages=(FakeStage("filters.range", limits="Z[0:10]"),),
        stats=None,
    )
    captured: list[str] = []

    def fake_validate(pipeline_json: str) -> subprocess.CompletedProcess[str]:
        captured.append(pipeline_json)
        return _valid_result()

    monkeypatch.setattr(
        "forest3d.io.lidar.pipeline.validate_pipeline",
        fake_validate,
    )

    assert (
        pipeline.validate(
            reader=FakeStage("readers.ept", filename="https://example.com/ept.json"),
            writer=FakeStage("writers.las", filename="tile.laz"),
        ).returncode
        == 0
    )
    assert json.loads(captured[0]) == [
        {"type": "readers.ept", "filename": "https://example.com/ept.json"},
        {"type": "filters.range", "limits": "Z[0:10]"},
        {"type": "writers.las", "filename": "tile.laz"},
    ]


@pytest.mark.usefixtures("patch_pdal_stage")
def test_lidar_pipeline_run_allows_none_writer(monkeypatch):
    pipeline = LidarPipeline(stats=_stats()).add(
        FakeStage("filters.range", limits="Z[0:10]")
    )

    monkeypatch.setattr(
        "forest3d.io.lidar.pipeline.validate_pipeline",
        lambda pipeline_json: _valid_result(),
    )

    result = pipeline.run(
        reader=FakeStage("readers.ept", filename="https://example.com/ept.json"),
        writer=None,
    )

    assert result.execute_calls == 1
    assert json.loads(result.toJSON()) == [
        {"type": "readers.ept", "filename": "https://example.com/ept.json"},
        {"type": "filters.range", "limits": "Z[0:10]"},
        {"type": "filters.stats", "count": "Classification"},
    ]


def test_lidar_pipeline_run_raises_for_empty_pipeline_execution():
    pipeline = LidarPipeline(stats=None)

    with pytest.raises(ValueError, match="Cannot execute an empty PDAL pipeline"):
        pipeline.run(reader=[], writer=None)


@pytest.mark.usefixtures("patch_pdal_stage")
def test_lidar_pipeline_run_raises_if_validation_fails(monkeypatch):
    pipeline = LidarPipeline(stats=_stats()).add(
        FakeStage("filters.range", limits="Z[0:10]")
    )

    monkeypatch.setattr(
        "forest3d.io.lidar.pipeline.validate_pipeline",
        lambda pipeline_json: subprocess.CompletedProcess(
            args=["pdal", "pipeline", "--stdin", "--validate"],
            returncode=1,
            stdout="",
            stderr="invalid pipeline",
        ),
    )

    with pytest.raises(
        subprocess.CalledProcessError, match="returned non-zero exit status 1"
    ):
        pipeline.run(
            reader=FakeStage("readers.ept", filename="https://example.com/ept.json"),
            writer=None,
        )


@pytest.mark.usefixtures("patch_pdal_stage")
def test_validate_pipeline_shells_out_over_stdin(monkeypatch):
    captured: dict[str, object] = {}

    expected = _valid_result()

    def fake_run(command, *, input, check, capture_output, text):
        captured["command"] = command
        captured["input"] = input
        captured["check"] = check
        captured["capture_output"] = capture_output
        captured["text"] = text
        return expected

    monkeypatch.setattr("forest3d.io.lidar.pipeline.subprocess.run", fake_run)

    result = validate_pipeline('{"pipeline": "value"}')

    assert result is expected
    assert captured["command"] == [
        "pdal",
        "pipeline",
        "--stdin",
        "--validate",
    ]
    assert captured["input"] == '{"pipeline": "value"}'
    assert captured["check"] is False
    assert captured["capture_output"] is True
    assert captured["text"] is True


### PDAL integration tests ###
# These tests should not use the monkeypatch fixture. They rely upon real PDAL stages,
# but may fail if the PDAL API changes or if PDAL is not configured correctly in the
# testing environment.
def test_validate_pipeline_succeeds_with_real_pdal_stages():
    """Test that validate_pipeline succeeds with real PDAL stages.

    Relies upon pdal.Reader.faux, pdal.Writer.null being available.
    """
    pipeline_json = (pdal.Reader.faux("ignored", count=4) | pdal.Writer.null()).toJSON()

    result = validate_pipeline(pipeline_json)

    assert result.returncode == 0
    assert json.loads(result.stdout)["valid"] is True


def test_lidar_pipeline_run_executes_with_real_pdal_stages():
    """Test that lidar_pipeline.run executes with real PDAL stages.

    Relies upon pdal.Reader.faux, pdal.Writer.null, pdal.Filter.range, and
    pdal.Filter.stats being available.
    """
    pipeline = LidarPipeline().add(pdal.Filter.range(limits="Z[0:10]"))

    result = pipeline.run(
        reader=pdal.Reader.faux("ignored", count=4),
        writer=pdal.Writer.null(),
    )

    assert isinstance(result, pdal.Pipeline)
    assert [stage["type"] for stage in json.loads(result.toJSON())] == [
        "readers.faux",
        "filters.range",
        "filters.stats",
        "writers.null",
    ]

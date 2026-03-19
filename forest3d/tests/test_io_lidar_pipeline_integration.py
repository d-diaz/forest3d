from __future__ import annotations

import json
import shutil

import pdal
import pytest

from forest3d.io.lidar.pipeline import LidarPipeline, validate_pipeline

HAS_PDAL_CLI = shutil.which("pdal") is not None
HAS_REQUIRED_STAGES = all(
    [
        hasattr(pdal.Reader, "faux"),
        hasattr(pdal.Writer, "null"),
        hasattr(pdal.Filter, "range"),
        hasattr(pdal.Filter, "stats"),
    ]
)

pytestmark = pytest.mark.skipif(
    not (HAS_PDAL_CLI and HAS_REQUIRED_STAGES),
    reason="Requires the PDAL CLI plus faux/null/range stage bindings",
)


def test_validate_pipeline_succeeds_with_real_pdal_stages():
    pipeline_json = (pdal.Reader.faux("ignored", count=4) | pdal.Writer.null()).toJSON()

    result = validate_pipeline(pipeline_json)

    assert result.returncode == 0
    assert json.loads(result.stdout)["valid"] is True


def test_lidar_pipeline_run_executes_with_real_pdal_stages():
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

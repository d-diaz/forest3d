from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import pdal


@dataclass(frozen=True)
class LidarPipeline:
    """An immutable template for composing and executing PDAL pipelines.

    The pipeline stores processing-stage configuration only. Reader stages are
    supplied when validating or running the pipeline, and writer stages are
    optional outputs appended at execution time.

    Attributes:
        stages: Ordered processing stages (PDAL filter stages). These exclude
            reader and writer stages, which are provided separately.
        stats: Optional ``filters.stats`` stage appended after the configured
            processing stages. When a writer is present, the stats stage is
            inserted immediately before it. Set to ``None`` to disable
            automatic stats collection. The default stats stage returns the
            mean, min, and max values for each set of point attributes, as
            well as the count of points for each Classification value.
    """

    stages: tuple[pdal.Stage, ...] = ()
    stats: pdal.Stage | None = pdal.Filter.stats(count="Classification")

    @property
    def pipeline_hash(self) -> str:
        """Compute a short hash of the processing pipeline and configured stats.

        The hash is computed from the JSON representation of the pipeline and
        configured stats. It does not include reader or writer stages.

        Returns:
            str: A short hash of the processing pipeline and configured stats.
        """

        return hashlib.sha1(self.to_json().encode()).hexdigest()[:12]

    def add(self, stage: pdal.Stage) -> LidarPipeline:
        """Return a new pipeline with one additional processing stage.

        Add a new PDAL filter stage to the pipeline. If the stage is a stats stage,
        it will replace the existing stats stage if it exists.

        Args:
            stage (pdal.Stage): A PDAL filter stage to add to the pipeline.

        Returns:
            LidarPipeline: A new pipeline with the additional stage.

        Raises:
            ValueError: If the stage is a PDAL reader or writer.
            UserWarning: If the stage is a stats stage and the pipeline already
                has a stats stage.
        """

        stage_type = getattr(stage, "type", "")
        if stage_type == "filters.stats":
            if self.stats is not None:
                warnings.warn(
                    "The previously configured stats stage is being replaced",
                    stacklevel=2,
                )
            return LidarPipeline(stages=self.stages, stats=stage)
        if stage_type.startswith("readers.") or stage_type.startswith("writers."):
            raise ValueError("Processing stages cannot be PDAL readers or writers")
        return LidarPipeline(self.stages + (stage,))

    def to_json(self) -> str:
        """Return PDAL-authored JSON string for the processing stages."""

        stages = list(self.stages)
        if self.stats is not None:
            stages.append(self.stats)
        pipeline = LidarPipeline._compose(stages)
        if pipeline is None:
            return json.dumps([])

        return pipeline.toJSON()

    def validate(
        self,
        *,
        reader: pdal.Stage | Iterable[pdal.Stage],
        writer: pdal.Stage | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Validate the processing stages or a composed pipeline with the PDAL CLI."""

        pipeline = self._compose(self._stages_with_io(reader=reader, writer=writer))
        pipeline_json = json.dumps([]) if pipeline is None else pipeline.toJSON()
        return validate_pipeline(pipeline_json)

    def run(
        self,
        *,
        reader: pdal.Stage | Iterable[pdal.Stage],
        writer: pdal.Stage | None = None,
    ) -> pdal.Pipeline:
        """Execute the composed pipeline after validating it with the PDAL CLI."""

        pipeline = self._compose(self._stages_with_io(reader=reader, writer=writer))
        if pipeline is None:
            raise ValueError("Cannot execute an empty PDAL pipeline")

        validation = validate_pipeline(pipeline.toJSON())
        if validation.returncode != 0:
            raise subprocess.CalledProcessError(
                validation.returncode,
                validation.args,
                output=validation.stdout,
                stderr=validation.stderr,
            )

        pipeline.execute()
        return pipeline

    @staticmethod
    def _compose(stages: Iterable[pdal.Stage]) -> pdal.Pipeline | None:
        """Compose a sequence of PDAL stages into a PDAL pipeline."""

        stage_list = list(stages)
        if not stage_list:
            return None

        composed: pdal.Stage | pdal.Pipeline = stage_list[0]
        for stage in stage_list[1:]:
            composed = composed | stage

        if isinstance(composed, pdal.Stage):
            return composed.pipeline()

        return composed

    def _stages_with_io(
        self,
        *,
        reader: pdal.Stage | Iterable[pdal.Stage],
        writer: pdal.Stage | None = None,
    ) -> tuple[pdal.Stage, ...]:
        """Return processing stages composed with input and optional output stages."""

        stages: list[pdal.Stage] = (
            [reader] if isinstance(reader, pdal.Stage) else list(reader)
        )
        stages.extend(self.stages)
        if self.stats is not None:
            stages.append(self.stats)
        if writer is not None:
            stages.append(writer)
        return tuple(stages)


def validate_pipeline(pipeline_json: str) -> subprocess.CompletedProcess[str]:
    """Validate pipeline JSON through the PDAL CLI."""

    return subprocess.run(
        ["pdal", "pipeline", "--stdin", "--validate"],
        input=pipeline_json,
        check=False,
        capture_output=True,
        text=True,
    )


def is_ept_json(src: str) -> bool:
    """Check if a file is an EPT JSON file."""

    return src.endswith(".json") and src.endswith("ept.json")


def is_las_laz(src: str) -> bool:
    """Check if a file is a LAS or LAZ file."""

    return src.endswith(".las") or src.endswith(".laz")

import logging
import os

import pdal
from pyproj import CRS

from forest3d.io.lidar.pipeline import LidarPipeline, is_ept_json, is_las_laz


def make_lidar_local_context(
    src: str | list[str], name: str, lon: float, lat: float, radius: float
) -> LidarPipeline:
    """Extract, clean, and reproject raw lidar into a local projection.

    This pipeline extracts points witin radius of the user-specified lat/lon
    coordinates, reprojects the point cloud into an AEQD projection, and crops
    the point cloud to a bounding box that encloses the specified radius.
    Existing classifications other than noise (18 and 7) are discarded.
    Low noise points are then detected and classified as noise. Ground points
    are thne classified using the ``smrf`` filter excluding any points
    classified as noise or withheld. Outliers are then flagged and classified
    as noise. Height above ground is calculated and added as a new dimension.

    Args:
        src (str | list[str]): The path(s) to the input LAS/LAZ point cloud file.
        name (str): The name of the output file (without extension, output file
            will be `{name}.copc.laz`)
        lon (float): The longitude of the center of the local context.
        lat (float): The latitude of the center of the local context.
        radius (float): The radius of the local context in meters.

    Returns:
        The pipeline run result.

    Raises:
        ValueError: If the source file is not EPT or LAS/LAZ files.
    """

    pipeline = LidarPipeline()
    aeqd_proj = (
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 "
        "+ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    )
    horiz = CRS.from_string(aeqd_proj)
    vert = CRS.from_epsg(5703)
    compound_crs = CRS.from_user_input(
        f"COMPOUNDCRS['AEQD_NAVD88', {horiz.to_wkt()}, {vert.to_wkt()}]"
    ).to_wkt()
    bounds = f"([{-radius}, {radius}], [-{radius}, {radius}]) / {compound_crs}"
    reproject = pdal.Filter.reprojection(out_srs=compound_crs)
    crop = pdal.Filter.crop(bounds=bounds)

    # if src is a str and is ept.json and make ept reader with filtered aoi
    if isinstance(src, str) and is_ept_json(src):
        reader = pdal.Reader.ept(filename=src, bounds=bounds)

    else:
        if isinstance(src, str):
            if not is_las_laz(src):
                msg = (
                    f"Invalid source file: {src}, only ept, las, and laz sources"
                    " are supported"
                )
                raise ValueError(msg)
            reader = [pdal.Reader.las(filename=src), pdal.Filter.crop(bounds=bounds)]
        else:
            for s in src:
                if not is_las_laz(s):
                    msg = (
                        f"Invalid source file: {s}, only ept, las and laz sources"
                        " are supported"
                    )
                    raise ValueError(msg)
            reader = [pdal.Reader.las(filename=s) for s in src] + [
                pdal.Filter.merge(),
                pdal.Filter.crop(bounds=bounds),
            ]

    # followed by final crop to desired bounds in AEQDafter reprojecting
    pipeline = (
        pipeline.add(reproject)
        .add(crop)
        .add(pdal.Filter.assign(value="ReturnNumber = 1", where="ReturnNumber < 1"))
        .add(
            pdal.Filter.assign(value="NumberOfReturns = 1", where="NumberOfReturns < 1")
        )
        .add(
            pdal.Filter.assign(
                value="Classification = 1",
                where="Classification != 7 && Classification != 18",
            )
        )
        .add(pdal.Filter.elm(where="Classification == 1"))
        .add(pdal.Filter.smrf(window=30, where="Classification == 1 && Withheld == 0"))
        .add(
            pdal.Filter.outlier(
                method="statistical",
                multiplier=3,
                mean_k=8,
                where="Classification == 1",
            )
        )
        .add(pdal.Filter.hag_delaunay())
    )
    result = pipeline.run(
        reader=reader,
        writer=pdal.Writer.copc(filename=f"{name}.copc.laz", extra_dims="all"),
    )

    if os.path.exists(f"{name}.copc.laz"):
        logging.info(f"Successfully wrote point cloud to {name}.copc.laz")
    else:
        logging.error(f"Failed to write point cloud to {name}.copc.laz")

    return result


def make_lidar_dem(
    src: str,
    dst: str,
    output_type: str = "idw",
    resolution: float = 0.5,
    radius: float = 3.0,
    window_size=5.0,
) -> LidarPipeline:
    """Generate a Digital Elevation Model (DEM) from a LAS/LAZ point cloud.

    Args:
        src: The path to the input LAS/LAZ point cloud file.
        dst: The path to the output DEM file.
        output_type: The type of interpolation to use (e.g., "idw", "mean")
        resolution: The resolution of the DEM.
        radius: The radius of the search window for points to use for
            interpolation.
        window_size: The size of the window to use as a fallback for pixels
            that have no points within the radius.

    Returns:
        The pipeline run result.
    """
    reader = pdal.Reader.las(filename=src)
    writer = pdal.Writer.gdal(
        gdaldriver="GTiff",
        dimension="Z",
        where="Classification == 2 && Withheld == 0",
        filename=dst,
        output_type=output_type,
        resolution=resolution,
        radius=radius,
        window_size=window_size,
    )
    pipeline = LidarPipeline()
    result = pipeline.run(reader=reader, writer=writer)

    if os.path.exists(dst):
        logging.info(f"Successfully wrote DEM to {dst}")
    else:
        logging.error(f"Failed to write DEM to {dst}")

    return result


def make_lidar_dsm(
    src: str, dst: str, resolution: float = 0.5, radius: float = 3.0
) -> LidarPipeline:
    """Generate a Digital Surface Model (DSM) from a LAS/LAZ point cloud.

    Args:
        src: The path to the input LAS/LAZ point cloud file.
        dst: The path to the output DSM file.
        resolution: The resolution of the DSM.
        radius: The radius of the search window for points to use for
            interpolation.

    Returns:
        The pipeline run result.
    """
    reader = pdal.Reader.las(filename=src)
    writer = pdal.Writer.gdal(
        gdaldriver="GTiff",
        dimension="Z",
        output_type="max",
        where="Classification != 7 && Classification != 18 && Withheld == 0",
        filename=dst,
        radius=radius,
        resolution=resolution,
    )
    pipeline = LidarPipeline()
    result = pipeline.run(reader=reader, writer=writer)

    if os.path.exists(dst):
        logging.info(f"Successfully wrote DSM to {dst}")
    else:
        logging.error(f"Failed to write DSM to {dst}")

    return result

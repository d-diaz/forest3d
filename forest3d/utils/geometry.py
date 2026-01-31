"""Functions for creating 3D geometric representations of trees."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import rasterio
from jax import Array
from jax.typing import ArrayLike
from shapely.geometry import Point, Polygon

from forest3d.models.hull_params import TreeHullParams


def _make_crown_hull_from_params(
    params: TreeHullParams, *, num_theta: int = 32, num_z: int = 50
) -> Array:
    """Convenience wrapper around `_make_crown_hull` that takes a params container."""
    return _make_crown_hull(
        stem_base=jnp.asarray(params.stem_base),
        top_height=jnp.asarray(params.top_height),
        crown_ratio=jnp.asarray(params.crown_ratio),
        lean_direction=jnp.asarray(params.lean_direction),
        lean_severity=jnp.asarray(params.lean_severity),
        crown_radii=jnp.asarray(params.crown_radii),
        crown_edge_heights=jnp.asarray(params.crown_edge_heights),
        crown_shapes=jnp.asarray(params.crown_shapes),
        num_theta=num_theta,
        num_z=num_z,
    )


def _make_crown_hull_batched(
    params: TreeHullParams, *, num_theta: int = 32, num_z: int = 50
) -> Array:
    """Vectorized crown hull over a batch of trees.

    Parameters
    ----------
    params : TreeHullParams
        A params container whose fields are stacked with a leading batch dimension `B`.
    num_theta, num_z : int
        Crown sampling resolution. For best performance under `jit`, treat these as
        static configuration (compile once per resolution).

    Returns
    -------
    points : jax.Array, shape (B, num_z * num_theta, 3)
        Batched crown hull points.
    """

    def _single(p: TreeHullParams) -> Array:
        return _make_crown_hull_from_params(p, num_theta=num_theta, num_z=num_z)

    return jax.vmap(_single)(params)


def _arrays_equal_shape(*args: np.ndarray, raise_exc: bool = True) -> bool:
    """Confirms all inputs, when converted  arrays, have equal shape.

    Parameters
    -----------
    args : array-like
        any arguments that can be converted to arrays with np.asanyarray
    raise_exc : boolean
        whether to raise a ValueError exception

    Returns:
    --------
    result : bool
        whether or not all args have same shape

    """
    arrs = [np.asanyarray(arg) for arg in args]
    shapes = np.array([arr.shape for arr in arrs])
    equal_shapes = np.all(shapes == shapes[0])

    if not equal_shapes and raise_exc:
        message = f"Input shapes mismatch: {shapes}"
        raise ValueError(message)

    return equal_shapes


def _get_raster_bbox_as_polygon(path_to_raster: str | os.PathLike) -> Polygon:
    """Returns a Shapely Polygon defining the bounding box of a raster.

    Parameters
    ----------
    path_to_raster : string, path to file
        A raster image that can be read by rasterio.

    Returns:
    --------
    bbox : shapely Polygon object
        A polygon describing the bounding box of the raster
    """
    with rasterio.open(path_to_raster) as raster_src:
        pass

    west_edge, south_edge, east_edge, north_edge = raster_src.bounds
    points = [
        Point(west_edge, south_edge),  # lower left corner
        Point(west_edge, north_edge),  # upper left corner
        Point(east_edge, north_edge),  # upper right corner
        Point(east_edge, south_edge),  # lower left corner
    ]

    return Polygon([(p.x, p.y) for p in points])


def get_elevation(
    dem: str | os.PathLike, x: float | np.ndarray, y: float | np.ndarray
) -> float | np.ndarray:
    """Calculates elevations from a DEM at specified (x, y) coordinates.

    Parameters
    ----------
    dem : string, path to file
        A digital elevation model in a format that can be read by rasterio.
    x : numeric, or numpy array of numeric values
        x-coordinate(s) of points to query
    y : numeric, or numpy array of numeric values
        y-coordinate(s) of points to query

    Returns:
    --------
    elev : numpy array
        elevation at specified (x, y) coordinates
    """
    x, y = np.asanyarray(x), np.asanyarray(y)
    with rasterio.open(dem) as src:
        BAND_ONE = 1
        terrain = src.read(BAND_ONE)

    # check that inputs are equal shape
    _arrays_equal_shape(x, y)

    coords = np.stack((x, y))
    # have rasterio identify the rows and columns where these coordinates occur
    if coords.shape == (2,):
        rows, cols = src.index(x, y)
    else:
        rows, cols = [], []
        for x_val, y_val in zip(x, y):
            row, col = src.index(x_val, y_val)
            rows.append(row)
            cols.append(row)
    # rows, cols = src.index(*coords)
    rows = np.array(rows)
    cols = np.array(cols)
    # index into the raster at these rows and columns
    try:
        elev = terrain[rows, cols]
    except IndexError:
        bounds = src.bounds
        error_msg = f"""
        (x,y) location outside bounds of elevation raster:
        {bounds}"""
        raise IndexError(error_msg)

    return elev


def _get_treetop_location(
    stem_base: ArrayLike,
    top_height: ArrayLike,
    lean_direction: ArrayLike | None = None,
    lean_severity: ArrayLike | None = None,
) -> Array:
    """Calculates 3D coordinates for the top of a tree.

    Allows specification of direction and severity of leaning. This location represents
    the translation in x, y, and z directions from (0,0,0) to identify the tree top

    Parameters
    -----------
    stem_base : array with shape(3,)
        (x,y,z) coordinates of stem base
    top_height : numeric, or array of numeric values
        vertical height of the tree apex from the base of the stem
    lean_direction : numeric, or array of numeric values
        direction of tree lean, in degrees with 0 = east, 90 = north, and
        180 = west
    lean_severity : numeric, or array of numeric values
        how much the tree is leaning, in degrees from vertical; 0 = no lean,
        and 90 meaning the tree is horizontal.

    Returns:
    --------
    top_translate_x, top_translate_y, top_translate_z : three values or arrays
        Coodrinates that define the translation of the tree top from (0,0,0)
    """
    stem_base = jnp.asarray(stem_base)
    top_height = jnp.asarray(top_height)
    stem_x, stem_y, stem_z = stem_base

    if lean_direction is None:
        lean_direction = jnp.zeros_like(stem_x)
    lean_direction = jnp.asarray(lean_direction)

    if lean_severity is None:
        lean_severity = jnp.zeros_like(stem_x)
    lean_severity = jnp.asarray(lean_severity)

    theta_lean = jnp.deg2rad(lean_direction)
    phi_lean = jnp.deg2rad(lean_severity)

    top_translate_x = stem_x + top_height * jnp.tan(phi_lean) * jnp.cos(theta_lean)
    top_translate_y = stem_y + top_height * jnp.tan(phi_lean) * jnp.sin(theta_lean)
    top_translate_z = stem_z

    return jnp.array((top_translate_x, top_translate_y, top_translate_z))


def _get_peripheral_points(
    crown_radii: ArrayLike,
    crown_edge_heights: ArrayLike,
    top_height: ArrayLike,
    crown_ratio: ArrayLike,
) -> Array:
    """Calculates the x,y,z coordinates of the points of maximum crown width.

    One point for E, N, W, and S directions around a tree.

    Parameters
    -----------
    crown_radii : array of numerics, shape (4,)
        distance from stem base to point of maximum crown width in each
        direction. Order of radii expected is E, N, W, S.
    crown_edge_heights : array of numerics, shape (4,)
        proportion of crown length above point of maximum crown width in each
        direction. Order expected is E, N, W, S. For example, values of
        (0, 0, 0, 0) would indicate that maximum crown width in all directions
        occurs at the base of the crown, while (0.5, 0.5, 0.5, 0.5) would
        indicate that maximum crown width in all directions occurs half way
        between crown base and crown apex.
    top_height : numeric, or array of numeric values
        vertical height of the tree apex from the base of the stem
    crown_ratio : numeric, or array of numeric values
        ratio of live crown length to total tree height

    Returns:
    --------
    periph_pts : array with shape (4, 3)
        (x,y,z) coordinates of points at maximum crown width
    """
    crown_base_height = top_height * (1 - crown_ratio)
    crown_length = crown_ratio * top_height

    (crown_radius_east, crown_radius_north, crown_radius_west, crown_radius_south) = (
        crown_radii
    )
    (crown_edgeht_east, crown_edgeht_north, crown_edgeht_west, crown_edgeht_south) = (
        crown_edge_heights
    )

    east_point = jnp.array(
        (crown_radius_east, 0, crown_base_height + crown_edgeht_east * crown_length),
        dtype=float,
    )

    north_point = jnp.array(
        (0, crown_radius_north, crown_base_height + crown_edgeht_north * crown_length),
        dtype=float,
    )

    west_point = jnp.array(
        (-crown_radius_west, 0, crown_base_height + crown_edgeht_west * crown_length),
        dtype=float,
    )

    south_point = jnp.array(
        (0, -crown_radius_south, crown_base_height + crown_edgeht_south * crown_length),
        dtype=float,
    )

    return jnp.stack((east_point, north_point, west_point, south_point))


def _get_hull_center_xy(crown_radii: ArrayLike) -> Array:
    """Calculates x,y coordinates of center of crown projection.

    The center of the crown projection is determined as the midpoint between
    points of maximum crown width in the x and y directions.

    Parameters
    -----------
    crown_radii : array of numerics, shape (4,)
        distance from stem base to point of maximum crown width in each
        direction. Order of radii expected is E, N, W, S.

    Returns:
    --------
    center_xy : array with shape (2,)
        x,y coordinates of the center of the crown hull
    """
    crown_radii = jnp.asarray(crown_radii)
    crown_radii_eastwest = crown_radii[0::2]
    crown_radii_northsouth = crown_radii[1::2]
    center_xy = jnp.array(
        (jnp.diff(crown_radii_eastwest / 2), jnp.diff(crown_radii_northsouth) / 2)
    )
    return center_xy[:, 0]


def _get_hull_eccentricity(crown_radii: np.ndarray, crown_ratio: float) -> np.ndarray:
    """Calculates eccentricity-index values for an asymmetric hull.

    Represents a tree crown, with eccentricity-index values used to determine
    the x,y positions of the base and the apex of a tree crown.

    The eccentricity-index is defined by Koop (1989, p.49-51) as 'the ratio of
    distance between tree base and centre point of the crown projection and
    crown radius'. Eccentricity-index values should range [-1, 1]. A value of 0
    indicates the x,y location of the tree apex or base is at the center of the
    horizontal crown projection. Values that approach -1 or 1 indicate the x,y
    location of the tree apex or base is near the edge of the crown.

        Koop, H. (1989). Forest Dynamics: SILVI-STAR: A Comprehensive
        Monitoring System. Springer: New York.

    Parameters
    -----------
    crown_radii : array of numerics, shape (4,)
        distance from stem base to point of maximum crown width in each
        direction. Order of radii expected is E, N, W, S.
    crown_ratio : numeric
        ratio of live crown length to total tree height

    Returns:
    --------
    idx : array with shape (2, 2)
        eccentricity-index values for the top (0, ) and bottom of a tree (1, ).
    """
    crown_radii_array = jnp.asarray(crown_radii)
    crown_ratio_array = jnp.asarray(crown_ratio)
    center_xy = _get_hull_center_xy(crown_radii_array)
    center_x, center_y = center_xy
    crown_radii_eastwest = crown_radii_array[0::2]
    crown_radii_northsouth = crown_radii_array[1::2]

    eccen = jnp.array(
        (
            center_x / crown_radii_eastwest.mean(),  # x direction
            center_y / crown_radii_northsouth.mean(),  # y direction
        )
    )
    idx = jnp.array(
        (
            -2 / jnp.pi * jnp.arctan(eccen) * crown_ratio_array,  # top of tree, x and y
            2 / jnp.pi * jnp.arctan(eccen) * crown_ratio_array,
        )
    )
    return idx


def _get_hull_apex_and_base(
    crown_radii: np.ndarray, top_height: float | np.ndarray, crown_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the (x,y,z) position of the apex and base of a tree crown.

    This models a tree crown as an asymmetric hull comprised of
    quarter-ellipses.

    Parameters
    -----------
    crown_radii : array of numerics, shape (4,)
        distance from stem base to point of maximum crown width in each
        direction. Order of radii expected is E, N, W, S.
    top_height : numeric, or array of numeric values
        vertical height of the tree apex from the base of the stem
    crown_ratio : numeric
        ratio of live crown length to total tree height


    Returns:
    --------
    hull_apex, hull_base : arrays with shape (3,)
        (x,y,z) coordinates for apex and base of hull representing tree crown
    """
    crown_radii_array = jnp.asarray(crown_radii)
    top_height_array = jnp.asarray(top_height)
    crown_ratio_array = jnp.asarray(crown_ratio)

    center_xy = _get_hull_center_xy(crown_radii_array)
    eccen_idx = _get_hull_eccentricity(crown_radii_array, crown_ratio_array)

    center_x, center_y = center_xy
    crown_radii_eastwest = crown_radii_array[0::2]
    crown_radii_northsouth = crown_radii_array[1::2]
    top_eccen_eastwest, top_eccen_northsouth = eccen_idx[0]
    bottom_eccen_eastwest, bottom_eccen_northsouth = eccen_idx[1]

    hull_apex = jnp.array(
        (
            center_x
            + jnp.diff(crown_radii_eastwest)[0]
            * top_eccen_eastwest,  # x location of crown apex
            center_x
            + jnp.diff(crown_radii_northsouth)[0]
            * top_eccen_northsouth,  # y location of crown apex
            top_height_array,
        ),
        dtype=float,
    )

    hull_base = jnp.array(
        (
            center_x
            + jnp.diff(crown_radii_eastwest)[0]
            * bottom_eccen_eastwest,  # x location of crown base
            center_y
            + jnp.diff(crown_radii_northsouth)[0]
            * bottom_eccen_northsouth,  # y location of crown base
            top_height_array * (1 - crown_ratio_array),
        ),
        dtype=float,
    )

    return hull_apex, hull_base


def _get_circular_plot_boundary(
    x: np.ndarray,
    y: np.ndarray,
    radius: np.ndarray,
    dem: str | os.PathLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns coordinates of 32 points along the circumference of a circular plot.

    If a digital elevation model readable by rasterio is also provided, the
    elevations of the circumerference points will also be calculated.

    Parameters
    -----------
    x : numeric, or numpy array of numeric values
        x-coordinate of plot center
    y : numeric, or numpy array of numeric values
        y-coordinate of plot center
    radius : numeric, or numpy array of numeric values
        radius of plot
    dem : string, path to file
        A digial elevation model in a format that can be read by rasterio

    Returns:
    --------
    xs, ys, zs : numpy arrays, each with shape (32,)
        x, y, and z coordinates of the plot boundary
    """
    thetas = np.linspace(0, 2 * np.pi, 32)
    xs = radius * np.cos(thetas) + x
    ys = radius * np.sin(thetas) + y

    zs = get_elevation(dem, xs, ys) if dem else np.zeros(32)

    return xs, ys, zs


def _make_crown_hull(
    stem_base: np.ndarray,
    top_height: float,
    crown_ratio: float,
    lean_direction: float,
    lean_severity: float,
    crown_radii: np.ndarray,
    crown_edge_heights: np.ndarray,
    crown_shapes: np.ndarray,
    num_theta: int = 32,
    num_z: int = 50,
) -> Array:
    """Makes a crown hull.

    Notes on required input constraints
    ----------------------------------
    This function is intentionally kept free of eager (Python-side) validation so it
    can be composed with JAX transforms like `jit`/`vmap`. As a result, **callers are
    responsible for ensuring inputs satisfy constraints**. Without them, some
    parameter combinations can yield NaNs/infs due to division-by-zero or invalid
    exponentiation.

    In practice, we enforce these constraints at API boundaries (e.g., the `Tree`
    model in `forest3d/models/dataclass.py`) and later via JIT-compatible checks.

    Constraints (single-tree)
    -------------------------
    - `stem_base`: shape `(3,)`, finite
    - `top_height`: finite, `> 0`
    - `crown_ratio`: in `(0, 1]` (avoid zero-length crowns where base==apex)
    - `lean_severity`: in `[0, 90)` (strictly < 90 to avoid `tan(pi/2)` overflow)
    - `crown_radii`: shape `(4,)`, finite, `>= 0`, not all zeros
    - `crown_edge_heights`: shape `(4,)`, finite, in `[0, 1)` (avoid peripheral line at apex)
    - `crown_shapes`: shape `(2,4)`, finite, strictly `> 0`

    Parameters
    ----------
    stem_base : array with shape(3,)
        (x,y,z) coordinates of stem base
    top_height : numeric
        vertical height of the tree apex from the base of the stem
    crown_ratio : numeric
        ratio of live crown length to total tree height
    lean_direction : numeric
        direction of tree lean, in degrees with 0 = east, 90 = north,
        180 = west, etc.
    lean_severity : numeric
        how much tree is leaning, in degrees from vertical; 0 = no lean,
        and 90 meaning the tree is horizontal
    crown_radii : array of numerics, shape (4,)
        distance from stem base to point of maximum crown width in each
        direction. Order of radii expected is E, N, W, S.
    crown_edge_heights : array of numerics, shape (4,)
        proportion of crown length above point of maximum crown width in each
        direction. Order expected is E, N, W, S. For example, values of
        (0, 0, 0, 0) would indicate that maximum crown width in all directions
        occurs at the base of the crown, while (0.5, 0.5, 0.5, 0.5) would
        indicate that maximum crown width in all directions occurs half way
        between crown base and crown apex.
    crown_shapes : array with shape (4,2)
        shape coefficients describing curvature of crown profiles
        in each direction (E, N, W, S) for top and bottom of crown
    num_theta : int, optional
        number of points along the circumference of the crown
    num_z : int, optional
        number of points along the height of the crown

    Returns
    -------
    points : numpy.ndarray, shape (num_z * num_theta, 3)
        3D points on the crown hull surface. Point order matches the previous
        `(x, y, z)` flattening order so downstream mesh topology remains stable.
    """
    translate_x, translate_y, translate_z = _get_treetop_location(
        stem_base, top_height, lean_direction, lean_severity
    )

    crown_radii_array = jnp.asarray(crown_radii)
    crown_edge_heights_array = jnp.asarray(crown_edge_heights)
    crown_shapes_array = jnp.asarray(crown_shapes)

    periph_points = _get_peripheral_points(
        crown_radii=crown_radii_array,
        crown_edge_heights=crown_edge_heights_array,
        top_height=jnp.asarray(top_height),
        crown_ratio=jnp.asarray(crown_ratio),
    )
    periph_points_xs = periph_points[:, 0]
    periph_points_ys = periph_points[:, 1]
    periph_points_zs = periph_points[:, 2]

    hull_apex, hull_base = _get_hull_apex_and_base(crown_radii, top_height, crown_ratio)
    apex_x, apex_y, apex_z = hull_apex
    base_x, base_y, base_z = hull_base

    # places where we'll calculate crown surface
    thetas = jnp.linspace(0, 2 * jnp.pi, num_theta)  # angles
    zs = jnp.linspace(base_z, apex_z, num_z)  # heights
    grid_thetas, grid_zs = jnp.meshgrid(thetas, zs)

    # calculate height difference between apex and peripheral points
    periph_points_height_from_apex = apex_z - periph_points_zs

    # calculate radial (horizontal) distance from apex axis to periph points
    top_periph_points_radii = jnp.hypot(
        periph_points_ys - apex_y, periph_points_xs - apex_x
    )

    # calculate the angle between peripheral points and apex axis
    apex_vs_periph_points_thetas = jnp.arctan2(
        periph_points_ys - apex_y, periph_points_xs - apex_x
    )

    # calculate radii along peripheral line (maximum crown widths by angle
    # theta using linear interpolation)
    apex_periph_line_radii = jnp.interp(
        grid_thetas,
        apex_vs_periph_points_thetas,
        top_periph_points_radii,
        period=2 * jnp.pi,
    )

    # convert peripheral line to x,y,z coords
    periph_line_xs = apex_periph_line_radii * jnp.cos(grid_thetas) + apex_x
    periph_line_ys = apex_periph_line_radii * jnp.sin(grid_thetas) + apex_y
    periph_line_zs = apex_z - jnp.interp(
        grid_thetas,
        apex_vs_periph_points_thetas,
        periph_points_height_from_apex,
        period=2 * jnp.pi,
    )

    # identify those points in the grid that are higher than the periph line
    grid_top = grid_zs >= periph_line_zs

    # calculate the shape coefficients at each angle theta (relative to apex)
    # using linear interpolation
    top_shapes_measured = crown_shapes_array[0]
    bottom_shapes_measured = crown_shapes_array[1]
    top_shapes_interp = jnp.interp(
        grid_thetas,
        apex_vs_periph_points_thetas,
        top_shapes_measured,
        period=2 * jnp.pi,
    )

    # calculate crown radius at each height z for top of crown
    #
    # Clamp the height deltas to ≥ 0 before exponentiation.
    # `grid_zs - periph_line_zs` is negative for points below the peripheral line.
    # Those values are later masked out by `grid_top`, but JAX may still evaluate
    # them during tracing. Negative bases to non-integer exponents produce NaNs.
    top_delta_z = jnp.maximum(grid_zs - periph_line_zs, 0.0)
    top_inner = (
        1
        - top_delta_z**top_shapes_interp
        / (apex_z - periph_line_zs) ** top_shapes_interp
    )
    # Clamp the “inner” term to ≥ 0 before taking a fractional power.
    # Floating point roundoff can make `top_inner` slightly negative (e.g. -1e-6),
    # and `negative ** fractional` yields NaNs.
    top_inner = jnp.maximum(top_inner, 0.0)
    top_hull_radii = ((top_inner) * apex_periph_line_radii**top_shapes_interp) ** (
        1 / top_shapes_interp
    )

    # generate the full crown
    # calculate the angle between peripheral points and base axis
    base_vs_periph_points_thetas = jnp.arctan2(
        periph_points_ys - base_y, periph_points_xs - base_x
    )

    # identify those points in the grid that are higher than the
    # peripheral line
    grid_bottom = grid_zs < periph_line_zs

    # calculate the angles between points on the peripheral line and crown
    # base
    bottom_periph_line_thetas = jnp.arctan2(
        periph_line_ys - base_y, periph_line_xs - base_x
    )

    # calculate radial distance between points on the peripheral line and
    # crown base
    base_periph_line_radii = jnp.hypot(periph_line_xs - base_x, periph_line_ys - base_y)

    # calculate the shape coefficients at each angle theta (relative to
    # crown base) using linear interpolation
    bottom_shapes_interp = jnp.interp(
        bottom_periph_line_thetas,
        base_vs_periph_points_thetas,
        bottom_shapes_measured,
        period=2 * jnp.pi,
    )

    # calculate crown radius at height z
    #
    # Clamp the height deltas to ≥ 0 before exponentiation.
    # `periph_line_zs - grid_zs` is negative for points above the peripheral line.
    # Those values are later masked out by `grid_bottom`, but JAX may still evaluate
    # them during tracing. Negative bases to non-integer exponents produce NaNs.
    bottom_delta_z = jnp.maximum(periph_line_zs - grid_zs, 0.0)
    bottom_denom_z = periph_line_zs - base_z
    # Make the bottom denominator safe when it’s exactly zero.
    # This can happen at azimuths where `crown_edge_height == 0` (peripheral line at
    # crown base). The bottom region at those azimuths is empty, but we still need
    # the expression to be numerically safe under tracing.
    bottom_denom_z_safe = jnp.where(bottom_denom_z == 0, 1.0, bottom_denom_z)
    bottom_inner = (
        1
        - bottom_delta_z**bottom_shapes_interp
        / bottom_denom_z_safe**bottom_shapes_interp
    )
    # Clamp the “inner” term to ≥ 0 before taking a fractional power.
    # Floating point roundoff can make `bottom_inner` slightly negative, and
    # `negative ** fractional` yields NaNs.
    bottom_inner = jnp.maximum(bottom_inner, 0.0)
    bottom_hull_radii = (
        (bottom_inner) * base_periph_line_radii**bottom_shapes_interp
    ) ** (1 / bottom_shapes_interp)

    hull_radii = jnp.where(grid_bottom, bottom_hull_radii, top_hull_radii)

    # calculate cartesian coordinates of crown edge points
    grid_xs = jnp.where(
        grid_top,
        hull_radii * jnp.cos(grid_thetas) + apex_x,
        hull_radii * jnp.cos(bottom_periph_line_thetas) + base_x,
    )
    grid_ys = jnp.where(
        grid_top,
        hull_radii * jnp.sin(grid_thetas) + apex_y,
        hull_radii * jnp.sin(bottom_periph_line_thetas) + base_y,
    )

    crown_xs = grid_xs + translate_x
    crown_ys = grid_ys + translate_y
    crown_zs = grid_zs + translate_z

    return jnp.column_stack((crown_xs.ravel(), crown_ys.ravel(), crown_zs.ravel()))

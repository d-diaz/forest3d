"""Functions for generating interactive visualizations of 3D models of trees."""

import os
import warnings
from typing import Any

import k3d
import numpy as np
import pandas as pd
import seaborn as sns
from ipywidgets import Accordion, FloatSlider, HBox, Layout, VBox, interactive_output

from forest3d.models.dataclass import Tree
from forest3d.models.dataframe import TreeListDataFrameModel
from forest3d.utils.geometry import get_elevation

warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", message="invalid value encountered in greater_equal")
warnings.filterwarnings("ignore", message="invalid value encountered in less")
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")


def _make_tree_all_params(
    species,
    dbh,
    top_height,
    stem_x,
    stem_y,
    stem_z,
    lean_direction,
    lean_severity,
    crown_ratio,
    crown_radius_east,
    crown_radius_north,
    crown_radius_west,
    crown_radius_south,
    crown_edge_height_east,
    crown_edge_height_north,
    crown_edge_height_west,
    crown_edge_height_south,
    shape_top_east,
    shape_top_north,
    shape_top_west,
    shape_top_south,
    shape_bottom_east,
    shape_bottom_north,
    shape_bottom_west,
    shape_bottom_south,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates a tree and returns its crown as a hull.

    Exposes all parameters used as individual arguments.

    This is used primarily for the plotting functions in the visualization.py
    script in this package. The parameters are the same as involved in
    instantiating a Tree object.

    Returns:
    --------
    x, y, z : numpy arrays
        the x, y, and z coordinates of points that occur along the edge of the
        tree crown.
    """
    crown_radii = np.array(
        (crown_radius_east, crown_radius_north, crown_radius_west, crown_radius_south)
    )

    crown_edge_heights = np.array(
        (
            crown_edge_height_east,
            crown_edge_height_north,
            crown_edge_height_west,
            crown_edge_height_south,
        )
    )

    crown_shapes = np.array(
        (
            (shape_top_east, shape_top_north, shape_top_west, shape_top_south),
            (
                shape_bottom_east,
                shape_bottom_north,
                shape_bottom_west,
                shape_bottom_south,
            ),
        )
    )

    tree = Tree(
        species=species,
        dbh=dbh,
        top_height=top_height,
        stem_x=stem_x,
        stem_y=stem_y,
        stem_z=stem_z,
        lean_direction=lean_direction,
        lean_severity=lean_severity,
        crown_ratio=crown_ratio,
        crown_radii=crown_radii,
        crown_edge_heights=crown_edge_heights,
        crown_shapes=crown_shapes,
    )

    return tree.crown()


def build_tree_controls() -> tuple[Accordion, dict[str, Any]]:
    """Builds the User Interface (UI) and controls for the single tree visualization."""

    top_height = FloatSlider(
        value=75, min=0, max=150, step=1.0, description="height", orientation="vertical"
    )
    stem_x = FloatSlider(value=0, min=-10, max=10, step=1.0, description="x")
    stem_y = FloatSlider(value=0, min=-10, max=10, step=1.0, description="y")
    stem_z = FloatSlider(value=0, min=-10, max=10, step=1.0, description="z")
    lean_direction = FloatSlider(min=0, max=360, step=1.0, description="direction")
    lean_severity = FloatSlider(min=0, max=89, step=1.0, description="severity")
    crown_ratio = FloatSlider(
        value=0.65,
        min=0,
        max=1.0,
        step=0.01,
        description="crown ratio",
        orientation="vertical",
    )
    crown_radius_east = FloatSlider(
        value=10, min=0, max=30, step=1.0, description="east"
    )
    crown_radius_north = FloatSlider(
        value=10, min=0, max=30, step=1.0, description="north"
    )
    crown_radius_west = FloatSlider(
        value=10, min=0, max=30, step=1.0, description="west"
    )
    crown_radius_south = FloatSlider(
        value=10, min=0, max=30, step=1.0, description="south"
    )
    crown_edge_height_east = FloatSlider(
        value=0.3, min=0, max=1, step=0.01, description="east", orientation="vertical"
    )
    crown_edge_height_north = FloatSlider(
        value=0.3, min=0, max=1, step=0.01, description="north", orientation="vertical"
    )
    crown_edge_height_west = FloatSlider(
        value=0.3, min=0, max=1, step=0.01, description="west", orientation="vertical"
    )
    crown_edge_height_south = FloatSlider(
        value=0.3, min=0, max=1, step=0.01, description="south", orientation="vertical"
    )
    shape_top_east = FloatSlider(
        value=2.0, min=0.0, max=3.0, step=0.1, description="top, east"
    )
    shape_top_north = FloatSlider(
        value=2.0, min=0.0, max=3.0, step=0.1, description="top, north"
    )
    shape_top_west = FloatSlider(
        value=2.0, min=0.0, max=3.0, step=0.1, description="top, west"
    )
    shape_top_south = FloatSlider(
        value=2.0, min=0.0, max=3.0, step=0.1, description="top, south"
    )
    shape_bottom_east = FloatSlider(
        value=2.0, min=0.0, max=3.0, step=0.1, description="bottom, east"
    )
    shape_bottom_north = FloatSlider(
        value=2.0, min=0.0, max=3.0, step=0.1, description="bottom, north"
    )
    shape_bottom_west = FloatSlider(
        value=2.0, min=0.0, max=3.0, step=0.1, description="bottom, west"
    )
    shape_bottom_south = FloatSlider(
        value=2.0, min=0.0, max=3.0, step=0.1, description="bottom, south"
    )

    # Group the parameter widgets into groups of controls
    height_controls = HBox([top_height, crown_ratio])
    edge_height_controls = HBox(
        [
            crown_edge_height_east,
            crown_edge_height_north,
            crown_edge_height_west,
            crown_edge_height_south,
        ]
    )
    location_controls = VBox([stem_x, stem_y, stem_z])
    lean_controls = VBox([lean_direction, lean_severity])
    radius_controls = VBox(
        [crown_radius_east, crown_radius_north, crown_radius_west, crown_radius_south]
    )
    shape_controls = VBox(
        [
            shape_top_east,
            shape_top_north,
            shape_top_west,
            shape_top_south,
            shape_bottom_east,
            shape_bottom_north,
            shape_bottom_west,
            shape_bottom_south,
        ]
    )
    # create and expandable user interface
    ui = Accordion(
        [
            location_controls,
            height_controls,
            lean_controls,
            radius_controls,
            edge_height_controls,
            shape_controls,
        ]
    )
    ui.set_title(0, "Stem Location")
    ui.set_title(1, "Tree Height")
    ui.set_title(2, "Tree Lean")
    ui.set_title(3, "Crown Radius")
    ui.set_title(4, "Crown Edge Heights")
    ui.set_title(5, "Crown Shapes")

    controls = {
        "top_height": top_height,
        "stem_x": stem_x,
        "stem_y": stem_y,
        "stem_z": stem_z,
        "lean_direction": lean_direction,
        "lean_severity": lean_severity,
        "crown_ratio": crown_ratio,
        "crown_radius_east": crown_radius_east,
        "crown_radius_north": crown_radius_north,
        "crown_radius_west": crown_radius_west,
        "crown_radius_south": crown_radius_south,
        "crown_edge_height_east": crown_edge_height_east,
        "crown_edge_height_north": crown_edge_height_north,
        "crown_edge_height_west": crown_edge_height_west,
        "crown_edge_height_south": crown_edge_height_south,
        "shape_top_east": shape_top_east,
        "shape_top_north": shape_top_north,
        "shape_top_west": shape_top_west,
        "shape_top_south": shape_top_south,
        "shape_bottom_east": shape_bottom_east,
        "shape_bottom_north": shape_bottom_north,
        "shape_bottom_west": shape_bottom_west,
        "shape_bottom_south": shape_bottom_south,
    }

    return ui, controls


def plot_single_tree_interactive(
    species: str = "Douglas-fir",
    dbh: float = 5.0,
    color: int = 0x2CA02C,
    point_size: float = 1.0,
) -> k3d.Plot:
    """Plots a tree with k3d and returns the plot."""
    ui, controls = build_tree_controls()

    plot = k3d.plot(grid_visible=False, height=600)
    plot.layout = Layout(
        width="100%",
        min_width="0px",
        height="600px",
        flex="1 1 auto",
    )
    pts = k3d.points(
        positions=np.zeros((1, 3), dtype=np.float32),
        color=color,
        point_size=point_size,
    )
    plot += pts

    def update(
        top_height,
        stem_x,
        stem_y,
        stem_z,
        lean_direction,
        lean_severity,
        crown_ratio,
        crown_radius_east,
        crown_radius_north,
        crown_radius_west,
        crown_radius_south,
        crown_edge_height_east,
        crown_edge_height_north,
        crown_edge_height_west,
        crown_edge_height_south,
        shape_top_east,
        shape_top_north,
        shape_top_west,
        shape_top_south,
        shape_bottom_east,
        shape_bottom_north,
        shape_bottom_west,
        shape_bottom_south,
    ):
        x, y, z = _make_tree_all_params(
            species=species,
            dbh=dbh,
            top_height=top_height,
            stem_x=stem_x,
            stem_y=stem_y,
            stem_z=stem_z,
            lean_direction=lean_direction,
            lean_severity=lean_severity,
            crown_ratio=crown_ratio,
            crown_radius_east=crown_radius_east,
            crown_radius_north=crown_radius_north,
            crown_radius_west=crown_radius_west,
            crown_radius_south=crown_radius_south,
            crown_edge_height_east=crown_edge_height_east,
            crown_edge_height_north=crown_edge_height_north,
            crown_edge_height_west=crown_edge_height_west,
            crown_edge_height_south=crown_edge_height_south,
            shape_top_east=shape_top_east,
            shape_top_north=shape_top_north,
            shape_top_west=shape_top_west,
            shape_top_south=shape_top_south,
            shape_bottom_east=shape_bottom_east,
            shape_bottom_north=shape_bottom_north,
            shape_bottom_west=shape_bottom_west,
            shape_bottom_south=shape_bottom_south,
        )

        positions = np.column_stack([x, y, z]).astype(np.float32)

        with plot.hold_sync():
            pts.positions = positions

    out = interactive_output(update, controls)
    out.layout.display = "none"  # keep it alive, but don’t show an empty output area

    update(**{k: w.value for k, w in controls.items()})

    return VBox(
        [
            HBox(
                [ui, plot],
                layout=Layout(
                    width="100%",
                    display="flex",
                    align_items="stretch",
                    justify_content="space-between",
                    gap="12px",
                ),
            ),
            out,
        ],
        layout=Layout(width="100%"),
    )


def plot_tree_list(
    trees: TreeListDataFrameModel,
    dem: str | os.PathLike | None = None,
    sample: int | None = None,
    crown_shape: tuple[int, int] = (50, 32),
) -> k3d.Plot:
    """Plots an interactive 3D view of a tree list.

    Parameters
    -----------
    trees (TreeListDataFrameModel): a validated dataframe or geodataframe
    dem (str, optional): path to elevation raster readable by rasterio,
        will be used to calculate elevation on a grid and produce a ground
        surface underneath the trees.
    sample (int, optional): number of trees to sample from the tree list.
    """
    df = pd.DataFrame(trees).copy()
    if sample is not None:
        df = df.sample(n=sample)

    show_dem = dem is not None

    # stem_z from DEM (avoid mutating caller)
    if dem is not None:
        df["stem_z"] = get_elevation(dem, df["stem_x"], df["stem_y"])
    elif "stem_z" not in df.columns:
        df["stem_z"] = 0.0

    spp = pd.unique(df["species"])
    palette = sns.color_palette("colorblind", len(spp))
    spp_to_color = {s: _rgb_to_k3d_int(palette[i]) for i, s in enumerate(spp)}

    plot = k3d.plot(grid_visible=False, height=700)

    # Optional DEM surface
    if show_dem and dem is not None:
        dem_grid = 100
        xs = np.linspace(df.stem_x.min(), df.stem_x.max(), dem_grid)
        ys = np.linspace(df.stem_y.min(), df.stem_y.max(), dem_grid)
        xx, yy = np.meshgrid(xs, ys)
        elev = (
            np.asanyarray(get_elevation(dem, xx.ravel(), yy.ravel()))
            .reshape(dem_grid, dem_grid)
            .astype(np.float32)
        )
        plot += k3d.surface(
            elev,
            xmin=float(xs.min()),
            xmax=float(xs.max()),
            ymin=float(ys.min()),
            ymax=float(ys.max()),
            color=0x8B4513,
            wireframe=False,
            opacity=0.8,
        )

    n_rows, n_cols = crown_shape

    # We’ll decide whether to “wrap” the mesh around the azimuth seam by checking
    # whether column 0 and last column are (nearly) identical.
    # (If they are identical, you do NOT want wrap_cols=True or you’ll double-connect.)
    wrap_cols_default = True

    # Group meshes by species to reduce number of k3d objects
    by_species: dict[str, list] = {s: [] for s in spp}

    for _, row in df.iterrows():
        tree = Tree(
            species=row["species"],
            dbh=float(row["dbh"]),
            top_height=float(row["top_height"]),
            stem_x=float(row["stem_x"]),
            stem_y=float(row["stem_y"]),
            stem_z=float(row.get("stem_z", 0.0)),
            crown_ratio=float(row.get("crown_ratio", 0.65)),
            crown_radius=row.get("crown_radius", None),
            lean_direction=float(row.get("lean_direction", 0.0)),
            lean_severity=float(row.get("lean_severity", 0.0)),
            num_theta=n_cols,
            num_z=n_rows,
        )
        x, y, z = tree.crown(num_theta=n_cols, num_z=n_rows)
        X = x.reshape((n_rows, n_cols))
        Y = y.reshape((n_rows, n_cols))
        Z = z.reshape((n_rows, n_cols))

        by_species[row["species"]].append((X, Y, Z))

    # Build one k3d.mesh per species (batched vertices + batched indices)
    for s, crowns in by_species.items():
        if not crowns:
            continue

        # decide wrap once using the first crown in this species
        X0, Y0, Z0 = crowns[0]
        seam_is_duplicate = (
            np.allclose(X0[:, 0], X0[:, -1])
            and np.allclose(Y0[:, 0], Y0[:, -1])
            and np.allclose(Z0[:, 0], Z0[:, -1])
        )
        wrap_cols = (not seam_is_duplicate) and wrap_cols_default
        base_faces = _grid_triangles(n_rows, n_cols, wrap_cols=wrap_cols)

        verts_list = []
        faces_list = []
        offset = 0

        for X, Y, Z in crowns:
            verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
                np.float32
            )
            verts_list.append(verts)

            faces_list.append(base_faces + offset)
            offset += verts.shape[0]

        vertices = np.vstack(verts_list)
        indices = np.vstack(faces_list)

        plot += k3d.mesh(
            vertices=vertices,
            indices=indices,
            color=spp_to_color[s],
            wireframe=False,
            opacity=1.0,
        )

    return plot


def random_forest(n_trees: int) -> k3d.Plot:
    """Plots a forest of randomly-generated tree crowns (k3d mesh) + flat DEM.

    Args:
        n_trees (int): number of trees to generate

    Returns:
        k3d.Plot: a plot of the forest
    """
    n_rows, n_cols = 50, 32
    faces = _grid_triangles(n_rows, n_cols, wrap_cols=True)

    palette = sns.color_palette("colorblind", n_trees)
    plot = k3d.plot(grid_visible=False, height=700)

    # bounds across ALL crowns
    x_min = y_min = z_min = np.inf
    x_max = y_max = z_max = -np.inf

    crowns_vertices = []  # store to avoid re-generating
    crowns_colors = []

    for _ in range(n_trees):
        x, y, z = _make_random_tree_crown()
        X = x.reshape((n_rows, n_cols))
        Y = y.reshape((n_rows, n_cols))
        Z = z.reshape((n_rows, n_cols))

        x_min = min(x_min, float(X.min()))
        x_max = max(x_max, float(X.max()))
        y_min = min(y_min, float(Y.min()))
        y_max = max(y_max, float(Y.max()))
        z_min = min(z_min, float(Z.min()))
        z_max = max(z_max, float(Z.max()))

        vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
        crowns_vertices.append(vertices)

        color = _rgb_to_k3d_int(palette[np.random.randint(0, len(palette))])
        crowns_colors.append(color)

    # Build a flat DEM
    flat = np.full((50, 50), float(0), dtype=np.float32)
    plot += k3d.surface(
        flat,
        xmin=x_min,
        xmax=x_max,
        ymin=y_min,
        ymax=y_max,
        color=0x8B4513,
        opacity=0.6,
        wireframe=False,
    )

    # Add crowns
    for vertices, color in zip(crowns_vertices, crowns_colors):
        plot += k3d.mesh(vertices=vertices, indices=faces, color=color, opacity=1.0)

    return plot


def _rgb_to_k3d_int(rgb_float_triplet) -> int:
    r, g, b = (int(255 * c) for c in rgb_float_triplet)
    return (r << 16) + (g << 8) + b


def _grid_triangles(n_rows: int, n_cols: int, wrap_cols: bool) -> np.ndarray:
    """Generates triangles for a grid.

    Args:
        n_rows : int
            The number of rows in the grid.
        n_cols : int
            The number of columns in the grid.
        wrap_cols : bool

    Returns:
        np.ndarray of shape (2 * n_rows * n_cols, 3)
    """
    # vertex ids laid out like the grid
    vid = np.arange(n_rows * n_cols, dtype=np.uint32).reshape(n_rows, n_cols)

    if wrap_cols:
        v00 = vid[:-1, :]  # (r, c)
        v10 = vid[1:, :]  # (r+1, c)
        v01 = np.roll(vid[:-1, :], -1, axis=1)  # (r, c+1 mod n_cols)
        v11 = np.roll(vid[1:, :], -1, axis=1)  # (r+1, c+1 mod n_cols)
    else:
        v00 = vid[:-1, :-1]
        v10 = vid[1:, :-1]
        v01 = vid[:-1, 1:]
        v11 = vid[1:, 1:]

    # two triangles per quad: (v00, v10, v01) and (v01, v10, v11)
    t1 = np.stack([v00, v10, v01], axis=-1).reshape(-1, 3)
    t2 = np.stack([v01, v10, v11], axis=-1).reshape(-1, 3)
    return np.vstack([t1, t2]).astype(np.uint32)


def _make_random_tree_crown() -> tuple[np.array, np.array, np.array]:
    """Generates a random tree crown."""
    dbh = np.random.rand() * 40
    top_height = np.random.randint(low=50, high=200)
    crown_ratio = np.random.randint(low=40, high=95) / 100
    stem_x = np.random.rand() * 500 - 250
    stem_y = np.random.rand() * 500 - 250
    stem_z = np.random.rand() * 10 - 5
    crown_radii = np.random.randint(low=10, high=60, size=4) / 100 * top_height
    crown_edge_heights = np.random.rand(4)
    crown_shapes = np.random.randint(low=50, high=300, size=(2, 4)) / 100
    lean_direction = np.random.rand() * 360
    lean_severity = np.random.rand() * 10

    tree = Tree(
        species="Douglas-fir",
        dbh=dbh,
        top_height=top_height,
        stem_x=stem_x,
        stem_y=stem_y,
        stem_z=stem_z,
        crown_ratio=crown_ratio,
        crown_radii=crown_radii,
        crown_edge_heights=crown_edge_heights,
        crown_shapes=crown_shapes,  # note: crown_shapes (not crown_shape)
        lean_direction=lean_direction,
        lean_severity=lean_severity,
    )
    return tree.crown()

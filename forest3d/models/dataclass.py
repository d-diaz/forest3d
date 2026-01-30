from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from forest3d.utils.geometry import _make_crown_hull


class Coordinate3D(BaseModel):
    """A single 3D coordinate."""

    x: float
    y: float
    z: float

    def to_numpy(self):
        """Returns coordinate as np.array with shape (3,)."""
        return np.array(self.x, self.y, self.z)


class CoordinateSet3D(BaseModel):
    """A set of 3D coordinates."""

    xs: NDArray[Shape["*"], float]  # noqa: F722
    ys: NDArray[Shape["*"], float]  # noqa: F722
    zs: NDArray[Shape["*"], float]  # noqa: F722


class Tree(BaseModel):
    """Class for tree attributes that can generate different types of crowns for it.

    Attributes:
    -----------
    species : string
        tree species code or name
    dbh : numeric
        diameter at breast height
    top_height : numeric
        vertical height of the tree apex from the base of the stem
    stem_x : numeric
        x-coordinate stem base
    stem_y : numeric
        y-coordinate of stem base
    stem_z : numeric
        z-coordinate of stem base
    lean_direction : numeric
        direction of tree lean, in degrees with 0 = east, 90 = north,
        180 = west, etc.
    lean_severity : numeric
        how much tree is leaning, in degrees from vertical; 0 = no lean,
        and 90 meaning the tree is horizontal
    crown_ratio : numeric
        ratio of live crown length to total tree height
    crown_radii : array of numerics, shape (4,)
        distance from stem base to point of maximum crown width in each
        direction. Order of radii expected is E, N, W, S.
    crown_edge_heights : array of numerics, shape (4,)
        proportion of crown length above point of maximum crown width in
        each direction. Order expected is E, N, W, S. For example, values
        of (0, 0, 0, 0) would indicate that maximum crown width in all
        directions occurs at the base of the crown, while (0.5, 0.5, 0.5,
        0.5) would indicate that maximum crown width in all directions
        occurs half way between crown base and crown apex.
    crown_shapes : array with shape (2,4)
        shape coefficients describing curvature of crown profiles
        in each direction (E, N, W, S) for top and bottom of crown. The
        crown_shapes[0, 0:4] describe the shape of the top of the crown.
        crown_shapes[1, 0:4] describe the shape of the bottom of the crown.
        Coef values of 1.0 produce a cone, values < 1 produce concave
        shapes, and values > 1 will produce convex shapes, with coef == 2.0
        producing an ellipse.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    species: str
    dbh: float = Field(gt=0)
    top_height: float = Field(gt=0)
    stem_x: float
    stem_y: float
    stem_z: float = Field(default=0)
    lean_direction: float = Field(default=0, ge=0, le=360)
    lean_severity: float = Field(default=0, ge=0, le=90)
    crown_ratio: float | int = Field(default=0.65, ge=0, le=1.0)
    crown_radius: float | int | None = None
    crown_radii: NDArray[Shape["4"], float | int] | None = None  # noqa: UP037
    crown_edge_heights: NDArray[Shape["4"], float] = np.array(  # noqa: UP037
        (0.3, 0.3, 0.3, 0.3)
    )
    crown_shapes: NDArray[Shape["2, 4"], float] = np.full((2, 4), fill_value=2.0)  # noqa: UP037

    @model_validator(mode="before")
    def crown_radii_from_radius(self):
        """Calculates crown radii if a single radius is given."""
        if "crown_radius" in self and "crown_radii" not in self:
            self["crown_radii"] = np.full(4, self["crown_radius"])
        return self

    @model_validator(mode="after")
    def default_crown_radii(self):
        """Sets default value for crown radii as 25% of height."""
        if self.crown_radii is None:
            self.crown_radii = np.full(4, 0.25 * self.top_height)
        return self

    @model_validator(mode="before")
    def percent_live_crown_to_ratio(self):
        """Divides crown_ratio by 100 if supplied value is greater than 1.0."""
        if "crown_ratio" in self:
            if self["crown_ratio"] > 1:
                self["crown_ratio"] /= 100.0
        return self

    @property
    def stem_base(self) -> NDArray[Shape["3"], float]:  # noqa: UP037,F722
        """Coordinates for the base of the stem."""
        return np.array((self.stem_x, self.stem_y, self.stem_z))

    stem_base = computed_field(stem_base)

    def crown(
        self, num_theta: int = 32, num_z: int = 50
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Generates a crown hull for this tree.

        Args:
        num_theta : int, optional
            number of points along the circumference of the crown
        num_z : int, optional
            number of points along the height of the crown

        Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray]
            x, y, z coordinates of the crown hull
        """
        return _make_crown_hull(
            stem_base=self.stem_base,
            top_height=self.top_height,
            crown_ratio=self.crown_ratio,
            lean_direction=self.lean_direction,
            lean_severity=self.lean_severity,
            crown_radii=self.crown_radii,
            crown_edge_heights=self.crown_edge_heights,
            crown_shapes=self.crown_shapes,
            num_theta=num_theta,
            num_z=num_z,
        )

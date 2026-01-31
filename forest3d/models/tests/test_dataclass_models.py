import numpy as np
import pytest
from pydantic import ValidationError

from forest3d.models.dataclass import Tree

TREE_WITH_RADIUS = {
    "species": "abc",
    "dbh": 10.0,
    "top_height": 10.0,
    "crown_ratio": 0.8,
    "crown_radius": 3.0,
    "stem_x": 5.0,
    "stem_y": 10.0,
}

TREE_WITH_RADIUS_AND_RADII = {
    "species": "abc",
    "dbh": 10.0,
    "top_height": 10.0,
    "crown_ratio": 0.8,
    "crown_radius": 3.0,
    "crown_radii": np.array((1.0, 2.0, 3.0, 4.0)),
    "stem_x": 5.0,
    "stem_y": 10.0,
}


def test_crown_radius_to_radii():
    tree = Tree.model_validate(TREE_WITH_RADIUS)
    assert tree.crown_radius == TREE_WITH_RADIUS["crown_radius"]
    assert np.all(tree.crown_radii == TREE_WITH_RADIUS["crown_radius"])


def test_tree_with_radius_and_radii():
    tree = Tree.model_validate(TREE_WITH_RADIUS_AND_RADII)
    assert tree.crown_radius == TREE_WITH_RADIUS_AND_RADII["crown_radius"]
    assert np.array_equal(tree.crown_radii, TREE_WITH_RADIUS_AND_RADII["crown_radii"])


def test_tree_rejects_lean_severity_90():
    bad = dict(TREE_WITH_RADIUS, lean_severity=90)
    with pytest.raises(ValidationError):
        Tree.model_validate(bad)


def test_tree_rejects_crown_shapes_non_positive():
    bad = dict(TREE_WITH_RADIUS, crown_shapes=np.zeros((2, 4)))
    with pytest.raises(ValidationError):
        Tree.model_validate(bad)


def test_tree_rejects_crown_edge_heights_at_or_above_one():
    bad = dict(TREE_WITH_RADIUS, crown_edge_heights=np.array((1.0, 0.3, 0.3, 0.3)))
    with pytest.raises(ValidationError):
        Tree.model_validate(bad)

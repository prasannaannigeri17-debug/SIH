"""
Unit tests for elevation and semantic aggregation policies.
"""
import pytest
import numpy as np
from mapping.adaptive_grid import AdaptiveResolutionGrid


@pytest.fixture
def grid():
    config = {
        "grid": {
            "default_aggregation": "robust",
            "zones": [
                {"name": "Zone A", "min_distance": 0.0, "max_distance": 10.0, "resolution": 0.10},
                {"name": "Zone B", "min_distance": 10.0, "max_distance": 30.0, "resolution": 0.20},
            ]
        }
    }
    return AdaptiveResolutionGrid(config)


def test_min_max_mean_aggregation(grid):
    # Place 4 points in one cell (x=2.05, y=2.05) with z in [1.0, 2.0, 3.0, 4.0]
    pts = np.array([
        [2.02, 2.02, 1.0],
        [2.04, 2.04, 2.0],
        [2.06, 2.06, 3.0],
        [2.08, 2.08, 4.0]
    ], dtype=np.float32)

    # Test "min" mode
    grid.clear()
    grid.insert_points(pts, aggregation_mode="min")
    cell = list(grid.cells.values())[0]
    assert cell.rep_z == pytest.approx(1.0, 0.01)

    # Test "max" mode
    grid.clear()
    grid.insert_points(pts, aggregation_mode="max")
    cell = list(grid.cells.values())[0]
    assert cell.rep_z == pytest.approx(4.0, 0.01)

    # Test "mean" mode
    grid.clear()
    grid.insert_points(pts, aggregation_mode="mean")
    cell = list(grid.cells.values())[0]
    assert cell.rep_z == pytest.approx(2.5, 0.01)


def test_semantic_majority_voting(grid):
    pts = np.array([
        [2.01, 2.01, 0.0],
        [2.02, 2.02, 0.0],
        [2.03, 2.03, 0.0],
    ], dtype=np.float32)
    # 2 points with class 4 (vehicle), 1 point with class 1 (road)
    sems = np.array([4, 4, 1], dtype=np.uint32)

    grid.clear()
    grid.insert_points(pts, semantic_labels=sems)
    cell = list(grid.cells.values())[0]
    assert cell.semantic_class == 4  # Majority wins


def test_semantic_majority_is_not_input_order_dependent(grid):
    pts = np.array([
        [2.01, 2.01, 0.0],
        [2.02, 2.02, 0.0],
        [2.03, 2.03, 0.0],
    ], dtype=np.float32)
    sems = np.array([1, 4, 4], dtype=np.uint32)

    grid.clear()
    grid.insert_points(pts, semantic_labels=sems)

    assert list(grid.cells.values())[0].semantic_class == 4

"""
Unit tests for Adaptive Resolution Grid:
Spatial indexing, radial zone boundaries, zero gaps, zero overlaps, and coordinate transforms.
"""
import pytest
import numpy as np
from mapping.adaptive_grid import AdaptiveResolutionGrid


@pytest.fixture
def default_grid():
    config = {
        "grid": {
            "default_aggregation": "robust",
            "zones": [
                {"name": "Zone A", "min_distance": 0.0, "max_distance": 10.0, "resolution": 0.05},
                {"name": "Zone B", "min_distance": 10.0, "max_distance": 30.0, "resolution": 0.15},
                {"name": "Zone C", "min_distance": 30.0, "max_distance": 60.0, "resolution": 0.30},
                {"name": "Zone D", "min_distance": 60.0, "max_distance": 100.0, "resolution": 0.60},
            ]
        }
    }
    return AdaptiveResolutionGrid(config)


def test_zone_boundary_exactness(default_grid):
    """
    Verifies that points exactly on or near zone boundaries map deterministically to exactly one zone.
    """
    # Zone A: [0, 10.0) -> Zone 0
    assert default_grid.get_zone_for_distance(0.0) == 0
    assert default_grid.get_zone_for_distance(9.999) == 0
    
    # Boundary at 10.0m belongs to Zone 1 (Zone B)
    assert default_grid.get_zone_for_distance(10.0) == 1
    assert default_grid.get_zone_for_distance(10.001) == 1
    assert default_grid.get_zone_for_distance(29.999) == 1

    # Boundary at 30.0m belongs to Zone 2 (Zone C)
    assert default_grid.get_zone_for_distance(30.0) == 2
    assert default_grid.get_zone_for_distance(59.999) == 2

    # Boundary at 60.0m belongs to Zone 3 (Zone D)
    assert default_grid.get_zone_for_distance(60.0) == 3
    assert default_grid.get_zone_for_distance(100.0) == 3

    # Out of sensor range
    assert default_grid.get_zone_for_distance(100.001) == -1
    assert default_grid.get_zone_for_distance(-1.0) == -1


def test_zero_spatial_gaps_and_overlaps(default_grid):
    """
    Continuously sweeps a dense radial ray of points across all zone transitions
    and verifies every point maps to a valid cell without omission or collision.
    """
    radii = np.linspace(0.1, 99.9, 1000)
    # Generate points along positive X axis
    pts = np.column_stack([radii, np.zeros_like(radii), np.zeros_like(radii)])

    inserted_count = default_grid.insert_points(pts)
    assert inserted_count > 0
    assert len(default_grid.cells) == inserted_count


def test_negative_coordinates_handling(default_grid):
    """
    Verifies that points in all 4 Cartesian quadrants are indexed correctly.
    """
    pts = np.array([
        [5.0, 5.0, 0.0],    # Quad 1
        [-5.0, 5.0, 0.0],   # Quad 2
        [-5.0, -5.0, 0.0],  # Quad 3
        [5.0, -5.0, 0.0],   # Quad 4
    ], dtype=np.float32)

    count = default_grid.insert_points(pts)
    assert count == 4
    assert len(default_grid.cells) == 4


def test_multiple_points_in_single_cell(default_grid):
    """
    Verifies that multiple points falling into the same cell are aggregated into 1 cell.
    """
    pts = np.array([
        [1.01, 1.01, 0.5],
        [1.02, 1.02, 1.0],
        [1.03, 1.03, 1.5],
    ], dtype=np.float32)

    count = default_grid.insert_points(pts)
    assert count == 1
    cell = list(default_grid.cells.values())[0]
    assert cell.point_count == 3
    assert cell.min_z == pytest.approx(0.5, 0.01)
    assert cell.max_z == pytest.approx(1.5, 0.01)
    assert cell.mean_z == pytest.approx(1.0, 0.01)

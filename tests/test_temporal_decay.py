"""
Unit tests for Temporal Updates, Decay, and Dynamic Object Eviction.
"""
import pytest
import numpy as np
from mapping.adaptive_grid import AdaptiveResolutionGrid
from mapping.temporal_map import TemporalMapManager


@pytest.fixture
def manager():
    config = {
        "grid": {
            "zones": [
                {"name": "Zone A", "min_distance": 0.0, "max_distance": 20.0, "resolution": 0.10}
            ]
        },
        "temporal": {
            "enable_decay": True,
            "static_decay_rate": 0.02,
            "dynamic_decay_rate": 0.80,
            "stale_timeout_sec": 1.5
        }
    }
    grid = AdaptiveResolutionGrid(config)
    return TemporalMapManager(config, grid)


def test_dynamic_cell_eviction(manager):
    # Insert dynamic obstacle (vehicle) at t=0.0s
    pts = np.array([[5.0, 5.0, 0.0]], dtype=np.float32)
    sem = np.array([4], dtype=np.uint32)
    conf = np.array([0.9], dtype=np.float32)
    is_dyn = np.array([True], dtype=bool)

    manager.update_frame(pts, sem, conf, is_dyn, timestamp=0.0)
    assert len(manager.grid.cells) == 1

    # Advance time by 2.0s (> stale_timeout_sec 1.5s) with no points at that location
    pts_new = np.array([[10.0, 10.0, 0.0]], dtype=np.float32)
    manager.update_frame(pts_new, sem, conf, is_dyn, timestamp=2.0)

    # Original cell at (5.0, 5.0) must be evicted!
    assert len(manager.grid.cells) == 1
    cell = list(manager.grid.cells.values())[0]
    assert cell.x_center == pytest.approx(10.05, 0.1)

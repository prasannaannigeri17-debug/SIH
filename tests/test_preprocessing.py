"""
Unit tests for Preprocessing & Ground extraction.
"""
import pytest
import numpy as np
from preprocessing.pointcloud_processor import PointCloudProcessor
from preprocessing.ground_detector import GroundDetector
from data.loaders.base_loader import PointCloudFrame


def test_pointcloud_processor_filters():
    cfg = {"sensor": {"min_range": 1.0, "max_range": 50.0, "min_z": -2.0, "max_z": 5.0}}
    processor = PointCloudProcessor(cfg)

    # Points with NaN, out of range, out of Z
    pts = np.array([
        [0.0, 0.0, 0.0],       # r=0 < min_range
        [5.0, 5.0, 0.0],       # Valid (r=7.07m)
        [60.0, 60.0, 0.0],     # r > 50m
        [5.0, 5.0, 10.0],      # z > 5m
        [np.nan, 5.0, 0.0],    # NaN
    ], dtype=np.float32)

    frame = PointCloudFrame(frame_id=0, timestamp=0.0, points=pts)
    filtered = processor.process(frame)

    assert len(filtered.points) == 1
    assert filtered.points[0, 0] == pytest.approx(5.0, 0.01)


def test_ground_detector():
    cfg = {"terrain": {"ground_elevation_tolerance": 0.15}}
    detector = GroundDetector(cfg)

    # Flat ground points at z=-1.73 and obstacle points at z=0.5
    pts = np.array([
        [5.0, 0.0, -1.73],
        [6.0, 1.0, -1.73],
        [5.0, 0.0, 0.5],
    ], dtype=np.float32)

    ground_mask, local_gz, delta_z = detector.segment_ground(pts)
    assert ground_mask[0] == True
    assert ground_mask[1] == True
    assert ground_mask[2] == False


def test_pointcloud_processor_filters_vertical_fov():
    cfg = {
        "sensor": {
            "min_range": 0.5,
            "max_range": 100.0,
            "min_z": -10.0,
            "max_z": 10.0,
            "hfov_deg": [-180.0, 180.0],
            "vfov_deg": [-10.0, 10.0],
        }
    }
    processor = PointCloudProcessor(cfg)
    frame = PointCloudFrame(
        frame_id=0,
        timestamp=0.0,
        points=np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 3.0]], dtype=np.float32),
    )

    filtered = processor.process(frame)

    assert len(filtered.points) == 1
    assert filtered.points[0, 2] == 0.0
